"""CPU training, held-out user selection, and independent cold-start testing."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .bundles import atomic_torch_save
from .data import Catalog, load_dataset, split_users, training_examples, user_sequences
from .metrics import mean_interval, rank, ranking_metrics, score_items
from .model import HistoryRecommender
from .storage import write_json


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    epochs: int = 5
    dimension: int = 32
    batch_size: int = 512
    learning_rate: float = 0.003
    negatives: int = 5
    max_train_users: int = 2500
    evaluation_users: int = 250
    history_size: int = 20
    threads: int = 2

    def validate(self) -> None:
        for name in (
            "epochs",
            "dimension",
            "batch_size",
            "negatives",
            "max_train_users",
            "evaluation_users",
            "history_size",
            "threads",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be between zero and one")


@torch.inference_mode()
def evaluate_users(
    model: HistoryRecommender,
    catalog: Catalog,
    sequences: dict[int, np.ndarray],
    users: np.ndarray,
    popularity: np.ndarray,
    history_counts: tuple[int, ...] = (1, 5, 10),
    k: int = 10,
    seed: int = 42,
) -> dict:
    model.eval()
    item_vectors = model.encode_items(torch.arange(1, len(catalog.ids) + 1)).numpy()
    normalized_genres = catalog.genres / np.maximum(
        np.linalg.norm(catalog.genres, axis=1, keepdims=True), 1
    )
    output = {}
    for count in history_counts:
        values = {
            name: {"recall": [], "ndcg": [], "catalog": set()}
            for name in ("popularity", "genre", "neural")
        }
        for user in users:
            items = sequences[int(user)]
            if len(items) <= count:
                continue
            history = items[:count]
            relevant = set(map(int, items[count:]))
            encoded = model.encode_history(torch.tensor(history + 1).unsqueeze(0)).numpy()[0]
            scores = {
                "popularity": popularity,
                "genre": score_items(normalized_genres, normalized_genres[history].mean(axis=0)),
                "neural": score_items(item_vectors, encoded),
            }
            for name, score in scores.items():
                recommendations = rank(score, history, k)
                recall, ndcg = ranking_metrics(recommendations, relevant, k)
                values[name]["recall"].append(recall)
                values[name]["ndcg"].append(ndcg)
                values[name]["catalog"].update(map(int, recommendations))
        if not values["neural"]["recall"]:
            raise ValueError(f"No held-out users have more than {count} liked items")
        output[str(count)] = {
            name: {
                "users": len(metrics["recall"]),
                f"recall_at_{k}": mean_interval(metrics["recall"], seed),
                f"ndcg_at_{k}": mean_interval(metrics["ndcg"], seed),
                "catalog_coverage": len(metrics["catalog"]) / len(catalog.ids),
            }
            for name, metrics in values.items()
        }
        output[str(count)]["paired_neural_minus_popularity"] = {
            f"recall_at_{k}": mean_interval(
                (np.array(values["neural"]["recall"]) - values["popularity"]["recall"]).tolist(),
                seed,
            ),
            f"ndcg_at_{k}": mean_interval(
                (np.array(values["neural"]["ndcg"]) - values["popularity"]["ndcg"]).tolist(), seed
            ),
            "interpretation": "Positive favors neural; bootstrap resamples paired users.",
        }
    return output


def train(data_path: Path, artifact_dir: Path, report_path: Path, config: TrainConfig) -> dict:
    config.validate()
    if (artifact_dir / "active.json").exists():
        raise ValueError("Versioned bundles exist; use a separate directory for a legacy train run")
    start = time.perf_counter()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    catalog, ratings = load_dataset(data_path)
    splits = split_users(ratings, config.seed)
    rng = np.random.default_rng(config.seed)
    users = rng.permutation(splits["train"])[: config.max_train_users]
    validation_users = rng.permutation(splits["validation"])[: config.evaluation_users]
    test_users = rng.permutation(splits["test"])[: config.evaluation_users]
    sequences, observed = user_sequences(ratings)
    histories, targets, owners = training_examples(sequences, users, config.history_size)
    popularity = np.zeros(len(catalog.ids), dtype=np.float32)
    for user in users:
        np.add.at(popularity, sequences[int(user)], 1)
    model = HistoryRecommender(torch.from_numpy(catalog.genres), config.dimension)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0001)
    best_score, best_state, learning_curve = -1.0, None, []
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for positions in np.array_split(
            rng.permutation(len(targets)), max(1, int(np.ceil(len(targets) / config.batch_size)))
        ):
            candidates = np.empty((len(positions), config.negatives + 1), dtype=np.int64)
            candidates[:, 0] = targets[positions]
            for row, owner in enumerate(owners[positions]):
                forbidden = observed[int(owner)]
                available = len(catalog.ids) - len(forbidden)
                if available < 1:
                    raise ValueError("A training user has no unobserved negative candidates")
                for column in range(1, config.negatives + 1):
                    while True:
                        item = int(rng.integers(len(catalog.ids)))
                        if item not in forbidden:
                            candidates[row, column] = item
                            break
            logits = model(torch.from_numpy(histories[positions]), torch.from_numpy(candidates + 1))
            loss = F.cross_entropy(logits, torch.zeros(len(positions), dtype=torch.long))
            if not torch.isfinite(loss):
                raise ValueError("Training loss is nonfinite; refusing to save a model")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total_loss += loss.item() * len(positions)
        validation = evaluate_users(
            model, catalog, sequences, validation_users, popularity, (5,), seed=config.seed
        )
        score = validation["5"]["neural"]["ndcg_at_10"]["mean"]
        learning_curve.append(
            {"epoch": epoch + 1, "loss": total_loss / len(targets), "validation_ndcg_at_10": score}
        )
        print(json.dumps(learning_curve[-1]), flush=True)
        if score > best_score:
            best_score, best_state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    test = evaluate_users(model, catalog, sequences, test_users, popularity, seed=config.seed)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(model.state_dict(), artifact_dir / "model.pt")
    np.save(artifact_dir / "popularity.npy", popularity, allow_pickle=False)
    metadata = {
        "format_version": 1,
        "config": asdict(config),
        "movie_ids": catalog.ids.tolist(),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256((artifact_dir / "model.pt").read_bytes()).hexdigest(),
        "train_users": list(map(int, users)),
        "validation_users": list(map(int, validation_users)),
        "test_users": list(map(int, test_users)),
    }
    write_json(artifact_dir / "metadata.json", metadata)
    report = {
        "config": asdict(config),
        "training_examples": len(targets),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "protocol": (
            "User-disjoint splits; prefixes of >=4-star ratings; full catalog ranking; "
            "unobserved candidates are not known dislikes."
        ),
        "tie_policy": "Timestamp ties are ordered by item index; no within-session ordering claim.",
        "selection": (
            "Best validation NDCG@10 at five ratings; test users evaluated after selection."
        ),
        "data_sha256": metadata["data_sha256"],
        "learning_curve": learning_curve,
        "runtime": {"numpy": np.__version__, "torch": torch.__version__},
        "limitations": [
            "User-disjoint offline split, not a global chronological deployment simulation.",
            "Catalog metadata is available for all items; this is not item cold start.",
            "Random unobserved negatives may include unlabeled positives.",
            "Training-only users' full observed histories exclude negative candidates.",
            "One seed and a limited user sample are a pilot rather than a leaderboard result.",
        ],
        "test": test,
        "elapsed_seconds": round(time.perf_counter() - start, 2),
    }
    write_json(report_path, report)
    return report
