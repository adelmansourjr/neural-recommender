"""Predeclared reserve-user experiment with strong baselines and score fusion."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .baselines import (
    diversified_rank,
    ease_scores,
    fit_ease,
    fit_item_knn,
    fuse_scores,
    interaction_gram,
    interaction_matrix,
    knn_scores,
    novel_item_share,
)
from .bundles import atomic_savez, atomic_torch_save
from .data import Catalog, load_dataset, split_users, user_sequences
from .isolation import validate_membership
from .metrics import mean_interval, rank, ranking_metrics, score_items
from .model import HistoryRecommender
from .storage import write_json


@dataclass(frozen=True)
class OptimizationConfig:
    split_seed: int = 42
    reserve_seed: int = 20261021
    seeds: tuple[int, ...] = (7, 19, 42)
    train_users: int = 2500
    evaluation_users: int = 200
    epochs: int = 5
    dimension: int = 32
    threads: int = 2


def select_accuracy_model(validation_scores: dict[str, float]) -> str:
    """Select among every candidate on the preselected five-history validation task."""
    if not validation_scores or not all(np.isfinite(value) for value in validation_scores.values()):
        raise ValueError("Accuracy selection requires finite validation measurements")
    return max(validation_scores, key=validation_scores.get)


def prepare_protocol(
    ratings: np.ndarray,
    data_hash: str,
    pilot_metadata: Path,
    path: Path,
    config: OptimizationConfig,
) -> dict:
    if path.exists():
        existing = json.loads(path.read_text())
        if (
            not isinstance(existing, dict)
            or existing.get("data_sha256") != data_hash
            or existing.get("config") != json.loads(json.dumps(asdict(config)))
        ):
            raise ValueError("Existing protocol differs; choose a new protocol path deliberately")
    if not pilot_metadata.exists():
        raise ValueError("Preserved original pilot metadata is required to exclude inspected users")
    pilot = json.loads(pilot_metadata.read_text())
    if pilot["data_sha256"] != data_hash:
        raise ValueError("Original pilot and upgrade must use the same data snapshot")
    if pilot["config"]["seed"] != config.split_seed:
        raise ValueError("User split must stay fixed relative to the inspected original pilot")
    splits = split_users(ratings, config.split_seed)
    validate_membership(pilot["test_users"], splits["test"], "Inspected pilot test users")
    rng = np.random.default_rng(config.reserve_seed)
    reserved = np.setdiff1d(splits["test"], pilot["test_users"])
    if len(reserved) < config.evaluation_users:
        raise ValueError("Not enough previously unevaluated reserve test users")
    if config.train_users < 1 or config.evaluation_users < 1 or config.epochs < 1:
        raise ValueError("Experiment sizes and epochs must be positive")
    protocol = {
        "created_at": datetime.now(UTC).isoformat(),
        "data_sha256": data_hash,
        "config": asdict(config),
        "train_users": rng.permutation(splits["train"])[: config.train_users].tolist(),
        "validation_users": rng.permutation(splits["validation"])[
            : config.evaluation_users
        ].tolist(),
        "reserve_test_users": rng.permutation(reserved)[: config.evaluation_users].tolist(),
        "excluded_pilot_test_users": pilot["test_users"],
        "ease_regularization_grid": [50.0, 200.0, 1000.0],
        "knn_shrinkage_grid": [0.0, 50.0, 200.0],
        "knn_neighbors": 80,
        "fusion_grid": [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.75, 0.25, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.8, 0.1, 0.1],
            [0.6, 0.3, 0.1],
            [0.0, 0.75, 0.25],
        ],
        "novelty_penalty_grid": [0.0, 0.05, 0.1, 0.2, 0.3],
        "selection": (
            "Tune all settings on validation NDCG@10 with five liked films. "
            "For discovery select maximum validation catalog coverage subject to "
            "NDCG@10 >=98% of the selected accuracy model, breaking ties by NDCG. "
            "Fit three fixed neural seeds on the same users; select each epoch on validation. "
            "Freeze settings before a single reserve-user evaluation at histories 1/5/10."
        ),
        "disclosure": (
            "The initial benchmark was inspected. These reserve users were excluded "
            "from that pilot's test sample; this is not an unseen-dataset claim. "
            "The user split is not a global historical deployment simulation."
        ),
    }
    # Reconstruct every membership and candidate grid even for cached protocols.
    # A matching dataset/config does not establish that the saved user lists or
    # tuning grids have not changed since the experiment was declared.
    if path.exists():
        existing = json.loads(path.read_text())
        for field, partition in (
            ("train_users", "train"),
            ("validation_users", "validation"),
            ("reserve_test_users", "test"),
        ):
            validate_membership(existing.get(field), splits[partition], field)
        expected = json.loads(json.dumps(protocol))
        if {key: value for key, value in existing.items() if key != "created_at"} != {
            key: value for key, value in expected.items() if key != "created_at"
        }:
            raise ValueError("Existing protocol differs; choose a new protocol path deliberately")
        return existing
    write_json(path, protocol)
    return protocol


def cases(sequences: dict[int, np.ndarray], users: list[int], count: int) -> list[tuple]:
    return [
        (user, sequences[user][:count], set(map(int, sequences[user][count:])))
        for user in users
        if len(sequences[user]) > count
    ]


def score_report(
    scores: list[np.ndarray],
    query_cases: list[tuple],
    popularity: np.ndarray,
    novelty: float = 0.0,
) -> tuple[dict, np.ndarray]:
    measurements, catalog = [], set()
    for values, (_, history, relevant) in zip(scores, query_cases, strict=True):
        selected = diversified_rank(values, history, popularity, novelty)
        recall, ndcg = ranking_metrics(selected, relevant, 10)
        catalog.update(map(int, selected))
        measurements.append([recall, ndcg, novel_item_share(selected, popularity)])
    if not measurements:
        raise ValueError("No eligible query users")
    measured = np.asarray(measurements)
    return {
        "users": len(query_cases),
        "recall_at_10": mean_interval(measured[:, 0].tolist()),
        "ndcg_at_10": mean_interval(measured[:, 1].tolist()),
        "long_tail_share_at_10": mean_interval(measured[:, 2].tolist()),
        "catalog_coverage": len(catalog) / len(popularity),
    }, measured


def _selection_score(scores: list[np.ndarray], query_cases: list[tuple]) -> float:
    return float(
        np.mean(
            [
                ranking_metrics(rank(value, history, 10), relevant, 10)[1]
                for value, (_, history, relevant) in zip(scores, query_cases, strict=True)
            ]
        )
    )


@torch.inference_mode()
def neural_scores(
    model: HistoryRecommender, query_cases: list[tuple], item_count: int
) -> list[np.ndarray]:
    model.eval()
    vectors = model.encode_items(torch.arange(1, item_count + 1)).numpy()
    output = []
    for _, history, _ in query_cases:
        user = model.encode_history(torch.tensor(history + 1).unsqueeze(0)).numpy()[0]
        output.append(score_items(vectors, user))
    return output


def _neural_training_inputs(
    catalog: Catalog,
    sequences: dict[int, np.ndarray],
    observed: dict[int, set[int]],
    users: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Construct only training-owner prefixes and negative pools, without fitting."""
    histories, targets, owners = [], [], []
    allowed = {}
    for user in users:
        items = sequences[user]
        if len(items) < 2:
            continue
        # Include short prefixes explicitly: the original late-prefix sampler did
        # not match the cold-start query distribution.
        positions = set(position for position in (1, 5, 10) if position < len(items))
        remainder = np.setdiff1d(np.arange(1, len(items)), list(positions))
        positions.update(
            rng.choice(remainder, min(15 - len(positions), len(remainder)), replace=False)
        )
        allowed[user] = np.setdiff1d(np.arange(len(catalog.ids)), list(observed[user]))
        if not len(allowed[user]):
            raise ValueError("Training user has no unobserved candidates")
        for position in sorted(positions):
            prefix = items[max(0, position - 20) : position] + 1
            padded = np.zeros(20, dtype=np.int64)
            padded[: len(prefix)] = prefix
            histories.append(padded)
            targets.append(items[position])
            owners.append(user)
    x, y, owners = np.stack(histories), np.asarray(targets), np.asarray(owners)
    return x, y, owners, allowed


def _train_neural(
    catalog: Catalog,
    sequences: dict[int, np.ndarray],
    observed: dict[int, set[int]],
    users: list[int],
    validation: list[tuple],
    seed: int,
    config: OptimizationConfig,
) -> tuple[HistoryRecommender, list[dict]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x, y, owners, allowed = _neural_training_inputs(catalog, sequences, observed, users, rng)
    model = HistoryRecommender(torch.from_numpy(catalog.genres), config.dimension)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
    best, state, curve = -1.0, None, []
    for epoch in range(config.epochs):
        model.train()
        total = 0.0
        for batch in np.array_split(rng.permutation(len(y)), max(1, int(np.ceil(len(y) / 512)))):
            negatives = np.stack([rng.choice(allowed[int(owner)], 5) for owner in owners[batch]])
            candidates = np.column_stack([y[batch], negatives]) + 1
            prediction = model(torch.from_numpy(x[batch]), torch.from_numpy(candidates))
            loss = F.cross_entropy(prediction, torch.zeros(len(batch), dtype=torch.long))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite neural loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.item()) * len(batch)
        score = _selection_score(neural_scores(model, validation, len(catalog.ids)), validation)
        curve.append({"epoch": epoch + 1, "loss": total / len(y), "validation_ndcg_at_10": score})
        print(json.dumps({"seed": seed, **curve[-1]}), flush=True)
        if score > best:
            best, state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return model, curve


def optimize(
    data_path: Path,
    artifact_dir: Path,
    report_path: Path,
    protocol_path: Path,
    pilot_metadata: Path,
    config: OptimizationConfig | None = None,
) -> dict:
    config = config or OptimizationConfig()
    if (artifact_dir / "active.json").exists():
        raise ValueError("Versioned bundles exist; use a separate directory for legacy optimize")
    start = time.perf_counter()
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    catalog, ratings = load_dataset(data_path)
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    protocol = prepare_protocol(ratings, digest, pilot_metadata, protocol_path, config)
    sequences, observed = user_sequences(ratings)
    validation = cases(sequences, protocol["validation_users"], 5)
    matrix = interaction_matrix(sequences, np.asarray(protocol["train_users"]), len(catalog.ids))
    popularity = matrix.sum(axis=0)
    gram = interaction_gram(matrix)
    del matrix
    search = {"ease": [], "item_knn": [], "fusion": [], "discovery": []}
    best, ease, selected_ridge = -1.0, None, None
    for regularization in protocol["ease_regularization_grid"]:
        coefficients = fit_ease(gram, regularization)
        values = [ease_scores(coefficients, history) for _, history, _ in validation]
        score = _selection_score(values, validation)
        search["ease"].append({"regularization": regularization, "validation_ndcg_at_10": score})
        print(
            json.dumps({"ease_ridge": regularization, "validation_ndcg_at_10": score}), flush=True
        )
        if score > best:
            best, ease, selected_ridge = score, coefficients, regularization
    best, knn, selected_shrinkage = -1.0, None, None
    for shrinkage in protocol["knn_shrinkage_grid"]:
        neighbors = fit_item_knn(gram, shrinkage, min(protocol["knn_neighbors"], len(gram) - 1))
        values = [knn_scores(*neighbors, history) for _, history, _ in validation]
        score = _selection_score(values, validation)
        search["item_knn"].append({"shrinkage": shrinkage, "validation_ndcg_at_10": score})
        if score > best:
            best, knn, selected_shrinkage = score, neighbors, shrinkage
    del gram
    models, curves = [], {}
    for seed in config.seeds:
        model, curve = _train_neural(
            catalog, sequences, observed, protocol["train_users"], validation, seed, config
        )
        models.append(model)
        curves[str(seed)] = curve
    validation_neural = [neural_scores(model, validation, len(catalog.ids)) for model in models]
    mean_neural = np.mean(validation_neural, axis=0)
    validation_ease = [ease_scores(ease, history) for _, history, _ in validation]
    best, weights, hybrid_validation = -1.0, None, None
    for candidate in protocol["fusion_grid"]:
        combined = [
            fuse_scores(a, b, popularity, tuple(candidate))
            for a, b in zip(validation_ease, mean_neural, strict=True)
        ]
        score = _selection_score(combined, validation)
        search["fusion"].append({"weights": candidate, "validation_ndcg_at_10": score})
        if score > best:
            best, weights, hybrid_validation = score, candidate, combined
    discovery_weight, best_coverage, best_discovery_ndcg = 0.0, -1.0, -1.0
    for penalty in protocol["novelty_penalty_grid"]:
        result, _ = score_report(hybrid_validation, validation, popularity, penalty)
        score, coverage = result["ndcg_at_10"]["mean"], result["catalog_coverage"]
        eligible = score >= 0.98 * best
        search["discovery"].append(
            {
                "penalty": penalty,
                "validation_ndcg_at_10": score,
                "catalog_coverage": coverage,
                "eligible": eligible,
            }
        )
        if eligible and (coverage, score) > (best_coverage, best_discovery_ndcg):
            discovery_weight, best_coverage, best_discovery_ndcg = penalty, coverage, score
    candidate_scores = {
        "popularity": _selection_score([popularity] * len(validation), validation),
        "ease": max(row["validation_ndcg_at_10"] for row in search["ease"]),
        "item_knn": max(row["validation_ndcg_at_10"] for row in search["item_knn"]),
        "neural": _selection_score(list(mean_neural), validation),
        "hybrid": best,
    }
    accuracy_model = select_accuracy_model(candidate_scores)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    atomic_savez(
        artifact_dir / "rankers.npz",
        ease=ease,
        knn_indices=knn[0],
        knn_values=knn[1],
        popularity=popularity,
    )
    for index, model in enumerate(models):
        atomic_torch_save(model.state_dict(), artifact_dir / f"neural_{index}.pt")
    metadata = {
        "format_version": 2,
        "data_sha256": digest,
        "movie_ids": catalog.ids.tolist(),
        "config": {"dimension": config.dimension, "history_size": 20},
        "seeds": list(config.seeds),
        "fusion_weights": weights,
        "discovery_penalty": discovery_weight,
        "accuracy_model": accuracy_model,
        "selected_ease_regularization": selected_ridge,
        "selected_knn_shrinkage": selected_shrinkage,
        "artifact_hashes": {
            name: hashlib.sha256((artifact_dir / name).read_bytes()).hexdigest()
            for name in ["rankers.npz"] + [f"neural_{i}.pt" for i in range(len(models))]
        },
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "selection": (
            "Validation selected EASE/neural/log-popularity fusion; disclosure in benchmark."
        ),
    }
    # Persist selected settings before evaluating the reserved users.
    write_json(artifact_dir / "metadata.json", metadata)
    results = {}
    for count in (1, 5, 10):
        query_cases = cases(sequences, protocol["reserve_test_users"], count)
        individual = [neural_scores(model, query_cases, len(catalog.ids)) for model in models]
        ensemble = np.mean(individual, axis=0)
        ease_values = [ease_scores(ease, history) for _, history, _ in query_cases]
        scores = {
            "popularity": [popularity] * len(query_cases),
            "item_knn": [knn_scores(*knn, history) for _, history, _ in query_cases],
            "ease": ease_values,
            "neural_ensemble": list(ensemble),
            "hybrid": [
                fuse_scores(a, b, popularity, tuple(weights))
                for a, b in zip(ease_values, ensemble, strict=True)
            ],
        }
        reports, measurements = {}, {}
        for name, values in scores.items():
            reports[name], measurements[name] = score_report(values, query_cases, popularity)
        reports["discovery"], measurements["discovery"] = score_report(
            scores["hybrid"], query_cases, popularity, discovery_weight
        )
        reports["neural_seed_results"] = {
            str(seed): score_report(values, query_cases, popularity)[0]
            for seed, values in zip(config.seeds, individual, strict=True)
        }
        for candidate, reference in (
            ("hybrid", "popularity"),
            ("hybrid", "ease"),
            ("neural_ensemble", "popularity"),
            ("item_knn", "popularity"),
        ):
            difference = measurements[candidate] - measurements[reference]
            reports[f"paired_{candidate}_minus_{reference}"] = {
                "ndcg_at_10": mean_interval(difference[:, 1].tolist()),
                "recall_at_10": mean_interval(difference[:, 0].tolist()),
                "interpretation": (
                    "Positive favors candidate; paired user bootstrap, conditional on models."
                ),
            }
        results[str(count)] = reports
    report = {
        "protocol": protocol,
        "validation_search": search,
        "neural_learning_curves": curves,
        "selected": {
            key: metadata[key]
            for key in (
                "fusion_weights",
                "discovery_penalty",
                "selected_ease_regularization",
                "selected_knn_shrinkage",
                "seeds",
                "accuracy_model",
            )
        },
        "serving_model_selection": {
            "validation_ndcg_at_10": candidate_scores,
            "winner": accuracy_model,
            "criterion": "Highest validation NDCG@10 at five likes among all evaluated candidates.",
            "timing": "Computed before reserve-user evaluation in this run.",
        },
        "reserve_test": results,
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "runtime": {"numpy": np.__version__, "torch": torch.__version__},
        "limitations": [
            "Original pilot test was already inspected; reserve protocol excludes those users.",
            "Every model uses the same user split; neural seeds do not estimate split uncertainty.",
            "Strong non-neural baselines are retained; selection may favor EASE-only fusion.",
            "Observed ratings are selection-biased; offline ranking is not online engagement.",
            "All catalog metadata is available; no item-cold-start claim is made.",
            "Reported paired comparisons are descriptive and not multiplicity-adjusted.",
        ],
    }
    write_json(report_path, report)
    return report
