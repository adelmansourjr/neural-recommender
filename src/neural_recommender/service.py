"""Validated inference across explicit legacy and atomic model-bundle formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .als import ALSConfig, als_scores
from .baselines import discovery_scores, ease_scores, fuse_scores, knn_scores
from .bundles import file_hash, resolve_bundle, verify_manifest
from .conditional import combine_scores, history_bucket, validate_selector
from .data import load_dataset
from .diversity import rerank, validate_weight
from .metrics import score_items
from .model import HistoryRecommender

MODES = frozenset(
    {
        "accuracy",
        "adaptive",
        "discovery",
        "blend",
        "hybrid",
        "neural",
        "ease",
        "item_knn",
        "popularity",
        "als",
    }
)
COMPONENTS = frozenset(
    {
        "ease",
        "item_knn",
        "popularity",
        "neural_ensemble",
        "previous_hybrid",
        "als",
    }
)


class RecommenderService:
    """Load a complete model generation once; requests use immutable in-memory state.

    Formats 1 and 2 remain readable as historical artifacts. Format 3 resolves an
    atomic active pointer and adds ALS plus a validation-fitted history selector.
    A process restart is required to load a newly published generation, ensuring
    an individual request never mixes artifacts from two different generations.
    """

    def __init__(self, data_path: Path, artifact_dir: Path):
        self.bundle_dir = resolve_bundle(artifact_dir)
        try:
            metadata = json.loads((self.bundle_dir / "metadata.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                "Model metadata is missing or malformed; train before serving"
            ) from error
        verify_manifest(self.bundle_dir, metadata)
        self.format_version = metadata["format_version"]
        if file_hash(data_path) != metadata.get("data_sha256"):
            raise ValueError("Dataset differs from the catalog used to train this checkpoint")
        self.catalog, _ = load_dataset(data_path)
        if self.catalog.ids.tolist() != metadata.get("movie_ids"):
            raise ValueError("Movie ordering does not match checkpoint metadata")
        config = metadata.get("config", {})
        if not isinstance(config, dict):
            raise ValueError("Checkpoint config must be a JSON object")
        dimension = config.get("dimension")
        self.max_history = config.get("history_size")
        if type(dimension) is not int or not 2 <= dimension <= 256:
            raise ValueError("Checkpoint neural dimension is invalid")
        if type(self.max_history) is not int or not 1 <= self.max_history <= 20:
            raise ValueError("Checkpoint history size is invalid")
        if self.format_version in {2, 3}:
            weights = metadata.get("fusion_weights")
            if (
                not isinstance(weights, list)
                or len(weights) != 3
                or any(type(weight) not in {int, float} for weight in weights)
                or not np.isfinite(weights).all()
                or min(weights) < 0
                or not np.isclose(sum(weights), 1)
            ):
                raise ValueError("Checkpoint fusion weights must be three finite simplex weights")
            penalty = metadata.get("discovery_penalty")
            if (
                type(penalty) not in {int, float}
                or not np.isfinite(penalty)
                or not 0 <= penalty <= 1
            ):
                raise ValueError("Checkpoint discovery penalty must be finite and in [0, 1]")
            if self.format_version == 2 and metadata.get("accuracy_model", "hybrid") not in {
                "hybrid",
                "neural",
                "ease",
                "item_knn",
                "popularity",
            }:
                raise ValueError("Checkpoint accuracy model is unavailable in format 2")
        self.models = []
        names = (
            ["model.pt"]
            if self.format_version == 1
            else [f"neural_{index}.pt" for index in range(len(metadata["seeds"]))]
        )
        for name in names:
            model = HistoryRecommender(torch.from_numpy(self.catalog.genres), dimension)
            state = torch.load(self.bundle_dir / name, map_location="cpu", weights_only=True)
            # The serialized feature buffer must not replace the verified catalog.
            if not isinstance(state, dict) or not isinstance(state.get("genres"), torch.Tensor):
                raise ValueError("Neural checkpoint is missing its catalog feature buffer")
            if not torch.equal(state["genres"], model.genres):
                raise ValueError("Neural checkpoint genre features differ from the catalog")
            try:
                model.load_state_dict(state)
            except (RuntimeError, TypeError) as error:
                raise ValueError("Neural checkpoint dimensions or fields are invalid") from error
            if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
                raise ValueError("Neural checkpoint contains nonfinite parameters or buffers")
            model.eval()
            self.models.append(model)
        self.model = self.models[0]  # Compatibility for benchmark helpers using the first member.
        with torch.inference_mode():
            item_indices = torch.arange(1, len(self.catalog.ids) + 1)
            self.vectors = [model.encode_items(item_indices).numpy() for model in self.models]
        if any(not np.isfinite(vectors).all() for vectors in self.vectors):
            raise ValueError("Neural checkpoint produces nonfinite item encodings")
        self.metadata = metadata
        self.rankers = None
        self.als_factors = None
        self.als_gram = None
        if self.format_version in {2, 3}:
            self.rankers = self._load_rankers()
        if self.format_version == 3:
            with np.load(self.bundle_dir / "als.npz", allow_pickle=False) as archive:
                if set(archive.files) != {"item_factors"}:
                    raise ValueError("ALS artifact has unexpected fields")
                self.als_factors = archive["item_factors"]
            raw_config = metadata.get("als_config")
            if not isinstance(raw_config, dict):
                raise ValueError("ALS checkpoint config must be a JSON object")
            try:
                als_config = ALSConfig(**raw_config)
            except TypeError as error:
                raise ValueError("ALS checkpoint config contains unsupported fields") from error
            als_config.validate()
            if self.als_factors.shape != (len(self.catalog.ids), als_config.factors):
                raise ValueError("ALS factors differ from the catalog/configuration")
            if self.als_factors.dtype.kind != "f" or not np.isfinite(self.als_factors).all():
                raise ValueError("ALS factors must be finite real floating-point values")
            self.als_config = als_config
            factors = self.als_factors.astype(np.float64)
            self.als_gram = np.einsum("if,ig->fg", factors, factors)
            if not np.isfinite(self.als_gram).all():
                raise ValueError("ALS factors overflow their cached Gram matrix")
            validate_selector(metadata.get("history_selector"), set(COMPONENTS))
            validate_selector(metadata.get("history_blends"), set(COMPONENTS))
        self.lookup = {int(identifier): index for index, identifier in enumerate(self.catalog.ids)}

    def _load_rankers(self) -> dict[str, np.ndarray]:
        """Check every shape and index before exposing a dense or sparse ranker."""
        with np.load(self.bundle_dir / "rankers.npz", allow_pickle=False) as archive:
            if set(archive.files) != {"ease", "knn_indices", "knn_values", "popularity"}:
                raise ValueError("Ranker archive has missing or unexpected fields")
            rankers = {name: archive[name] for name in archive.files}
        count = len(self.catalog.ids)
        indices, values = rankers["knn_indices"], rankers["knn_values"]
        if rankers["ease"].shape != (count, count) or rankers["popularity"].shape != (count,):
            raise ValueError("Ranker dimensions differ from catalog")
        if indices.ndim != 2 or indices.shape != values.shape or indices.shape[0] != count:
            raise ValueError("kNN neighbor/value dimensions differ from catalog")
        if indices.dtype.kind not in {"i", "u"} or np.any(indices < 0) or np.any(indices >= count):
            raise ValueError("kNN contains invalid catalog indices")
        if any(array.dtype.kind not in {"f", "i", "u"} for array in rankers.values()):
            raise ValueError("Ranker arrays must contain real numbers")
        if any(not np.isfinite(array).all() for array in rankers.values()):
            raise ValueError("Ranker arrays must be finite")
        if np.any(rankers["popularity"] < 0) or np.any(values < 0):
            raise ValueError("Popularity and kNN weights must be nonnegative")
        if np.any(np.diag(rankers["ease"]) != 0):
            raise ValueError("EASE self-reconstruction weights must be zero")
        return rankers

    def selection(self, mode: str, history_length: int) -> dict[str, Any]:
        """Describe the actual ranker selected without reading or scoring any film."""
        if mode not in MODES:
            raise ValueError("Unknown ranking mode")
        if type(history_length) is not int or not 1 <= history_length <= self.max_history:
            raise ValueError(f"Provide one to {self.max_history} liked films")
        if self.format_version == 1:
            if mode not in {"accuracy", "adaptive", "neural"}:
                raise ValueError(
                    "Format 1 only supports neural ranking; run optimize for baselines"
                )
            return {
                "selected_ranker": "neural",
                "history_bucket": None,
                "weights": {"neural_ensemble": 1.0},
            }
        if mode in {"accuracy", "adaptive"}:
            if self.format_version == 3:
                bucket = history_bucket(history_length)
                chosen = self.metadata["history_selector"][bucket]
                return {
                    "selected_ranker": chosen["name"],
                    "history_bucket": bucket,
                    "weights": dict(chosen["weights"]),
                }
            mode = self.metadata.get("accuracy_model", "hybrid")
        if mode == "blend":
            if self.format_version != 3:
                raise ValueError("History-specific blending requires format 3; run refine")
            bucket = history_bucket(history_length)
            chosen = self.metadata["history_blends"][bucket]
            return {
                "selected_ranker": chosen["name"],
                "history_bucket": bucket,
                "weights": dict(chosen["weights"]),
            }
        if mode == "als" and self.format_version != 3:
            raise ValueError("ALS is absent from this older bundle; run refine")
        component = {
            "neural": "neural_ensemble",
            "hybrid": "previous_hybrid",
            "discovery": "previous_hybrid",
        }.get(mode, mode)
        return {"selected_ranker": mode, "history_bucket": None, "weights": {component: 1.0}}

    @torch.inference_mode()
    def _component_scores(self, indices: np.ndarray, needed: set[str]) -> dict[str, np.ndarray]:
        """Run only components needed by the requested ranker or frozen blend."""
        output = {}
        if "previous_hybrid" in needed:
            needed = needed | {"ease", "neural_ensemble", "popularity"}
        if "neural_ensemble" in needed:
            tensor = torch.tensor(indices + 1).unsqueeze(0)
            output["neural_ensemble"] = np.mean(
                [
                    score_items(vectors, model.encode_history(tensor).numpy()[0])
                    for model, vectors in zip(self.models, self.vectors, strict=True)
                ],
                axis=0,
            )
        if self.rankers is not None:
            if "popularity" in needed:
                output["popularity"] = self.rankers["popularity"]
            if "ease" in needed:
                output["ease"] = ease_scores(self.rankers["ease"], indices)
            if "item_knn" in needed:
                output["item_knn"] = knn_scores(
                    self.rankers["knn_indices"], self.rankers["knn_values"], indices
                )
        if "als" in needed:
            output["als"] = als_scores(
                self.als_factors,
                indices,
                self.als_config.regularization,
                self.als_config.alpha,
                self.als_gram,
            )
        if "previous_hybrid" in needed:
            output["previous_hybrid"] = fuse_scores(
                output["ease"],
                output["neural_ensemble"],
                output["popularity"],
                tuple(self.metadata["fusion_weights"]),
            )
        return output

    def recommend(
        self,
        movie_ids: list[int],
        k: int = 10,
        mode: str = "accuracy",
        relevance_weight: float = 1.0,
    ) -> list[dict]:
        """Rank unseen catalog items; reject ambiguous or unsupported requests."""
        validate_weight(relevance_weight)
        if (
            not movie_ids
            or len(movie_ids) > self.max_history
            or type(k) is not int
            or not 1 <= k <= 50
        ):
            raise ValueError(
                f"Provide 1–{self.max_history} liked movies and request 1–50 recommendations"
            )
        if any(type(identifier) is not int for identifier in movie_ids):
            raise ValueError("Movie identifiers must be integers, not booleans or coerced strings")
        if len(set(movie_ids)) != len(movie_ids):
            raise ValueError("Liked movie identifiers must be unique")
        if any(identifier not in self.lookup for identifier in movie_ids):
            raise ValueError("Unknown movie identifier")
        indices = np.array([self.lookup[identifier] for identifier in movie_ids])
        choice = self.selection(mode, len(indices))
        components = self._component_scores(indices, set(choice["weights"]))
        scores = combine_scores(components, choice["weights"])
        if mode == "discovery":
            scores = discovery_scores(
                scores, self.rankers["popularity"], self.metadata["discovery_penalty"]
            )
        selected = rerank(scores, self.catalog.genres, indices, k, relevance_weight)
        return [
            {
                "movie_id": int(self.catalog.ids[index]),
                "title": self.catalog.titles[index],
                "score": round(float(scores[index]), 6),
                "genres": [
                    name
                    for name, active in zip(
                        self.catalog.genre_names, self.catalog.genres[index], strict=True
                    )
                    if active
                ],
            }
            for index in selected
        ]

    def search(self, query: str, limit: int = 15) -> list[dict]:
        """Bounded case-insensitive catalog search for the local demo."""
        if (
            type(limit) is not int
            or not 1 <= limit <= 100
            or not isinstance(query, str)
            or len(query) > 100
        ):
            raise ValueError("Search needs at most 100 characters and a limit from one to 100")
        return [
            {"movie_id": int(identifier), "title": title}
            for identifier, title in zip(self.catalog.ids, self.catalog.titles, strict=True)
            if query.lower() in title.lower()
        ][:limit]
