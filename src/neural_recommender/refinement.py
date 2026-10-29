"""A frozen ALS and history-conditional selection experiment on new reserve users.

All training/validation memberships remain the same as the preceding upgrade.
The new reserve sample excludes every previously scored test user. Its protocol
is recorded before fitting candidates; every selection uses validation only.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .als import ALSConfig, als_scores, fit_implicit_als
from .baselines import ease_scores, fuse_scores, interaction_matrix, knn_scores
from .bundles import file_hash, publish_bundle
from .conditional import combine_scores
from .data import load_dataset, split_users, user_sequences
from .isolation import validate_membership
from .metrics import mean_interval
from .optimization import (
    OptimizationConfig,
    _selection_score,
    cases,
    neural_scores,
    prepare_protocol,
    score_report,
)
from .service import RecommenderService
from .storage import write_json

ALS_GRID = (
    ALSConfig(factors=16, regularization=10, alpha=10),
    ALSConfig(factors=32, regularization=10, alpha=10),
    ALSConfig(factors=32, regularization=10, alpha=40),
    ALSConfig(factors=32, regularization=1, alpha=40),
)

CANDIDATES = {
    "popularity": {"popularity": 1.0},
    "item_knn": {"item_knn": 1.0},
    "ease": {"ease": 1.0},
    "neural_ensemble": {"neural_ensemble": 1.0},
    "als": {"als": 1.0},
    "previous_hybrid": {"previous_hybrid": 1.0},
    "knn_ease_equal": {"item_knn": 0.5, "ease": 0.5},
    "knn_neural": {"item_knn": 0.75, "neural_ensemble": 0.25},
    "knn_als": {"item_knn": 0.75, "als": 0.25},
    "ease_als": {"ease": 0.75, "als": 0.25},
    "neural_als": {"neural_ensemble": 0.75, "als": 0.25},
    "balanced": {"item_knn": 0.4, "ease": 0.3, "neural_ensemble": 0.2, "als": 0.1},
}


def prepare_refinement_protocol(
    ratings: np.ndarray,
    data_digest: str,
    upgrade_protocol_path: Path,
    pilot_metadata_path: Path,
    output: Path,
    reserve_users: int = 200,
) -> dict:
    """Freeze membership and grids before any new candidate or reserve score."""
    previous = json.loads(upgrade_protocol_path.read_text())
    pilot = json.loads(pilot_metadata_path.read_text())
    if data_digest != previous["data_sha256"] or data_digest != pilot["data_sha256"]:
        raise ValueError("Refinement and earlier protocols must use the same dataset snapshot")
    if reserve_users < 1:
        raise ValueError("Reserve user count must be positive")
    # Validate the entire inherited contract before propagating its memberships.
    # This only reads an existing upgrade protocol; it does not refit anything.
    prepare_protocol(
        ratings,
        data_digest,
        pilot_metadata_path,
        upgrade_protocol_path,
        OptimizationConfig(**previous["config"]),
    )
    excluded = set(previous["reserve_test_users"]) | set(pilot["test_users"])
    splits = split_users(ratings, 42)
    if not set(previous["train_users"]).issubset(splits["train"]):
        raise ValueError("Earlier training users do not match the fixed training split")
    if not set(previous["validation_users"]).issubset(splits["validation"]):
        raise ValueError("Earlier validation users do not match the fixed validation split")
    if not excluded.issubset(splits["test"]):
        raise ValueError("Earlier inspected users do not match the fixed test split")
    available = np.setdiff1d(splits["test"], list(excluded))
    expected = {
        "data_sha256": data_digest,
        "train_users": previous["train_users"],
        "validation_users": previous["validation_users"],
        "excluded_previously_inspected_test_users": sorted(excluded),
        "available_reserve_users": len(available),
        "reserve_test_users": np.random.default_rng(20261022)
        .permutation(available)[:reserve_users]
        .tolist(),
        "requested_reserve_users": reserve_users,
        "als_grid": [asdict(config) for config in ALS_GRID],
        "candidate_blends": CANDIDATES,
        "selector_rule": "Nearest history bucket among 1/5/10; ties select the lower bucket.",
        "primary_history": 5,
        "sensitivity_histories": [1, 10],
        "selection_rule": (
            "Select ALS on five-like validation NDCG@10. Then select each history bucket's "
            "highest validation NDCG@10 candidate from the complete frozen grid. "
            "No reserve data may influence fitting, epoch selection, or blend weights."
        ),
        "discovery_penalty": previous["novelty_penalty_grid"][-1],
        "limitations": (
            "Prior reports were inspected, and validation is reused. New reserve users are "
            "disjoint from all prior scored test samples, not an unseen-dataset certification. "
            "Offline ratings remain selection-biased; conditional selection can overfit validation."
        ),
    }
    if len(expected["reserve_test_users"]) < reserve_users:
        raise ValueError("Insufficient uninspected reserve users for the requested protocol")
    if output.exists():
        recorded = json.loads(output.read_text())
        for field, partition in (
            ("train_users", "train"),
            ("validation_users", "validation"),
            ("reserve_test_users", "test"),
        ):
            validate_membership(recorded.get(field), splits[partition], field)
        comparable = {name: value for name, value in recorded.items() if name != "created_at"}
        if comparable != expected:
            raise ValueError(
                "Refinement protocol changed; deliberately create a new experiment path"
            )
        return recorded
    protocol = {"created_at": datetime.now(UTC).isoformat(), **expected}
    write_json(output, protocol)
    return protocol


def _fixed_scores(
    service: RecommenderService, query_cases: list[tuple]
) -> dict[str, list[np.ndarray]]:
    """Cache fixed preceding models, avoiding training or refitting on validation."""
    rankers = service.rankers
    if rankers is None:
        raise ValueError("Refinement requires a previously optimized format 2 or 3 bundle")
    neural = np.mean(
        [neural_scores(model, query_cases, len(service.catalog.ids)) for model in service.models],
        axis=0,
    )
    ease = [ease_scores(rankers["ease"], history) for _, history, _ in query_cases]
    previous = [
        fuse_scores(left, right, rankers["popularity"], tuple(service.metadata["fusion_weights"]))
        for left, right in zip(ease, neural, strict=True)
    ]
    return {
        "popularity": [rankers["popularity"]] * len(query_cases),
        "item_knn": [
            knn_scores(rankers["knn_indices"], rankers["knn_values"], history)
            for _, history, _ in query_cases
        ],
        "ease": ease,
        "neural_ensemble": list(neural),
        "previous_hybrid": previous,
    }


def _candidate_scores(
    components: dict[str, list[np.ndarray]], weights: dict[str, float]
) -> list[np.ndarray]:
    count = len(next(iter(components.values())))
    return [
        combine_scores({name: values[index] for name, values in components.items()}, weights)
        for index in range(count)
    ]


def refine(
    data_path: Path = Path("data/ml-1m.zip"),
    artifact_dir: Path = Path("artifacts"),
    protocol_path: Path = Path("reports/refinement_protocol.json"),
    report_path: Path = Path("reports/refined_benchmark.json"),
    upgrade_protocol_path: Path = Path("reports/upgrade_protocol.json"),
    pilot_metadata_path: Path = Path("reports/initial_pilot_metadata.json"),
    reserve_users: int = 200,
) -> dict:
    """Fit bounded new candidates and publish only a complete validated bundle."""
    start = time.perf_counter()
    torch.set_num_threads(2)
    catalog, ratings = load_dataset(data_path)
    protocol = prepare_refinement_protocol(
        ratings,
        file_hash(data_path),
        upgrade_protocol_path,
        pilot_metadata_path,
        protocol_path,
        reserve_users,
    )
    service = RecommenderService(data_path, artifact_dir)
    # Use the exact generation loaded into memory, even if another completed
    # experiment activates a different pointer while this experiment runs.
    source = service.bundle_dir
    source_metadata = service.metadata
    if source_metadata.get("protocol_sha256") != file_hash(upgrade_protocol_path):
        raise ValueError("Baseline bundle does not match the preserved upgrade protocol")
    if list(service.metadata.get("seeds", [])) != [7, 19, 42]:
        raise ValueError("Refinement baseline must use the recorded three-seed upgrade ensemble")
    sequences, _ = user_sequences(ratings)
    validation_cases = {
        str(count): cases(sequences, protocol["validation_users"], count) for count in (1, 5, 10)
    }
    fixed = {
        bucket: _fixed_scores(service, queries) for bucket, queries in validation_cases.items()
    }
    interactions = interaction_matrix(
        sequences, np.asarray(protocol["train_users"]), len(catalog.ids)
    )
    als_search, best_score, selected_factors, selected_config = [], -1.0, None, None
    for entry in protocol["als_grid"]:
        config = ALSConfig(**entry)
        factors, details = fit_implicit_als(interactions, config)
        scores = [
            als_scores(factors, history, config.regularization, config.alpha)
            for _, history, _ in validation_cases["5"]
        ]
        score = _selection_score(scores, validation_cases["5"])
        als_search.append({**details, "validation_ndcg_at_10": score})
        print(json.dumps({"als": entry, "validation_ndcg_at_10": score}), flush=True)
        if score > best_score:
            best_score, selected_factors, selected_config = score, factors, config
    del interactions
    for bucket, queries in validation_cases.items():
        fixed[bucket]["als"] = [
            als_scores(
                selected_factors, history, selected_config.regularization, selected_config.alpha
            )
            for _, history, _ in queries
        ]
    selectors, best_blends, validation_search = {}, {}, {}
    for bucket, queries in validation_cases.items():
        measurements = {}
        for name, weights in protocol["candidate_blends"].items():
            scores = _candidate_scores(fixed[bucket], weights)
            measurements[name] = _selection_score(scores, queries)
        winner = max(measurements, key=measurements.get)
        blends = [
            name for name, weights in protocol["candidate_blends"].items() if len(weights) > 1
        ]
        blend_winner = max(blends, key=measurements.get)
        selectors[bucket] = {"name": winner, "weights": protocol["candidate_blends"][winner]}
        best_blends[bucket] = {
            "name": blend_winner,
            "weights": protocol["candidate_blends"][blend_winner],
        }
        validation_search[bucket] = measurements
    metadata = {
        **{
            key: value
            for key, value in source_metadata.items()
            if key not in {"artifact_hashes", "accuracy_model"}
        },
        "format_version": 3,
        "history_selector": selectors,
        "history_blends": best_blends,
        "als_config": asdict(selected_config),
        "source_bundle": str(source.relative_to(artifact_dir))
        if source != artifact_dir
        else "legacy-root",
        "refinement_protocol_sha256": file_hash(protocol_path),
        "selector_rule": protocol["selector_rule"],
        "selection_note": (
            "History-conditional choices use validation only; "
            "old reports and bundle remain available."
        ),
    }
    # Freeze every selected setting before any access to the new reserve users.
    write_json(
        report_path.with_name("refinement_selection.json"),
        {
            "created_at": datetime.now(UTC).isoformat(),
            "als_config": asdict(selected_config),
            "history_selector": selectors,
            "history_blends": best_blends,
            "validation_search": validation_search,
            "protocol_sha256": file_hash(protocol_path),
        },
    )
    results = {}
    for count in (1, 5, 10):
        bucket = str(count)
        query_cases = cases(sequences, protocol["reserve_test_users"], count)
        components = _fixed_scores(service, query_cases)
        components["als"] = [
            als_scores(
                selected_factors, history, selected_config.regularization, selected_config.alpha
            )
            for _, history, _ in query_cases
        ]
        reports, individual = {}, {}
        candidate_scores = {
            name: _candidate_scores(components, weights)
            for name, weights in protocol["candidate_blends"].items()
        }
        for name, scores in candidate_scores.items():
            reports[name], individual[name] = score_report(
                scores, query_cases, service.rankers["popularity"]
            )
        winner = selectors[bucket]["name"]
        reports["adaptive"] = {**reports[winner], "selected_ranker": winner}
        individual["adaptive"] = individual[winner]
        for candidate, reference in (
            ("adaptive", "popularity"),
            ("adaptive", "item_knn"),
            ("adaptive", "neural_ensemble"),
            ("als", "popularity"),
        ):
            difference = individual[candidate] - individual[reference]
            reports[f"paired_{candidate}_minus_{reference}"] = {
                "ndcg_at_10": mean_interval(difference[:, 1].tolist()),
                "recall_at_10": mean_interval(difference[:, 0].tolist()),
                "interpretation": (
                    "Positive favors candidate; paired-user descriptive intervals, "
                    "no multiplicity adjustment."
                ),
            }
        results[bucket] = reports
    report = {
        "protocol": protocol,
        "als_validation_search": als_search,
        "validation_candidates": validation_search,
        "history_selector": selectors,
        "history_blends": best_blends,
        "selected_als": asdict(selected_config),
        "reserve_test": results,
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "source_artifact_hashes": source_metadata["artifact_hashes"],
        "runtime": {"numpy": np.__version__, "torch": torch.__version__},
        "limitations": [
            "Prior results were inspected; previously scored test users are excluded here.",
            "Reusing validation across generations allows selection overfit.",
            "User intervals condition on models and do not adjust for multiple contrasts.",
            "Intermediate history lengths use the nearest bucket and were not separately tested.",
            "ALS/kNN are non-neural baselines and may be selected over neural components.",
        ],
    }
    # Publish once scoring succeeds; a failing experiment leaves the serving pointer untouched.
    published = publish_bundle(artifact_dir, source, metadata, selected_factors)
    report["published_bundle"] = published.name
    write_json(report_path, report)
    return report
