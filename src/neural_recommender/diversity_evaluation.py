"""Descriptive reranking tradeoffs on previously inspected reserve users.

The fixed grid is recorded before scoring. This comparison does not select a
default, fit a model, or claim that the already inspected users are a new test.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .bundles import file_hash
from .conditional import combine_scores
from .data import load_dataset, user_sequences
from .diversity import CANDIDATE_POOL, intra_list_diversity, rerank
from .metrics import mean_interval, ranking_metrics
from .optimization import cases
from .service import RecommenderService
from .storage import write_json

WEIGHTS = (1.0, 0.9, 0.75, 0.5, 0.0)


def evaluate_diversity(
    data_path: Path,
    artifact_dir: Path,
    frozen_report_path: Path,
    protocol_path: Path,
    output_path: Path,
) -> dict:
    """Measure relevance, list diversity, and catalog coverage without selection."""
    if output_path.exists():
        raise ValueError("Refusing to overwrite an existing diversity evaluation")
    if protocol_path.resolve() in {output_path.resolve(), frozen_report_path.resolve()}:
        raise ValueError("Protocol, output, and frozen report paths must be distinct")
    frozen = json.loads(frozen_report_path.read_text())
    service = RecommenderService(data_path, artifact_dir)
    if service.format_version != 3 or service.bundle_dir.name != frozen["published_bundle"]:
        raise ValueError("Diversity evaluation requires the frozen refined generation")
    if (
        service.metadata["history_selector"] != frozen["history_selector"]
        or service.metadata["als_config"] != frozen["selected_als"]
        or any(
            service.metadata["artifact_hashes"].get(name) != value
            for name, value in frozen["source_artifact_hashes"].items()
        )
    ):
        raise ValueError("Diversity evaluation models differ from the frozen experiment")
    if file_hash(data_path) != frozen["protocol"]["data_sha256"]:
        raise ValueError("Diversity evaluation data differs from the frozen experiment")
    specification = {
        "version": 1,
        "mode": "accuracy",
        "history_length": 5,
        "k": 10,
        "relevance_weights": list(WEIGHTS),
        "candidate_pool": CANDIDATE_POOL,
        "similarity": "genre Jaccard; first item is highest relevance; stable relevance-order ties",
        "relevance_scaling": "min-max within the 100 unseen relevance candidates",
        "bundle": service.bundle_dir.name,
        "artifact_hashes": service.metadata["artifact_hashes"],
        "frozen_report_sha256": file_hash(frozen_report_path),
        "data_sha256": file_hash(data_path),
        "reserve_users": frozen["protocol"]["reserve_test_users"],
        "selection": "None; every grid result retained; serving default remains relevance_weight=1",
        "scope": (
            "Descriptive evaluation on already inspected reserve users, not an untouched test. "
            "Genre distance is a metadata proxy, not measured user satisfaction or fairness."
        ),
    }
    if protocol_path.exists():
        recorded = json.loads(protocol_path.read_text())
        if {key: value for key, value in recorded.items() if key != "created_at"} != specification:
            raise ValueError("Diversity protocol changed; choose a deliberately new protocol path")
    else:
        write_json(protocol_path, {"created_at": datetime.now(UTC).isoformat(), **specification})
    # Only now read outcomes and compute scores, after the grid is on disk.
    _, ratings = load_dataset(data_path)
    sequences, _ = user_sequences(ratings)
    queries = cases(sequences, specification["reserve_users"], 5)
    measurements = {str(weight): [] for weight in WEIGHTS}
    coverage = {str(weight): set() for weight in WEIGHTS}
    latency = {str(weight): [] for weight in WEIGHTS}
    for _, history, relevant in queries:
        selection = service.selection("accuracy", len(history))
        scores = combine_scores(
            service._component_scores(history, set(selection["weights"])), selection["weights"]
        )
        for weight in WEIGHTS:
            key = str(weight)
            started = time.perf_counter()
            selected = rerank(scores, service.catalog.genres, history, 10, weight)
            latency[key].append((time.perf_counter() - started) * 1000)
            recall, ndcg = ranking_metrics(selected, set(map(int, relevant)), 10)
            measurements[key].append(
                (recall, ndcg, intra_list_diversity(service.catalog.genres[selected]))
            )
            coverage[key].update(map(int, selected))
    results = {}
    baseline = np.asarray(measurements["1.0"])
    for key, rows in measurements.items():
        values = np.asarray(rows)
        results[key] = {
            "users": len(rows),
            "recall_at_10": mean_interval(values[:, 0].tolist()),
            "ndcg_at_10": mean_interval(values[:, 1].tolist()),
            "genre_intra_list_diversity": mean_interval(values[:, 2].tolist()),
            "catalog_coverage": len(coverage[key]) / len(service.catalog.ids),
            "paired_ndcg_minus_default": mean_interval((values[:, 1] - baseline[:, 1]).tolist()),
            "paired_diversity_minus_default": mean_interval(
                (values[:, 2] - baseline[:, 2]).tolist()
            ),
            "reranking_p95_ms": float(np.percentile(latency[key], 95)),
        }
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_sha256": file_hash(protocol_path),
        "protocol": specification,
        "results": results,
        "interval_note": "Paired user bootstrap; descriptive, not multiplicity-adjusted.",
    }
    write_json(output_path, report)
    return report
