"""Verify deployed rankings against an existing frozen experiment, without fitting.

This is a reproducibility check on already inspected reserve users. It must never
be used to choose models, weights, buckets, or a new experiment's hyperparameters.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .als import als_scores
from .baselines import discovery_scores
from .bundles import file_hash
from .data import load_dataset, user_sequences
from .optimization import cases, score_report
from .refinement import _candidate_scores, _fixed_scores
from .service import MODES, RecommenderService
from .storage import write_json


def audit_serving(
    data_path: Path,
    artifact_dir: Path,
    frozen_report_path: Path,
    output: Path,
) -> dict:
    """Check all ten API rankers and every frozen candidate at histories 1/5/10."""
    if output.resolve() == frozen_report_path.resolve():
        raise ValueError("An audit must not overwrite its frozen experiment report")
    frozen = json.loads(frozen_report_path.read_text())
    service = RecommenderService(data_path, artifact_dir)
    if service.format_version != 3:
        raise ValueError("This audit requires the refined format 3 generation")
    if (
        service.bundle_dir.name != frozen["published_bundle"]
        or service.metadata["history_selector"] != frozen["history_selector"]
        or service.metadata["history_blends"] != frozen["history_blends"]
        or service.metadata["als_config"] != frozen["selected_als"]
    ):
        raise ValueError("Active generation differs from the frozen experiment")
    if any(
        service.metadata["artifact_hashes"].get(name) != digest
        for name, digest in frozen["source_artifact_hashes"].items()
    ):
        raise ValueError("Reused source artifacts differ from the frozen experiment")
    protocol = frozen["protocol"]
    if protocol["data_sha256"] != file_hash(data_path):
        raise ValueError("Frozen experiment and serving dataset differ")
    _, ratings = load_dataset(data_path)
    sequences, _ = user_sequences(ratings)
    evidence = {}
    for length in (1, 5, 10):
        bucket = str(length)
        queries = cases(sequences, protocol["reserve_test_users"], length)
        components = _fixed_scores(service, queries)
        components["als"] = [
            als_scores(
                service.als_factors,
                history,
                service.als_config.regularization,
                service.als_config.alpha,
            )
            for _, history, _ in queries
        ]
        candidates = {
            name: _candidate_scores(components, weights)
            for name, weights in protocol["candidate_blends"].items()
        }
        # Reproduce all reported candidates, not only the deployed winner.
        for name, values in candidates.items():
            measured, _ = score_report(values, queries, service.rankers["popularity"])
            expected = frozen["reserve_test"][bucket][name]
            if measured["users"] != expected["users"]:
                raise ValueError("Frozen evaluation eligibility changed")
            for metric in ("recall_at_10", "ndcg_at_10", "long_tail_share_at_10"):
                np.testing.assert_allclose(
                    measured[metric]["mean"], expected[metric]["mean"], atol=1e-10, rtol=0
                )
            np.testing.assert_allclose(
                measured["catalog_coverage"], expected["catalog_coverage"], atol=1e-10, rtol=0
            )
        for index, (_, history, _) in enumerate(queries):
            identifiers = service.catalog.ids[history].tolist()
            for mode in sorted(MODES):
                if mode in {"accuracy", "adaptive", "blend"}:
                    selector = (
                        frozen["history_blends"] if mode == "blend" else frozen["history_selector"]
                    )
                    values = candidates[selector[bucket]["name"]][index]
                else:
                    component = {
                        "hybrid": "previous_hybrid",
                        "discovery": "previous_hybrid",
                        "neural": "neural_ensemble",
                    }.get(mode, mode)
                    values = components[component][index]
                if mode == "discovery":
                    values = discovery_scores(
                        values, service.rankers["popularity"], service.metadata["discovery_penalty"]
                    )
                # Independently sort with an explicit catalog-index tie breaker,
                # avoiding the serving rank() helper for the parity assertion.
                order = np.lexsort((np.arange(len(values)), -values))
                expected_indices = order[~np.isin(order, history)][:10]
                actual = service.recommend(identifiers, 10, mode)
                if [row["movie_id"] for row in actual] != service.catalog.ids[
                    expected_indices
                ].tolist():
                    raise ValueError(
                        f"Serving/evaluation ranking mismatch in {mode}, bucket {bucket}"
                    )
                np.testing.assert_allclose(
                    [row["score"] for row in actual], values[expected_indices], atol=5.1e-7, rtol=0
                )
        evidence[bucket] = {
            "users": len(queries),
            "candidate_metrics_checked": len(candidates),
            "mode_rankings_checked": len(queries) * len(MODES),
        }
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "passed",
        "bundle": service.bundle_dir.name,
        "frozen_report_sha256": file_hash(frozen_report_path),
        "data_sha256": protocol["data_sha256"],
        "artifact_hashes": service.metadata["artifact_hashes"],
        "histories": evidence,
        "scope": (
            "Reproduction of existing frozen results and serving parity; "
            "no fitting, selection, or new reserve users."
        ),
    }
    write_json(output, result)
    return result
