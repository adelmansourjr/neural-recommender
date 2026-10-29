"""Reconstruct split and fitting isolation without training or tuning on reserve users.

This checks the recorded experiment, not a claim that an observational dataset
can establish online performance, global temporal isolation, or absence of all
possible overfitting. The output contains descriptive learning-curve diagnostics.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .baselines import interaction_matrix
from .bundles import file_hash
from .data import load_dataset, user_sequences
from .optimization import OptimizationConfig, _neural_training_inputs, prepare_protocol
from .refinement import prepare_refinement_protocol
from .service import RecommenderService
from .storage import write_json


def array_digest(array: np.ndarray) -> str:
    """Hash shape, dtype, and values so equal bytes cannot hide a shape change."""
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def validate_case_isolation(sequences: dict, users: list[int]) -> dict:
    """Check prefix/target disjointness and disclose timestamp-tie ambiguity later."""
    checked = {}
    for count in (1, 5, 10):
        eligible = 0
        for user in users:
            items = sequences[user]
            if len(items) <= count:
                continue
            if set(map(int, items[:count])) & set(map(int, items[count:])):
                raise ValueError("Evaluation history contains a held-out target")
            eligible += 1
        checked[str(count)] = eligible
    return checked


def validate_selection(selection: dict, frozen: dict, protocol: dict) -> None:
    """Recompute every selected bucket from recorded validation, never reserve scores."""
    if selection["validation_search"] != frozen["validation_candidates"]:
        raise ValueError("Selection and frozen validation measurements differ")
    for bucket, values in selection["validation_search"].items():
        candidates = protocol["candidate_blends"]
        if set(values) != set(candidates) or not all(np.isfinite(v) for v in values.values()):
            raise ValueError("Validation candidate grid is incomplete or nonfinite")
        winner = max(values, key=values.get)
        blends = [name for name, weights in candidates.items() if len(weights) > 1]
        blend = max(blends, key=values.get)
        for field, name in (("history_selector", winner), ("history_blends", blend)):
            expected = {"name": name, "weights": candidates[name]}
            if selection[field][bucket] != expected or frozen[field][bucket] != expected:
                raise ValueError("Saved selector does not match validation-only selection")
    search = frozen["als_validation_search"]
    best = max(search, key=lambda row: row["validation_ndcg_at_10"])
    if best["config"] != selection["als_config"] or best["config"] != frozen["selected_als"]:
        raise ValueError("ALS selection differs from recorded validation optimum")


def audit_leakage(
    data_path: Path,
    artifact_dir: Path,
    report_dir: Path,
    output: Path,
) -> dict:
    """Audit the existing refined experiment and active generation; refuse overwrites."""
    if output.exists():
        raise ValueError("A leakage audit must use a new output path")
    paths = {
        name: report_dir / filename
        for name, filename in {
            "pilot": "initial_pilot_metadata.json",
            "upgrade_protocol": "upgrade_protocol.json",
            "refinement_protocol": "refinement_protocol.json",
            "upgrade_report": "optimized_benchmark.json",
            "refinement_report": "refined_benchmark.json",
            "selection": "refinement_selection.json",
        }.items()
    }
    if not all(path.is_file() for path in paths.values()):
        raise ValueError("All preserved experiment receipts are required before auditing")
    sources = {name: json.loads(path.read_text()) for name, path in paths.items()}
    catalog, ratings = load_dataset(data_path)
    digest = file_hash(data_path)
    upgrade = prepare_protocol(
        ratings,
        digest,
        paths["pilot"],
        paths["upgrade_protocol"],
        OptimizationConfig(**sources["upgrade_protocol"]["config"]),
    )
    refined = prepare_refinement_protocol(
        ratings,
        digest,
        paths["upgrade_protocol"],
        paths["pilot"],
        paths["refinement_protocol"],
        sources["refinement_protocol"]["requested_reserve_users"],
    )
    if sources["refinement_report"]["protocol"] != refined:
        raise ValueError("Frozen results contain a different refinement protocol")
    if sources["selection"]["protocol_sha256"] != file_hash(paths["refinement_protocol"]):
        raise ValueError("Selection receipt does not bind the refinement protocol")
    validate_selection(sources["selection"], sources["refinement_report"], refined)
    service = RecommenderService(data_path, artifact_dir)
    metadata = service.metadata
    if (
        service.format_version != 3
        or service.bundle_dir.name != sources["refinement_report"]["published_bundle"]
        or metadata["protocol_sha256"] != file_hash(paths["upgrade_protocol"])
        or metadata["refinement_protocol_sha256"] != file_hash(paths["refinement_protocol"])
        or metadata["history_selector"] != sources["selection"]["history_selector"]
        or metadata["history_blends"] != sources["selection"]["history_blends"]
        or metadata["als_config"] != sources["selection"]["als_config"]
    ):
        raise ValueError("Active bundle differs from the audited experiment/selection")
    sequences, observed = user_sequences(ratings)
    train_users = np.asarray(upgrade["train_users"])
    held_out = sorted(
        set(upgrade["validation_users"])
        | set(upgrade["reserve_test_users"])
        | set(refined["reserve_test_users"])
        | set(sources["pilot"]["test_users"])
    )
    matrix = interaction_matrix(sequences, train_users, len(catalog.ids))
    matrix_hash = array_digest(matrix)
    popularity = matrix.sum(axis=0)
    np.testing.assert_array_equal(popularity, service.rankers["popularity"])
    del matrix
    # Poison *every* accessed held-out group, including observed negatives. The
    # exact production training-input builder must ignore all poisoned entries.
    poisoned = {**sequences, **{user: np.array([len(catalog.ids) - 1]) for user in held_out}}
    poisoned_observed = {**observed, **{user: set(range(len(catalog.ids))) for user in held_out}}
    poisoned_matrix = interaction_matrix(poisoned, train_users, len(catalog.ids))
    if array_digest(poisoned_matrix) != matrix_hash:
        raise ValueError("Held-out interactions influence fitted matrix features")
    del poisoned_matrix
    example_hashes = {}
    negative_pool_count = 0
    for seed in upgrade["config"]["seeds"]:
        original = _neural_training_inputs(
            catalog, sequences, observed, upgrade["train_users"], np.random.default_rng(seed)
        )
        changed = _neural_training_inputs(
            catalog,
            poisoned,
            poisoned_observed,
            upgrade["train_users"],
            np.random.default_rng(seed),
        )
        hashes = [array_digest(array) for array in original[:3]]
        if hashes != [array_digest(array) for array in changed[:3]]:
            raise ValueError("Held-out interactions influence training examples")
        if not set(map(int, original[2])).issubset(set(map(int, train_users))):
            raise ValueError("Training examples include held-out users")
        for history, target in zip(original[0], original[1], strict=True):
            if int(target) + 1 in history:
                raise ValueError("Training history contains its target")
        for user, pool in original[3].items():
            if not len(pool) or set(map(int, pool)) & observed[user]:
                raise ValueError("Training negatives include observed items or no candidates")
            np.testing.assert_array_equal(pool, changed[3][user])
            if seed == upgrade["config"]["seeds"][0]:
                negative_pool_count += len(pool)
        example_hashes[str(seed)] = hashes
        del original, changed
    curve_diagnostics = {}
    for seed, curve in sources["upgrade_report"]["neural_learning_curves"].items():
        best = max(curve, key=lambda row: row["validation_ndcg_at_10"])
        curve_diagnostics[seed] = {
            "epochs": len(curve),
            "selected_epoch": best["epoch"],
            "best_epoch_is_last": best == curve[-1],
            "first_loss": curve[0]["loss"],
            "last_loss": curve[-1]["loss"],
            "validation_ndcg_change": curve[-1]["validation_ndcg_at_10"]
            - curve[0]["validation_ndcg_at_10"],
        }
    eligible = validate_case_isolation(sequences, refined["reserve_test_users"])
    # A per-user positive prefix is not a globally chronological train/test split.
    train_timestamps = ratings[np.isin(ratings[:, 0], train_users), 3]
    reserve_rows = ratings[np.isin(ratings[:, 0], refined["reserve_test_users"])]
    tied_boundaries = {str(count): 0 for count in (1, 5, 10)}
    for user in refined["reserve_test_users"]:
        rows = reserve_rows[(reserve_rows[:, 0] == user) & (reserve_rows[:, 2] >= 4)]
        rows = rows[np.lexsort((rows[:, 1], rows[:, 3]))]
        for count in (1, 5, 10):
            if len(rows) > count and rows[count - 1, 3] == rows[count, 3]:
                tied_boundaries[str(count)] += 1
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "passed_recorded_user_isolation_checks",
        "scope": "Read-only reconstruction; no fitting, tuning, or new evaluation cohort.",
        "data_sha256": digest,
        "input_sha256": {name: file_hash(path) for name, path in paths.items()},
        "implementation_sha256": {
            name: file_hash(Path(__file__).with_name(name))
            for name in (
                "leakage_audit.py",
                "isolation.py",
                "data.py",
                "optimization.py",
                "refinement.py",
                "baselines.py",
            )
        },
        "bundle": service.bundle_dir.name,
        "artifact_hashes": metadata["artifact_hashes"],
        "membership": {
            "train": len(train_users),
            "validation": len(upgrade["validation_users"]),
            "pilot_inspected_test": len(sources["pilot"]["test_users"]),
            "upgrade_reserve": len(upgrade["reserve_test_users"]),
            "refinement_reserve": len(refined["reserve_test_users"]),
            "poisoned_held_out_users": len(held_out),
            "refinement_eligible_by_history": eligible,
        },
        "fitting": {
            "train_matrix_sha256": matrix_hash,
            "train_only_popularity_matches_active_artifact": True,
            "held_out_poisoning_preserves_matrix_and_example_hashes": True,
            "training_example_sha256": example_hashes,
            "negative_pool_candidates_checked": negative_pool_count,
            "negative_exclusion_scope": "Full observed histories of training users only.",
        },
        "selection": {
            "recorded_validation_argmax_matches_active_selectors": True,
            "validation_reused_across_generations": True,
            "reserve_now_inspected_not_available_for_future_tuning": True,
        },
        "neural_learning_curves": curve_diagnostics,
        "temporal_limits": {
            "global_chronological_split": False,
            "latest_training_timestamp": int(train_timestamps.max()),
            "earliest_refinement_timestamp": int(reserve_rows[:, 3].min()),
            "refinement_target_boundaries_with_timestamp_ties": tied_boundaries,
        },
        "limitations": [
            "The audit verifies recorded artifacts, not independent training-execution "
            "attestation.",
            "All catalog metadata is available; no item cold-start or future-catalog claim.",
            "Training users' future interactions exclude negative candidates: offline user split, "
            "not a time-causal next-event training protocol.",
            "Timestamp ties cross some prefix boundaries; item-index ordering is deterministic, "
            "not evidence of within-session chronology.",
            "Validation reuse and candidate searches can overfit selection. Reserve users are "
            "now inspected and cannot be treated as a fresh confirmatory test in future tuning.",
            "All three neural validation curves peak at the epoch budget. Underfitting or lack "
            "of convergence cannot be ruled out; no tuning is justified by reserve performance.",
            "No repeated split, online test, popularity-bias correction, or "
            "overfitting-free claim.",
        ],
    }
    write_json(output, result)
    return result
