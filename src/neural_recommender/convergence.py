"""Bounded validation-only diagnostics, without promoting or retaining fitted models.

The existing five-epoch experiment is immutable. This separate diagnostic asks
whether its three neural seeds had exhausted a larger, predeclared budget. It
cannot establish a generalization bound or authorize reuse of inspected tests.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .bundles import file_hash
from .data import load_dataset, user_sequences
from .optimization import OptimizationConfig, _train_neural, cases, prepare_protocol
from .storage import write_json

RULE = {"max_epochs": 20, "min_epochs": 10, "patience": 5, "min_delta": 0.002}


def curve_diagnostics(curve: list[dict], rule: dict) -> dict:
    """Apply a fixed practical-improvement rule to complete, finite curves.

    The anchor advances only after an improvement exceeding ``min_delta``.
    Continuing to the fixed budget avoids mistaking an early flat region for
    final convergence. Neither significance nor parameter optimality is implied.
    """
    if (
        rule != RULE
        or len(curve) != rule["max_epochs"]
        or any(row.get("epoch") != i + 1 for i, row in enumerate(curve))
        or any(
            type(row.get(key)) not in (float, int) or not np.isfinite(row[key])
            for row in curve
            for key in ("loss", "validation_ndcg_at_10")
        )
        or any(row["loss"] < 0 or not 0 <= row["validation_ndcg_at_10"] <= 1 for row in curve)
    ):
        raise ValueError("Diagnostic requires the frozen rule and a complete finite learning curve")
    anchor = -1.0
    last_improvement = 0
    first_plateau = None
    for row in curve:
        if row["validation_ndcg_at_10"] > anchor + rule["min_delta"]:
            anchor = row["validation_ndcg_at_10"]
            last_improvement = row["epoch"]
        if (
            first_plateau is None
            and row["epoch"] >= rule["min_epochs"]
            and row["epoch"] - last_improvement >= rule["patience"]
        ):
            first_plateau = row["epoch"]
    best = max(curve, key=lambda row: row["validation_ndcg_at_10"])
    historical_best = max(row["validation_ndcg_at_10"] for row in curve[:5])
    plateau = len(curve) - last_improvement >= rule["patience"]
    return {
        "best_epoch": best["epoch"],
        "best_validation_ndcg_at_10": best["validation_ndcg_at_10"],
        "five_epoch_best_validation_ndcg_at_10": historical_best,
        "validation_gain_over_five_epoch_budget": best["validation_ndcg_at_10"] - historical_best,
        "last_practical_improvement_epoch": last_improvement,
        "first_temporary_plateau_epoch": first_plateau,
        "plateau_at_fixed_budget": plateau,
        "first_loss": curve[0]["loss"],
        "last_loss": curve[-1]["loss"],
        "last_validation_ndcg_at_10": curve[-1]["validation_ndcg_at_10"],
        "status": "practical_validation_plateau_observed" if plateau else "budget_limited",
    }


def development_view(ratings: np.ndarray, training: list[int], validation: list[int]) -> np.ndarray:
    """Remove all other owners before constructing histories or negative pools."""
    if (
        not training
        or not validation
        or any(type(user) is not int for user in training + validation)
        or len(set(training)) != len(training)
        or len(set(validation)) != len(validation)
        or set(training) & set(validation)
        or not set(training + validation).issubset(set(map(int, ratings[:, 0])))
    ):
        raise ValueError("Development owners must be unique, disjoint, known integer users")
    return ratings[np.isin(ratings[:, 0], training + validation)].copy()


def _contract(project: Path) -> dict:
    """Bind data, production training helpers, execution environment and active files."""
    reports = project / "reports"
    data = project / "data/ml-1m.zip"
    upgrade = json.loads((reports / "upgrade_protocol.json").read_text())
    if upgrade["config"]["seeds"] != [7, 19, 42] or upgrade["config"]["epochs"] != 5:
        raise ValueError("Diagnostic is restricted to the preserved three-seed five-epoch run")
    active = json.loads((project / "artifacts/active.json").read_text())["bundle"]
    if not isinstance(active, str) or Path(active).name != active:
        raise ValueError("Active generation must be a local bundle name")
    bundle = project / "artifacts/bundles" / active
    if not bundle.is_dir():
        raise ValueError("Active bundle is unavailable")
    source = Path(__file__).parent
    paths = {
        "upgrade_protocol": reports / "upgrade_protocol.json",
        "pilot_metadata": reports / "initial_pilot_metadata.json",
        "historical_curves_report": reports / "optimized_benchmark.json",
        "runner": project / "scripts/convergence_diagnostic.py",
        **{
            f"source/{name}": source / name
            for name in (
                "convergence.py",
                "optimization.py",
                "model.py",
                "metrics.py",
                "data.py",
                "isolation.py",
                "storage.py",
                "bundles.py",
                "baselines.py",
            )
        },
    }
    return {
        "data_sha256": file_hash(data),
        "input_sha256": {name: file_hash(path) for name, path in paths.items()},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "rule": RULE,
        "training_users": upgrade["train_users"],
        "validation_users": upgrade["validation_users"],
        "training_configuration": {**upgrade["config"], "epochs": RULE["max_epochs"]},
        "active_bundle": active,
        "active_artifact_sha256": {
            str(path.relative_to(project)): file_hash(path)
            for path in sorted([project / "artifacts/active.json", *bundle.rglob("*")])
            if path.is_file()
        },
        "prefix_reproduction_absolute_tolerance": 1e-10,
        "selection_policy": (
            "Train each seed for exactly 20 epochs; record the earliest maximal validation "
            "NDCG@10 epoch. Fixed minimum 10 epochs, patience 5, practical delta 0.002. "
            "Report temporary and terminal plateaus; no early stop, model retention, "
            "reserve scoring, default change, or promotion."
        ),
    }


def freeze(project: Path, protocol: Path) -> dict:
    """Save the prospective diagnostic contract before any new fitting begins."""
    if protocol.exists():
        raise ValueError("Refusing to overwrite a frozen convergence protocol")
    contract = _contract(project)
    _, ratings = load_dataset(project / "data/ml-1m.zip")
    upgrade = json.loads((project / "reports/upgrade_protocol.json").read_text())
    prepare_protocol(
        ratings,
        contract["data_sha256"],
        project / "reports/initial_pilot_metadata.json",
        project / "reports/upgrade_protocol.json",
        OptimizationConfig(**upgrade["config"]),
    )
    development_view(ratings, contract["training_users"], contract["validation_users"])
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "format_version": 1,
        "scope": "Training and previously used validation users only; no new test outcomes.",
        "contract": contract,
    }
    write_json(protocol, result)
    return result


def run(project: Path, protocol: Path, output: Path) -> dict:
    """Replay the original seeds with a larger fixed budget and preserve all active artifacts."""
    if output.exists():
        raise ValueError("Refusing to overwrite a convergence result")
    frozen = json.loads(protocol.read_text())
    contract = _contract(project)
    if frozen.get("format_version") != 1 or frozen.get("contract") != contract:
        raise ValueError("Frozen convergence protocol identity changed; refusing to fit")
    started = time.perf_counter()
    catalog, ratings = load_dataset(project / "data/ml-1m.zip")
    permitted = development_view(ratings, contract["training_users"], contract["validation_users"])
    sequences, observed = user_sequences(permitted)
    validation = cases(sequences, contract["validation_users"], 5)
    historical = json.loads((project / "reports/optimized_benchmark.json").read_text())
    original_curves = historical["neural_learning_curves"]
    config = OptimizationConfig(**contract["training_configuration"])
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    results = {}
    for seed in config.seeds:
        model, curve = _train_neural(
            catalog,
            sequences,
            observed,
            contract["training_users"],
            validation,
            seed,
            config,
        )
        # No checkpoint is published. The production trainer returns a selected
        # in-memory model, but this diagnostic consumes only its learning curve.
        del model
        deviations = [
            abs(new[key] - old[key])
            for new, old in zip(curve[:5], original_curves[str(seed)], strict=True)
            for key in ("loss", "validation_ndcg_at_10")
        ]
        largest = max(deviations)
        if largest > contract["prefix_reproduction_absolute_tolerance"]:
            raise ValueError(
                "First five epochs differ from preserved experiment; comparison invalid"
            )
        results[str(seed)] = {
            "historical_prefix_reproduced": True,
            "largest_original_prefix_absolute_deviation": largest,
            "learning_curve": curve,
            "diagnostics": curve_diagnostics(curve, contract["rule"]),
        }
    if _contract(project) != contract:
        raise ValueError("Bound inputs or active artifacts changed during convergence diagnostic")
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_sha256": file_hash(protocol),
        "status": "completed_validation_only_diagnostic",
        "rule": contract["rule"],
        "data_sha256": contract["data_sha256"],
        "training_users": len(contract["training_users"]),
        "validation_users": len(validation),
        "excluded_other_rating_rows": len(ratings) - len(permitted),
        "development_rating_rows": len(permitted),
        "active_artifacts_unchanged": True,
        "models_saved": False,
        "reserve_users_scored": 0,
        "results": results,
        "all_seeds_have_terminal_practical_plateau": all(
            row["diagnostics"]["plateau_at_fixed_budget"] for row in results.values()
        ),
        "limitations": [
            "The previously reused validation set is a development resource, not an unseen test.",
            "The plateau rule is a practical diagnostic, not proof of global optimum "
            "or calibration.",
            "Continued validation improvement indicates budget sensitivity, "
            "not proven underfitting.",
            "Training cross-entropy and validation ranking metrics are different objectives; "
            "their numerical difference is not a train/test generalization gap.",
            "No generalization or improvement claim is made for these unpromoted fitted models.",
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(output, result)
    return result
