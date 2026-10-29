"""Counterexamples for premature convergence and leakage into development views."""

import copy
import json

import numpy as np
import pytest

from neural_recommender import convergence
from neural_recommender.convergence import RULE, curve_diagnostics, development_view


def curve(scores):
    return [
        {"epoch": i + 1, "loss": 1 / (i + 1), "validation_ndcg_at_10": score}
        for i, score in enumerate(scores)
    ]


def test_late_improvement_revokes_a_temporary_plateau():
    # Stopping after the initially flat region would miss continued learning.
    rows = curve([0.3] * 17 + [0.31, 0.32, 0.33])
    result = curve_diagnostics(rows, RULE)
    assert result["first_temporary_plateau_epoch"] == 10
    assert result["last_practical_improvement_epoch"] == 20
    assert not result["plateau_at_fixed_budget"]
    assert result["status"] == "budget_limited"
    assert result["validation_gain_over_five_epoch_budget"] == pytest.approx(0.03)


def test_practical_plateau_ignores_sub_delta_jitter_but_records_exact_best():
    rows = curve([0.3] * 19 + [0.301])
    result = curve_diagnostics(rows, RULE)
    assert result["plateau_at_fixed_budget"]
    assert result["last_practical_improvement_epoch"] == 1
    assert result["best_epoch"] == 20
    assert result["best_validation_ndcg_at_10"] == 0.301


def test_sustained_improvement_remains_budget_limited():
    result = curve_diagnostics(curve([0.2 + i * 0.005 for i in range(20)]), RULE)
    assert result["first_temporary_plateau_epoch"] is None
    assert not result["plateau_at_fixed_budget"]


@pytest.mark.parametrize("mutation", ["short", "nonfinite", "boolean", "order", "range"])
def test_malformed_or_partial_curves_cannot_receive_a_plateau_status(mutation):
    rows = curve([0.3] * 20)
    if mutation == "short":
        rows.pop()
    elif mutation == "nonfinite":
        rows[3]["loss"] = float("inf")
    elif mutation == "boolean":
        rows[3]["validation_ndcg_at_10"] = True
    elif mutation == "order":
        rows.reverse()
    else:
        rows[3]["validation_ndcg_at_10"] = 1.1
    with pytest.raises(ValueError, match="complete finite"):
        curve_diagnostics(rows, RULE)


def test_relaxed_plateau_rule_is_rejected():
    relaxed = {**RULE, "min_delta": 0.1}
    with pytest.raises(ValueError, match="frozen rule"):
        curve_diagnostics(curve([0.3] * 20), relaxed)


def test_reserve_poisoning_is_removed_before_histories_are_constructed():
    ratings = np.array([[1, 0, 5, 100], [2, 1, 4, 200], [99, 2, 5, 300]])
    before = development_view(ratings, [1], [2])
    ratings[2, 1:] = [3000, 1, 9999999]
    after = development_view(ratings, [1], [2])
    np.testing.assert_array_equal(before, after)
    assert set(after[:, 0]) == {1, 2}


@pytest.mark.parametrize(
    "training,validation", [([1], [1]), ([1, 1], [2]), ([True], [2]), ([1], [3])]
)
def test_development_view_rejects_overlapping_coerced_and_unknown_owners(training, validation):
    with pytest.raises(ValueError, match="unique, disjoint"):
        development_view(np.array([[1, 0, 5, 1], [2, 1, 4, 2]]), training, validation)


def test_source_or_active_artifact_drift_prevents_fitting(tmp_path, monkeypatch):
    contract = {"source": "frozen", "active_artifacts": "frozen"}
    frozen = tmp_path / "protocol.json"
    frozen.write_text(json.dumps({"format_version": 1, "contract": contract}))
    monkeypatch.setattr(convergence, "_contract", lambda _: {**contract, "source": "changed"})

    def forbidden(*args, **kwargs):
        raise AssertionError("Training must never start after a frozen identity changes")

    monkeypatch.setattr(convergence, "_train_neural", forbidden)
    with pytest.raises(ValueError, match="identity changed"):
        convergence.run(tmp_path, frozen, tmp_path / "result.json")
    assert not (tmp_path / "result.json").exists()


def test_existing_experiment_records_are_not_overwritten(tmp_path):
    output = tmp_path / "preserved.json"
    original = {"immutable": True}
    output.write_text(json.dumps(copy.deepcopy(original)))
    with pytest.raises(ValueError, match="overwrite"):
        convergence.freeze(tmp_path, output)
    with pytest.raises(ValueError, match="overwrite"):
        convergence.run(tmp_path, tmp_path / "unused.json", output)
    assert json.loads(output.read_text()) == original
