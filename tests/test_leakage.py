"""Counterfactual leakage tests: changing held-out data must not change fitting inputs."""

import copy
import json

import numpy as np
import pytest

from neural_recommender.baselines import interaction_matrix
from neural_recommender.data import Catalog, split_users
from neural_recommender.leakage_audit import (
    array_digest,
    validate_case_isolation,
    validate_selection,
)
from neural_recommender.optimization import (
    OptimizationConfig,
    _neural_training_inputs,
    prepare_protocol,
)
from neural_recommender.refinement import prepare_refinement_protocol


@pytest.fixture
def declared_protocol(tmp_path):
    ratings = np.array([[user, user % 24, 5, 1000 + user] for user in range(1, 61)])
    splits = split_users(ratings, 42)
    pilot = tmp_path / "pilot.json"
    pilot.write_text(
        json.dumps(
            {
                "data_sha256": "snapshot",
                "config": {"seed": 42},
                "test_users": [int(splits["test"][0])],
            }
        )
    )
    path = tmp_path / "upgrade.json"
    config = OptimizationConfig(train_users=10, evaluation_users=2, epochs=1)
    protocol = prepare_protocol(ratings, "snapshot", pilot, path, config)
    return ratings, pilot, path, config, protocol


@pytest.mark.parametrize(
    "mutation",
    ["train_overlap", "train_duplicate", "train_boolean", "train_order", "pilot_reuse", "grid"],
)
def test_cached_protocol_reconstructs_memberships_and_grids(declared_protocol, mutation):
    ratings, pilot, path, config, protocol = declared_protocol
    if mutation == "train_overlap":
        protocol["train_users"][0] = protocol["validation_users"][0]
    elif mutation == "train_duplicate":
        protocol["train_users"][0] = protocol["train_users"][1]
    elif mutation == "train_boolean":
        protocol["train_users"][0] = True
    elif mutation == "train_order":
        protocol["train_users"].reverse()
    elif mutation == "pilot_reuse":
        protocol["reserve_test_users"][0] = protocol["excluded_pilot_test_users"][0]
    else:
        protocol["fusion_grid"].append([0.1, 0.1, 0.8])
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError):
        prepare_protocol(ratings, "snapshot", pilot, path, config)


def test_refinement_revalidates_inherited_upgrade_contract(declared_protocol, tmp_path):
    ratings, pilot, path, _, protocol = declared_protocol
    protocol["train_users"].append(protocol["train_users"][0])
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="unique"):
        prepare_refinement_protocol(ratings, "snapshot", path, pilot, tmp_path / "refine.json", 2)
    assert not (tmp_path / "refine.json").exists()


def test_exact_production_training_inputs_ignore_held_out_positives_and_negatives():
    catalog = Catalog(np.arange(1, 25), tuple(map(str, range(24))), np.ones((24, 1)), ("Drama",))
    sequences = {1: np.arange(12), 2: np.arange(3, 15), 99: np.arange(15, 24)}
    observed = {1: set(range(15)), 2: set(range(3, 18)), 99: set(range(15, 24))}
    first = _neural_training_inputs(catalog, sequences, observed, [1, 2], np.random.default_rng(42))
    matrix = interaction_matrix(sequences, np.array([1, 2]), 24)
    # An unsafe sampler that used any held-out owner's history/observed set
    # would now either change its arrays or find no negative candidates.
    sequences[99] = np.arange(24)[::-1]
    observed[99] = set(range(24))
    second = _neural_training_inputs(
        catalog, sequences, observed, [1, 2], np.random.default_rng(42)
    )
    for left, right in zip(first[:3], second[:3], strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(matrix, interaction_matrix(sequences, np.array([1, 2]), 24))
    for user in (1, 2):
        np.testing.assert_array_equal(first[3][user], second[3][user])
        assert not set(map(int, first[3][user])) & observed[user]
    for history, target in zip(first[0], first[1], strict=True):
        assert target + 1 not in history
    assert set(first[2]) == {1, 2}


def test_case_isolation_rejects_target_in_prefix():
    with pytest.raises(ValueError, match="target"):
        validate_case_isolation({1: np.array([3, 4, 3])}, [1])


def test_array_digest_binds_dtype_and_shape():
    values = np.arange(4)
    assert array_digest(values) != array_digest(values.reshape(2, 2))
    assert array_digest(values) != array_digest(values.astype(np.float64))


def test_validation_audit_rejects_selector_chosen_from_reserve_results():
    candidates = {"a": {"a": 1.0}, "blend": {"a": 0.5, "b": 0.5}}
    selection = {
        "validation_search": {"5": {"a": 0.5, "blend": 0.4}},
        "history_selector": {"5": {"name": "a", "weights": candidates["a"]}},
        "history_blends": {"5": {"name": "blend", "weights": candidates["blend"]}},
        "als_config": {"factors": 16},
    }
    frozen = {
        "validation_candidates": selection["validation_search"],
        "history_selector": copy.deepcopy(selection["history_selector"]),
        "history_blends": selection["history_blends"],
        "als_validation_search": [{"config": {"factors": 16}, "validation_ndcg_at_10": 0.3}],
        "selected_als": selection["als_config"],
    }
    validate_selection(selection, frozen, {"candidate_blends": candidates})
    # A reserve-preferred but validation-inferior blend cannot silently become
    # the deployed default, even if both saved selectors are edited together.
    selection["history_selector"]["5"] = selection["history_blends"]["5"]
    frozen["history_selector"] = selection["history_selector"]
    with pytest.raises(ValueError, match="validation-only"):
        validate_selection(selection, frozen, {"candidate_blends": candidates})
