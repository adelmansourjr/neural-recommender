"""MMR is explicit, bounded, deterministic, and preserves the original endpoint."""

import numpy as np
import pytest
from pydantic import ValidationError

from neural_recommender.api import RecommendationRequest
from neural_recommender.diversity import (
    CANDIDATE_POOL,
    genre_similarity,
    intra_list_diversity,
    rerank,
)
from neural_recommender.metrics import rank


def test_relevance_endpoint_exactly_preserves_original_ties_and_history_exclusion():
    scores = np.array([5.0, 5.0, 4.0, 4.0, 3.0])
    genres = np.array([[1, 0], [1, 0], [0, 1], [1, 1], [1, 0]])
    excluded = np.array([0, 3])
    np.testing.assert_array_equal(rerank(scores, genres, excluded, 4), rank(scores, excluded, 4))
    np.testing.assert_array_equal(rerank(scores, genres, excluded, 4, 1.0), [1, 2, 4])


def test_diversity_endpoint_uses_top_relevance_seed_then_minimizes_redundancy():
    scores = np.array([10.0, 9.0, 8.0, 7.0])
    genres = np.array([[1, 0], [1, 0], [0, 1], [1, 1]])
    result = rerank(scores, genres, np.array([], dtype=int), 3, 0.0)
    np.testing.assert_array_equal(result, [0, 2, 3])
    assert intra_list_diversity(genres[result[:2]]) > intra_list_diversity(genres[:2])
    for _ in range(5):
        np.testing.assert_array_equal(rerank(scores, genres, np.array([], dtype=int), 3, 0), result)


def test_diversity_candidates_remain_bounded_and_never_include_history():
    scores = np.arange(150, 0, -1, dtype=float)
    genres = np.tile([1, 0], (150, 1))
    genres[149] = [0, 1]  # A tempting distant genre outside the relevance pool.
    result = rerank(scores, genres, np.array([0, 2]), 50, 0)
    assert len(result) == len(set(result)) == 50
    assert not set(result) & {0, 2, 149}
    assert result.max() < CANDIDATE_POOL + 2


def test_genre_jaccard_has_known_values_and_handles_absent_features():
    features = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    similarity = genre_similarity(features)
    assert similarity[0, 1] == similarity[1, 2] == 0.5
    assert similarity[0, 2] == similarity[3, 3] == 0
    assert intra_list_diversity(features[:3]) == pytest.approx(2 / 3)
    assert intra_list_diversity(features[:1]) == 0


@pytest.mark.parametrize("weight", [True, "0.5", None, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_weight_cannot_silently_change_ranking(weight):
    with pytest.raises(ValueError, match="weight"):
        rerank(np.ones(3), np.eye(3), np.array([], dtype=int), 2, weight)


def test_all_seen_catalog_returns_no_results_and_constant_scores_use_stable_ties():
    genres = np.eye(4)
    assert not len(rerank(np.ones(4), genres, np.arange(4), 2, 0.5))
    np.testing.assert_array_equal(
        rerank(np.ones(4), genres, np.array([], dtype=int), 3, 0.5), [0, 1, 2]
    )


def test_normalization_of_finite_extreme_scores_does_not_overflow():
    scores = np.array([1e308, 0.0, -1e308])
    with np.errstate(all="raise"):
        result = rerank(scores, np.eye(3), np.array([], dtype=int), 3, 0.75)
    np.testing.assert_array_equal(result, [0, 1, 2])


@pytest.mark.parametrize("weight", [True, "0.75", None, float("nan"), float("inf"), -1, 2])
def test_api_schema_rejects_nonfinite_or_coerced_weights(weight):
    with pytest.raises(ValidationError):
        RecommendationRequest.model_validate({"movie_ids": [1], "relevance_weight": weight})


def test_pairwise_genre_work_is_bounded_before_allocation(monkeypatch):
    from neural_recommender import diversity

    original = diversity.genre_similarity
    observed = []

    def checked(features):
        observed.append(features.shape)
        assert features.shape[0] <= CANDIDATE_POOL
        return original(features)

    monkeypatch.setattr(diversity, "genre_similarity", checked)
    scores = np.arange(10_000, 0, -1, dtype=float)
    genres = np.tile([1, 0, 1], (10_000, 1))
    result = rerank(scores, genres, np.array([0]), 50, 0.5)
    assert len(result) == 50
    assert observed == [(CANDIDATE_POOL, 3)]
