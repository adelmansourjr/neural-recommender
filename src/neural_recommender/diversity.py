"""Opt-in bounded MMR reranking using interpretable genre-set similarity.

The relevance model is unchanged. A user-selected weight trades its normalized
scores against redundant genres within the first 100 unseen relevance candidates.
"""

from __future__ import annotations

import numpy as np

from .metrics import rank

CANDIDATE_POOL = 100


def validate_weight(relevance_weight: float) -> None:
    """Reject coercions and nonfinite values at HTTP and direct-call boundaries."""
    if (
        type(relevance_weight) not in {int, float}
        or not np.isfinite(relevance_weight)
        or not 0 <= relevance_weight <= 1
    ):
        raise ValueError("Relevance weight must be a finite number between zero and one")


def genre_similarity(genres: np.ndarray) -> np.ndarray:
    """Pairwise Jaccard similarity; two absent genre sets have similarity zero."""
    if genres.ndim != 2 or not np.isin(genres, (0, 1)).all():
        raise ValueError("Genre features must be a binary item-by-genre matrix")
    features = genres.astype(np.float64)
    intersection = np.einsum("ig,jg->ij", features, features, optimize=False)
    counts = features.sum(axis=1)
    union = counts[:, None] + counts[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def rerank(
    scores: np.ndarray,
    genres: np.ndarray,
    excluded: np.ndarray,
    k: int,
    relevance_weight: float = 1.0,
) -> np.ndarray:
    """Greedily maximize λ relevance − (1−λ) maximum selected-genre similarity.

    The first item is always the top relevance candidate, including at λ=0.
    Ties preserve relevance order and then catalog order. λ=1 delegates exactly
    to the historical full-catalog ranker, preserving every default result.
    """
    validate_weight(relevance_weight)
    if type(k) is not int or not 1 <= k <= 50:
        raise ValueError("Diversity reranking supports one to fifty recommendations")
    if genres.ndim != 2 or genres.shape[0] != len(scores):
        raise ValueError("Genre features and scores must share a catalog")
    if relevance_weight == 1:
        return rank(scores, excluded, k)
    candidates = rank(scores, excluded, CANDIDATE_POOL)
    if not len(candidates):
        return candidates
    similarity = genre_similarity(genres[candidates])
    values = scores[candidates].astype(np.float64)
    # Scale before subtraction: two finite extreme scores can have an infinite
    # difference even though their relative ordering remains well-defined.
    magnitude = float(np.max(np.abs(values)))
    scaled = values / magnitude if magnitude > 0 else np.zeros(len(values))
    span = float(scaled.max() - scaled.min())
    relevance = (scaled - scaled.min()) / span if span > 0 else np.zeros(len(values))
    chosen = [0]
    available = np.ones(len(candidates), dtype=bool)
    available[0] = False
    redundancy = similarity[:, 0].copy()
    for _ in range(1, min(k, len(candidates))):
        objective = relevance_weight * relevance - (1 - relevance_weight) * redundancy
        objective[~available] = -np.inf
        position = int(np.argmax(objective))
        chosen.append(position)
        available[position] = False
        redundancy = np.maximum(redundancy, similarity[:, position])
    return candidates[np.asarray(chosen)]


def intra_list_diversity(genres: np.ndarray) -> float:
    """Mean pairwise genre Jaccard distance; zero for lists shorter than two."""
    if len(genres) < 2:
        return 0.0
    similarities = genre_similarity(genres)
    return float((1 - similarities[np.triu_indices(len(genres), k=1)]).mean())
