"""Full-catalog ranking metrics and user-level bootstrap uncertainty."""

from __future__ import annotations

import numpy as np


def score_items(items: np.ndarray, user: np.ndarray) -> np.ndarray:
    """Row-wise dot products without platform-specific BLAS vector-matrix dispatch.

    NumPy 2.2 / Accelerate on Apple silicon can set spurious floating-point flags
    for bounded float32 matrix-vector products. Explicit contraction is stable
    here and tested against scalar dot products; do not suppress numeric errors.
    """
    if items.ndim != 2 or user.shape != (items.shape[1],):
        raise ValueError("Item and user embedding dimensions must match")
    if not np.isfinite(items).all() or not np.isfinite(user).all():
        raise ValueError("Embeddings must be finite")
    return np.einsum("ij,j->i", items, user, optimize=False)


def rank(scores: np.ndarray, excluded: np.ndarray, k: int) -> np.ndarray:
    if k < 1:
        raise ValueError("k must be positive")
    values = scores.astype(float, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Ranking scores must be finite")
    values[excluded] = -np.inf
    available = np.flatnonzero(np.isfinite(values))
    return available[np.argsort(-values[available], kind="stable")[:k]]


def ranking_metrics(recommended: np.ndarray, relevant: set[int], k: int) -> tuple[float, float]:
    if k < 1 or len(set(recommended)) != len(recommended):
        raise ValueError("Recommendations must be unique and k positive")
    if not relevant:
        raise ValueError("Ranking evaluation needs at least one relevant item")
    hits = np.array([int(int(item) in relevant) for item in recommended[:k]])
    recall = float(hits.sum() / len(relevant))
    discounts = 1 / np.log2(np.arange(2, len(hits) + 2))
    ideal = (1 / np.log2(np.arange(2, min(k, len(relevant)) + 2))).sum()
    return recall, float(np.dot(hits, discounts) / ideal)


def mean_interval(values: list[float], seed: int = 42) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        raise ValueError("Cannot estimate an interval from an empty evaluation")
    rng = np.random.default_rng(seed)
    bootstrap = np.mean(rng.choice(array, size=(500, len(array)), replace=True), axis=1)
    return {
        "mean": float(array.mean()),
        "ci95": list(map(float, np.quantile(bootstrap, [0.025, 0.975]))),
    }
