"""Train-only closed-form recommenders and validation-calibrated score fusion."""

from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve

from .metrics import rank


def interaction_matrix(
    sequences: dict[int, np.ndarray], users: np.ndarray, item_count: int
) -> np.ndarray:
    matrix = np.zeros((len(users), item_count), dtype=np.float32)
    for row, user in enumerate(users):
        matrix[row, sequences[int(user)]] = 1
    return matrix


def interaction_gram(matrix: np.ndarray) -> np.ndarray:
    """Count co-occurrences with Torch's tested CPU BLAS path."""
    tensor = torch.from_numpy(np.ascontiguousarray(matrix))
    return (tensor.T @ tensor).numpy().astype(np.float64)


def fit_ease(gram: np.ndarray, regularization: float) -> np.ndarray:
    """EASE: ridge reconstruction with a zero diagonal, not a neural model.

    B[i,j] is the contribution of observed item i to target item j. See
    Steck (2019), Embarrassingly Shallow Autoencoders for Sparse Data.
    """
    if regularization <= 0 or gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("EASE requires a square Gram matrix and positive regularization")
    regularized = gram.copy()
    regularized.flat[:: len(gram) + 1] += regularization
    factor = cho_factor(regularized, overwrite_a=True, check_finite=True)
    precision = cho_solve(factor, np.eye(len(gram)), check_finite=False)
    coefficients = -precision / np.diag(precision)[None, :]
    np.fill_diagonal(coefficients, 0)
    if not np.isfinite(coefficients).all():
        raise ValueError("EASE produced nonfinite coefficients")
    return coefficients.astype(np.float32)


def fit_item_knn(
    gram: np.ndarray, shrinkage: float, neighbors: int = 80
) -> tuple[np.ndarray, np.ndarray]:
    if shrinkage < 0 or not 1 <= neighbors < len(gram):
        raise ValueError("Invalid kNN shrinkage or neighbor count")
    norm = np.sqrt(np.diag(gram))
    denominator = norm[:, None] * norm[None, :]
    similarity = np.divide(gram, denominator, out=np.zeros_like(gram), where=denominator > 0)
    if shrinkage:
        similarity *= gram / (gram + shrinkage)
    np.fill_diagonal(similarity, 0)
    values, indices = torch.topk(torch.from_numpy(similarity), neighbors, dim=1)
    return indices.numpy().astype(np.int32), values.numpy().astype(np.float32)


def ease_scores(coefficients: np.ndarray, history: np.ndarray) -> np.ndarray:
    return coefficients[history].mean(axis=0)


def knn_scores(indices: np.ndarray, values: np.ndarray, history: np.ndarray) -> np.ndarray:
    scores = np.zeros(len(indices), dtype=np.float32)
    np.add.at(scores, indices[history].ravel(), values[history].ravel() / len(history))
    return scores


def normalize_scores(values: np.ndarray) -> np.ndarray:
    """Per-query standardization puts each ranker's scores on a common scale."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Cannot calibrate nonfinite scores")
    return ((values - values.mean()) / max(float(values.std()), 1e-8)).astype(np.float32)


def fuse_scores(
    ease: np.ndarray,
    neural: np.ndarray,
    popularity: np.ndarray,
    weights: tuple[float, float, float],
) -> np.ndarray:
    if min(weights) < 0 or not np.isclose(sum(weights), 1):
        raise ValueError("Fusion weights must be nonnegative and sum to one")
    return sum(
        weight * normalize_scores(values)
        for weight, values in zip(weights, (ease, neural, np.log1p(popularity)), strict=True)
    )


def diversified_rank(
    scores: np.ndarray,
    history: np.ndarray,
    popularity: np.ndarray,
    novelty_weight: float,
    k: int = 10,
) -> np.ndarray:
    if not 0 <= novelty_weight <= 1:
        raise ValueError("Novelty weight must be between zero and one")
    if novelty_weight == 0:
        return rank(scores, history, k)
    adjusted = discovery_scores(scores, popularity, novelty_weight)
    return rank(adjusted, history, k)


def discovery_scores(scores: np.ndarray, popularity: np.ndarray, weight: float) -> np.ndarray:
    if not 0 <= weight <= 1:
        raise ValueError("Novelty weight must be between zero and one")
    return normalize_scores(scores) - weight * normalize_scores(np.log1p(popularity))


def novel_item_share(recommendations: np.ndarray, popularity: np.ndarray) -> float:
    """Share outside the most popular 20% of the catalog, defined on training only."""
    head_count = max(1, int(np.ceil(len(popularity) * 0.2)))
    head = np.argsort(-popularity, kind="stable")[:head_count]
    return float(np.mean(~np.isin(recommendations, head)))
