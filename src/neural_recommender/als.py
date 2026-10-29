"""Weighted implicit matrix factorization with explicit new-user fold-in.

The objective follows Hu, Koren, and Volinsky (2008): an observed positive has
preference one and confidence ``1 + alpha``; unobserved entries have preference
zero and confidence one. That is an optimization convention, not a statement
that an unrated film is disliked. Each alternating update solves a regularized
positive-definite system. No held-out user identifier is fitted during training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.linalg import solve

from .metrics import score_items


@dataclass(frozen=True)
class ALSConfig:
    """A bounded CPU configuration, serialized into the experiment protocol."""

    factors: int = 32
    regularization: float = 10.0
    alpha: float = 10.0
    iterations: int = 5
    seed: int = 27

    def validate(self) -> None:
        if any(type(value) is not int for value in (self.factors, self.iterations, self.seed)):
            raise ValueError("ALS factors, iterations, and seed must be integers")
        if any(type(value) not in {int, float} for value in (self.regularization, self.alpha)):
            raise ValueError("ALS regularization and alpha must be numbers")
        if not 2 <= self.factors <= 128 or not 1 <= self.iterations <= 30:
            raise ValueError("ALS factors/iterations exceed the supported CPU budget")
        if not np.isfinite(self.regularization) or self.regularization <= 0:
            raise ValueError("ALS regularization must be finite and positive")
        if not np.isfinite(self.alpha) or not 0 < self.alpha <= 100:
            raise ValueError("ALS confidence alpha must be finite and in (0, 100]")


def fold_in(
    item_factors: np.ndarray,
    history: np.ndarray,
    regularization: float,
    alpha: float,
    gram: np.ndarray | None = None,
) -> np.ndarray:
    """Solve a new user's factors from their supplied history only.

    The all-item Gram term represents the unobserved confidence-one entries.
    ``gram`` may be cached because item factors are frozen at serving time.
    """
    if (
        item_factors.ndim != 2
        or not min(item_factors.shape)
        or item_factors.dtype.kind not in {"f", "i", "u"}
        or not np.isfinite(item_factors).all()
    ):
        raise ValueError("ALS item factors must be a finite matrix")
    if history.ndim != 1 or not len(history) or len(np.unique(history)) != len(history):
        raise ValueError("ALS fold-in requires a nonempty unique item history")
    if any(type(value) not in {int, float} for value in (regularization, alpha)):
        raise ValueError("ALS regularization and alpha must be finite numbers")
    if (
        history.dtype.kind not in {"i", "u"}
        or np.any(history < 0)
        or np.any(history >= len(item_factors))
    ):
        raise ValueError("ALS history indexes are outside the catalog")
    if (
        not np.isfinite(regularization)
        or not np.isfinite(alpha)
        or regularization <= 0
        or alpha <= 0
    ):
        raise ValueError("ALS regularization and alpha must be positive")
    factors = item_factors.astype(np.float64, copy=False)
    base = gram if gram is not None else np.einsum("if,ig->fg", factors, factors)
    if (
        base.shape != (factors.shape[1], factors.shape[1])
        or base.dtype.kind not in {"f", "i", "u"}
        or not np.isfinite(base).all()
    ):
        raise ValueError("Cached ALS Gram matrix has invalid dimensions or nonfinite values")
    if not np.allclose(base, base.T):
        raise ValueError("Cached ALS Gram matrix must be symmetric")
    return _solve_factors(factors[history], base, regularization, alpha)


def _solve_factors(
    selected: np.ndarray, base: np.ndarray, regularization: float, alpha: float
) -> np.ndarray:
    """Solve a validated row update; training validates shared factors once."""
    try:
        with np.errstate(over="raise", invalid="raise"):
            system = base + alpha * np.einsum("if,ig->fg", selected, selected)
            system = system + regularization * np.eye(base.shape[0])
            target = (1 + alpha) * selected.sum(axis=0)
        # Finite input factors can still overflow a contraction. Never pass a
        # nonfinite system to LAPACK with its repeated finite scan disabled.
        if not np.isfinite(system).all() or not np.isfinite(target).all():
            raise ValueError("ALS normal equations overflowed")
        result = solve(system, target, assume_a="pos", check_finite=False)
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError("ALS normal equations are not finite positive-definite systems") from error
    if not np.isfinite(result).all():
        raise ValueError("ALS solver produced nonfinite user factors")
    return result


def fit_implicit_als(interactions: np.ndarray, config: ALSConfig) -> tuple[np.ndarray, dict]:
    """Fit item factors using only the supplied training-user interaction matrix."""
    config.validate()
    if interactions.ndim != 2 or not np.isin(interactions, (0, 1)).all():
        raise ValueError("ALS expects a binary user/item interaction matrix")
    if min(interactions.shape) < 2:
        raise ValueError("ALS requires at least two users and items")
    user_items = [np.flatnonzero(row) for row in interactions]
    item_users = [np.flatnonzero(column) for column in interactions.T]
    rng = np.random.default_rng(config.seed)
    item_factors = rng.normal(0, 0.01, size=(interactions.shape[1], config.factors))
    user_factors = np.zeros((interactions.shape[0], config.factors), dtype=np.float64)

    def update(histories: list[np.ndarray], fixed: np.ndarray) -> np.ndarray:
        gram = np.einsum("if,ig->fg", fixed, fixed)
        result = np.zeros((len(histories), config.factors), dtype=np.float64)
        for row, history in enumerate(histories):
            if len(history):
                result[row] = _solve_factors(
                    fixed[history], gram, config.regularization, config.alpha
                )
        return result

    for _ in range(config.iterations):
        user_factors = update(user_items, item_factors)
        item_factors = update(item_users, user_factors)
    if not np.isfinite(item_factors).all():
        raise ValueError("ALS produced nonfinite factors")
    return item_factors.astype(np.float32), {
        "config": asdict(config),
        "training_users": interactions.shape[0],
        "catalog_items": interactions.shape[1],
        "positive_interactions": int(interactions.sum()),
    }


def als_scores(
    item_factors: np.ndarray,
    history: np.ndarray,
    regularization: float,
    alpha: float,
    gram: np.ndarray | None = None,
) -> np.ndarray:
    """Score the complete catalog after folding in the current observed history."""
    user = fold_in(item_factors, history, regularization, alpha, gram)
    return score_items(item_factors, user)
