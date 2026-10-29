"""Transparent history buckets and fixed score blends used in research and serving."""

from __future__ import annotations

import numpy as np

from .baselines import normalize_scores


def history_bucket(length: int) -> str:
    """Map one to twenty likes to the nearest validated history size (ties lower)."""
    if type(length) is not int or not 1 <= length <= 20:
        raise ValueError("Conditional rankers support histories of one to twenty liked movies")
    return str(min((1, 5, 10), key=lambda value: abs(value - length)))


def combine_scores(scores: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Fuse normalized component scores using frozen nonnegative weights."""
    if not weights or any(
        type(value) not in {int, float} or not np.isfinite(value) or value < 0
        for value in weights.values()
    ):
        raise ValueError("Blend weights must be finite and nonnegative")
    if not np.isclose(sum(weights.values()), 1) or set(weights) - scores.keys():
        raise ValueError("Blend must sum to one and reference available rankers")
    active = {name: value for name, value in weights.items() if value > 0}
    if len(active) == 1:
        # Preserve the original component score units when no blending is needed.
        return scores[next(iter(active))].copy()
    return sum(weight * normalize_scores(scores[name]) for name, weight in active.items())


def validate_selector(selector: dict[str, dict], available: set[str]) -> None:
    """Reject malformed, incomplete, or recursive model-selection metadata."""
    if not isinstance(selector, dict) or set(selector) != {"1", "5", "10"}:
        raise ValueError("History selector must define exactly the validated 1/5/10 buckets")
    for choice in selector.values():
        if not isinstance(choice, dict) or not isinstance(choice.get("name"), str):
            raise ValueError("History selector choices need an explicit candidate name")
        weights = choice.get("weights")
        if not isinstance(weights, dict) or set(weights) - available:
            raise ValueError("History selector references unavailable components")
        combine_scores({name: np.zeros(2) for name in available}, weights)
