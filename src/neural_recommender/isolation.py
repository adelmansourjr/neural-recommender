"""Fail-closed membership checks shared by experiment creation and audits."""

from __future__ import annotations

import numpy as np


def validate_membership(values: object, allowed: np.ndarray, name: str) -> None:
    """Reject duplicate, coerced, unknown, or wrongly partitioned user IDs.

    JSON booleans compare equal to integer IDs in Python, so checking set
    membership alone would silently accept a malformed frozen protocol.
    """
    if (
        not isinstance(values, list)
        or not values
        or any(type(user) is not int for user in values)
        or len(set(values)) != len(values)
        or not set(values).issubset(set(map(int, allowed)))
    ):
        raise ValueError(f"{name} must contain unique integer users from its fixed split")
