"""Turn-level state decoding and consistency checks for the 151-d policy state."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np


STATE_ATOL = 1e-6


def states_equal(left: Sequence[float], right: Sequence[float]) -> bool:
    """Return whether two policy states are numerically identical for validation."""
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    return left_array.shape == right_array.shape and bool(
        np.allclose(left_array, right_array, rtol=0.0, atol=STATE_ATOL)
    )


def state_focus(env: Any, state: Sequence[float]) -> Dict[str, Any]:
    """Decode the leading focus one-hot vector from a policy state."""
    vector = np.asarray(state, dtype=np.float32)[: env.n_exhibits + 1]
    index = int(np.argmax(vector))
    exhibit = env.exhibit_keys[index] if index < env.n_exhibits else None
    return {
        "index": index,
        "exhibit": exhibit,
        "one_hot": vector.astype(float).tolist(),
    }


def state_coverage(env: Any, state: Sequence[float]) -> Dict[str, float]:
    """Decode the five coverage entries at the start of history features."""
    start = env.n_exhibits + 1
    values = np.asarray(state, dtype=np.float32)[start : start + env.n_exhibits]
    return {
        exhibit: float(values[index])
        for index, exhibit in enumerate(env.exhibit_keys)
    }


def expected_coverage(env: Any, coverage: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    return {
        exhibit: float(coverage.get(exhibit, {}).get("coverage", 0.0))
        for exhibit in env.exhibit_keys
    }


def validate_state(
    env: Any,
    state: Sequence[float],
    current_exhibit: str | None,
    coverage: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Validate that encoded focus and coverage describe the supplied snapshot."""
    encoded_focus = state_focus(env, state)
    encoded_coverage = state_coverage(env, state)
    expected = expected_coverage(env, coverage)
    coverage_match = all(
        abs(encoded_coverage[name] - expected[name]) <= STATE_ATOL
        for name in env.exhibit_keys
    )
    return {
        "focus_encoded_in_state": encoded_focus,
        "coverage_encoded_in_state": encoded_coverage,
        "expected_exhibit": current_exhibit,
        "expected_coverage": expected,
        "exhibit_match": encoded_focus["exhibit"] == current_exhibit,
        "coverage_match": coverage_match,
    }

