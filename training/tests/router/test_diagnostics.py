"""Tests for `router.diagnostics`.

Avoids `import pytest` (per `training/tests/datagen/sampling/test_complexity.py`'s
learning: importing pytest here makes mypy follow `_pytest`'s numpy
integration into a stub incompatible with this project's venv). Plain
`def test_...()` functions are enough for pytest to collect and run these.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from router.diagnostics import compute_feature_diagnostics


def _synthetic_df(n: int = 50, seed: int = 0) -> pd.DataFrame:
    """`gap` plus a perfectly monotonic feature, an inversely monotonic one,
    and an independent random one — enough to exercise rho near +1, -1, and 0.
    """
    rng = np.random.default_rng(seed)
    gap = rng.uniform(-1.0, 1.0, size=n)
    return pd.DataFrame(
        {
            "gap": gap,
            "monotonic_up": gap * 2.0 + 5.0,
            "monotonic_down": -gap * 3.0,
            "independent": rng.uniform(-1.0, 1.0, size=n),
        }
    )


def test_monotonic_feature_gives_rho_of_plus_one() -> None:
    df = _synthetic_df()
    result = compute_feature_diagnostics(df, ["monotonic_up"])
    row = result.iloc[0]
    assert round(row["spearman_rho"], 9) == 1.0


def test_inversely_monotonic_feature_gives_rho_of_minus_one() -> None:
    df = _synthetic_df()
    result = compute_feature_diagnostics(df, ["monotonic_down"])
    row = result.iloc[0]
    assert round(row["spearman_rho"], 9) == -1.0


def test_independent_random_feature_gives_rho_near_zero() -> None:
    df = _synthetic_df(n=2000, seed=1)
    result = compute_feature_diagnostics(df, ["independent"])
    row = result.iloc[0]
    assert abs(row["spearman_rho"]) < 0.1


def test_ordering_is_by_absolute_rho_descending() -> None:
    df = _synthetic_df()
    result = compute_feature_diagnostics(df, ["independent", "monotonic_down", "monotonic_up"])
    assert list(result["feature"]) == ["monotonic_down", "monotonic_up", "independent"]
    abs_rhos = result["spearman_rho"].abs().tolist()
    assert abs_rhos == sorted(abs_rhos, reverse=True)


def test_result_has_expected_columns() -> None:
    df = _synthetic_df()
    result = compute_feature_diagnostics(df, ["monotonic_up", "independent"])
    assert list(result.columns) == ["feature", "spearman_rho", "p_value"]
    assert len(result) == 2
