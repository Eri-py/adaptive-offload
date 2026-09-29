"""Feature diagnostics: how strongly each candidate feature relates to the
per-frame accuracy gap (`offload_accuracy - local_accuracy`), via Spearman
rank correlation. Pure — no DB access; callers pass in `load_frame_table`'s
DataFrame (or any DataFrame with a `gap` column) and the feature columns to
check.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from scipy.stats import spearmanr


def compute_feature_diagnostics(
    df: pd.DataFrame, feature_columns: Sequence[str], *, gap_column: str = "gap"
) -> pd.DataFrame:
    """Spearman rho and p-value of each of `feature_columns` against `gap_column`,
    as a `feature`/`spearman_rho`/`p_value` DataFrame ordered by |rho| descending.
    """
    rows = []
    gap = df[gap_column]
    for feature in feature_columns:
        rho, p_value = spearmanr(df[feature], gap)
        rows.append({"feature": feature, "spearman_rho": float(rho), "p_value": float(p_value)})
    result = pd.DataFrame(rows, columns=["feature", "spearman_rho", "p_value"])
    order = result["spearman_rho"].abs().sort_values(ascending=False).index
    return result.reindex(order).reset_index(drop=True)
