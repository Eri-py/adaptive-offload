"""Stage 1 (per-frame gap/confidence models), stage 2 (per-row LOCAL/OFFLOAD
or ACCEPT/ESCALATE classifier), the cascade label, and the two router
designs — decide-first and cascade — each in a linear and a
gradient-boosted variant.

Stage 1 is fit only on frames that never appear in `simulation_results`, so
its outputs are genuinely out-of-sample for every simulated row (stricter
than "test photos aren't used in training": no simulated photo, train or
test, ever reaches stage 1's training data). Stage 2 is fit only on the
frame-level split's *train* photos, so no held-out test photo's rows reach
it either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from router.evaluation import escalated_utility, local_utility
from router.frame_dataset import CONFIDENCE_COLUMNS, IMAGE_COLUMNS

Design = Literal["decide_first", "cascade"]
Family = Literal["linear", "gbt"]

# Stage 1's feature columns per design: decide-first is image-side only;
# cascade adds the local model's own confidence features (Task 1 constants).
DESIGN_FEATURE_COLUMNS: dict[Design, list[str]] = {
    "decide_first": list(IMAGE_COLUMNS),
    "cascade": [*IMAGE_COLUMNS, *CONFIDENCE_COLUMNS],
}

# Stage 2's non-stage-1 inputs: network + device conditions. No scene
# complexity — that's already folded into stage 1's predictions.
STAGE2_NETWORK_COLUMNS = [
    "network_bandwidth_mbps",
    "network_latency_ms",
    "network_packet_loss_pct",
    "device_load_pct",
]
STAGE2_FEATURE_COLUMNS = ["predicted_gap", "p_local_good_enough", *STAGE2_NETWORK_COLUMNS]

_StageRegressor: TypeAlias = Pipeline | HistGradientBoostingRegressor
_StageClassifier: TypeAlias = Pipeline | HistGradientBoostingClassifier


def _make_stage1_models(family: Family) -> tuple[_StageRegressor, _StageClassifier]:
    """A regressor for `gap` and a classifier for `gap <= 0`, same family."""
    if family == "linear":
        return (
            make_pipeline(StandardScaler(), LinearRegression()),
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        )
    return (
        HistGradientBoostingRegressor(random_state=42),
        HistGradientBoostingClassifier(random_state=42),
    )


def _make_stage2_classifier(family: Family) -> _StageClassifier:
    if family == "linear":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return HistGradientBoostingClassifier(random_state=42)


def _positive_class_proba(
    classifier: _StageClassifier, features: pd.DataFrame
) -> npt.NDArray[np.float64]:
    """`predict_proba`'s column for the `True` class, regardless of `classes_`
    order (Pipeline/HistGradientBoostingClassifier both sort `classes_`, so
    this is normally column 1, but looking it up avoids relying on that)."""
    proba = classifier.predict_proba(features)
    classes = list(classifier.classes_)
    return np.asarray(proba[:, classes.index(True)], dtype=np.float64)


@dataclass(frozen=True)
class Stage1Models:
    """Stage 1's fitted regressor + classifier, the feature columns they were
    fit on, and the exact frame `file_name`s that reached `.fit()` — the last
    field is what the leakage-guard test inspects, rather than re-deriving
    the simulated-frame filter itself.
    """

    regressor: _StageRegressor
    classifier: _StageClassifier
    feature_columns: list[str]
    trained_frame_ids: frozenset[str]


def train_stage1(
    frame_table: pd.DataFrame,
    feature_columns: list[str],
    family: Family,
    simulated_frame_ids: set[str],
) -> Stage1Models:
    """Fit stage 1 on every frame in `frame_table` except the simulated ones —
    no simulated photo, train or test, ever reaches this fit."""
    training_frames = frame_table[~frame_table["file_name"].isin(simulated_frame_ids)]
    regressor, classifier = _make_stage1_models(family)
    regressor.fit(training_frames[feature_columns], training_frames["gap"])
    classifier.fit(training_frames[feature_columns], training_frames["gap"] <= 0)
    return Stage1Models(
        regressor=regressor,
        classifier=classifier,
        feature_columns=feature_columns,
        trained_frame_ids=frozenset(training_frames["file_name"]),
    )


def predict_stage1(models: Stage1Models, df: pd.DataFrame) -> pd.DataFrame:
    """Add `predicted_gap` and `p_local_good_enough` columns to `df`, which
    must carry `models.feature_columns`. Works on the per-frame table or
    directly on the simulated rows — both already carry the same feature
    columns (`frame_dataset.load_simulated_rows` merges them in), so no
    frame-table merge is needed here.
    """
    features = df[models.feature_columns]
    predicted_gap = models.regressor.predict(features)
    p_local_good_enough = _positive_class_proba(models.classifier, features)
    return df.assign(predicted_gap=predicted_gap, p_local_good_enough=p_local_good_enough)


def cascade_label(df: pd.DataFrame) -> pd.Series:
    """Per-row ACCEPT/ESCALATE: ESCALATE if the escalated utility (offload
    accuracy, local + offload latency) beats local utility, else ACCEPT."""
    escalate = escalated_utility(df) > local_utility(df)
    return pd.Series(np.where(escalate, "ESCALATE", "ACCEPT"), index=df.index)


@dataclass(frozen=True)
class Stage2Model:
    """Either a fitted classifier, or — when the training label had only one
    class (fitting `LogisticRegression`/`HistGradientBoostingClassifier`
    would raise `ValueError` on a single class) — a constant pick returned
    for every row instead. `trained_frame_ids` is the exact set of `frame_id`s
    in the rows that reached (or would have reached) `.fit()`.
    """

    classifier: _StageClassifier | None
    constant_pick: str | None
    trained_frame_ids: frozenset[str]


def train_stage2(df_train: pd.DataFrame, label_column: str, family: Family) -> Stage2Model:
    """Fit stage 2 on `df_train`'s stage-1 outputs + network/device columns.

    If `label_column` has only one distinct value in `df_train` (e.g. every
    train-photo cascade label is ACCEPT), skip fitting — both model families
    raise on a single class — and return that value as a constant pick for
    every row instead, which is the only sensible prediction when the
    training data contains no example of the other class.
    """
    labels = df_train[label_column]
    trained_frame_ids = frozenset(df_train["frame_id"])
    unique_labels = labels.unique()
    if len(unique_labels) == 1:
        return Stage2Model(
            classifier=None,
            constant_pick=str(unique_labels[0]),
            trained_frame_ids=trained_frame_ids,
        )
    classifier = _make_stage2_classifier(family)
    classifier.fit(df_train[STAGE2_FEATURE_COLUMNS], labels)
    return Stage2Model(
        classifier=classifier, constant_pick=None, trained_frame_ids=trained_frame_ids
    )


def predict_stage2(model: Stage2Model, df_test: pd.DataFrame) -> npt.NDArray[np.str_]:
    """One pick per row of `df_test`."""
    if model.classifier is None:
        assert model.constant_pick is not None
        return np.full(len(df_test), model.constant_pick, dtype=f"<U{len(model.constant_pick)}")
    predictions = model.classifier.predict(df_test[STAGE2_FEATURE_COLUMNS])
    return np.asarray(predictions, dtype=np.str_)


@dataclass(frozen=True)
class RouterResult:
    """A router's picks for the test rows, plus the exact frame ids each stage
    actually fit on — what the leakage-guard tests inspect, instead of
    re-deriving the simulated/train/test filters themselves in the test.
    """

    picks: npt.NDArray[np.str_]
    stage1_trained_frame_ids: frozenset[str]
    stage2_trained_frame_ids: frozenset[str]


def run_router(
    frame_table: pd.DataFrame,
    simulated_rows: pd.DataFrame,
    train_frame_ids: set[str],
    test_frame_ids: set[str],
    simulated_frame_ids: set[str],
    design: Design,
    family: Family,
) -> RouterResult:
    """Train and evaluate one router design/family combination:

    1. Stage 1 fits on `frame_table` minus every simulated frame.
    2. Stage 1's outputs are computed for every simulated row (which already
       carries the feature columns `frame_dataset.load_simulated_rows` merged
       in), then stage 2 fits on the train-photo rows only, labelled with the
       stored `label` column (decide-first) or the cascade label (cascade).
    3. Stage 2 predicts picks for the test-photo rows.
    """
    feature_columns = DESIGN_FEATURE_COLUMNS[design]
    stage1 = train_stage1(frame_table, feature_columns, family, simulated_frame_ids)
    rows = predict_stage1(stage1, simulated_rows)

    train_rows = rows[rows["frame_id"].isin(train_frame_ids)].reset_index(drop=True)
    test_rows = rows[rows["frame_id"].isin(test_frame_ids)].reset_index(drop=True)

    if design == "decide_first":
        label_column = "label"
    else:
        train_rows = train_rows.assign(_cascade_label=cascade_label(train_rows))
        label_column = "_cascade_label"

    stage2 = train_stage2(train_rows, label_column, family)
    picks = predict_stage2(stage2, test_rows)
    return RouterResult(
        picks=picks,
        stage1_trained_frame_ids=stage1.trained_frame_ids,
        stage2_trained_frame_ids=stage2.trained_frame_ids,
    )
