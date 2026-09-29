"""Stage 1 (per-frame gap/confidence models), stage 2 (per-row LOCAL/OFFLOAD
or ACCEPT/ESCALATE picker — a 0/1 classifier by default, or a cost-aware
utility-margin regressor), the cascade label, and the two router designs —
decide-first and cascade — each in a linear and a gradient-boosted variant.

Stage 1 is fit only on frames that never appear in `simulation_results`, so
its outputs are genuinely out-of-sample for every simulated row (stricter
than "test photos aren't used in training": no simulated photo, train or
test, ever reaches stage 1's training data). Stage 2 is fit only on the
frame-level split's *train* photos, so no held-out test photo's rows reach
it either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from coco.router.evaluation import escalated_utility, local_utility, offload_utility
from coco.router.frame_dataset import CONFIDENCE_COLUMNS, IMAGE_COLUMNS

Design = Literal["decide_first", "cascade"]
Family = Literal["linear", "gbt"]
# "classifier" (default): stage 2 fits a 0/1 classifier on the stored/cascade
# label, as before. "cost_aware": stage 2 regresses the per-row utility
# margin (`stage2_margin`) instead, so a costlier mistake pulls the fit more
# than a cheap one, then picks by the predicted margin's sign.
Objective = Literal["classifier", "cost_aware"]

# The pick each design returns when the margin (classifier's positive class,
# or cost-aware's predicted margin) favors staying local.
STAGE2_POSITIVE_PICK: dict[Design, str] = {"decide_first": "LOCAL", "cascade": "ACCEPT"}
STAGE2_NEGATIVE_PICK: dict[Design, str] = {"decide_first": "OFFLOAD", "cascade": "ESCALATE"}

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

# Stage 2's GBT variants (classifier and cost-aware regressor) must not be
# able to isolate a single training photo in a leaf: `predicted_gap` and
# `p_local_good_enough` are constant within a photo's rows, so an
# unconstrained tree can split on them to memorise each photo's own outcome
# instead of learning a generalizable relationship (train accuracy ~98%,
# test ~55% — see findings.md's S1 section). Requiring at least this many
# photos' worth of rows per leaf makes that impossible by construction.
STAGE2_MIN_LEAF_PHOTOS = 20

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


def _stage2_min_samples_leaf(df_train: pd.DataFrame) -> int:
    """At least `STAGE2_MIN_LEAF_PHOTOS` photos' worth of rows per leaf,
    computed from `df_train` itself (rows per photo = len(df_train) /
    number of unique train frame_ids) rather than a hard-coded row count —
    stays correct if the simulated rows-per-photo count ever changes."""
    n_photos = int(df_train["frame_id"].nunique())
    rows_per_photo = len(df_train) / n_photos
    return int(math.ceil(rows_per_photo * STAGE2_MIN_LEAF_PHOTOS))


def _make_stage2_classifier(family: Family, min_samples_leaf: int) -> _StageClassifier:
    if family == "linear":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return HistGradientBoostingClassifier(random_state=42, min_samples_leaf=min_samples_leaf)


def _make_stage2_regressor(family: Family, min_samples_leaf: int) -> _StageRegressor:
    if family == "linear":
        return make_pipeline(StandardScaler(), LinearRegression())
    return HistGradientBoostingRegressor(random_state=42, min_samples_leaf=min_samples_leaf)


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


def stage2_margin(design: Design, df: pd.DataFrame) -> pd.Series:
    """Per-row utility margin the cost-aware objective regresses: local
    utility minus the alternative path's utility (offload for decide-first,
    the escalated cascade path for cascade). Positive means LOCAL/ACCEPT is
    the better pick, and by how much — the classifier objective's 0/1 label
    throws that magnitude away, which is what makes it cost-blind."""
    alternative = offload_utility(df) if design == "decide_first" else escalated_utility(df)
    return local_utility(df) - alternative


@dataclass(frozen=True)
class Stage2Model:
    """Either a fitted classifier (`objective="classifier"`), a fitted margin
    regressor (`objective="cost_aware"`), or — when the classifier objective's
    training label had only one class (fitting `LogisticRegression`/
    `HistGradientBoostingClassifier` would raise `ValueError` on a single
    class) — a constant pick returned for every row instead.
    `positive_pick`/`negative_pick` are the design's two picks (e.g.
    LOCAL/OFFLOAD), used by `predict_stage2` to translate a cost-aware
    regressor's predicted-margin sign into a pick. `trained_frame_ids` is the
    exact set of `frame_id`s in the rows that reached (or would have reached)
    `.fit()`.
    """

    objective: Objective
    classifier: _StageClassifier | None
    regressor: _StageRegressor | None
    constant_pick: str | None
    positive_pick: str
    negative_pick: str
    trained_frame_ids: frozenset[str]


def train_stage2(
    df_train: pd.DataFrame,
    design: Design,
    family: Family,
    *,
    objective: Objective = "classifier",
    label_column: str = "label",
) -> Stage2Model:
    """Fit stage 2 on `df_train`'s stage-1 outputs + network/device columns.

    `objective="classifier"` (default, unchanged behaviour): fits on the 0/1
    label in `label_column` ("label" for decide-first, "_cascade_label" for
    cascade). If that column has only one distinct value in `df_train` (e.g.
    every train-photo cascade label is ACCEPT), skip fitting — both
    classifier families raise on a single class — and return that value as a
    constant pick for every row instead, which is the only sensible
    prediction when the training data contains no example of the other
    class.

    `objective="cost_aware"`: fits a regressor on `stage2_margin(design,
    df_train)` instead of a 0/1 label, so a mistake's cost (not just its
    direction) drives the fit; `label_column` is unused. A regressor never
    raises on "single-class" data, so there is no constant-pick fallback here.

    Both objectives' GBT family use `min_samples_leaf =
    _stage2_min_samples_leaf(df_train)` (`STAGE2_MIN_LEAF_PHOTOS` photos'
    worth of rows), so a leaf can never be small enough to isolate one
    training photo via `predicted_gap`/`p_local_good_enough` (see that
    constant's docstring). The linear family is unaffected.
    """
    trained_frame_ids = frozenset(df_train["frame_id"])
    positive_pick = STAGE2_POSITIVE_PICK[design]
    negative_pick = STAGE2_NEGATIVE_PICK[design]
    min_samples_leaf = _stage2_min_samples_leaf(df_train)

    if objective == "cost_aware":
        regressor = _make_stage2_regressor(family, min_samples_leaf)
        regressor.fit(df_train[STAGE2_FEATURE_COLUMNS], stage2_margin(design, df_train))
        return Stage2Model(
            objective=objective,
            classifier=None,
            regressor=regressor,
            constant_pick=None,
            positive_pick=positive_pick,
            negative_pick=negative_pick,
            trained_frame_ids=trained_frame_ids,
        )

    labels = df_train[label_column]
    unique_labels = labels.unique()
    if len(unique_labels) == 1:
        return Stage2Model(
            objective=objective,
            classifier=None,
            regressor=None,
            constant_pick=str(unique_labels[0]),
            positive_pick=positive_pick,
            negative_pick=negative_pick,
            trained_frame_ids=trained_frame_ids,
        )
    classifier = _make_stage2_classifier(family, min_samples_leaf)
    classifier.fit(df_train[STAGE2_FEATURE_COLUMNS], labels)
    return Stage2Model(
        objective=objective,
        classifier=classifier,
        regressor=None,
        constant_pick=None,
        positive_pick=positive_pick,
        negative_pick=negative_pick,
        trained_frame_ids=trained_frame_ids,
    )


def predict_stage2(model: Stage2Model, df_test: pd.DataFrame) -> npt.NDArray[np.str_]:
    """One pick per row of `df_test`."""
    if model.objective == "cost_aware":
        assert model.regressor is not None
        predicted_margin = model.regressor.predict(df_test[STAGE2_FEATURE_COLUMNS])
        picks = np.where(predicted_margin > 0, model.positive_pick, model.negative_pick)
        return np.asarray(picks, dtype=np.str_)
    if model.classifier is None:
        assert model.constant_pick is not None
        return np.full(len(df_test), model.constant_pick, dtype=f"<U{len(model.constant_pick)}")
    predictions = model.classifier.predict(df_test[STAGE2_FEATURE_COLUMNS])
    return np.asarray(predictions, dtype=np.str_)


@dataclass(frozen=True)
class RouterResult:
    """A router's picks for the test rows, the exact `test_rows` they were
    predicted on (`picks[i]` is the pick for `test_rows.iloc[i]`), and the
    exact frame ids each stage actually fit on. Score `picks` against this
    `test_rows`, not a separately re-filtered DataFrame — `picks` has no
    index of its own, so a different filter risks a silent row-order
    mismatch. `stage1_trained_frame_ids`/`stage2_trained_frame_ids` are what
    the leakage-guard tests inspect, instead of re-deriving the filters.
    """

    picks: npt.NDArray[np.str_]
    test_rows: pd.DataFrame
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
    objective: Objective = "classifier",
) -> RouterResult:
    """Train and evaluate one router design/family/objective combination:

    1. Stage 1 fits on `frame_table` minus every simulated frame.
    2. Stage 1's outputs are computed for every simulated row (which already
       carries the feature columns `frame_dataset.load_simulated_rows` merged
       in), then stage 2 fits on the train-photo rows only: the classifier
       objective labels them with the stored `label` column (decide-first) or
       the cascade label (cascade); the cost-aware objective instead
       regresses the per-row utility margin (`stage2_margin`) and ignores
       both labels.
    3. Stage 2 predicts picks for the test-photo rows.
    """
    feature_columns = DESIGN_FEATURE_COLUMNS[design]
    stage1 = train_stage1(frame_table, feature_columns, family, simulated_frame_ids)
    rows = predict_stage1(stage1, simulated_rows)

    train_rows = rows[rows["frame_id"].isin(train_frame_ids)].reset_index(drop=True)
    test_rows = rows[rows["frame_id"].isin(test_frame_ids)].reset_index(drop=True)

    if objective == "cost_aware":
        stage2 = train_stage2(train_rows, design, family, objective=objective)
    else:
        if design == "decide_first":
            label_column = "label"
        else:
            train_rows = train_rows.assign(_cascade_label=cascade_label(train_rows))
            label_column = "_cascade_label"
        stage2 = train_stage2(
            train_rows, design, family, objective=objective, label_column=label_column
        )

    picks = predict_stage2(stage2, test_rows)
    return RouterResult(
        picks=picks,
        test_rows=test_rows,
        stage1_trained_frame_ids=stage1.trained_frame_ids,
        stage2_trained_frame_ids=stage2.trained_frame_ids,
    )
