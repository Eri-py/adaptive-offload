"""Tests for `router.two_stage`, against seeded synthetic frames and rows —
no database.

Avoids `import pytest` (per `training/tests/datagen/sampling/test_complexity.py`'s
learning: importing pytest here makes mypy follow `_pytest`'s numpy
integration into a stub incompatible with this project's venv). Plain
`def test_...()` functions are enough for pytest to collect and run these.

`_make_frame_table` gives frame `gap` an alternating sign by frame index
(even indices offload-better, odd indices local-better) so that *any*
contiguous slice of frames used below — the non-simulated pool, or the
simulated train/test subsets — contains both classes for stage 1's `gap <=
0` classifier and, via fixed local/offload latencies, for the cascade
label too. That keeps every test deterministic regardless of which slice
it uses, instead of relying on chance from an unconstrained RNG draw.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from router.evaluation import escalated_utility, local_utility
from router.frame_dataset import CONFIDENCE_COLUMNS, IMAGE_COLUMNS
from router.two_stage import (
    DESIGN_FEATURE_COLUMNS,
    STAGE2_FEATURE_COLUMNS,
    Design,
    Family,
    Objective,
    cascade_label,
    predict_stage2,
    run_router,
    train_stage1,
    train_stage2,
)

_IMAGE_STAT_COLUMNS = [c for c in IMAGE_COLUMNS if c != "scene_complexity"]
_FRAME_FEATURE_COLUMNS = [*_IMAGE_STAT_COLUMNS, *CONFIDENCE_COLUMNS]

N_FRAMES = 30
# Frames 18..29 are "simulated"; 0..17 never appear in simulation_results.
SIMULATED_START = 18
TRAIN_FRAME_END = 26  # 18..25 train, 26..29 test
LOCAL_LATENCY_MS = 50.0
OFFLOAD_LATENCY_MS = 120.0


def _frame_name(i: int) -> str:
    return f"frame_{i:03d}.jpg"


def _make_frame_table(n_frames: int = N_FRAMES, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    local_accuracy = rng.uniform(0.4, 0.9, n_frames)
    sign = np.where(np.arange(n_frames) % 2 == 0, 1.0, -1.0)
    gap_magnitude = rng.uniform(0.05, 0.25, n_frames)
    offload_accuracy = np.clip(local_accuracy + sign * gap_magnitude, 0.0, 1.0)
    gap = offload_accuracy - local_accuracy

    data: dict[str, object] = {
        "file_name": [_frame_name(i) for i in range(n_frames)],
        "scene_complexity": rng.uniform(0.0, 1.0, n_frames),
        "sharpness": rng.uniform(0.0, 200.0, n_frames),
        "brightness": rng.uniform(0.0, 1.0, n_frames),
        "contrast": rng.uniform(0.0, 1.0, n_frames),
        "colorfulness": rng.uniform(0.0, 100.0, n_frames),
        "entropy": rng.uniform(0.0, 8.0, n_frames),
        "detection_count": rng.integers(0, 10, n_frames).astype(float),
        "max_confidence": rng.uniform(0.0, 1.0, n_frames),
        "mean_confidence": rng.uniform(0.0, 1.0, n_frames),
        "min_confidence": rng.uniform(0.0, 1.0, n_frames),
        "mean_box_area": rng.uniform(0.0, 1.0, n_frames),
        "min_box_area": rng.uniform(0.0, 1.0, n_frames),
        "local_accuracy": local_accuracy,
        "offload_accuracy": offload_accuracy,
        "gap": gap,
    }
    return pd.DataFrame(data)


def _make_simulated_rows(
    frame_table: pd.DataFrame,
    simulated_frame_ids: list[str],
    *,
    rows_per_frame: int = 2,
    seed: int = 1,
) -> pd.DataFrame:
    """Two rows per simulated frame. `label` alternates LOCAL/OFFLOAD by frame
    (matching the alternating gap sign), and latencies are fixed so the
    cascade label is hand-verifiable: with DEFAULT_LAMBDA = 0.3, escalating
    beats accepting whenever gap > 0.036, which every even-indexed (gap > 0)
    frame here clears and every odd-indexed (gap < 0) frame fails.
    """
    rng = np.random.default_rng(seed)
    frames = frame_table.set_index("file_name")
    records = []
    for frame_id in simulated_frame_ids:
        frow = frames.loc[frame_id]
        frame_index = int(frame_id.split("_")[1].split(".")[0])
        label = "OFFLOAD" if frame_index % 2 == 0 else "LOCAL"
        for _ in range(rows_per_frame):
            record = {
                "frame_id": frame_id,
                "network_bandwidth_mbps": float(rng.uniform(1.0, 50.0)),
                "network_latency_ms": float(rng.uniform(5.0, 100.0)),
                "network_packet_loss_pct": float(rng.uniform(0.0, 5.0)),
                "device_load_pct": float(rng.uniform(0.0, 100.0)),
                "local_latency_ms": LOCAL_LATENCY_MS,
                "local_accuracy": frow["local_accuracy"],
                "offload_latency_ms": OFFLOAD_LATENCY_MS,
                "offload_accuracy": frow["offload_accuracy"],
                "label": label,
                **{col: frow[col] for col in ["scene_complexity", *_FRAME_FEATURE_COLUMNS]},
            }
            records.append(record)
    return pd.DataFrame.from_records(records)


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame, set[str], set[str], set[str]]:
    frame_table = _make_frame_table()
    simulated_frame_ids_list = [_frame_name(i) for i in range(SIMULATED_START, N_FRAMES)]
    simulated_rows = _make_simulated_rows(frame_table, simulated_frame_ids_list)
    train_frame_ids = {
        _frame_name(i) for i in range(SIMULATED_START, TRAIN_FRAME_END)
    }
    test_frame_ids = {_frame_name(i) for i in range(TRAIN_FRAME_END, N_FRAMES)}
    simulated_frame_ids = set(simulated_frame_ids_list)
    return frame_table, simulated_rows, train_frame_ids, test_frame_ids, simulated_frame_ids


def test_stage1_training_frames_exclude_every_simulated_frame() -> None:
    frame_table, _, _, _, simulated_frame_ids = _fixture()
    stage1 = train_stage1(
        frame_table, DESIGN_FEATURE_COLUMNS["decide_first"], "linear", simulated_frame_ids
    )
    assert stage1.trained_frame_ids.isdisjoint(simulated_frame_ids)
    assert stage1.trained_frame_ids == set(frame_table["file_name"]) - simulated_frame_ids


def test_stage2_training_rows_exclude_every_test_photo() -> None:
    frame_table, simulated_rows, train_frame_ids, test_frame_ids, simulated_frame_ids = _fixture()
    result = run_router(
        frame_table,
        simulated_rows,
        train_frame_ids,
        test_frame_ids,
        simulated_frame_ids,
        "decide_first",
        "linear",
    )
    assert result.stage2_trained_frame_ids.isdisjoint(test_frame_ids)
    assert result.stage2_trained_frame_ids == train_frame_ids


def test_stage2_cost_aware_training_rows_exclude_every_test_photo() -> None:
    """Same leakage guard as the classifier objective above, for the
    cost-aware objective's regressor path through `run_router`."""
    frame_table, simulated_rows, train_frame_ids, test_frame_ids, simulated_frame_ids = _fixture()
    result = run_router(
        frame_table,
        simulated_rows,
        train_frame_ids,
        test_frame_ids,
        simulated_frame_ids,
        "cascade",
        "gbt",
        "cost_aware",
    )
    assert result.stage2_trained_frame_ids.isdisjoint(test_frame_ids)
    assert result.stage2_trained_frame_ids == train_frame_ids


def test_picks_are_identical_across_two_runs() -> None:
    frame_table, simulated_rows, train_frame_ids, test_frame_ids, simulated_frame_ids = _fixture()
    args = (frame_table, simulated_rows, train_frame_ids, test_frame_ids, simulated_frame_ids)
    first = run_router(*args, "cascade", "gbt")
    second = run_router(*args, "cascade", "gbt")
    assert np.array_equal(first.picks, second.picks)


def test_cascade_label_matches_hand_computed_cases() -> None:
    df = pd.DataFrame(
        {
            # Row 0: offload clearly better -> ESCALATE.
            # Row 1: offload not worth the extra latency -> ACCEPT.
            "local_accuracy": [0.8, 0.8],
            "local_latency_ms": [100.0, 100.0],
            "offload_accuracy": [0.9, 0.8],
            "offload_latency_ms": [200.0, 500.0],
        }
    )
    labels = cascade_label(df)

    # Row 0: escalated = 0.9 - 0.3*(100+200)/1000 = 0.81 > local = 0.77.
    assert math.isclose(escalated_utility(df).iloc[0], 0.81)
    assert math.isclose(local_utility(df).iloc[0], 0.77)
    # Row 1: escalated = 0.8 - 0.3*(100+500)/1000 = 0.62 < local = 0.77.
    assert math.isclose(escalated_utility(df).iloc[1], 0.62)
    assert math.isclose(local_utility(df).iloc[1], 0.77)

    assert list(labels) == ["ESCALATE", "ACCEPT"]


def test_all_design_family_objective_combinations_return_one_pick_per_test_row() -> None:
    frame_table, simulated_rows, train_frame_ids, test_frame_ids, simulated_frame_ids = _fixture()
    n_test_rows = len(simulated_rows[simulated_rows["frame_id"].isin(test_frame_ids)])
    assert n_test_rows > 0

    valid_picks = {
        "decide_first": {"LOCAL", "OFFLOAD"},
        "cascade": {"ACCEPT", "ESCALATE"},
    }
    designs: list[Design] = ["decide_first", "cascade"]
    families: list[Family] = ["linear", "gbt"]
    objectives: list[Objective] = ["classifier", "cost_aware"]
    for design in designs:
        for family in families:
            for objective in objectives:
                result = run_router(
                    frame_table,
                    simulated_rows,
                    train_frame_ids,
                    test_frame_ids,
                    simulated_frame_ids,
                    design,
                    family,
                    objective,
                )
                assert len(result.picks) == n_test_rows
                assert set(result.picks) <= valid_picks[design]


def test_cost_aware_picks_are_identical_across_two_runs() -> None:
    frame_table, simulated_rows, train_frame_ids, test_frame_ids, simulated_frame_ids = _fixture()
    args = (frame_table, simulated_rows, train_frame_ids, test_frame_ids, simulated_frame_ids)
    first = run_router(*args, "cascade", "gbt", "cost_aware")
    second = run_router(*args, "cascade", "gbt", "cost_aware")
    assert np.array_equal(first.picks, second.picks)


def test_stage2_single_class_label_returns_that_constant_for_every_row() -> None:
    df_train = pd.DataFrame(
        {
            "frame_id": ["frame_a.jpg", "frame_a.jpg", "frame_b.jpg"],
            "predicted_gap": [0.1, 0.1, -0.1],
            "p_local_good_enough": [0.4, 0.4, 0.6],
            "network_bandwidth_mbps": [10.0, 10.0, 20.0],
            "network_latency_ms": [30.0, 30.0, 40.0],
            "network_packet_loss_pct": [0.0, 0.0, 1.0],
            "device_load_pct": [10.0, 10.0, 20.0],
            "label": ["ACCEPT", "ACCEPT", "ACCEPT"],
        }
    )
    df_test = pd.DataFrame(
        {
            "predicted_gap": [0.2, -0.2],
            "p_local_good_enough": [0.3, 0.7],
            "network_bandwidth_mbps": [15.0, 25.0],
            "network_latency_ms": [35.0, 45.0],
            "network_packet_loss_pct": [0.5, 1.5],
            "device_load_pct": [15.0, 25.0],
        }
    )

    model = train_stage2(df_train, "cascade", "linear", label_column="label")
    assert model.objective == "classifier"
    assert model.classifier is None
    assert model.regressor is None
    assert model.constant_pick == "ACCEPT"
    assert model.positive_pick == "ACCEPT"
    assert model.negative_pick == "ESCALATE"
    assert model.trained_frame_ids == {"frame_a.jpg", "frame_b.jpg"}

    picks = predict_stage2(model, df_test)
    assert list(picks) == ["ACCEPT", "ACCEPT"]


def test_stage2_cost_aware_picks_follow_sign_of_predicted_margin() -> None:
    """Cost-aware stage 2 regresses the utility margin (local minus the
    alternative path) instead of a 0/1 label, then picks LOCAL/OFFLOAD by the
    sign of the predicted margin. `predicted_gap` is the only feature that
    varies between the two training frames, and is set to fully determine the
    margin's sign (frame_a: local wins big; frame_b: offload wins big), so a
    linear regressor fits an exact line through the two (predicted_gap,
    margin) points and its test-time prediction is hand-verifiable without
    needing the fitted coefficients."""
    constant_columns = {
        "network_bandwidth_mbps": [10.0, 10.0, 10.0, 10.0],
        "network_latency_ms": [30.0, 30.0, 30.0, 30.0],
        "network_packet_loss_pct": [0.0, 0.0, 0.0, 0.0],
        "device_load_pct": [10.0, 10.0, 10.0, 10.0],
        "p_local_good_enough": [0.5, 0.5, 0.5, 0.5],
    }
    df_train = pd.DataFrame(
        {
            "frame_id": ["frame_a.jpg", "frame_a.jpg", "frame_b.jpg", "frame_b.jpg"],
            "predicted_gap": [-1.0, -1.0, 1.0, 1.0],
            # local_utility - offload_utility: frame_a's local pass is fast
            # and accurate (margin ~= +1.0), frame_b's is slow and inaccurate
            # (margin ~= -1.0) — offload wins there instead.
            "local_accuracy": [0.95, 0.95, 0.10, 0.10],
            "local_latency_ms": [10.0, 10.0, 500.0, 500.0],
            "offload_accuracy": [0.10, 0.10, 0.95, 0.95],
            "offload_latency_ms": [500.0, 500.0, 10.0, 10.0],
            **constant_columns,
        }
    )
    df_test = pd.DataFrame(
        {"predicted_gap": [-1.0, 1.0], **{k: v[:2] for k, v in constant_columns.items()}}
    )

    model = train_stage2(df_train, "decide_first", "linear", objective="cost_aware")
    assert model.objective == "cost_aware"
    assert model.classifier is None
    assert model.regressor is not None
    assert model.constant_pick is None
    assert model.positive_pick == "LOCAL"
    assert model.negative_pick == "OFFLOAD"
    assert model.trained_frame_ids == {"frame_a.jpg", "frame_b.jpg"}

    picks = predict_stage2(model, df_test)
    assert list(picks) == ["LOCAL", "OFFLOAD"]


def test_gbt_stage2_classifier_does_not_memorise_per_photo_noise() -> None:
    """Guards S1: `predicted_gap`/`p_local_good_enough` are constant within a
    photo, so an unconstrained `HistGradientBoostingClassifier` can split on
    them to learn each training photo's own label instead of a generalizable
    relationship. Here the label is pure per-photo noise (an even LOCAL/
    OFFLOAD split fixed by photo index), independent of every feature (drawn
    from a separate RNG stream), so any train accuracy above the 50% base
    rate is memorisation, not signal. First confirms the unconstrained
    premise directly (sklearn's default `min_samples_leaf=20` reaches
    perfect train accuracy by isolating each photo's 10 rows). Then asserts
    `train_stage2`'s constrained GBT stays at the base rate: with exactly 20
    photos here, `_stage2_min_samples_leaf` (`STAGE2_MIN_LEAF_PHOTOS` = 20
    photos' worth of rows) requires a leaf as large as the whole training
    set, so the tree cannot split at all and just predicts the majority
    class for every row — the strongest possible demonstration that no leaf
    can isolate a single photo."""
    n_photos = 20
    rows_per_photo = 10
    feature_rng = np.random.default_rng(8)

    frame_ids = [f"frame_{i:03d}.jpg" for i in range(n_photos)]
    photo_labels = np.array(["LOCAL"] * (n_photos // 2) + ["OFFLOAD"] * (n_photos // 2))

    records = []
    for i, frame_id in enumerate(frame_ids):
        # Constant within the photo, like real `predicted_gap`/
        # `p_local_good_enough` — and drawn independently of `photo_labels`,
        # so they carry zero real signal about the label.
        predicted_gap = feature_rng.uniform(-1.0, 1.0)
        p_local_good_enough = feature_rng.uniform(0.0, 1.0)
        for _ in range(rows_per_photo):
            records.append(
                {
                    "frame_id": frame_id,
                    "predicted_gap": predicted_gap,
                    "p_local_good_enough": p_local_good_enough,
                    "network_bandwidth_mbps": feature_rng.uniform(1.0, 50.0),
                    "network_latency_ms": feature_rng.uniform(5.0, 100.0),
                    "network_packet_loss_pct": feature_rng.uniform(0.0, 5.0),
                    "device_load_pct": feature_rng.uniform(0.0, 100.0),
                    "label": str(photo_labels[i]),
                }
            )
    df_train = pd.DataFrame.from_records(records)

    unconstrained = HistGradientBoostingClassifier(random_state=42)
    unconstrained.fit(df_train[STAGE2_FEATURE_COLUMNS], df_train["label"])
    unconstrained_accuracy = float(
        (unconstrained.predict(df_train[STAGE2_FEATURE_COLUMNS]) == df_train["label"]).mean()
    )
    assert unconstrained_accuracy > 0.95

    model = train_stage2(df_train, "decide_first", "gbt")
    assert model.classifier is not None
    picks = predict_stage2(model, df_train)
    constrained_accuracy = float((picks == df_train["label"].to_numpy()).mean())
    assert math.isclose(constrained_accuracy, 0.5, abs_tol=0.05)
