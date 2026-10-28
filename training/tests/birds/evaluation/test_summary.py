"""Tests for `birds.evaluation.summary`, checked against hand computations."""

import math

import pandas as pd
import pytest
from sqlalchemy import Engine

from birds.evaluation.records import DATASET_NAME, PhotoPrediction
from birds.evaluation.store import write_predictions
from birds.evaluation.summary import (
    ESCALATION_PERCENTS,
    format_summary,
    load_summary,
    main,
    run_params,
    summarize,
)


def _frame(
    rows: list[tuple[str, bool, float, float]], run_ids: list[str | None] | None = None
) -> pd.DataFrame:
    """Hand-built rows; every row carries training run "run-1" unless `run_ids` says otherwise."""
    frame = pd.DataFrame(rows, columns=["photo_id", "is_correct", "max_probability", "top2_margin"])
    frame["training_run_id"] = run_ids if run_ids is not None else ["run-1"] * len(rows)
    return frame


# Ten photos. Small-model confidence rises with photo id, so the sweep order is 1..10.
#  id: small_ok large_ok
#   1:  F  T      2:  F  T      3:  F  F      4:  T  T      5:  T  F
#   6:  T  T      7:  T  T      8:  T  T      9:  T  T     10:  F  F
_SMALL_OK = [False, False, False, True, True, True, True, True, True, False]
_LARGE_OK = [True, True, False, True, False, True, True, True, True, False]


def _small() -> pd.DataFrame:
    return _frame([(str(i), ok, i / 10, (11 - i) / 20) for i, ok in enumerate(_SMALL_OK, start=1)])


def _large() -> pd.DataFrame:
    return _frame([(str(i), ok, 0.5, 0.5) for i, ok in enumerate(_LARGE_OK, start=1)])


def test_agreement_matches_hand_computation() -> None:
    a = summarize(_small(), _large()).agreement

    assert a.small_accuracy == pytest.approx(0.6)
    assert a.large_accuracy == pytest.approx(0.7)
    assert a.gap == pytest.approx(0.1)
    assert a.both_right == pytest.approx(0.5)  # 4, 6, 7, 8, 9
    assert a.only_large_right == pytest.approx(0.2)  # 1, 2
    assert a.only_small_right == pytest.approx(0.1)  # 5
    assert a.both_wrong == pytest.approx(0.2)  # 3, 10
    assert a.oracle == pytest.approx(0.8)


def test_separation_matches_hand_computation() -> None:
    separation = summarize(_small(), _large()).separation

    # max_probability rises with id: correct {4,5,6,7,8,9} vs incorrect {1,2,3,10}.
    # Pairs (correct > incorrect): each correct beats 1,2,3 and loses to 10 -> 18/24.
    assert separation[("small", "max_probability")] == pytest.approx(0.75)
    # top2_margin falls with id, the exact mirror: 6 correct x 4 incorrect, 6 wins.
    assert separation[("small", "top2_margin")] == pytest.approx(6 / 24)
    # The large model's confidence is constant: every pair ties, AUC is 0.5.
    assert separation[("large", "max_probability")] == pytest.approx(0.5)


def test_separation_is_nan_when_one_outcome_only() -> None:
    small = _frame([("1", True, 0.9, 0.5), ("2", True, 0.8, 0.4)])
    large = _frame([("1", True, 0.9, 0.5), ("2", False, 0.8, 0.4)])

    separation = summarize(small, large).separation

    assert math.isnan(separation[("small", "max_probability")])
    assert not math.isnan(separation[("large", "max_probability")])


def test_cascade_matches_hand_computation() -> None:
    rows = {r.percent: r for r in summarize(_small(), _large()).cascade["max_probability"]}

    assert [r for r in rows] == list(ESCALATION_PERCENTS)
    assert rows[0].accuracy == pytest.approx(0.6)
    assert rows[0].gap_recovered == pytest.approx(0.0)
    # 10%: escalate photo 1 (large right): 6 + 1 = 7 of 10.
    assert rows[10].escalated == 1
    assert rows[10].accuracy == pytest.approx(0.7)
    assert rows[10].gap_recovered == pytest.approx(1.0)
    # 20%: photos 1, 2 -> 8/10, recovers 2.0 (beats the large model, not clamped).
    assert rows[20].accuracy == pytest.approx(0.8)
    assert rows[20].gap_recovered == pytest.approx(2.0)
    # 30%: photos 1-3 -> 8/10 (photo 3 wrong in both).
    assert rows[30].accuracy == pytest.approx(0.8)
    # 50%: photos 1-5 -> large right on 1,2,4; wrong on 3,5; small right 6-9, wrong 10 -> 7/10.
    assert rows[50].accuracy == pytest.approx(0.7)
    assert rows[5].escalated == 0
    assert rows[5].accuracy == pytest.approx(0.6)


def test_cascade_endpoints_equal_the_two_models() -> None:
    summary = summarize(_small(), _large())

    for rows in summary.cascade.values():
        assert rows[0].accuracy == summary.agreement.small_accuracy
        assert rows[-1].accuracy == summary.agreement.large_accuracy
        assert rows[-1].gap_recovered == pytest.approx(1.0)


def test_zero_gap_gives_nan_recovery() -> None:
    summary = summarize(_small(), _small())

    assert all(math.isnan(r.gap_recovered) for r in summary.cascade["top2_margin"])


def test_tied_confidences_break_on_numeric_photo_id() -> None:
    # Equal confidence everywhere: escalation order must follow numeric id (2 before 10).
    ids = ["10", "2", "1"]
    small = _frame([(i, False, 0.5, 0.5) for i in ids])
    large = _frame([("1", False, 0.5, 0.5), ("2", True, 0.5, 0.5), ("10", False, 0.5, 0.5)])

    rows = {r.percent: r for r in summarize(small, large).cascade["max_probability"]}

    # 50% of 3 photos = 1 escalated, which must be photo 1 (wrong in large): accuracy 0.
    assert rows[50].accuracy == 0.0
    # Reversed input order must give identical output.
    reversed_rows = summarize(small.iloc[::-1], large.iloc[::-1]).cascade["max_probability"]
    assert reversed_rows == summarize(small, large).cascade["max_probability"]


def test_repeated_calls_format_identically() -> None:
    tied = _frame([(str(i), i % 2 == 0, 0.5, 0.5) for i in range(1, 21)])
    other = _frame([(str(i), i % 3 == 0, 0.5, 0.5) for i in range(1, 21)])

    first = format_summary(summarize(tied, other))
    second = format_summary(summarize(tied.sample(frac=1, random_state=1), other))

    assert first == second


def test_mismatched_photo_sets_raise() -> None:
    with pytest.raises(
        ValueError, match="different photos.*python -m birds.evaluation.predictions"
    ):
        summarize(_frame([("1", True, 0.5, 0.5)]), _frame([("2", True, 0.5, 0.5)]))


def test_no_stored_predictions_names_the_recording_command() -> None:
    with pytest.raises(
        ValueError, match="no stored predictions.*python -m birds.evaluation.predictions"
    ):
        summarize(_frame([]), _frame([]))


def _two_photos(run_ids: list[str | None]) -> pd.DataFrame:
    return _frame([("1", True, 0.9, 0.5), ("2", False, 0.8, 0.4)], run_ids)


def test_summary_names_each_models_training_run() -> None:
    out = format_summary(summarize(_two_photos(["a", "a"]), _two_photos(["b", "b"])))

    assert "small   a\n" in out
    assert "large   b\n" in out
    assert "note:" not in out


def test_mixed_training_runs_are_flagged_for_that_model_only() -> None:
    summary = summarize(_two_photos(["a", "a"]), _two_photos(["b", "c"]))
    out = format_summary(summary)

    assert summary.training_runs == {"small": ("a",), "large": ("b", "c")}
    assert "large   b, c" in out
    assert "note: large rows come from 2 training runs" in out
    assert "python -m birds.evaluation.predictions" in out
    assert "note: small" not in out


def test_untraceable_rows_do_not_claim_a_checkpoint() -> None:
    out = format_summary(summarize(_two_photos(["a", "a"]), _two_photos([None, None])))

    assert "large   none recorded" in out
    assert "note: large rows carry no training run, so no checkpoint is named." in out
    assert "note: small" not in out


def test_partly_untraceable_rows_are_noted() -> None:
    summary = summarize(_two_photos(["a", None]), _two_photos(["b", "b"]))

    assert summary.training_runs["small"] == ("a", None)
    assert "note: some small rows carry no training run." in format_summary(summary)


def test_summary_run_params_link_the_checkpoints() -> None:
    summary = summarize(_two_photos(["a", "a"]), _two_photos(["b", "c"]))

    params = run_params(summary)

    assert params["small_training_run_id"] == "a"
    assert params["large_training_run_id"] == "b,c"
    assert params["photos"] == 2


def test_summary_run_params_omit_an_unknown_run() -> None:
    summary = summarize(_two_photos(["a", "a"]), _two_photos([None, None]))

    assert "large_training_run_id" not in run_params(summary)


def _record(photo_id: str, ok: bool) -> PhotoPrediction:
    return PhotoPrediction(photo_id, 1, 1 if ok else 2, ok, 0.6, 0.3)


def test_summary_prints_with_tracking_off(
    postgres_engine: Engine, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for model, flags in (("small", [True, False]), ("large", [True, True])):
        write_predictions(
            postgres_engine,
            DATASET_NAME,
            "test",
            model,
            None,
            [_record("1", flags[0]), _record("2", flags[1])],
        )
    monkeypatch.setattr("birds.evaluation.summary.get_engine", lambda: postgres_engine)
    assert load_summary(postgres_engine).agreement.gap == pytest.approx(0.5)

    main()

    out = capsys.readouterr().out
    assert "Agreement" in out
    assert "Cascade by small-model top2_margin" in out
