"""Tests for `router.budget_plot`.

Avoids `import pytest` (per `training/tests/datagen/sampling/test_complexity.py`'s
learning: importing pytest here makes mypy follow `_pytest`'s numpy
integration into a stub incompatible with this project's venv) — `tmp_path`
is still injected by pytest's built-in fixture without importing the package.
"""

from __future__ import annotations

from pathlib import Path

from router.budget_plot import (
    BudgetPanelData,
    BudgetPoint,
    _panel_axis_limits,  # exercised directly for the out-of-range axis-clipping case
    plot_budget_curves,
)


def _panel(budget_ms: float) -> BudgetPanelData:
    return BudgetPanelData(
        budget_ms=budget_ms,
        sweep_curves={
            "raw": [BudgetPoint(50.0, 0.4), BudgetPoint(120.0, 0.6), BudgetPoint(200.0, 0.7)],
            "learned": [
                BudgetPoint(60.0, 0.5),
                BudgetPoint(110.0, 0.65),
                BudgetPoint(190.0, 0.72),
            ],
        },
        tuned_points={
            "raw": BudgetPoint(120.0, 0.6),
            "learned": BudgetPoint(110.0, 0.65),
        },
        single_points={
            "budget-only": BudgetPoint(90.0, 0.55),
            "always-local": BudgetPoint(30.0, 0.45),
            "always-offload": BudgetPoint(180.0, 0.68),
            "oracle": BudgetPoint(100.0, 0.75),
        },
    )


def test_plot_budget_curves_writes_a_non_empty_png(tmp_path: Path) -> None:
    panels = [_panel(100.0), _panel(200.0)]
    output_path = tmp_path / "curves.png"

    plot_budget_curves(panels, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    with output_path.open("rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


def test_plot_budget_curves_handles_a_score_missing_from_some_budgets(tmp_path: Path) -> None:
    # A panel whose sweep/tuned data only has one score, and no single points at all.
    sparse_panel = BudgetPanelData(
        budget_ms=150.0,
        sweep_curves={"raw": [BudgetPoint(80.0, 0.5), BudgetPoint(140.0, 0.62)]},
        tuned_points={"raw": BudgetPoint(140.0, 0.62)},
        single_points={},
    )
    output_path = tmp_path / "sparse.png"

    plot_budget_curves([sparse_panel], output_path)

    assert output_path.stat().st_size > 0


def test_plot_budget_curves_raises_on_no_panels(tmp_path: Path) -> None:
    output_path = tmp_path / "empty.png"
    try:
        plot_budget_curves([], output_path)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an empty panel list")


def _panel_with_offload_at(
    budget_ms: float, offload_latency_ms: float, offload_accuracy: float
) -> BudgetPanelData:
    return BudgetPanelData(
        budget_ms=budget_ms,
        sweep_curves={"raw": [BudgetPoint(50.0, 0.4), BudgetPoint(80.0, 0.5)]},
        tuned_points={"raw": BudgetPoint(80.0, 0.5)},
        single_points={
            "budget-only": BudgetPoint(70.0, 0.45),
            "always-local": BudgetPoint(40.0, 0.4),
            "always-offload": BudgetPoint(offload_latency_ms, offload_accuracy),
            "oracle": BudgetPoint(75.0, 0.55),
        },
    )


def test_panel_axis_limits_clips_far_off_always_offload_latency() -> None:
    # always-offload sits at 300ms while every other point is under 100ms;
    # its accuracy (0.5) is otherwise well within the other points' range.
    panel = _panel_with_offload_at(budget_ms=100.0, offload_latency_ms=300.0, offload_accuracy=0.5)

    xlim, _ylim, offload_off_panel = _panel_axis_limits(panel)

    assert offload_off_panel is True
    assert xlim[1] < 300.0


def test_panel_axis_limits_clips_far_off_always_offload_accuracy() -> None:
    # always-offload's latency (90ms) is in range but its accuracy (0.06) is
    # far below every other point — the real 300ms-budget panel's shape,
    # where the marker would otherwise silently clip past the y-axis.
    panel = _panel_with_offload_at(budget_ms=100.0, offload_latency_ms=90.0, offload_accuracy=0.06)

    xlim, ylim, offload_off_panel = _panel_axis_limits(panel)

    assert offload_off_panel is True
    assert xlim[0] <= 90.0 <= xlim[1]  # latency itself is in range
    assert 0.06 < ylim[0]  # but accuracy falls below the y range


def test_panel_axis_limits_keeps_fully_in_range_always_offload() -> None:
    # always-offload sits within the span the other points already require,
    # on both axes.
    panel = _panel_with_offload_at(budget_ms=100.0, offload_latency_ms=95.0, offload_accuracy=0.42)

    xlim, ylim, offload_off_panel = _panel_axis_limits(panel)

    assert offload_off_panel is False
    assert xlim[0] <= 95.0 <= xlim[1]
    assert ylim[0] <= 0.42 <= ylim[1]


def test_plot_budget_curves_writes_a_valid_png_with_far_off_always_offload(
    tmp_path: Path,
) -> None:
    panel = _panel_with_offload_at(budget_ms=100.0, offload_latency_ms=300.0, offload_accuracy=0.36)
    output_path = tmp_path / "far_offload.png"

    plot_budget_curves([panel], output_path)

    assert output_path.exists()
    with output_path.open("rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"
