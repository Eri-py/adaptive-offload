"""Accuracy-vs-latency figure for the latency-budget router.

One panel per budget, x = mean latency (ms), y = on-time accuracy. Each panel
draws the threshold-sweep curve for each confidence score, that score's tuned
operating point, single markers for budget-only/always-local/always-offload/
the oracle, and a vertical line at the budget.

Takes only plain point data (`BudgetPoint`/`BudgetPanelData`) — no dependency
on the router or the database, so it can be tested with synthetic data.
`budget_experiment.py` (Task 5) builds these from
`router.latency_policies.RouterMetrics`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: this module only ever saves a file

import matplotlib.pyplot as plt  # noqa: E402 (must follow matplotlib.use)
from matplotlib.artist import Artist  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402


@dataclass(frozen=True)
class BudgetPoint:
    """One (mean latency ms, on-time accuracy) point on a panel."""

    mean_latency_ms: float
    on_time_accuracy: float


@dataclass(frozen=True)
class BudgetPanelData:
    """Everything one budget's panel needs, already reduced to plain points."""

    budget_ms: float
    sweep_curves: dict[str, list[BudgetPoint]]  # score name -> ordered threshold sweep
    tuned_points: dict[str, BudgetPoint]  # score name -> its tuned operating point
    single_points: dict[str, BudgetPoint]  # "budget-only"/"always-local"/"always-offload"/"oracle"


# Categorical hues in a fixed order (never cycled/reassigned per-panel) so a
# score keeps the same color across every panel — Okabe-Ito colorblind-safe set.
_KNOWN_SCORE_COLORS: dict[str, str] = {
    "raw": "#0072B2",  # blue
    "learned": "#D55E00",  # vermillion
}
_FALLBACK_SCORE_COLORS = ("#009E73", "#CC79A7", "#56B4E9")  # for score names beyond the two above

# label -> (marker, color), also fixed and never cycled.
_SINGLE_POINT_STYLE: dict[str, tuple[str, str]] = {
    "budget-only": ("s", "#000000"),
    "always-local": ("^", "#999999"),
    "always-offload": ("v", "#56B4E9"),
    "oracle": ("*", "#CC79A7"),
}
_SINGLE_POINT_ORDER = ("budget-only", "always-local", "always-offload", "oracle")

# Fraction of the data span reserved as padding on each axis, with a floor so a
# near-zero span (e.g. every x value equal) still gets a visible margin.
_AXIS_PAD_FRAC = 0.08
_X_PAD_FLOOR_MS = 5.0
_Y_PAD_FLOOR = 0.02
# Fraction of the (already padded) axis span an edge-clamped marker/annotation
# is inset from the border, so it never sits on or past the frame.
_EDGE_INSET_FRAC = 0.04


def _padded_range(values: Sequence[float], pad_floor: float) -> tuple[float, float]:
    """Min/max of `values` expanded by `_AXIS_PAD_FRAC` of the span (or
    `pad_floor` if that span is ~0)."""
    low, high = min(values), max(values)
    pad = max((high - low) * _AXIS_PAD_FRAC, pad_floor)
    return low - pad, high + pad


def _panel_axis_limits(
    panel: BudgetPanelData,
) -> tuple[tuple[float, float], tuple[float, float], bool]:
    """x/y limits sized to every point except always-offload plus the budget
    line (for x only), and whether always-offload's true (latency, accuracy)
    falls outside the resulting x or y range and needs edge treatment instead
    of being plotted at its true, possibly off-panel (and so invisibly
    clipped), position."""
    xs: list[float] = [panel.budget_ms]
    ys: list[float] = []
    for points in panel.sweep_curves.values():
        xs.extend(p.mean_latency_ms for p in points)
        ys.extend(p.on_time_accuracy for p in points)
    for point in panel.tuned_points.values():
        xs.append(point.mean_latency_ms)
        ys.append(point.on_time_accuracy)
    for name, point in panel.single_points.items():
        if name == "always-offload":
            continue
        xs.append(point.mean_latency_ms)
        ys.append(point.on_time_accuracy)

    xlim = _padded_range(xs, _X_PAD_FLOOR_MS)
    ylim = _padded_range(ys, _Y_PAD_FLOOR) if ys else (0.0, 1.0)

    offload = panel.single_points.get("always-offload")
    offload_off_panel = offload is not None and (
        offload.mean_latency_ms > xlim[1]
        or offload.on_time_accuracy < ylim[0]
        or offload.on_time_accuracy > ylim[1]
    )
    return xlim, ylim, offload_off_panel


def _score_color(score_name: str, all_score_names: Sequence[str]) -> str:
    """Fixed color per score name; a name beyond the known two gets a
    deterministic fallback color based on its sorted position among the
    other unknown names, so repeated calls never reassign colors."""
    if score_name in _KNOWN_SCORE_COLORS:
        return _KNOWN_SCORE_COLORS[score_name]
    unknown = sorted(name for name in all_score_names if name not in _KNOWN_SCORE_COLORS)
    return _FALLBACK_SCORE_COLORS[unknown.index(score_name) % len(_FALLBACK_SCORE_COLORS)]


def _plot_panel(ax: Axes, panel: BudgetPanelData, handles_by_label: dict[str, Artist]) -> None:
    xlim, ylim, offload_off_panel = _panel_axis_limits(panel)
    score_names = sorted(panel.sweep_curves)  # deterministic order, independent of dict insertion

    for score_name in score_names:
        points = panel.sweep_curves[score_name]
        if not points:
            continue
        color = _score_color(score_name, score_names)

        sweep_label = f"{score_name} sweep"
        (line,) = ax.plot(
            [p.mean_latency_ms for p in points],
            [p.on_time_accuracy for p in points],
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
            label=sweep_label,
        )
        handles_by_label.setdefault(sweep_label, line)

        tuned = panel.tuned_points.get(score_name)
        if tuned is not None:
            tuned_label = f"{score_name} tuned"
            scatter = ax.scatter(
                [tuned.mean_latency_ms],
                [tuned.on_time_accuracy],
                color=color,
                edgecolor="black",
                marker="D",
                s=90,
                zorder=5,
                label=tuned_label,
            )
            handles_by_label.setdefault(tuned_label, scatter)

    for name in _SINGLE_POINT_ORDER:
        point = panel.single_points.get(name)
        if point is None:
            continue
        marker, color = _SINGLE_POINT_STYLE[name]
        plot_x, plot_y = point.mean_latency_ms, point.on_time_accuracy
        if name == "always-offload" and offload_off_panel:
            # True (latency, accuracy) is off-panel on at least one axis:
            # clamp only the axis that's actually out of range to its inset
            # edge, and label it with the real values, instead of it silently
            # vanishing past the axis limits.
            x_inset = (xlim[1] - xlim[0]) * _EDGE_INSET_FRAC
            y_inset = (ylim[1] - ylim[0]) * _EDGE_INSET_FRAC
            if plot_x > xlim[1]:
                plot_x = xlim[1] - x_inset
            if plot_y < ylim[0]:
                plot_y = ylim[0] + y_inset
            elif plot_y > ylim[1]:
                plot_y = ylim[1] - y_inset
        scatter = ax.scatter(
            [plot_x],
            [plot_y],
            marker=marker,
            color=color,
            edgecolor="black",
            s=110,
            zorder=6,
            label=name,
        )
        handles_by_label.setdefault(name, scatter)
        if name == "always-offload" and offload_off_panel:
            ax.annotate(
                f"always-offload → {point.mean_latency_ms:.0f} ms, "
                f"{point.on_time_accuracy:.2f}",
                xy=(plot_x, plot_y),
                xytext=(-6, 6),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=7,
                color=color,
            )

    budget_line = ax.axvline(
        panel.budget_ms, color="#999999", linestyle="--", linewidth=1, label="budget"
    )
    handles_by_label.setdefault("budget", budget_line)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"{panel.budget_ms:g} ms budget")
    ax.set_xlabel("Mean latency (ms)")
    ax.set_ylabel("On-time accuracy")


def plot_budget_curves(panels: Sequence[BudgetPanelData], output_path: str | Path) -> None:
    """Write the accuracy-vs-latency figure (one panel per budget) to
    `output_path` as a PNG, creating parent directories if needed."""
    if not panels:
        raise ValueError("plot_budget_curves requires at least one budget panel")

    ncols = min(3, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    # No sharey: each panel's y-range is now clipped to its own points (see
    # `_panel_axis_limits`), so a shared y-axis would defeat that clipping.
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    handles_by_label: dict[str, Artist] = {}
    for ax, panel in zip(axes_flat, panels, strict=False):
        _plot_panel(ax, panel, handles_by_label)

    # More grid cells than panels (e.g. 6 budgets in a 2x3 grid always fills it,
    # but a smaller/odd panel count wouldn't) — hide the unused trailing axes.
    for ax in axes_flat[len(panels) :]:
        ax.set_visible(False)

    fig.suptitle("Accuracy vs. latency under a hard budget")
    fig.legend(
        handles_by_label.values(),
        handles_by_label.keys(),
        loc="lower center",
        ncol=min(len(handles_by_label), 6),
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout(rect=(0.0, 0.1, 1.0, 1.0))

    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches="tight" so a legend row that spills past the tight_layout
    # rect (e.g. many score names) is still fully included, never clipped.
    fig.savefig(resolved_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
