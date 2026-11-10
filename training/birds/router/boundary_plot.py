"""Boundary heatmap for the birds latency-budget cascade.

One panel per budget over the local-cost (x) by offload-cost (y) plane. The fill says how the
cascade compares with the best *deployable* policy in each cell, and the lines drawn over it are
the budget's band edges, which is where the structure of this experiment actually lives.

Why the fill is "cascade vs best deployable" and not an argmax over policies:
- The within-budget oracle is a bound that knows each photo's correctness, so it wins every cell
  by construction. It is not an input here.
- Budget-only equals max(always-local, always-offload) in every cell, and escalation costs
  local + offload, so any budget that admits escalation also admits plain offload. The cascade
  cannot win on latency, only (barely) on accuracy, so an argmax would read "budget-only
  everywhere". Ties are therefore first-class: a cascade tie, loss and win look different.

Takes only plain grid data (`BoundaryPanelData`) and marked positions (`BoundaryMarks`); no
router or database import, so it is testable with synthetic input. Offload positions of the
named conditions depend on a server-compute value this module must not invent, so the caller
passes them in.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: this module only ever saves a file

import matplotlib.pyplot as plt  # noqa: E402 (must follow matplotlib.use)
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from birds.router.conditions import SMALL_DESKTOP_CPU_MS  # noqa: E402

# Accuracy gap at or below which the cascade counts as tied with the best deployable policy.
# 0.001 is about six of the 5,794 test photos: the real cascade's best case beats always-offload
# by two photos (0.8702 vs 0.8699), which is noise, not a win. Also below bootstrap resolution.
TIE_TOLERANCE = 0.001

CASCADE = "cascade"
# The deployable policies; the oracle is deliberately absent (a bound, not a policy).
DEPLOYABLE_POLICIES = ("always-local", "always-offload", "budget-only", CASCADE)

# Fill codes, in legend order.
NO_POLICY_ON_TIME, CASCADE_LOSES, CASCADE_TIES, CASCADE_WINS = 0, 1, 2, 3
_FILL_LABELS = (
    "no policy on time",
    "cascade loses to best deployable",
    "cascade ties best deployable",
    "cascade beats best deployable",
)
# Okabe-Ito, matching budget_plot.py; neutral grey for the empty region.
_FILL_COLORS = ("#DDDDDD", "#56B4E9", "#E69F00", "#009E73")


@dataclass(frozen=True)
class BoundaryPanelData:
    """One budget's grid. `accuracy[policy][i, j]` is on-time accuracy at offload_ms[i],
    local_ms[j]."""

    budget_ms: float
    local_ms: Sequence[float]
    offload_ms: Sequence[float]
    accuracy: dict[str, npt.NDArray[np.float64]]


@dataclass(frozen=True)
class BoundaryMarks:
    """What is marked on every panel: the one real birds measurement (desktop CPU, a reference
    point on the local axis, not an on-device anchor) and each named condition's offload cost."""

    desktop_local_ms: float = SMALL_DESKTOP_CPU_MS
    condition_offload_ms: dict[str, float] = field(default_factory=dict)


def classify_cells(
    accuracy: dict[str, npt.NDArray[np.float64]], tolerance: float = TIE_TOLERANCE
) -> npt.NDArray[np.int_]:
    """Fill code per cell from the deployable policies' accuracies (the oracle is ignored)."""
    cascade = accuracy[CASCADE]
    best_other = np.maximum.reduce([accuracy[p] for p in DEPLOYABLE_POLICIES if p != CASCADE])
    gap = cascade - best_other
    codes = np.full(cascade.shape, CASCADE_LOSES, dtype=np.int_)
    codes[np.abs(gap) <= tolerance] = CASCADE_TIES
    codes[gap > tolerance] = CASCADE_WINS
    best = np.maximum(cascade, best_other)
    codes[best <= tolerance] = NO_POLICY_ON_TIME
    return codes


def _draw_panel(ax: Axes, panel: BoundaryPanelData, marks: BoundaryMarks) -> None:
    for policy in DEPLOYABLE_POLICIES:
        if policy not in panel.accuracy:
            raise ValueError(f"panel for {panel.budget_ms:g} ms is missing policy {policy!r}")
    local = np.asarray(panel.local_ms, dtype=np.float64)
    offload = np.asarray(panel.offload_ms, dtype=np.float64)
    codes = classify_cells(panel.accuracy)
    if codes.shape != (len(offload), len(local)):
        raise ValueError("accuracy grids must be shaped (len(offload_ms), len(local_ms))")

    cmap = ListedColormap(list(_FILL_COLORS))
    norm = BoundaryNorm(np.arange(-0.5, 4.5), cmap.N)
    ax.pcolormesh(local, offload, codes, cmap=cmap, norm=norm, shading="nearest")

    budget = panel.budget_ms
    # The three band edges. Escalation edge sits above the offload edge for every local > 0, and
    # the gap between them is exactly the local cost.
    ax.axvline(budget, color="black", linewidth=1, linestyle=":")  # local alone stops fitting
    ax.axhline(budget, color="black", linewidth=1, linestyle="--")  # offload stops fitting
    ax.plot(local, budget - local, color="black", linewidth=1.2)  # local + offload stops fitting

    # Marks: desktop measurement on the local axis, named conditions on the offload axis.
    ax.axvline(marks.desktop_local_ms, color="#CC79A7", linewidth=1.5)
    ax.text(
        marks.desktop_local_ms,
        offload[-1],
        f" desktop CPU {marks.desktop_local_ms:g} ms",
        color="#CC79A7",
        fontsize=7,
        rotation=90,
        va="top",
        ha="left",
    )
    for name, offload_ms in marks.condition_offload_ms.items():
        ax.axhline(offload_ms, color="#222222", linewidth=0.8, linestyle="-.")
        ax.text(
            local[-1],
            offload_ms,
            f"{name} ({offload_ms:.0f} ms) ",
            fontsize=7,
            ha="right",
            va="bottom",
        )

    ax.set_xlim(local[0], local[-1])
    ax.set_ylim(offload[0], offload[-1])
    ax.set_title(f"{budget:g} ms budget")
    ax.set_xlabel("Local cost (ms)")
    ax.set_ylabel("Offload cost (ms)")


def build_boundary_figure(panels: Sequence[BoundaryPanelData], marks: BoundaryMarks) -> Figure:
    """One panel per budget, ascending by budget, with a shared legend."""
    if not panels:
        raise ValueError("build_boundary_figure requires at least one budget panel")
    ordered = sorted(panels, key=lambda p: p.budget_ms)

    ncols = min(3, len(ordered))
    nrows = math.ceil(len(ordered) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    for ax, panel in zip(axes_flat, ordered, strict=False):
        _draw_panel(ax, panel, marks)
    for ax in axes_flat[len(ordered) :]:
        ax.set_visible(False)

    handles: list[Patch | Line2D] = [
        Patch(facecolor=c, label=label) for c, label in zip(_FILL_COLORS, _FILL_LABELS, strict=True)
    ]
    handles += [
        Line2D([], [], color="black", linestyle=":", label="local = budget"),
        Line2D([], [], color="black", linestyle="--", label="offload = budget"),
        Line2D([], [], color="black", label="local + offload = budget"),
    ]
    fig.suptitle("Where the cascade can act: local cost by offload cost, per budget")
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0.0, 0.1, 1.0, 1.0))
    return fig


def plot_boundary_maps(
    panels: Sequence[BoundaryPanelData], marks: BoundaryMarks, output_path: str | Path
) -> None:
    """Write the boundary figure to `output_path` as a PNG, creating parent directories."""
    fig = build_boundary_figure(panels, marks)
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(resolved_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
