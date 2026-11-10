import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from birds.router.boundary_plot import (
    CASCADE_LOSES,
    CASCADE_TIES,
    CASCADE_WINS,
    NO_POLICY_ON_TIME,
    BoundaryMarks,
    BoundaryPanelData,
    build_boundary_figure,
    classify_cells,
    plot_boundary_maps,
)

_LOCAL = [1.0, 5.0, 10.0, 20.0]
_OFFLOAD = [10.0, 50.0, 100.0]
_MARKS = BoundaryMarks(
    desktop_local_ms=5.877,
    condition_offload_ms={"good": 40.0, "medium": 80.0, "poor": 150.0},
)


def _grid(value: float) -> npt.NDArray[np.float64]:
    return np.full((len(_OFFLOAD), len(_LOCAL)), value)


def _panel(budget_ms: float, cascade: float = 0.5) -> BoundaryPanelData:
    return BoundaryPanelData(
        budget_ms=budget_ms,
        local_ms=_LOCAL,
        offload_ms=_OFFLOAD,
        accuracy={
            "always-local": _grid(0.4),
            "always-offload": _grid(0.5),
            "budget-only": _grid(0.5),
            "cascade": _grid(cascade),
        },
    )


def test_writes_non_empty_file_for_synthetic_grid() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "nested" / "boundary.png"
        plot_boundary_maps([_panel(50.0)], _MARKS, out)
        assert out.stat().st_size > 0


def test_panels_appear_in_budget_order_one_per_budget() -> None:
    fig = build_boundary_figure([_panel(100.0), _panel(25.0), _panel(50.0)], _MARKS)
    titles = [ax.get_title() for ax in fig.axes if ax.get_visible() and ax.get_title()]
    plt.close(fig)
    assert titles == ["25 ms budget", "50 ms budget", "100 ms budget"]


def test_marks_are_desktop_measurement_and_three_named_conditions_only() -> None:
    fig = build_boundary_figure([_panel(50.0)], _MARKS)
    texts = [t.get_text() for t in fig.axes[0].texts]
    plt.close(fig)
    assert len(texts) == 4
    assert "desktop CPU 5.877 ms" in texts[0]
    for text, name in zip(texts[1:], ("good", "medium", "poor"), strict=True):
        assert text.strip().startswith(name)
    assert not any("17.34" in t or "106" in t for t in texts)  # attempt 1 phone figures


def test_default_desktop_mark_is_the_real_measurement() -> None:
    assert BoundaryMarks().desktop_local_ms == 5.877


def test_cells_are_tie_loss_win_or_empty_and_oracle_is_ignored() -> None:
    acc = {
        "always-local": np.array([[0.0, 0.4, 0.4, 0.4]]),
        "always-offload": np.array([[0.0, 0.5, 0.5, 0.5]]),
        "budget-only": np.array([[0.0, 0.5, 0.5, 0.5]]),
        "cascade": np.array([[0.0, 0.4, 0.5003, 0.52]]),
        "oracle": np.array([[1.0, 1.0, 1.0, 1.0]]),
    }
    codes = classify_cells(acc)[0].tolist()
    assert codes == [NO_POLICY_ON_TIME, CASCADE_LOSES, CASCADE_TIES, CASCADE_WINS]
