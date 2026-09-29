"""Tests for `birds.evaluate`: missing checkpoint, per-model transform, timing sanity."""

import json
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from birds import evaluate as ev
from shared import tracking


def test_mean_latency_ms_is_positive_and_finite() -> None:
    device = torch.device("cpu")
    model = nn.Linear(4, 3)
    images = torch.randn(6, 4)
    labels = torch.randint(0, 3, (6,))
    dataset = TensorDataset(images, labels)

    latency_ms = ev.mean_latency_ms(model, dataset, device, warmup=2)

    assert latency_ms > 0
    assert math.isfinite(latency_ms)


def test_mean_latency_ms_warmup_exceeds_dataset_size() -> None:
    """`warmup` larger than the dataset must still work, cycling through items."""
    device = torch.device("cpu")
    model = nn.Linear(4, 3)
    dataset = TensorDataset(torch.randn(3, 4), torch.randint(0, 3, (3,)))

    latency_ms = ev.mean_latency_ms(model, dataset, device, warmup=10)

    assert latency_ms > 0
    assert math.isfinite(latency_ms)


def test_load_checkpoint_missing_names_the_train_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ev, "WEIGHTS_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match=r"python -m birds\.train --model small"):
        ev._load_checkpoint("small", torch.device("cpu"))


def test_main_uses_each_models_own_eval_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    transform_names: list[str] = []
    loader_transforms: list[Any] = []

    def fake_eval_transform(name: str) -> str:
        transform_names.append(name)
        return f"transform-{name}"

    def fake_make_data_loader(split: str, transform: Any, *args: Any, **kwargs: Any) -> str:
        loader_transforms.append(transform)
        return "loader"

    monkeypatch.setattr(ev, "require_cuda", lambda: torch.device("cpu"))
    monkeypatch.setattr(ev, "eval_transform", fake_eval_transform)
    monkeypatch.setattr(ev, "make_data_loader", fake_make_data_loader)
    monkeypatch.setattr(ev, "load_test", lambda transform: transform)
    monkeypatch.setattr(ev, "_load_checkpoint", lambda name, device: nn.Identity())
    monkeypatch.setattr(ev, "top1_accuracy", lambda *args: 1.0)
    monkeypatch.setattr(ev, "mean_latency_ms", lambda *args: 1.0)

    ev.main()

    assert loader_transforms == ["transform-small", "transform-large"]
    assert transform_names.count("small") == 2 and transform_names.count("large") == 2


def test_main_builds_data_loaders_before_checking_for_a_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_make_data_loader(*args: Any, **kwargs: Any) -> str:
        calls.append("loader")
        return "loader"

    def fake_require_cuda() -> torch.device:
        calls.append("require_cuda")
        raise RuntimeError("no gpu")

    monkeypatch.setattr(ev, "eval_transform", lambda name: name)
    monkeypatch.setattr(ev, "make_data_loader", fake_make_data_loader)
    monkeypatch.setattr(ev, "require_cuda", fake_require_cuda)

    with pytest.raises(RuntimeError, match="no gpu"):
        ev.main()

    assert calls == ["loader", "loader", "require_cuda"]


def _patch_main_for_tracking(monkeypatch: pytest.MonkeyPatch, weights_dir: Path) -> list[Any]:
    """Stubs main's heavy steps; returns the list that records each tracking.run call."""
    calls: list[Any] = []

    @contextmanager
    def fake_run(
        run_name: str,
        params: Mapping[str, Any],
        *,
        experiment: str,
        run_id: str | None = None,
    ) -> Iterator[tracking.RunHandle]:
        calls.append((run_name, dict(params), run_id))
        yield tracking.RunHandle(id=run_id)

    monkeypatch.setattr(ev, "WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr(ev, "require_cuda", lambda: torch.device("cpu"))
    monkeypatch.setattr(ev, "eval_transform", lambda name: name)
    monkeypatch.setattr(ev, "make_data_loader", lambda *args, **kwargs: "loader")
    monkeypatch.setattr(ev, "load_test", lambda transform: transform)
    monkeypatch.setattr(ev, "_load_checkpoint", lambda name, device: nn.Identity())
    monkeypatch.setattr(ev, "top1_accuracy", lambda *args: 0.9)
    monkeypatch.setattr(ev, "mean_latency_ms", lambda *args: 2.0)
    monkeypatch.setattr(tracking, "run", fake_run)
    monkeypatch.setattr(tracking, "log_metrics", lambda handle, metrics: calls.append(metrics))
    monkeypatch.setattr(
        tracking,
        "log_to_run",
        lambda run_id, **kwargs: calls.append((run_id, kwargs)),
    )
    return calls


def test_main_attaches_results_to_the_training_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _patch_main_for_tracking(monkeypatch, tmp_path)
    (tmp_path / "small.pt.run.json").write_text(json.dumps({"run_id": "abc"}))
    (tmp_path / "large.pt.run.json").write_text(json.dumps({"run_id": "def"}))

    ev.main()

    # Attached results are written straight to the run id, never by resuming the run.
    attached = [c for c in calls if isinstance(c, tuple) and len(c) == 2]
    assert [run_id for run_id, _ in attached] == ["abc", "def"]
    assert attached[0][1]["metrics"] == {"test_accuracy": 0.9, "mean_latency_ms": 2.0}
    assert "eval_device" in attached[0][1]["tags"]
    assert not [c for c in calls if isinstance(c, tuple) and len(c) == 3]
    assert "not attached" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "sidecar", [None, "not json", "{}", '{"run_id": 5}', "[]", '{"run_id": ""}']
)
def test_main_without_a_usable_sidecar_runs_standalone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sidecar: str | None,
) -> None:
    calls = _patch_main_for_tracking(monkeypatch, tmp_path)
    if sidecar is not None:
        (tmp_path / "small.pt.run.json").write_text(sidecar)

    ev.main()

    out = capsys.readouterr().out
    runs = [c for c in calls if isinstance(c, tuple) and len(c) == 3]
    assert runs[0][0] == "evaluate-small" and runs[0][2] is None
    assert "test accuracy" in out and "0.9000" in out
    assert "small results are not attached to a training run" in out
    assert "large results are not attached to a training run" in out
