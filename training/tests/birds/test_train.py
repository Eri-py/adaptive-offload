"""Tests for `birds.train`: best-validation checkpointing and data-before-model order."""

import copy
import inspect
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from birds import train as train_module
from birds.metrics import EvalResult
from shared import tracking
from shared.tracking import RunHandle

_NUM_CLASSES = 3
_NUM_FEATURES = 4


def _tiny_model() -> nn.Module:
    return nn.Linear(_NUM_FEATURES, _NUM_CLASSES)


def _synthetic_loader(num_samples: int, batch_size: int) -> "DataLoader[Any]":
    images = torch.randn(num_samples, _NUM_FEATURES)
    labels = torch.randint(0, _NUM_CLASSES, (num_samples,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def test_fit_saves_a_checkpoint(tmp_path: Path) -> None:
    model = _tiny_model()
    train_loader = _synthetic_loader(num_samples=8, batch_size=4)
    val_loader = _synthetic_loader(num_samples=4, batch_size=4)
    checkpoint_path = tmp_path / "small.pt"

    result = train_module.fit(
        model,
        train_loader,
        val_loader,
        device=torch.device("cpu"),
        epochs=2,
        lr=1e-3,
        weight_decay=0.0,
        checkpoint_path=checkpoint_path,
    )

    assert checkpoint_path.exists()
    assert len(result.history) == 2


def test_fit_keeps_the_best_epoch_not_the_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scripted validation accuracies [0.5, 0.9, 0.7]: epoch 2 (1-based) wins."""
    model = _tiny_model()
    train_loader = _synthetic_loader(num_samples=8, batch_size=4)
    val_loader = _synthetic_loader(num_samples=4, batch_size=4)
    checkpoint_path = tmp_path / "small.pt"

    scripted_accuracies = iter([0.5, 0.9, 0.7])
    snapshots_by_epoch: dict[int, dict[str, torch.Tensor]] = {}
    call_count = 0

    def fake_loss_and_accuracy(
        model: nn.Module, loader: "DataLoader[Any]", device: torch.device
    ) -> EvalResult:
        nonlocal call_count
        call_count += 1
        snapshots_by_epoch[call_count] = copy.deepcopy(model.state_dict())
        return EvalResult(loss=0.0, accuracy=next(scripted_accuracies))

    monkeypatch.setattr(train_module, "loss_and_accuracy", fake_loss_and_accuracy)
    result = train_module.fit(
        model,
        train_loader,
        val_loader,
        device=torch.device("cpu"),
        epochs=3,
        lr=1e-3,
        weight_decay=0.0,
        checkpoint_path=checkpoint_path,
    )

    assert result.best_epoch == 2
    assert result.best_val_accuracy == 0.9

    saved_state = torch.load(checkpoint_path)
    expected_state = snapshots_by_epoch[2]
    for key, expected_value in expected_state.items():
        assert torch.equal(saved_state[key], expected_value)


def test_fit_signature_has_no_test_split_parameter() -> None:
    parameters = list(inspect.signature(train_module.fit).parameters)

    assert parameters == [
        "model",
        "train_loader",
        "val_loader",
        "device",
        "epochs",
        "lr",
        "weight_decay",
        "checkpoint_path",
        "on_epoch_end",
    ]


def _patch_main(
    monkeypatch: pytest.MonkeyPatch, calls: list[str], weights_dir: Path, *, data_missing: bool
) -> None:
    def fake_make_data_loader(split: str, *args: Any, **kwargs: Any) -> str:
        calls.append(f"data:{split}")
        if data_missing:
            raise FileNotFoundError("run `python -m birds.download`")
        return split

    def fake_require_cuda() -> torch.device:
        calls.append("cuda")
        return torch.device("cpu")

    def fake_build_model(name: str, pretrained: bool = True) -> nn.Module:
        calls.append("model")
        return _tiny_model()

    monkeypatch.setattr("sys.argv", ["train", "--model", "small"])
    monkeypatch.setattr(train_module, "WEIGHTS_DIR", weights_dir)
    monkeypatch.setattr(train_module, "make_data_loader", fake_make_data_loader)
    monkeypatch.setattr(train_module, "require_cuda", fake_require_cuda)
    monkeypatch.setattr(train_module, "build_model", fake_build_model)

    def fake_fit(*args: Any, **kwargs: Any) -> train_module.FitResult:
        calls.append("fit")
        return train_module.FitResult(best_val_accuracy=0.5, best_epoch=1)

    monkeypatch.setattr(train_module, "fit", fake_fit)


def test_main_builds_data_loaders_before_the_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    _patch_main(monkeypatch, calls, tmp_path, data_missing=False)

    train_module.main()

    assert calls == ["data:train", "data:val", "cuda", "model", "fit"]


def test_main_builds_no_model_when_the_dataset_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    _patch_main(monkeypatch, calls, tmp_path, data_missing=True)

    with pytest.raises(FileNotFoundError, match="birds.download"):
        train_module.main()

    assert "model" not in calls
    assert "cuda" not in calls


def test_fit_calls_on_epoch_end_once_per_epoch_with_that_epochs_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validations = iter(
        [
            EvalResult(loss=2.0, accuracy=0.1),
            EvalResult(loss=1.5, accuracy=0.4),
            EvalResult(loss=1.7, accuracy=0.3),
        ]
    )
    monkeypatch.setattr(
        train_module, "loss_and_accuracy", lambda *args, **kwargs: next(validations)
    )
    seen: list[train_module.EpochResult] = []

    result = train_module.fit(
        _tiny_model(),
        _synthetic_loader(num_samples=8, batch_size=4),
        _synthetic_loader(num_samples=4, batch_size=4),
        device=torch.device("cpu"),
        epochs=3,
        lr=1e-3,
        weight_decay=0.0,
        checkpoint_path=tmp_path / "small.pt",
        on_epoch_end=seen.append,
    )

    assert seen == result.history
    assert [e.epoch for e in seen] == [1, 2, 3]
    assert [e.val_accuracy for e in seen] == [0.1, 0.4, 0.3]
    assert [e.val_loss for e in seen] == [2.0, 1.5, 1.7]


def test_run_sidecar_path_sits_next_to_the_checkpoint() -> None:
    assert tracking.run_sidecar_path(Path("/w/small.pt")) == Path("/w/small.pt.run.json")


def test_main_writes_the_run_id_sidecar_when_tracking_is_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_main(monkeypatch, [], tmp_path, data_missing=False)
    logged: list[dict[str, float]] = []

    @contextmanager
    def fake_run(name: str, params: Mapping[str, Any], *, experiment: str) -> Iterator[RunHandle]:
        assert params["model"] == "small"
        assert params["dataset"] == "cub200"
        yield RunHandle(id="abc123")

    def fake_log_metrics(
        handle: RunHandle, metrics: Mapping[str, float], *, step: int | None = None
    ) -> None:
        logged.append(dict(metrics))

    monkeypatch.setattr(tracking, "run", fake_run)
    monkeypatch.setattr(tracking, "log_metrics", fake_log_metrics)

    train_module.main()

    sidecar = tmp_path / "small.pt.run.json"
    assert json.loads(sidecar.read_text()) == {"run_id": "abc123"}
    assert logged == [{"best_val_accuracy": 0.5, "best_epoch": 1.0}]


def test_main_writes_no_sidecar_and_removes_a_stale_one_when_tracking_is_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_main(monkeypatch, [], tmp_path, data_missing=False)
    stale = tmp_path / "small.pt.run.json"
    stale.write_text('{"run_id": "old"}')

    train_module.main()  # the suite-wide TRAINING_TRACKING=off applies

    assert not stale.exists()
