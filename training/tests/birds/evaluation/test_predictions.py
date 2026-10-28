"""Tests for `birds.evaluation.predictions`: hand-computed per-photo records and alignment."""

from pathlib import Path
from typing import Any, get_args

import pytest
import torch
from PIL import Image
from sqlalchemy import Engine
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from birds.config import ModelName
from birds.dataset import splits as splits_module
from birds.dataset.splits import Split
from birds.evaluation import predictions as predictions_module
from birds.evaluation.predictions import predict_photos
from birds.evaluation.records import DATASET_NAME, PhotoPrediction
from birds.evaluation.store import read_predictions
from birds.modeling import build as build_module

_LOGITS = torch.tensor(
    [
        [0.0, 5.0, 1.0],  # argmax 1, label 1: correct
        [5.0, 0.0, 1.0],  # argmax 0, label 2: wrong
        [0.0, 0.0, 0.0],  # uniform: probabilities 1/3 each, margin 0
    ]
)
_LABELS = [1, 2, 0]
_IDS = ["10", "11", "12"]


def _loader(batch_size: int = 2) -> "DataLoader[Any]":
    return DataLoader(TensorDataset(_LOGITS, torch.tensor(_LABELS)), batch_size=batch_size)


def _run(batch_size: int = 2) -> list[PhotoPrediction]:
    return predict_photos(nn.Identity(), _loader(batch_size), torch.device("cpu"), _IDS, _LABELS)


def test_records_match_hand_computed_values() -> None:
    probs = _LOGITS.softmax(dim=1)
    records = _run()

    assert [r.photo_id for r in records] == _IDS
    assert [r.true_label for r in records] == _LABELS
    assert [r.predicted_label for r in records] == [1, 0, 0]
    assert [r.is_correct for r in records] == [True, False, True]
    assert records[0].max_probability == pytest.approx(probs[0, 1].item())
    assert records[0].top2_margin == pytest.approx((probs[0, 1] - probs[0, 2]).item())
    assert records[1].max_probability == pytest.approx(probs[1, 0].item())
    assert records[1].top2_margin == pytest.approx((probs[1, 0] - probs[1, 2]).item())
    assert records[2].max_probability == pytest.approx(1 / 3)
    assert records[2].top2_margin == pytest.approx(0.0, abs=1e-7)


def test_batch_size_does_not_change_the_records() -> None:
    assert _run(batch_size=1) == _run(batch_size=3)


def test_misaligned_label_order_raises() -> None:
    swapped_ids_labels = [_LABELS[1], _LABELS[0], _LABELS[2]]

    with pytest.raises(ValueError, match="does not match"):
        predict_photos(nn.Identity(), _loader(), torch.device("cpu"), _IDS, swapped_ids_labels)


def test_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="photo ids"):
        predict_photos(nn.Identity(), _loader(), torch.device("cpu"), _IDS[:2], _LABELS)


def test_loader_yielding_fewer_photos_than_the_split_raises() -> None:
    short = DataLoader(TensorDataset(_LOGITS[:2], torch.tensor(_LABELS[:2])), batch_size=2)

    with pytest.raises(ValueError, match="does not match"):
        predict_photos(nn.Identity(), short, torch.device("cpu"), _IDS, _LABELS)


# species index -> (official-train photos, official-test photos); every split gets photos.
_LAYOUT = {0: (12, 1), 1: (12, 1)}


class _ConstantModel(nn.Module):
    """Always predicts class `favorite` of 2, whatever the image."""

    def __init__(self, favorite: int) -> None:
        super().__init__()
        self._favorite = favorite

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(len(images), 2)
        logits[:, self._favorite] = 3.0
        return logits


@pytest.fixture
def fake_cub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "CUB_200_2011"
    (root / "images").mkdir(parents=True)
    images, labels, flags = [], [], []
    photo_id = 0
    for species, (n_train, n_test) in _LAYOUT.items():
        (root / "images" / f"s{species}").mkdir()
        for is_train in [1] * n_train + [0] * n_test:
            photo_id += 1
            relative = f"s{species}/p{photo_id}.png"
            Image.new("RGB", (4, 4)).save(root / "images" / relative)
            images.append(f"{photo_id} {relative}")
            labels.append(f"{photo_id} {species + 1}")
            flags.append(f"{photo_id} {is_train}")
    (root / "images.txt").write_text("\n".join(images) + "\n")
    (root / "image_class_labels.txt").write_text("\n".join(labels) + "\n")
    (root / "train_test_split.txt").write_text("\n".join(flags) + "\n")
    monkeypatch.setattr(splits_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(splits_module, "NUM_WORKERS", 0)
    monkeypatch.setattr(
        predictions_module, "eval_transform", lambda name: lambda image: torch.zeros(3, 4, 4)
    )


def test_each_photo_gets_one_record_per_model_across_all_splits(
    fake_cub: None, postgres_engine: Engine
) -> None:
    device = torch.device("cpu")
    train_ids, val_ids, test_ids = splits_module.split_ids()
    models: dict[ModelName, nn.Module] = {"small": _ConstantModel(0), "large": _ConstantModel(1)}
    expected_sizes: dict[Split, int] = {
        "train": len(train_ids),
        "val": len(val_ids),
        "test": len(test_ids),
    }

    for name, model in models.items():
        for split, size in expected_sizes.items():
            count = predictions_module.record_split(
                postgres_engine, model, name, split, device, None
            )
            assert count == size

    for split, size in expected_sizes.items():
        small = read_predictions(postgres_engine, DATASET_NAME, split, "small")
        large = read_predictions(postgres_engine, DATASET_NAME, split, "large")
        assert len(small) == len(large) == size
        assert small["photo_id"].tolist() == large["photo_id"].tolist()
        assert set(small["predicted_label"]) == {0}
        assert set(large["predicted_label"]) == {1}
        assert small["is_correct"].tolist() == (small["true_label"] == 0).tolist()


def test_missing_dataset_stops_before_gpu_checkpoints_and_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(splits_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(predictions_module, "require_cuda", _fail("require_cuda"))
    monkeypatch.setattr(predictions_module, "load_checkpoint", _fail("load_checkpoint"))

    with pytest.raises(FileNotFoundError, match="python -m birds.dataset.download"):
        predictions_module.main()


def test_missing_checkpoint_names_the_train_command(
    fake_cub: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(predictions_module, "require_cuda", lambda: torch.device("cpu"))
    monkeypatch.setattr(predictions_module, "get_engine", lambda: None)
    monkeypatch.setattr(build_module, "WEIGHTS_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="python -m birds.train --model small"):
        predictions_module.main()


def test_one_missing_checkpoint_writes_no_records(
    fake_cub: None, postgres_engine: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "small.pt").touch()  # small present, large absent
    monkeypatch.setattr(predictions_module, "require_cuda", lambda: torch.device("cpu"))
    monkeypatch.setattr(predictions_module, "get_engine", lambda: postgres_engine)
    monkeypatch.setattr(build_module, "WEIGHTS_DIR", tmp_path)
    monkeypatch.setattr(predictions_module, "load_checkpoint", _fail("load_checkpoint"))

    with pytest.raises(FileNotFoundError, match="python -m birds.train --model large"):
        predictions_module.main()

    for split in get_args(Split):
        for name in ("small", "large"):
            assert read_predictions(postgres_engine, DATASET_NAME, split, name).empty


def test_no_gpu_refuses_to_proceed(fake_cub: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(predictions_module, "load_checkpoint", _fail("load_checkpoint"))

    with pytest.raises(RuntimeError, match="No CUDA GPU"):
        predictions_module.main()


def _fail(name: str) -> Any:
    def _raise(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"{name} must not be reached")

    return _raise

