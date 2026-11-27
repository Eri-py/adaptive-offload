"""Tests for `birds.export`: label derivation, parity comparison and resolved paths."""

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from birds import export
from birds.config import NUM_CLASSES
from birds.dataset import splits
from birds.modeling import build


def _write_classes(path: Path, count: int = NUM_CLASSES) -> Path:
    lines = [f"{i} {i:03d}.Species_Number_{i}" for i in range(1, count + 1)]
    lines[0] = "1 001.Black_footed_Albatross"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_species_names_are_ordered_by_zero_based_class_index(tmp_path: Path) -> None:
    names = export.species_names(_write_classes(tmp_path / "classes.txt"))
    assert names[0] == "Black footed Albatross"
    assert names[1] == "Species Number 2"
    assert names[199] == "Species Number 200"


def test_species_names_strip_prefix_and_underscores(tmp_path: Path) -> None:
    names = export.species_names(_write_classes(tmp_path / "classes.txt"))
    assert all("_" not in name and not name[0].isdigit() and "." not in name for name in names)


def test_species_names_has_200_entries(tmp_path: Path) -> None:
    assert len(export.species_names(_write_classes(tmp_path / "classes.txt"))) == 200


def test_species_names_rejects_wrong_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected 200"):
        export.species_names(_write_classes(tmp_path / "classes.txt", count=199))


def test_species_names_missing_file_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="birds.dataset.download"):
        export.species_names(tmp_path / "classes.txt")


def _logits(top_class: list[int], margin: float = 5.0) -> np.ndarray[Any, Any]:
    logits = np.zeros((len(top_class), 4))
    logits[np.arange(len(top_class)), top_class] = margin
    return logits


def test_parity_passes_within_tolerance() -> None:
    reference = _logits([0, 1, 2])
    report = export.compare_outputs(reference, reference + 1e-6)
    assert report.num_disagreements == 0
    export.require_parity(report)


def test_parity_fails_when_predictions_disagree() -> None:
    report = export.compare_outputs(_logits([0, 1, 2]), _logits([0, 3, 2]))
    assert report.num_disagreements == 1
    with pytest.raises(export.ParityError, match="1/3 predictions disagree"):
        export.require_parity(report)


def test_parity_fails_when_probability_drifts_too_far() -> None:
    report = export.compare_outputs(_logits([0, 1]), _logits([0, 1], margin=4.9))
    assert report.num_disagreements == 0
    with pytest.raises(export.ParityError, match="drift"):
        export.require_parity(report)


def test_default_output_dir_resolves_to_the_app_models_folder() -> None:
    assert export.DEFAULT_OUTPUT_DIR.parts[-4:] == ("adaptive-offload", "app", "assets", "models")
    assert export.DEFAULT_OUTPUT_DIR.parent.parent.parent.is_dir()


class _FakeEdgeModel:
    def export(self, path: str) -> None:
        Path(path).write_bytes(b"truncated garbage")


def test_unusable_staged_file_fails_export_and_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _unreadable(path: str) -> None:
        raise ValueError(f"not a model: {path}")

    fake_litert = types.SimpleNamespace(convert=lambda *_: _FakeEdgeModel(), load=_unreadable)
    monkeypatch.setitem(sys.modules, "litert_torch", fake_litert)
    monkeypatch.setattr(export, "load_checkpoint", lambda *_: torch.nn.Identity())
    monkeypatch.setattr(export, "species_names", lambda _: ["x"] * NUM_CLASSES)
    monkeypatch.setattr(splits, "dataset_root", lambda: tmp_path)
    monkeypatch.setattr(export, "_sample_test_batch", lambda n: torch.zeros(n, 3, 224, 224))

    output_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="not a model"):
        export.export(output_dir, num_parity_samples=2)

    assert list(output_dir.iterdir()) == []


def test_missing_checkpoint_names_training_command_even_without_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "litert_torch", types.SimpleNamespace())
    monkeypatch.setattr(build, "WEIGHTS_DIR", tmp_path / "no_weights")
    monkeypatch.setattr(splits, "dataset_root", lambda: tmp_path / "no_dataset")

    output_dir = tmp_path / "out"
    with pytest.raises(FileNotFoundError, match=r"birds\.train --model small"):
        export.export(output_dir, num_parity_samples=2)

    assert not output_dir.exists()
