"""Exports the small bird classifier to TFLite plus its species-name list, gated on parity."""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from birds.config import NUM_CLASSES, SEED, TRAINING_DIR
from birds.dataset import splits
from birds.modeling.build import load_checkpoint

DEFAULT_OUTPUT_DIR = TRAINING_DIR.parent / "app" / "assets" / "models"
MODEL_FILENAME = "small.tflite"
SPECIES_FILENAME = "cub200-species.json"
DEFAULT_PARITY_SAMPLES = 50
MAX_PROB_DRIFT = 1e-3
_INPUT_SHAPE = (1, 3, 224, 224)
_DOWNLOAD_COMMAND = "python -m birds.dataset.download"


class ParityError(RuntimeError):
    """The TFLite model disagrees with the PyTorch model beyond tolerance."""


@dataclass(frozen=True)
class ParityReport:
    num_samples: int
    num_disagreements: int  # photos whose predicted class differs
    max_prob_drift: float  # worst |max-softmax difference| across photos


def _softmax(logits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    result: np.ndarray[Any, Any] = exps / exps.sum(axis=1, keepdims=True)
    return result


def compare_outputs(
    torch_logits: np.ndarray[Any, Any], tflite_logits: np.ndarray[Any, Any]
) -> ParityReport:
    """Compares (N, classes) logits from the two models by argmax and max-softmax drift."""
    torch_probs, tflite_probs = _softmax(torch_logits), _softmax(tflite_logits)
    disagreements = int((torch_probs.argmax(axis=1) != tflite_probs.argmax(axis=1)).sum())
    drift = np.abs(torch_probs.max(axis=1) - tflite_probs.max(axis=1))
    return ParityReport(
        num_samples=len(torch_logits),
        num_disagreements=disagreements,
        max_prob_drift=float(drift.max()),
    )


def require_parity(report: ParityReport, tolerance: float = MAX_PROB_DRIFT) -> None:
    """Raises `ParityError` unless every prediction agrees and drift is within `tolerance`."""
    if report.num_disagreements == 0 and report.max_prob_drift <= tolerance:
        return
    raise ParityError(
        f"TFLite parity check failed: {report.num_disagreements}/{report.num_samples} "
        f"predictions disagree; worst max-softmax drift {report.max_prob_drift:.2e} "
        f"(tolerance {tolerance:.0e})."
    )


def species_names(classes_file: Path) -> list[str]:
    """Species names ordered by 0-based class index, e.g. `1 001.Black_footed_Albatross`."""
    if not classes_file.is_file():
        raise FileNotFoundError(
            f"Missing {classes_file}. Run `{_DOWNLOAD_COMMAND}` to fetch the dataset first."
        )
    names: list[str] = []
    for line in classes_file.read_text().splitlines():
        if not line.strip():
            continue
        class_id, folder = line.split()
        if int(class_id) != len(names) + 1:  # file order must match splits.py's id - 1
            raise ValueError(f"{classes_file}: expected class id {len(names) + 1}, got {class_id}")
        names.append(folder.split(".", 1)[1].replace("_", " "))
    if len(names) != NUM_CLASSES:
        raise ValueError(f"{classes_file}: expected {NUM_CLASSES} species, found {len(names)}")
    return names


def _sample_test_batch(num_samples: int) -> torch.Tensor:
    """Preprocessed (N, 3, 224, 224) tensor of seeded random test-split photos."""
    dataset = splits.load_test(splits.eval_transform("small"))
    indices = np.random.default_rng(SEED).choice(len(dataset), num_samples, replace=False)
    return torch.stack([dataset[int(i)][0] for i in indices])


def _tflite_logits(edge_model: Any, batch: torch.Tensor) -> np.ndarray[Any, Any]:
    outputs = [np.asarray(edge_model(image.unsqueeze(0))).reshape(1, -1) for image in batch]
    return np.concatenate(outputs, axis=0)


def export(output_dir: Path, num_parity_samples: int) -> tuple[Path, Path]:
    """Converts, parity-checks, then publishes the model and names; nothing lands on failure."""
    import litert_torch  # type: ignore[import-untyped]  # deferred: heavy, noisy import

    # checkpoint first: a clean machine should be told to train before it is told to download
    model: nn.Module = load_checkpoint("small", torch.device("cpu")).eval()
    names = species_names(splits.dataset_root() / "classes.txt")
    edge_model = litert_torch.convert(model, (torch.randn(*_INPUT_SHAPE),))

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path, species_path = output_dir / MODEL_FILENAME, output_dir / SPECIES_FILENAME
    staging_path = model_path.with_name(model_path.name + ".partial")
    try:
        edge_model.export(str(staging_path))
        batch = _sample_test_batch(num_parity_samples)
        with torch.no_grad():
            torch_logits = model(batch).numpy()
        published_model = litert_torch.load(str(staging_path))  # verify the file we ship
        require_parity(compare_outputs(torch_logits, _tflite_logits(published_model, batch)))
        os.replace(staging_path, model_path)
    finally:
        staging_path.unlink(missing_ok=True)
    species_path.write_text(json.dumps(names, indent=2) + "\n")
    return model_path, species_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--parity-samples", type=int, default=DEFAULT_PARITY_SAMPLES)
    args = parser.parse_args()
    model_path, species_path = export(args.output_dir, args.parity_samples)
    print(f"Wrote {model_path} and {species_path}")


if __name__ == "__main__":
    main()
