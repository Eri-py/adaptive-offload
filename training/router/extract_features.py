"""Frame-feature extraction: image statistics + YOLOv8n confidence features, persisted.

Real invocation (once `training`'s package is installed, per this repo's
`datagen.*`/`router.*`-not-`training.*` import-root convention):

    python -m router.extract_features --folder <path> --dataset <name>

Not a registered `[project.scripts]` entry — `.claude/coding-guidelines.md`
caps that list at `run-simulation`/`score-complexity`; this is a router
input, run as a module like `router.baseline`/`router.analyze`.

Computes only the frames `router.feature_store.list_dataset_frames` (i.e.
`scene_complexity`) knows about that don't already have a `frame_features`
row, using the same YOLOv8n weights/settings as the stored local accuracy
(`load_model` + `model(path, device="cpu", verbose=False)`), so detections
match what that accuracy was scored on.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from common.db import get_engine
from dotenv import load_dotenv
from sqlalchemy import Engine
from ultralytics.engine.results import Results

from datagen import config
from datagen.simulate.yolo_inference import load_model
from router.feature_store import (
    FeatureRow,
    get_known_feature_file_names,
    list_dataset_frames,
    store_features,
)
from router.frame_features.confidence import confidence_features, sequences_from_results
from router.frame_features.image_stats import image_stats

# Rows are flushed in batches so an interrupted long run keeps its progress
# (see the implementation plan's "Rows are flushed in batches of 100").
BATCH_SIZE = 100

# Returns (confidences, normalised box areas) for one image's detections —
# not `Results` directly, so tests can stub this without a real model/torch.
Predict = Callable[[Path], tuple[Sequence[float], Sequence[float]]]

# Builds a `Predict`, e.g. loading model weights — deferred to a factory so
# `extract_features` can call it only once pending/missing-image checks pass.
PredictFactory = Callable[[], Predict]


@dataclass(frozen=True)
class TimingSummary:
    """Average per-frame seconds spent in each timed segment, plus group totals.

    `image_stats_seconds` is the image-features group's full per-frame cost.
    `confidence_group_seconds` is the confidence-features group's full
    per-frame cost — `model_seconds + confidence_seconds`, since that group's
    cost is the model call plus the confidence derivation, not either alone.
    """

    image_stats_seconds: float
    model_seconds: float
    confidence_seconds: float
    confidence_group_seconds: float


def extract_features(
    engine: Engine, dataset: str, folder: Path, predict_factory: PredictFactory
) -> TimingSummary | None:
    """Compute and store `frame_features` rows for every pending frame of `dataset`.

    "Pending" means listed in `scene_complexity` for `dataset` but without a
    `frame_features` row yet — a re-run only fills in the gap, like
    `score-complexity`. Every pending frame's image is checked for existence
    up front, so a missing image fails before any (possibly slow)
    computation runs, per the spec's "never leaves partial/missing values"
    and "fails clearly on a missing image" requirements. Returns `None` when
    there's nothing to compute.

    `predict_factory` is only called once those checks pass, so a real
    predictor's model load (weights + warm-up inference) is never paid on a
    no-op re-run or a bad `folder`. It's called before the per-frame timing
    loop below starts, so the load never skews the reported averages.
    """
    known_frames = list_dataset_frames(engine, dataset)
    already_stored = get_known_feature_file_names(engine, dataset)
    pending = [name for name in known_frames if name not in already_stored]

    if not pending:
        print("Nothing to compute: every known frame already has stored features.")
        return None

    pending_paths = {name: folder / name for name in pending}
    missing = sorted(name for name, path in pending_paths.items() if not path.is_file())
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} image file(s) under {folder}: {', '.join(missing)}"
        )

    predict = predict_factory()

    image_stats_total = 0.0
    model_total = 0.0
    confidence_total = 0.0
    batch: list[FeatureRow] = []
    for name in pending:
        path = pending_paths[name]

        start = time.perf_counter()
        stats = image_stats(path)
        image_stats_total += time.perf_counter() - start

        start = time.perf_counter()
        confidences, box_areas = predict(path)
        model_total += time.perf_counter() - start

        start = time.perf_counter()
        confidence = confidence_features(confidences, box_areas)
        confidence_total += time.perf_counter() - start

        batch.append(FeatureRow(file_name=name, image_stats=stats, confidence=confidence))
        if len(batch) >= BATCH_SIZE:
            store_features(engine, dataset, batch)
            batch = []
    if batch:
        store_features(engine, dataset, batch)

    frame_count = len(pending)
    summary = TimingSummary(
        image_stats_seconds=image_stats_total / frame_count,
        model_seconds=model_total / frame_count,
        confidence_seconds=confidence_total / frame_count,
        confidence_group_seconds=(model_total + confidence_total) / frame_count,
    )
    print(
        f"Computed features for {frame_count} frame(s). Average per-frame seconds: "
        f"image_features={summary.image_stats_seconds:.4f}, "
        f"confidence_features (model+derive)={summary.confidence_group_seconds:.4f} "
        f"(model={summary.model_seconds:.4f}, confidence={summary.confidence_seconds:.4f})"
    )
    return summary


def _make_real_predictor(weights_path: Path, device: str) -> Predict:
    """Build a `Predict` that runs real YOLOv8n inference for `main()`.

    Same call shape as the simulator's local path
    (`model(path, device="cpu", verbose=False)`), so the detections these
    features come from match what the stored local accuracy was scored on.
    """
    model = load_model(weights_path, device)

    def predict(path: Path) -> tuple[Sequence[float], Sequence[float]]:
        # `stream=False` (the default) always returns a `list[Results]`.
        results = cast(list[Results], model(path, device=device, verbose=False))
        return sequences_from_results(results[0])

    return predict


def main() -> None:
    # CLI-only convenience: load DATABASE_URL from training/.env if it isn't
    # already in the environment. Mirrors `score_complexity.main()`.
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    parser = argparse.ArgumentParser(
        description="Compute image-statistics and YOLOv8n confidence features for a dataset."
    )
    parser.add_argument(
        "--folder",
        required=True,
        type=Path,
        help="Folder the dataset's images live in (non-recursive).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name frame features are stored under.",
    )
    args = parser.parse_args()

    engine = get_engine()
    extract_features(
        engine,
        args.dataset,
        args.folder,
        lambda: _make_real_predictor(config.LOCAL_MODEL_WEIGHTS_PATH, "cpu"),
    )


if __name__ == "__main__":
    main()
