"""Read/write access to the three shared Postgres tables, via `common.models`.

Every function takes a SQLAlchemy `Engine` explicitly (never a module-level
connection) so tests can substitute the ephemeral-Postgres fixture engine and
a future simulator script can pass whatever engine `common.db.get_engine()`
builds.
"""

from __future__ import annotations

from dataclasses import dataclass

from common.models import Label, ModelInference, SceneComplexity, SimulationResult, SimulationRun
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from datagen.simulate.inference import DetectionResult


@dataclass(frozen=True)
class RunConfig:
    """Resolved config snapshot for one simulator invocation.

    Mirrors `SimulationRun`'s columns (minus `run_id`/`created_at`, which are
    generated on insert) — see the spec's requirement that every run's exact
    preset name, per-axis ranges, frame count, condition-vector count, seed,
    and lambda be persisted before any result rows are written.
    """

    dataset: str
    preset_name: str
    frame_count: int
    condition_vector_count: int
    condition_ranges: dict[str, list[float]]
    seed: int
    lambda_value: float


@dataclass(frozen=True)
class ResultRow:
    """One (frame, condition-vector) row to persist under a run."""

    frame_id: str
    network_bandwidth_mbps: float
    network_latency_ms: float
    network_packet_loss_pct: float
    device_load_pct: float
    local_latency_ms: float
    local_accuracy: float
    offload_latency_ms: float
    offload_accuracy: float
    label: Label


def get_known_complexity(engine: Engine, dataset: str) -> dict[str, float]:
    """Read all cached `scene_complexity` rows for `dataset` as {file_name: score}.

    Ordered by `file_name` — a second line of defence against Postgres's
    unordered `SELECT` row order feeding non-deterministic dict-insertion
    order into `sampling.stratified_sample`'s tie-breaking (the sort there is
    already a total order over `(score, file_name)`, so this ordering isn't
    load-bearing for that specific bug, but keeping this function's own
    output deterministic avoids depending on that elsewhere).
    """
    with Session(engine) as session:
        rows = session.execute(
            select(SceneComplexity)
            .where(SceneComplexity.dataset == dataset)
            .order_by(SceneComplexity.file_name)
        ).scalars()
        return {row.file_name: row.scene_complexity for row in rows}


def store_complexity_scores(engine: Engine, dataset: str, scores: dict[str, float]) -> None:
    """Insert new `scene_complexity` rows, skipping file names already present."""
    with Session(engine) as session:
        known_file_names = set(
            session.execute(
                select(SceneComplexity.file_name).where(SceneComplexity.dataset == dataset)
            ).scalars()
        )
        new_rows = [
            SceneComplexity(dataset=dataset, file_name=file_name, scene_complexity=score)
            for file_name, score in scores.items()
            if file_name not in known_file_names
        ]
        session.add_all(new_rows)
        session.commit()


def get_known_model_inference(
    engine: Engine, dataset: str
) -> dict[str, dict[Label, DetectionResult]]:
    """Read all cached `model_inference` rows for `dataset`.

    Returns `{file_name: {model_path: DetectionResult(...)}}` — a
    `file_name` key is present only for whichever `Label` values actually
    have a persisted row for it, so a file may map to just one of the two
    paths rather than necessarily both. Ordered by `file_name` — same
    determinism reasoning as `get_known_complexity`.
    """
    with Session(engine) as session:
        rows = session.execute(
            select(ModelInference)
            .where(ModelInference.dataset == dataset)
            .order_by(ModelInference.file_name)
        ).scalars()
        results: dict[str, dict[Label, DetectionResult]] = {}
        for row in rows:
            results.setdefault(row.file_name, {})[row.model_path] = DetectionResult(
                latency_ms=row.latency_ms, accuracy=row.accuracy
            )
        return results


def store_model_inference(
    engine: Engine, dataset: str, results: dict[str, dict[Label, DetectionResult]]
) -> None:
    """Insert new `model_inference` rows, skipping `(file_name, model_path)` pairs already present.

    Mirrors `store_complexity_scores`'s already-known-check shape, but keyed
    on the two-column `(file_name, model_path)` pair rather than a single
    column, since a `model_inference` row's identity within a dataset also
    depends on which model produced it.
    """
    with Session(engine) as session:
        known_pairs = set(
            session.execute(
                select(ModelInference.file_name, ModelInference.model_path).where(
                    ModelInference.dataset == dataset
                )
            ).all()
        )
        new_rows = [
            ModelInference(
                dataset=dataset,
                file_name=file_name,
                model_path=model_path,
                latency_ms=result.latency_ms,
                accuracy=result.accuracy,
            )
            for file_name, by_model in results.items()
            for model_path, result in by_model.items()
            if (file_name, model_path) not in known_pairs
        ]
        session.add_all(new_rows)
        session.commit()


def create_run(engine: Engine, config: RunConfig) -> str:
    """Insert one `SimulationRun` row from a resolved config snapshot, return its run id."""
    with Session(engine) as session:
        run = SimulationRun(
            dataset=config.dataset,
            preset_name=config.preset_name,
            frame_count=config.frame_count,
            condition_vector_count=config.condition_vector_count,
            condition_ranges=config.condition_ranges,
            seed=config.seed,
            lambda_value=config.lambda_value,
        )
        session.add(run)
        session.commit()
        return run.run_id


def store_results(engine: Engine, run_id: str, rows: list[ResultRow]) -> None:
    """Bulk-insert `SimulationResult` rows tagged with `run_id`.

    Assumes `run_id` already names a committed `SimulationRun` row (via
    `create_run`) — Postgres enforces the FK immediately on insert.
    """
    with Session(engine) as session:
        session.add_all(
            SimulationResult(
                run_id=run_id,
                frame_id=row.frame_id,
                network_bandwidth_mbps=row.network_bandwidth_mbps,
                network_latency_ms=row.network_latency_ms,
                network_packet_loss_pct=row.network_packet_loss_pct,
                device_load_pct=row.device_load_pct,
                local_latency_ms=row.local_latency_ms,
                local_accuracy=row.local_accuracy,
                offload_latency_ms=row.offload_latency_ms,
                offload_accuracy=row.offload_accuracy,
                label=row.label,
            )
            for row in rows
        )
        session.commit()
