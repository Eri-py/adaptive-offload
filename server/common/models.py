"""Shared ORM models for the data-gen simulator and (later) the offload API.

Three tables, all owned here per `.claude/coding-guidelines.md`'s Database
section: `simulation_runs` (one row per simulator invocation, holding the
resolved config snapshot), `simulation_results` (one row per frame ×
condition pair, FK'd to its run), and `scene_complexity` (one row per frame,
computed once and reused across runs).
"""

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Enum, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    """Timezone-aware UTC now, used as a Python-side default for timestamp columns."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base shared by every table in this module."""


class Label(enum.Enum):
    """Which path a `SimulationResult` row's computed win/loss label favors."""

    LOCAL = "local"
    OFFLOAD = "offload"


class SimulationRun(Base):
    """One row per simulator invocation, snapshotting the config that produced it."""

    __tablename__ = "simulation_runs"

    run_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    preset_name: Mapped[str] = mapped_column(String, nullable=False)
    frame_count: Mapped[int] = mapped_column(Integer, nullable=False)
    condition_vector_count: Mapped[int] = mapped_column(Integer, nullable=False)
    condition_ranges: Mapped[dict[str, list[float]]] = mapped_column(JSON, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    lambda_value: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class SimulationResult(Base):
    """One row per (frame, condition-vector) pair produced by a simulator run."""

    __tablename__ = "simulation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String, ForeignKey("simulation_runs.run_id"), nullable=False
    )
    frame_id: Mapped[str] = mapped_column(String, nullable=False)
    network_bandwidth_mbps: Mapped[float] = mapped_column(Float, nullable=False)
    network_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    network_packet_loss_pct: Mapped[float] = mapped_column(Float, nullable=False)
    device_load_pct: Mapped[float] = mapped_column(Float, nullable=False)
    local_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    local_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    offload_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    offload_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    label: Mapped[Label] = mapped_column(Enum(Label), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class SceneComplexity(Base):
    """One row per (dataset, file_name), computed once and reused across runs."""

    __tablename__ = "scene_complexity"

    dataset: Mapped[str] = mapped_column(String, primary_key=True)
    file_name: Mapped[str] = mapped_column(String, primary_key=True)
    scene_complexity: Mapped[float] = mapped_column(Float, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
