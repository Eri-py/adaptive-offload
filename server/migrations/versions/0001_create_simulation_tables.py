"""create simulation tables

Revision ID: 0001
Revises:
Create Date: 2026-09-07 00:00:00.000000

Creates the three tables owned by `server/common/models.py`: `simulation_runs`
(one row per simulator invocation), `simulation_results` (one row per frame x
condition pair, FK'd to its run), and `scene_complexity` (one row per frame,
computed once and reused across runs). Authored to match those models
exactly; not applied here — see `.claude/coding-guidelines.md` and this
repo's `CLAUDE.md` for who runs migrations.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "scene_complexity",
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("file_name", sa.String(), nullable=False),
        sa.Column("scene_complexity", sa.Float(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("dataset", "file_name"),
    )
    op.create_table(
        "simulation_runs",
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("preset_name", sa.String(), nullable=False),
        sa.Column("frame_count", sa.Integer(), nullable=False),
        sa.Column("condition_vector_count", sa.Integer(), nullable=False),
        sa.Column("condition_ranges", sa.JSON(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("lambda_value", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_table(
        "simulation_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("frame_id", sa.String(), nullable=False),
        sa.Column("network_bandwidth_mbps", sa.Float(), nullable=False),
        sa.Column("network_latency_ms", sa.Float(), nullable=False),
        sa.Column("network_packet_loss_pct", sa.Float(), nullable=False),
        sa.Column("device_load_pct", sa.Float(), nullable=False),
        sa.Column("local_latency_ms", sa.Float(), nullable=False),
        sa.Column("local_accuracy", sa.Float(), nullable=False),
        sa.Column("offload_latency_ms", sa.Float(), nullable=False),
        sa.Column("offload_accuracy", sa.Float(), nullable=False),
        sa.Column("label", sa.Enum("LOCAL", "OFFLOAD", name="label"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["simulation_runs.run_id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("simulation_results")
    op.drop_table("simulation_runs")
    op.drop_table("scene_complexity")
