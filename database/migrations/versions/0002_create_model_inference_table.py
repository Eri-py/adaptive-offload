"""create model_inference table

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-22 00:00:00.000000

Creates `model_inference`, the table owned by `database/common/models.py`'s
`ModelInference`: one row per (dataset, frame, model) triple, caching real
inference results. `model_path` reuses the `label` Postgres enum type that
migration `0001` already created for `simulation_results.label` — this
migration must not recreate it. Authored to match `ModelInference` exactly;
not applied here — see `.claude/coding-guidelines.md` and this repo's
`CLAUDE.md` for who runs migrations.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "model_inference",
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("file_name", sa.String(), nullable=False),
        sa.Column(
            "model_path",
            sa.Enum("LOCAL", "OFFLOAD", name="label", create_type=False),
            nullable=False,
        ),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("accuracy", sa.Float(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("dataset", "file_name", "model_path"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("model_inference")
