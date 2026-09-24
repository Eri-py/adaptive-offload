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

`model_path` uses `sqlalchemy.dialects.postgresql.ENUM`, not the generic
`sqlalchemy.Enum` — the generic type has no real concept of `create_type`
(it silently drops that kwarg; only the Postgres-specific `ENUM` class
actually implements it), so a generic `sa.Enum(..., create_type=False)` here
would still try to re-run `CREATE TYPE label` and fail with
`DuplicateObject` the moment this migration runs as its own invocation
against a database already at revision 0001 (confirmed by reproducing it
against a real, disposable Postgres database, not just `alembic ... --sql`
offline rendering, which never touches a real connection and can't catch
this class of bug).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

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
            postgresql.ENUM("LOCAL", "OFFLOAD", name="label", create_type=False),
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
