"""create classify_requests table

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-27 00:00:00.000000

Creates `classify_requests`, the table owned by `database/common/models.py`'s
`ClassifyRequest`: one row per request served by the bird-classification
endpoint, holding the payload size, whether a geometric transform was applied,
the prediction and confidence measures, and compute/total timings. Authored to
match `ClassifyRequest` exactly; not applied here — see
`.claude/coding-guidelines.md` and this repo's `CLAUDE.md` for who runs
migrations. No enum columns, so none of migration `0002`'s `postgresql.ENUM`
pitfall applies here.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "classify_requests",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload_bytes", sa.Integer(), nullable=False),
        sa.Column("was_transformed", sa.Boolean(), nullable=False),
        sa.Column("predicted_label", sa.Integer(), nullable=False),
        sa.Column("predicted_name", sa.String(), nullable=False),
        sa.Column("max_probability", sa.Float(), nullable=False),
        sa.Column("top2_margin", sa.Float(), nullable=False),
        sa.Column("compute_ms", sa.Float(), nullable=False),
        sa.Column("total_ms", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("classify_requests")
