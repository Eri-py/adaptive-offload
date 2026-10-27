"""create bird_predictions table

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-27 00:00:00.000000

Creates `bird_predictions`, the table owned by `database/common/models.py`'s
`BirdPrediction`: one row per (dataset, photo_id, model_name), holding a bird
classifier's per-photo prediction, confidence, and correctness. Authored to
match `BirdPrediction` exactly; not applied here — see
`.claude/coding-guidelines.md` and this repo's `CLAUDE.md` for who runs
migrations. No enum columns, so none of migration `0002`'s `postgresql.ENUM`
pitfall applies here.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "bird_predictions",
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("photo_id", sa.String(), nullable=False),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column("split", sa.String(), nullable=False),
        sa.Column("true_label", sa.Integer(), nullable=False),
        sa.Column("predicted_label", sa.Integer(), nullable=False),
        sa.Column("is_correct", sa.Boolean(), nullable=False),
        sa.Column("max_probability", sa.Float(), nullable=False),
        sa.Column("top2_margin", sa.Float(), nullable=False),
        sa.Column("training_run_id", sa.String(), nullable=True),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("dataset", "photo_id", "model_name"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("bird_predictions")
