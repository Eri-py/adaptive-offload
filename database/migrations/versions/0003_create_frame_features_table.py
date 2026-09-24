"""create frame_features table

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-24 00:00:00.000000

Creates `frame_features`, the table owned by `database/common/models.py`'s
`FrameFeatures`: one row per (dataset, frame), holding candidate router
features — five cheap image statistics and six YOLOv8n confidence/box-size
features. Authored to match `FrameFeatures` exactly; not applied here — see
`.claude/coding-guidelines.md` and this repo's `CLAUDE.md` for who runs
migrations. No enum columns, so none of migration `0002`'s `postgresql.ENUM`
pitfall applies here.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "frame_features",
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("file_name", sa.String(), nullable=False),
        sa.Column("sharpness", sa.Float(), nullable=False),
        sa.Column("brightness", sa.Float(), nullable=False),
        sa.Column("contrast", sa.Float(), nullable=False),
        sa.Column("colorfulness", sa.Float(), nullable=False),
        sa.Column("entropy", sa.Float(), nullable=False),
        sa.Column("detection_count", sa.Integer(), nullable=False),
        sa.Column("max_confidence", sa.Float(), nullable=False),
        sa.Column("mean_confidence", sa.Float(), nullable=False),
        sa.Column("min_confidence", sa.Float(), nullable=False),
        sa.Column("mean_box_area", sa.Float(), nullable=False),
        sa.Column("min_box_area", sa.Float(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("dataset", "file_name"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("frame_features")
