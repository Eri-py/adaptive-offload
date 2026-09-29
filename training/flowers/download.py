"""Explicit dataset download step: `python -m flowers.download`.

The only place in this package that downloads anything. Training and
evaluation (`data.py`) always load with `download=False` and fail with a
message pointing back here, per the coding guidelines' rule that acquiring a
dataset is a separate, explicitly run concern from the pipeline that uses it.
"""

from torchvision.datasets import Flowers102

from flowers.config import DATA_DIR

_SPLITS = ("train", "val", "test")


def main() -> None:
    for split in _SPLITS:
        dataset = Flowers102(root=DATA_DIR, split=split, download=True)
        print(f"{split}: {len(dataset)} images")


if __name__ == "__main__":
    main()
