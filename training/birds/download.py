"""Explicit dataset download step: `python -m birds.download`.

The only place in this package that downloads anything. Training and
evaluation (`data.py`) never download and fail with a message pointing back
here, per the coding guidelines' rule that acquiring a dataset is a separate,
explicitly run concern from the pipeline that uses it.
"""

import tarfile
import urllib.request
from pathlib import Path

from birds.config import CUB_URL, DATA_DIR
from birds.data import INDEX_FILES, dataset_root

_ARCHIVE_NAME = "CUB_200_2011.tgz"
_CHUNK_BYTES = 1 << 20


def _missing_files(root: Path) -> tuple[int, int]:
    """Returns (photos listed in images.txt, missing photos plus missing index files)."""
    missing_indexes = sum(1 for name in INDEX_FILES if not (root / name).is_file())
    index = root / "images.txt"
    if not index.is_file():
        return 0, missing_indexes
    relative_paths = [line.split()[1] for line in index.read_text().splitlines() if line.strip()]
    missing_photos = sum(1 for path in relative_paths if not (root / "images" / path).is_file())
    return len(relative_paths), missing_photos + missing_indexes


def _download(archive: Path) -> None:
    print(f"downloading {CUB_URL}")
    with urllib.request.urlopen(CUB_URL) as response, archive.open("wb") as out:
        while chunk := response.read(_CHUNK_BYTES):
            out.write(chunk)


def main() -> None:
    root = dataset_root()
    total, missing = _missing_files(root)
    if total and not missing:
        print(f"CUB-200-2011 already present: {total} images at {root}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = DATA_DIR / _ARCHIVE_NAME
    _download(archive)
    with tarfile.open(archive) as tar:
        tar.extractall(DATA_DIR, filter="data")
    archive.unlink()

    total, missing = _missing_files(root)
    if not total or missing:
        raise RuntimeError(
            f"CUB-200-2011 incomplete after extraction: {missing} files missing of {total} images"
        )
    print(f"CUB-200-2011 ready: {total} images at {root}")


if __name__ == "__main__":
    main()
