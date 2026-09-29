"""`mlflow-ui` — open the MLflow UI on the project's tracking database.

Saves sourcing `training/.env` by hand just to get `MLFLOW_TRACKING_URI` onto the
command line. Extra arguments are passed straight through, so `mlflow-ui --port 5001`
works. The user runs this; the agent never does (see `CLAUDE.md`, Infrastructure).
"""

import os
import shutil
import sys
from pathlib import Path

from shared.tracking import tracking_uri


def _mlflow_executable() -> str | None:
    """The `mlflow` beside this interpreter, else whatever is on PATH.

    Checking alongside `sys.executable` first means `.venv/bin/mlflow-ui` works without
    the venv being activated, which is how the command is easiest to reach.
    """
    sibling = Path(sys.executable).parent / "mlflow"
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return shutil.which("mlflow")


def main() -> None:
    try:
        uri = tracking_uri()
    except RuntimeError as exc:
        sys.exit(f"error: {exc}")

    executable = _mlflow_executable()
    if executable is None:
        sys.exit("error: `mlflow` was not found — install it into the shared venv (.venv).")

    print(f"Starting the MLflow UI on the '{uri.rsplit('/', 1)[-1]}' database. Ctrl-C to stop.")
    # execv rather than subprocess: the UI is the point of the command, so let it take over
    # this process and own the terminal's signals directly.
    os.execv(executable, ["mlflow", "ui", "--backend-store-uri", uri, *sys.argv[1:]])
