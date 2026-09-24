"""Wires Tasks 1-4 together: loads the frame/simulated data, reports feature
diagnostics, trains and evaluates all four routers on the frame-level
held-out split, and prints the results. Read-only against the database — no
model files are saved (unlike `router/baseline.py`).

Run as `python -m router.feature_experiment` from `training/`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from common.db import get_engine
from dotenv import load_dotenv
from sqlalchemy import Engine

from router.dataset import frame_level_split
from router.diagnostics import compute_feature_diagnostics
from router.evaluation import (
    BootstrapResult,
    RouterSummary,
    bootstrap_router_vs_offload,
    score_cascade,
    score_decide_first,
    summarize,
)
from router.frame_dataset import (
    CONFIDENCE_COLUMNS,
    IMAGE_COLUMNS,
    load_frame_table,
    load_simulated_frame_ids,
    load_simulated_rows,
)
from router.two_stage import Design, Family, run_router

# Every router this experiment trains and evaluates: one per design/family
# combination, matching `router/baseline.py`'s model families.
ROUTER_CONFIGS: list[tuple[Design, Family]] = [
    ("decide_first", "linear"),
    ("decide_first", "gbt"),
    ("cascade", "linear"),
    ("cascade", "gbt"),
]


@dataclass(frozen=True)
class RouterEvaluation:
    """One design/family combination's evaluation against the held-out test rows."""

    name: str
    design: Design
    family: Family
    summary: RouterSummary
    bootstrap: BootstrapResult


@dataclass(frozen=True)
class ExperimentResult:
    """Everything the experiment prints: the diagnostics table, the held-out
    split's photo counts, and every router's evaluation."""

    diagnostics: pd.DataFrame
    train_frame_count: int
    test_frame_count: int
    routers: list[RouterEvaluation]


def run_experiment(engine: Engine, dataset: str) -> ExperimentResult:
    """Loads the data, computes diagnostics, splits by photo, trains and
    evaluates all four routers on the held-out test rows.
    """
    frame_table = load_frame_table(engine, dataset)
    simulated_frame_ids = load_simulated_frame_ids(engine, dataset)
    simulated_rows = load_simulated_rows(engine, dataset)

    diagnostics = compute_feature_diagnostics(frame_table, [*IMAGE_COLUMNS, *CONFIDENCE_COLUMNS])

    train_split, test_split = frame_level_split(simulated_rows, test_size=0.2, seed=42)
    train_frame_ids = set(train_split["frame_id"])
    test_frame_ids = set(test_split["frame_id"])

    # Re-derive the test rows from `simulated_rows` by frame-id membership
    # (same set of rows as `test_split`, but in `simulated_rows`'s original
    # row order) rather than reusing `test_split` directly: `run_router`
    # filters its own picks the same way internally, so this keeps the
    # scoring DataFrame's row order identical to `picks`'s order. Scoring
    # against `test_split`'s (possibly shuffled) order would silently
    # mis-align the two under pandas's index-based Series arithmetic.
    test_df = simulated_rows[simulated_rows["frame_id"].isin(test_frame_ids)].reset_index(
        drop=True
    )

    routers = []
    for design, family in ROUTER_CONFIGS:
        result = run_router(
            frame_table,
            simulated_rows,
            train_frame_ids,
            test_frame_ids,
            simulated_frame_ids,
            design,
            family,
        )
        score = score_decide_first if design == "decide_first" else score_cascade
        router_utility = score(test_df, result.picks)
        routers.append(
            RouterEvaluation(
                name=f"{design}_{family}",
                design=design,
                family=family,
                summary=summarize(test_df, router_utility),
                bootstrap=bootstrap_router_vs_offload(test_df, router_utility, seed=42),
            )
        )

    return ExperimentResult(
        diagnostics=diagnostics,
        train_frame_count=len(train_frame_ids),
        test_frame_count=len(test_frame_ids),
        routers=routers,
    )


def _print_report(result: ExperimentResult) -> None:
    print("--- Feature diagnostics (Spearman rho vs. offload-local gap, all frames) ---")
    print(result.diagnostics.to_string(index=False))
    print()

    print(
        f"Held-out split: {result.train_frame_count} train photos / "
        f"{result.test_frame_count} test photos.\n"
    )

    print("--- Router results (held-out photos) ---")
    header = (
        f"{'router':<20} {'avg util':>10} {'local':>10} {'offload':>10} {'oracle':>10} "
        f"{'headroom':>10} {'95% CI':>22} {'useful':>8}"
    )
    print(header)
    for router in result.routers:
        s = router.summary
        b = router.bootstrap
        ci = f"[{b.ci_low:.4f}, {b.ci_high:.4f}]"
        print(
            f"{router.name:<20} {s.avg_utility_router:>10.4f} {s.avg_utility_always_local:>10.4f} "
            f"{s.avg_utility_always_offload:>10.4f} {s.avg_utility_oracle:>10.4f} "
            f"{s.headroom_share:>10.2%} {ci:>22} {str(b.useful):>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostics and router evaluation for the spec-01 frame features."
    )
    parser.add_argument(
        "--dataset", default="coco_val2017", help="Dataset name to evaluate (default: coco_val2017)"
    )
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    engine = get_engine()

    result = run_experiment(engine, args.dataset)
    _print_report(result)


if __name__ == "__main__":
    main()
