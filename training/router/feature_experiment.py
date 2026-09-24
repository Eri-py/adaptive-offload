"""Wires Tasks 1-4 together: loads the frame/simulated data, reports feature
diagnostics, trains and evaluates all eight routers (four design/family
combinations, each under the classifier and cost-aware stage-2 objectives)
on the frame-level held-out split, and prints the results. Read-only against
the database — no model files are saved (unlike `router/baseline.py`).

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
    cascade_ceiling_utility,
    escalated_utility,
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
from router.two_stage import Design, Family, Objective, run_router

# Every router this experiment trains and evaluates: the four classifier
# design/family combinations (matching `router/baseline.py`'s model
# families), plus the same four combinations again under the cost-aware
# objective (regresses the utility margin instead of a 0/1 label — see
# `router.two_stage.stage2_margin`).
ROUTER_CONFIGS: list[tuple[Design, Family, Objective]] = [
    ("decide_first", "linear", "classifier"),
    ("decide_first", "gbt", "classifier"),
    ("cascade", "linear", "classifier"),
    ("cascade", "gbt", "classifier"),
    ("decide_first", "linear", "cost_aware"),
    ("decide_first", "gbt", "cost_aware"),
    ("cascade", "linear", "cost_aware"),
    ("cascade", "gbt", "cost_aware"),
]


def _router_name(design: Design, family: Family, objective: Objective) -> str:
    suffix = "" if objective == "classifier" else "_costaware"
    return f"{design}_{family}{suffix}"


@dataclass(frozen=True)
class RouterEvaluation:
    """One design/family/objective combination's evaluation against the
    held-out test rows."""

    name: str
    design: Design
    family: Family
    objective: Objective
    summary: RouterSummary
    bootstrap: BootstrapResult
    # The cascade design's own ceiling (`cascade_ceiling_utility`'s mean over
    # the held-out rows) — `None` for decide_first rows, since that ceiling
    # only applies to the ACCEPT/ESCALATE design. Identical across every
    # cascade router here (it depends only on the held-out rows, not on any
    # router's picks), computed once in `run_experiment` and copied onto each
    # cascade `RouterEvaluation` so it prints alongside that router's row.
    cascade_ceiling: float | None


@dataclass(frozen=True)
class ExperimentResult:
    """Everything the experiment prints: the diagnostics table, the held-out
    split's photo counts, and every router's evaluation."""

    diagnostics: pd.DataFrame
    train_frame_count: int
    test_frame_count: int
    routers: list[RouterEvaluation]
    # Cascade design ceiling and always-escalate average over the held-out
    # test rows (see `RouterEvaluation.cascade_ceiling`'s docstring) — kept
    # at the experiment level too since both are single numbers shared by
    # every cascade router, not something that varies per router config.
    cascade_ceiling: float
    always_escalate: float


def run_experiment(engine: Engine, dataset: str) -> ExperimentResult:
    """Loads the data, computes diagnostics, splits by photo, trains and
    evaluates all eight routers on the held-out test rows.
    """
    frame_table = load_frame_table(engine, dataset)
    simulated_frame_ids = load_simulated_frame_ids(engine, dataset)
    simulated_rows = load_simulated_rows(engine, dataset)

    diagnostics = compute_feature_diagnostics(frame_table, [*IMAGE_COLUMNS, *CONFIDENCE_COLUMNS])

    train_split, test_split = frame_level_split(simulated_rows, test_size=0.2, seed=42)
    train_frame_ids = set(train_split["frame_id"])
    test_frame_ids = set(test_split["frame_id"])

    routers = []
    # Depend only on the held-out rows, not any router's picks, so computed
    # once from the first router's `test_rows` rather than re-filtering here.
    cascade_ceiling: float | None = None
    always_escalate: float | None = None
    for design, family, objective in ROUTER_CONFIGS:
        result = run_router(
            frame_table,
            simulated_rows,
            train_frame_ids,
            test_frame_ids,
            simulated_frame_ids,
            design,
            family,
            objective,
        )
        if cascade_ceiling is None or always_escalate is None:
            cascade_ceiling = float(cascade_ceiling_utility(result.test_rows).mean())
            always_escalate = float(escalated_utility(result.test_rows).mean())
        score = score_decide_first if design == "decide_first" else score_cascade
        router_utility = score(result.test_rows, result.picks)
        routers.append(
            RouterEvaluation(
                name=_router_name(design, family, objective),
                design=design,
                family=family,
                objective=objective,
                summary=summarize(result.test_rows, router_utility),
                bootstrap=bootstrap_router_vs_offload(result.test_rows, router_utility, seed=42),
                cascade_ceiling=cascade_ceiling if design == "cascade" else None,
            )
        )

    assert cascade_ceiling is not None
    assert always_escalate is not None
    return ExperimentResult(
        diagnostics=diagnostics,
        train_frame_count=len(train_frame_ids),
        test_frame_count=len(test_frame_ids),
        routers=routers,
        cascade_ceiling=cascade_ceiling,
        always_escalate=always_escalate,
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
    # 29-char width fits the longest router name ("decide_first_linear_costaware").
    header = (
        f"{'router':<29} {'avg util':>10} {'local':>10} {'offload':>10} {'oracle':>10} "
        f"{'headroom':>10} {'95% CI':>22} {'useful':>8} {'casc ceiling':>13}"
    )
    print(header)
    for router in result.routers:
        s = router.summary
        b = router.bootstrap
        ci = f"[{b.ci_low:.4f}, {b.ci_high:.4f}]"
        ceiling = "-" if router.cascade_ceiling is None else f"{router.cascade_ceiling:.4f}"
        print(
            f"{router.name:<29} {s.avg_utility_router:>10.4f} {s.avg_utility_always_local:>10.4f} "
            f"{s.avg_utility_always_offload:>10.4f} {s.avg_utility_oracle:>10.4f} "
            f"{s.headroom_share:>10.2%} {ci:>22} {str(b.useful):>8} {ceiling:>13}"
        )
    print(
        f"\n'casc ceiling' (cascade rows only) is the cascade design's own ceiling — "
        f"max(local_utility, escalated_utility) per row, averaged over the held-out "
        f"rows: {result.cascade_ceiling:.4f}. Always-escalate (every row ESCALATEs): "
        f"{result.always_escalate:.4f}. Both are below the 'oracle' column above because "
        f"ESCALATE always pays the local pass's latency on top of the offload pass's, so "
        f"a cascade router can never earn the oracle's uncharged OFFLOAD latency."
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
