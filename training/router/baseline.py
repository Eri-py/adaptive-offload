"""Direct classifier: the "simple baseline" router design from the research notes.

5 features in (network bandwidth/latency/packet-loss, device load, scene
complexity), one of LOCAL/OFFLOAD out. Trained + evaluated on a frame-level
train/test split so no photo's rows leak across the split. Quick-and-dirty
first pass, not yet cleaned up.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from common.db import get_engine
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from datagen.config import DEFAULT_LAMBDA
from router.dataset import FEATURE_COLUMNS, frame_level_split, load_training_data

MODEL_PATH = Path(__file__).resolve().parent / "models" / "direct_classifier.joblib"
NONLINEAR_MODEL_PATH = Path(__file__).resolve().parent / "models" / "nonlinear_classifier.joblib"
UTILITY_MODEL_PATH = Path(__file__).resolve().parent / "models" / "utility_regressors.joblib"

_UTILITY_TARGETS = ["local_latency_ms", "local_accuracy", "offload_latency_ms", "offload_accuracy"]


def _utility(accuracy: float, latency_ms: float, lambda_value: float = DEFAULT_LAMBDA) -> float:
    """Same formula as datagen.simulate.labeling's private `_utility` — reimplemented
    here rather than importing a leading-underscore name across packages."""
    return accuracy - lambda_value * (latency_ms / 1000.0)


def train_baseline_classifier(df_train: pd.DataFrame) -> Pipeline:
    """Fit a scaled logistic regression on the 5 features -> LOCAL/OFFLOAD label."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(df_train[FEATURE_COLUMNS], df_train["label"])
    return model


def train_nonlinear_classifier(df_train: pd.DataFrame) -> HistGradientBoostingClassifier:
    """Same 5 features -> label task as `train_baseline_classifier`, but a
    non-linear model (gradient-boosted trees) instead of a linear one — no
    feature scaling needed, trees don't care about feature scale. Everything
    else (features, split, evaluation) is identical, so any performance
    difference from `train_baseline_classifier` isolates model capacity, not
    a confound with the data or evaluation methodology.
    """
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(df_train[FEATURE_COLUMNS], df_train["label"])
    return model


def _score_picks(picks: np.ndarray, df_test: pd.DataFrame) -> dict[str, float]:
    """Given one LOCAL/OFFLOAD pick per test row, score it against always-local/
    always-offload using each row's *real* measured latency/accuracy — shared by
    every router design below, so the evaluation methodology never drifts between
    them, only the picks themselves do.
    """
    accuracy = float((picks == df_test["label"].to_numpy()).mean())

    router_utility = []
    local_utility = []
    offload_utility = []
    for (_, row), pick in zip(df_test.iterrows(), picks, strict=True):
        local_u = _utility(row["local_accuracy"], row["local_latency_ms"])
        offload_u = _utility(row["offload_accuracy"], row["offload_latency_ms"])
        router_utility.append(local_u if pick == "LOCAL" else offload_u)
        local_utility.append(local_u)
        offload_utility.append(offload_u)

    return {
        "label_accuracy": accuracy,
        "avg_utility_router": sum(router_utility) / len(router_utility),
        "avg_utility_always_local": sum(local_utility) / len(local_utility),
        "avg_utility_always_offload": sum(offload_utility) / len(offload_utility),
    }


def evaluate(
    model: Pipeline | HistGradientBoostingClassifier, df_test: pd.DataFrame
) -> dict[str, float]:
    """Compare the router's picks against always-local/always-offload on held-out rows."""
    picks = model.predict(df_test[FEATURE_COLUMNS])
    return _score_picks(picks, df_test)


def train_utility_regressors(df_train: pd.DataFrame) -> dict[str, LinearRegression]:
    """One linear regressor per target (local/offload latency + accuracy), same
    features and same (linear) capacity as `train_baseline_classifier` — isolates
    the effect of predicting utility's inputs directly instead of predicting the
    label, without also changing model capacity, so it's comparable to the
    logistic-regression result specifically.
    """
    return {
        target: LinearRegression().fit(df_train[FEATURE_COLUMNS], df_train[target])
        for target in _UTILITY_TARGETS
    }


def predict_utility_regression_picks(
    regressors: dict[str, LinearRegression], df: pd.DataFrame
) -> np.ndarray:
    """Predict all four latency/accuracy values, combine into predicted utility per
    path via the same formula real labels were generated with, and pick whichever
    path's *predicted* utility is higher. Shared by `evaluate_utility_regression`
    and by the per-condition breakdown analysis, so both look at the exact same
    picks.
    """
    features = df[FEATURE_COLUMNS]
    predicted = {target: regressors[target].predict(features) for target in _UTILITY_TARGETS}

    predicted_local_utility = predicted["local_accuracy"] - DEFAULT_LAMBDA * (
        predicted["local_latency_ms"] / 1000.0
    )
    predicted_offload_utility = predicted["offload_accuracy"] - DEFAULT_LAMBDA * (
        predicted["offload_latency_ms"] / 1000.0
    )
    picks: np.ndarray = np.where(
        predicted_offload_utility > predicted_local_utility, "OFFLOAD", "LOCAL"
    )
    return picks


def evaluate_utility_regression(
    regressors: dict[str, LinearRegression], df_test: pd.DataFrame
) -> dict[str, float]:
    """Score `predict_utility_regression_picks`'s picks against always-local/always-offload."""
    picks = predict_utility_regression_picks(regressors, df_test)
    return _score_picks(picks, df_test)


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    engine = get_engine()

    df = load_training_data(engine, dataset="coco_val2017")
    print(f"Loaded {len(df)} rows across {df['frame_id'].nunique()} unique photos.")

    df_train, df_test = frame_level_split(df, test_size=0.2, seed=42)
    print(
        f"Train: {len(df_train)} rows / {df_train['frame_id'].nunique()} photos. "
        f"Test: {len(df_test)} rows / {df_test['frame_id'].nunique()} photos."
    )

    linear_model = train_baseline_classifier(df_train)
    linear_metrics = evaluate(linear_model, df_test)

    nonlinear_model = train_nonlinear_classifier(df_train)
    nonlinear_metrics = evaluate(nonlinear_model, df_test)

    utility_regressors = train_utility_regressors(df_train)
    utility_metrics = evaluate_utility_regression(utility_regressors, df_test)

    print("\n--- Results (held-out photos only) ---")
    print(f"{'metric':<32} {'logistic':>12} {'HistGB':>12} {'utility-reg':>14}")
    for key, label in [
        ("label_accuracy", "Label accuracy"),
        ("avg_utility_router", "Avg utility — router"),
        ("avg_utility_always_local", "Avg utility — always local"),
        ("avg_utility_always_offload", "Avg utility — always offload"),
    ]:
        print(
            f"{label:<32} {linear_metrics[key]:>12.4f} {nonlinear_metrics[key]:>12.4f} "
            f"{utility_metrics[key]:>14.4f}"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(linear_model, MODEL_PATH)
    joblib.dump(nonlinear_model, NONLINEAR_MODEL_PATH)
    joblib.dump(utility_regressors, UTILITY_MODEL_PATH)
    print(f"\nSaved models to {MODEL_PATH}, {NONLINEAR_MODEL_PATH}, and {UTILITY_MODEL_PATH}")


if __name__ == "__main__":
    main()
