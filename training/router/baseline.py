"""Direct classifier: the "simple baseline" router design from the research notes.

5 features in (network bandwidth/latency/packet-loss, device load, scene
complexity), one of LOCAL/OFFLOAD out. Trained + evaluated on a frame-level
train/test split so no photo's rows leak across the split. Quick-and-dirty
first pass, not yet cleaned up.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from common.db import get_engine
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from datagen.config import DEFAULT_LAMBDA
from router.dataset import FEATURE_COLUMNS, frame_level_split, load_training_data

MODEL_PATH = Path(__file__).resolve().parent / "models" / "direct_classifier.joblib"


def _utility(accuracy: float, latency_ms: float, lambda_value: float = DEFAULT_LAMBDA) -> float:
    """Same formula as datagen.simulate.labeling's private `_utility` — reimplemented
    here rather than importing a leading-underscore name across packages."""
    return accuracy - lambda_value * (latency_ms / 1000.0)


def train_baseline_classifier(df_train: pd.DataFrame) -> Pipeline:
    """Fit a scaled logistic regression on the 5 features -> LOCAL/OFFLOAD label."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(df_train[FEATURE_COLUMNS], df_train["label"])
    return model


def evaluate(model: Pipeline, df_test: pd.DataFrame) -> dict[str, float]:
    """Compare the router's picks against always-local/always-offload on held-out rows."""
    predicted = model.predict(df_test[FEATURE_COLUMNS])
    accuracy = float((predicted == df_test["label"].to_numpy()).mean())

    router_utility = []
    local_utility = []
    offload_utility = []
    for (_, row), pick in zip(df_test.iterrows(), predicted, strict=True):
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

    model = train_baseline_classifier(df_train)
    metrics = evaluate(model, df_test)

    print("\n--- Results (held-out photos only) ---")
    print(f"Label accuracy (router matches the real winner): {metrics['label_accuracy']:.1%}")
    print(f"Avg utility — router:          {metrics['avg_utility_router']:.4f}")
    print(f"Avg utility — always local:    {metrics['avg_utility_always_local']:.4f}")
    print(f"Avg utility — always offload:  {metrics['avg_utility_always_offload']:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
