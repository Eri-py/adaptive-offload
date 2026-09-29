"""Reports test loss/accuracy (GPU batches) and per-photo latency (small on CPU, large on GPU)."""

import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset

from birds.config import EXPERIMENT_NAME, TRAIN_SETTINGS, WEIGHTS_DIR, ModelName
from birds.data import eval_transform, load_test, make_data_loader
from birds.metrics import EvalResult, loss_and_accuracy
from birds.models import build_model, require_cuda
from shared import tracking

_MODEL_NAMES: tuple[ModelName, ModelName] = ("small", "large")
_TRAIN_COMMAND = "python -m birds.train --model {name}"


def mean_latency_ms(
    model: nn.Module,
    dataset: "Dataset[Any]",
    device: torch.device,
    warmup: int = 10,
) -> float:
    """Mean batch-1 forward-pass ms over `dataset`; warm-up cycles items, CUDA synced."""
    model.eval()
    model.to(device)
    num_items = len(dataset)  # type: ignore[arg-type]

    with torch.inference_mode():
        for i in range(warmup):
            image, _ = dataset[i % num_items]
            image = image.unsqueeze(0).to(device)
            model(image)
        if device.type == "cuda":
            torch.cuda.synchronize()

        total_ms = 0.0
        for i in range(num_items):
            image, _ = dataset[i]
            image = image.unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(image)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            total_ms += (end - start) * 1000

    return total_ms / num_items


def _load_checkpoint(name: ModelName, device: torch.device) -> nn.Module:
    """Builds `name`'s model (untrained head) and loads its trained checkpoint."""
    checkpoint_path = WEIGHTS_DIR / f"{name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint for the '{name}' model at {checkpoint_path}. "
            f"Run `{_TRAIN_COMMAND.format(name=name)}` first."
        )
    model = build_model(name, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def _log_results(
    name: ModelName, result: EvalResult, latency_ms: float, device_label: str
) -> bool:
    """Logs the results to the model's training run; True if attached, else a standalone run."""
    training_run_id = tracking.read_run_id(WEIGHTS_DIR / f"{name}.pt")
    metrics = {
        "test_accuracy": result.accuracy,
        "test_loss": result.loss,
        "mean_latency_ms": latency_ms,
    }
    if training_run_id is not None:
        tracking.log_to_run(training_run_id, metrics=metrics, tags={"eval_device": device_label})
        return True
    with tracking.run(
        f"evaluate-{name}", {"eval_device": device_label}, experiment=EXPERIMENT_NAME
    ) as handle:
        tracking.log_metrics(handle, metrics)
    return False


def main() -> None:
    test_loaders = {
        name: make_data_loader(
            "test", eval_transform(name), TRAIN_SETTINGS[name].batch_size, shuffle=False
        )
        for name in _MODEL_NAMES
    }

    gpu_device = require_cuda()
    cpu_device = torch.device("cpu")
    thread_count = torch.get_num_threads()

    rows: list[tuple[str, EvalResult, float, str]] = []
    unattached: list[str] = []
    for name in _MODEL_NAMES:
        model = _load_checkpoint(name, gpu_device)
        result = loss_and_accuracy(model, test_loaders[name], gpu_device)

        if name == "small":
            # Phone stand-in: CPU.
            latency_model = _load_checkpoint(name, cpu_device)
            latency_device = cpu_device
            device_label = f"cpu ({thread_count} threads)"
        else:
            # Server stand-in: GPU.
            latency_model = model
            latency_device = gpu_device
            device_label = "cuda"

        test_dataset = load_test(eval_transform(name))
        latency_ms = mean_latency_ms(latency_model, test_dataset, latency_device)
        rows.append((name, result, latency_ms, device_label))
        if not _log_results(name, result, latency_ms, device_label):
            unattached.append(name)

    header = f"{'model':<8} {'test accuracy':>14} {'test loss':>10} {'mean ms/photo':>15}  device"
    print(header)
    for row_name, row_result, row_latency_ms, row_label in rows:
        print(
            f"{row_name:<8} {row_result.accuracy:>14.4f} {row_result.loss:>10.4f} "
            f"{row_latency_ms:>15.3f}  {row_label}"
        )
    for unattached_name in unattached:
        print(f"note: {unattached_name} results are not attached to a training run.")


if __name__ == "__main__":
    main()
