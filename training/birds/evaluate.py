"""Reports test accuracy (GPU batches) and per-photo latency (small on CPU, large on GPU)."""

import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset

from birds.config import TRAIN_SETTINGS, WEIGHTS_DIR, ModelName
from birds.data import eval_transform, load_test, make_data_loader
from birds.metrics import top1_accuracy
from birds.models import build_model, require_cuda

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
    # Weights are overwritten by the checkpoint, so skip the ImageNet download.
    model = build_model(name, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def main() -> None:
    gpu_device = require_cuda()
    cpu_device = torch.device("cpu")
    thread_count = torch.get_num_threads()

    # Loaders come first: a missing dataset must fail before any checkpoint loads.
    test_loaders = {
        name: make_data_loader(
            "test", eval_transform(name), TRAIN_SETTINGS[name].batch_size, shuffle=False
        )
        for name in _MODEL_NAMES
    }

    rows: list[tuple[str, float, float, str]] = []
    for name in _MODEL_NAMES:
        model = _load_checkpoint(name, gpu_device)
        accuracy = top1_accuracy(model, test_loaders[name], gpu_device)

        if name == "small":
            # Small model stands in for the phone: CPU.
            latency_model = _load_checkpoint(name, cpu_device)
            latency_device = cpu_device
            device_label = f"cpu ({thread_count} threads)"
        else:
            # Large model stands in for the server: GPU, reusing the loaded model.
            latency_model = model
            latency_device = gpu_device
            device_label = "cuda"

        test_dataset = load_test(eval_transform(name))
        latency_ms = mean_latency_ms(latency_model, test_dataset, latency_device)
        rows.append((name, accuracy, latency_ms, device_label))

    header = f"{'model':<8} {'test accuracy':>14} {'mean ms/photo':>15}  device"
    print(header)
    for row_name, row_accuracy, row_latency_ms, row_label in rows:
        print(f"{row_name:<8} {row_accuracy:>14.4f} {row_latency_ms:>15.3f}  {row_label}")


if __name__ == "__main__":
    main()
