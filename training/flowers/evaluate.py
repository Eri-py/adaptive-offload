"""Reports test accuracy and per-photo inference time for a trained flower model.

Accuracy is measured in batches on the GPU for both models (accuracy doesn't
depend on the deployment device, only latency does). Latency is measured one
photo at a time, matching how the model would actually be deployed: the small
model stands in for the phone (CPU), the large one for the server (GPU) — the
same split the COCO simulator used for its two models.
"""

import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from flowers.config import TRAIN_SETTINGS, WEIGHTS_DIR, ModelName
from flowers.data import eval_transform, load_test, make_data_loader
from flowers.models import build_model, require_cuda

_TRAIN_COMMAND = "python -m flowers.train --model {name}"


def test_accuracy(model: nn.Module, loader: "DataLoader[Any]", device: torch.device) -> float:
    """Top-1 accuracy of `model` over every batch in `loader`.

    Plain fp32, not autocast: this runs once per model over the full test
    split, so there's no reason to trade the accuracy figure's precision for
    autocast's speed (unlike training, which runs this every epoch).
    """
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.size(0))
    return correct / total


def mean_latency_ms(
    model: nn.Module,
    dataset: "Dataset[Any]",
    device: torch.device,
    warmup: int = 10,
) -> float:
    """Mean forward-pass latency (ms) over every item in `dataset`, batch of 1.

    `warmup` forward passes run first and are discarded (they reuse the
    dataset's own items, cycling with modulo so this works even when the
    dataset is smaller than `warmup`, e.g. in tests). Only the forward pass
    itself is timed, not moving the tensor to `device`. `torch.cuda
    .synchronize()` brackets each timed forward pass on CUDA, since kernels
    launch asynchronously there and would otherwise be measured incomplete.
    """
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
    # pretrained=False: we're about to overwrite every weight with the
    # checkpoint's state dict, so there's no reason to also download the
    # ImageNet weights first.
    model = build_model(name, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def main() -> None:
    gpu_device = require_cuda()
    cpu_device = torch.device("cpu")
    thread_count = torch.get_num_threads()

    rows: list[tuple[str, float, float, str]] = []
    for name in ("small", "large"):
        model = _load_checkpoint(name, gpu_device)
        settings = TRAIN_SETTINGS[name]
        test_loader = make_data_loader(
            "test", eval_transform(name), settings.batch_size, shuffle=False
        )
        accuracy = test_accuracy(model, test_loader, gpu_device)

        if name == "small":
            # Small model's latency stands in for the phone: CPU.
            latency_model = _load_checkpoint(name, cpu_device)
            latency_device = cpu_device
            device_label = f"cpu ({thread_count} threads)"
        else:
            # Large model's latency stands in for the server: GPU. Reuse the
            # already-loaded, already-on-GPU model from the accuracy step.
            latency_model = model
            latency_device = gpu_device
            device_label = "cuda"

        test_dataset = load_test(eval_transform(name))
        latency_ms = mean_latency_ms(latency_model, test_dataset, latency_device)
        rows.append((name, accuracy, latency_ms, device_label))

    header = f"{'model':<8} {'test accuracy':>14} {'mean ms/photo':>15}  device"
    print(header)
    for name, accuracy, latency_ms, device_label in rows:
        print(f"{name:<8} {accuracy:>14.4f} {latency_ms:>15.3f}  {device_label}")


if __name__ == "__main__":
    main()
