"""Turns uploaded JPEG bytes into the normalized tensor the classifier expects."""

from dataclasses import dataclass
from io import BytesIO

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.transforms import InterpolationMode

INPUT_SIZE = 224
RESIZE_SIZE = 232
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Same as training's eval transform: both models use bilinear Resize(232) + CenterCrop(224).
_GEOMETRY = transforms.Compose(
    [
        transforms.Resize(RESIZE_SIZE, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(INPUT_SIZE),
    ]
)
_TO_NORMALIZED_TENSOR = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
)


class ImageDecodeError(ValueError):
    """The request body is not an image Pillow can decode."""


@dataclass(frozen=True)
class PreprocessedImage:
    tensor: torch.Tensor  # (3, 224, 224), float32, ImageNet-normalized
    geometric_transform_applied: bool


def preprocess_image(jpeg_bytes: bytes) -> PreprocessedImage:
    """Decodes to RGB; resizes/crops unless the phone already sent an exact 224x224 crop."""
    try:
        image = Image.open(BytesIO(jpeg_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ImageDecodeError("Request body is not a decodable image.") from error

    # The phone sends a pre-cropped 224x224 (the cost model's payload); re-cropping would drift.
    already_cropped = image.size == (INPUT_SIZE, INPUT_SIZE)
    if not already_cropped:
        image = _GEOMETRY(image)
    return PreprocessedImage(
        tensor=_TO_NORMALIZED_TENSOR(image),
        geometric_transform_applied=not already_cropped,
    )
