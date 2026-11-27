from io import BytesIO

import pytest
from PIL import Image

from api.services.classification.preprocess import ImageDecodeError, preprocess_image


def _jpeg(size: tuple[int, int], color: tuple[int, int, int] = (200, 100, 50)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def test_224_input_is_not_geometrically_transformed() -> None:
    result = preprocess_image(_jpeg((224, 224)))

    assert result.geometric_transform_applied is False
    assert tuple(result.tensor.shape) == (3, 224, 224)


def test_non_224_input_is_resized_and_cropped() -> None:
    result = preprocess_image(_jpeg((640, 480)))

    assert result.geometric_transform_applied is True
    assert tuple(result.tensor.shape) == (3, 224, 224)


def test_square_non_224_input_is_still_transformed() -> None:
    assert preprocess_image(_jpeg((300, 300))).geometric_transform_applied is True


def test_normalization_uses_imagenet_statistics() -> None:
    # Solid white: (1.0 - mean) / std per channel; JPEG keeps white exact.
    result = preprocess_image(_jpeg((224, 224), (255, 255, 255)))

    expected = [(1.0 - 0.485) / 0.229, (1.0 - 0.456) / 0.224, (1.0 - 0.406) / 0.225]
    for channel, value in enumerate(expected):
        assert result.tensor[channel].mean().item() == pytest.approx(value, abs=0.02)


def test_grayscale_input_is_converted_to_three_channels() -> None:
    buffer = BytesIO()
    Image.new("L", (224, 224), 128).save(buffer, format="JPEG")

    assert tuple(preprocess_image(buffer.getvalue()).tensor.shape) == (3, 224, 224)


@pytest.mark.parametrize("body", [b"", b"not an image at all"])
def test_undecodable_body_raises_decode_error(body: bytes) -> None:
    with pytest.raises(ImageDecodeError):
        preprocess_image(body)


def test_truncated_jpeg_raises_decode_error() -> None:
    with pytest.raises(ImageDecodeError):
        preprocess_image(_jpeg((224, 224))[:100])
