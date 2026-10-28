"""Tests for `birds.evaluation.records`."""

import pytest

from birds.evaluation.records import numeric_photo_id


def test_numeric_photo_id_parses_stringified_ints() -> None:
    assert numeric_photo_id("10") > numeric_photo_id("2")


def test_non_numeric_photo_id_raises_an_actionable_error() -> None:
    with pytest.raises(ValueError, match=r"photo id 'img_7' is not an integer"):
        numeric_photo_id("img_7")
