"""
tests/test_dedup.py
--------------------
Unit tests for DuplicateFilter.
"""
import numpy as np
import pytest
from PIL import Image

from src.preprocessing.image_processing import DuplicateFilter


def _solid_image(r: int, g: int, b: int, size: int = 50) -> Image.Image:
    """Create a solid-colour PIL image for testing."""
    return Image.new("RGB", (size, size), color=(r, g, b))


class TestDuplicateFilter:
    def test_empty_filter_is_not_duplicate(self):
        f = DuplicateFilter()
        img = _solid_image(255, 0, 0)
        assert f.is_duplicate(img) is False

    def test_same_image_is_duplicate_after_add(self):
        f = DuplicateFilter(threshold=0.4)
        img = _solid_image(200, 100, 50)
        f.add(img)
        assert f.is_duplicate(img) is True

    def test_different_image_is_not_duplicate(self):
        f = DuplicateFilter(threshold=0.4)
        red = _solid_image(255, 0, 0)
        blue = _solid_image(0, 0, 255)
        f.add(red)
        assert f.is_duplicate(blue) is False

    def test_len_tracks_added_images(self):
        f = DuplicateFilter()
        assert len(f) == 0
        f.add(_solid_image(10, 20, 30))
        f.add(_solid_image(40, 50, 60))
        assert len(f) == 2

    def test_reset_clears_seen_images(self):
        f = DuplicateFilter()
        img = _solid_image(128, 128, 128)
        f.add(img)
        f.reset()
        assert len(f) == 0
        assert f.is_duplicate(img) is False
