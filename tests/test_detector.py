"""
tests/test_detector.py
-----------------------
Unit tests for FashionDetector.

These tests mock YOLO and cv2 to avoid requiring GPU or real weights.
"""
from unittest.mock import MagicMock, patch, PropertyMock
import pytest
from PIL import Image

from src.models.yolo_detector import FashionDetector


@pytest.fixture
def mock_detector(tmp_path):
    """Create a FashionDetector with a mocked YOLO model."""
    with patch("models.detector.YOLO") as MockYOLO:
        mock_model = MagicMock()
        mock_model.names = {0: "Casual_Top", 1: "Casual_Jeans"}
        MockYOLO.return_value = mock_model
        detector = FashionDetector(
            weights_path="fake_weights.pt",
            device="cpu",
        )
        yield detector, mock_model, tmp_path


class TestFashionDetectorInit:
    def test_loads_model(self, mock_detector):
        detector, mock_model, _ = mock_detector
        mock_model.to.assert_called_once_with("cpu")

    def test_default_thresholds(self, mock_detector):
        detector, _, _ = mock_detector
        assert 0 < detector.conf_threshold < 1
        assert detector.frame_skip >= 1


class TestDefaultOutputDir:
    def test_output_dir_contains_crops(self):
        out = FashionDetector._default_output_dir()
        assert "crops_" in out
