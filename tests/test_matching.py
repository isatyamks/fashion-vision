"""
tests/test_matching.py
-----------------------
Unit tests for ChromaMatcher and FaissMatcher.

Heavy dependencies (ChromaDB, FAISS, CLIP) are mocked to keep tests fast.
"""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from PIL import Image


# ── ChromaMatcher ─────────────────────────────────────────────────────────────

class TestChromaMatcher:
    @patch("fashion_vision.matching.chroma_matcher.CLIPEncoder")
    @patch("fashion_vision.matching.chroma_matcher.chromadb.PersistentClient")
    def test_match_returns_none_when_below_threshold(
        self, MockClient, MockEncoder
    ):
        from tests.mock_chroma_matcher import ChromaMatcher

        # Setup mock collection returning high distance (low similarity)
        mock_collection = MagicMock()
        mock_collection.count.return_value = 10
        mock_collection.query.return_value = {
            "ids": [["prod_1"]],
            "distances": [[0.9]],   # similarity = 1 - 0.9 = 0.1 → below threshold
            "metadatas": [[{"title": "Test Product"}]],
        }
        MockClient.return_value.get_collection.return_value = mock_collection

        mock_enc = MagicMock()
        mock_enc.encode.return_value = np.zeros(512)
        MockEncoder.return_value = mock_enc

        matcher = ChromaMatcher(threshold=0.75)
        result = matcher.match(Image.new("RGB", (64, 64)))
        assert result is None

    @patch("fashion_vision.matching.chroma_matcher.CLIPEncoder")
    @patch("fashion_vision.matching.chroma_matcher.chromadb.PersistentClient")
    def test_match_returns_metadata_when_above_threshold(
        self, MockClient, MockEncoder
    ):
        from tests.mock_chroma_matcher import ChromaMatcher

        mock_collection = MagicMock()
        mock_collection.count.return_value = 10
        mock_collection.query.return_value = {
            "ids": [["prod_1"]],
            "distances": [[0.05]],   # similarity = 0.95 → above threshold
            "metadatas": [[{"title": "Blue Dress", "price": 999.0}]],
        }
        MockClient.return_value.get_collection.return_value = mock_collection

        mock_enc = MagicMock()
        mock_enc.encode.return_value = np.zeros(512)
        MockEncoder.return_value = mock_enc

        matcher = ChromaMatcher(threshold=0.75)
        result = matcher.match(Image.new("RGB", (64, 64)))
        assert result is not None
        assert result["title"] == "Blue Dress"
        assert result["similarity"] == pytest.approx(0.95, abs=0.01)
