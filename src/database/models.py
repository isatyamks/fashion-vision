"""
services/faiss/metadata.py
---------------------------
Handles loading and serving product metadata.
"""
import csv
from typing import Dict


class MetadataManager:
    """Manages product details loaded from the Shopify CSV export."""

    def __init__(self, product_csv: str):
        self.product_csv = product_csv
        self.product_details: Dict[str, Dict] = {}
        self._load_metadata()

    def _load_metadata(self):
        with open(self.product_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.product_details[row["id"]] = dict(row)

    def get_details(self, product_id: str) -> Dict:
        """Fetch metadata for a given product ID."""
        return self.product_details.get(product_id, {})
