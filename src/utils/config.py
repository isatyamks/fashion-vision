"""
fashion_vision/config.py
------------------------
Centralized configuration for Fashion Vision.
All paths, thresholds, and runtime settings live here.
Override defaults via environment variables where noted.
"""
import os
from pathlib import Path

import torch

# ── Root & Layout ──────────────────────────────────────────────────────────────
ROOT: Path = Path(__file__).parent.parent.parent

DATA_DIR: Path = ROOT / "data"
OUTPUTS_DIR: Path = ROOT / "outputs"
VECTOR_DB_DIR: Path = ROOT / "vectordb"
WEIGHTS_DIR: Path = ROOT / "weights"

# Generated output sub-directories
PRODUCT_IMAGES_DIR: Path = DATA_DIR / "product"
VIDEO_CROPS_DIR: Path = OUTPUTS_DIR / "video_crops"
RESULTS_DIR: Path = OUTPUTS_DIR / "results"

# ── Data Sources ───────────────────────────────────────────────────────────────
SHOPIFY_URL_CSV: Path = DATA_DIR / "shopify_data" / "url_data.csv"
SHOPIFY_URL_SMALL_CSV: Path = DATA_DIR / "shopify_data" / "url_data_small.csv"
SHOPIFY_PRODUCT_CSV: Path = DATA_DIR / "shopify_data" / "product_data.csv"

# ── Model ─────────────────────────────────────────────────────────────────────
# Override YOLO_WEIGHTS via env: FASHION_VISION_WEIGHTS=/path/to/yolo11n.pt
YOLO_WEIGHTS: Path = Path(
    os.environ.get("FASHION_VISION_WEIGHTS", str(WEIGHTS_DIR / "yolo11n.pt"))
)

# ── Detection ─────────────────────────────────────────────────────────────────
CONF_THRESHOLD: float = 0.2   # YOLO confidence threshold
FRAME_SKIP: int = 2           # Process every Nth frame (1 = every frame)
DEDUP_THRESHOLD: float = 0.4  # Histogram correlation threshold for duplicate frames

# ── Matching ──────────────────────────────────────────────────────────────────
MATCH_THRESHOLD: float = 0.75  # Minimum cosine similarity to accept a match
BATCH_SIZE: int = 32           # Embedding batch size
TOP_K: int = 3                 # Number of nearest neighbours to retrieve

# ── Embeddings (SigLIP) ───────────────────────────────────────────────────────
SIGLIP_MODEL_NAME: str = os.environ.get(
    "FASHION_VISION_SIGLIP_MODEL", "google/siglip-base-patch16-224"
)

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE: str = os.environ.get(
    "FASHION_VISION_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
)
