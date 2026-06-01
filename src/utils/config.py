import os
from pathlib import Path
import torch

ROOT: Path = Path(__file__).parent.parent.parent
DATA_DIR: Path = ROOT / "data"
OUTPUTS_DIR: Path = ROOT / "outputs"
VECTOR_DB_DIR: Path = ROOT / "vectordb"
WEIGHTS_DIR: Path = ROOT / "weights"
PRODUCT_IMAGES_DIR: Path = DATA_DIR / "product"
VIDEO_CROPS_DIR: Path = OUTPUTS_DIR / "video_crops"
RESULTS_DIR: Path = OUTPUTS_DIR / "results"
SHOPIFY_URL_CSV: Path = DATA_DIR / "shopify_data" / "url_data.csv"
SHOPIFY_URL_SMALL_CSV: Path = DATA_DIR / "shopify_data" / "url_data_small.csv"
SHOPIFY_PRODUCT_CSV: Path = DATA_DIR / "shopify_data" / "product_data.csv"
YOLO_WEIGHTS: Path = Path(
    os.environ.get("FASHION_VISION_WEIGHTS", str(WEIGHTS_DIR / "yolo11n.pt"))
)
CONF_THRESHOLD: float = 0.2
FRAME_SKIP: int = 2
DEDUP_THRESHOLD: float = 0.4
MATCH_THRESHOLD: float = 0.75
BATCH_SIZE: int = 32
TOP_K: int = 3
SIGLIP_MODEL_NAME: str = os.environ.get(
    "FASHION_VISION_SIGLIP_MODEL", "google/siglip-base-patch16-224"
)
DEVICE: str = os.environ.get(
    "FASHION_VISION_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
)
