from pathlib import Path
from typing import List, Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
CATALOG_DIR = DATA_DIR / "catalog"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
CATALOG_DIR.mkdir(exist_ok=True)

# Model settings
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
SPACY_MODEL = "en_core_web_sm"

# FAISS settings
FAISS_INDEX_PATH = CATALOG_DIR / "product_index.faiss"
EMBEDDING_DIM = 512  # CLIP embedding dimension

# Vibe classification settings
SUPPORTED_VIBES = [
    "Coquette",
    "Clean Girl",
    "Cottagecore",
    "Streetcore",
    "Y2K",
    "Boho",
    "Party Glam"
]

# Matching thresholds
MATCH_THRESHOLDS = {
    "exact": 0.9,
    "similar": 0.75
}

# API settings
API_V1_PREFIX = "/api/v1"
PROJECT_NAME = "Flickd Smart Tagging Engine"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Video processing settings
FRAME_EXTRACTION_INTERVAL = 1  # Extract 1 frame per second
MAX_FRAMES = 30  # Maximum number of frames to process per video 