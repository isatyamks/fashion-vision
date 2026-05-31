# Fashion Vision — Setup Guide

## Prerequisites

- Python 3.9+
- CUDA-compatible GPU (optional but recommended)
- Git

## Installation

```bash
# 1. Clone
git clone https://github.com/isatyamks/fashion-vision.git
cd fashion-vision

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the package in editable mode
pip install -e .
```

## Environment Variables (optional overrides)

| Variable | Default | Description |
|---|---|---|
| `FASHION_VISION_WEIGHTS` | `weights/best.pt` | Path to YOLO weights |
| `FASHION_VISION_CLIP_MODEL` | `clip-ViT-B-32` | SentenceTransformer model name |
| `FASHION_VISION_DEVICE` | auto (cuda/cpu) | Force device |

## Data Setup

Place your Shopify CSVs at:
```
data/
└── shopify_data/
    ├── url_data.csv        # columns: id, image_url
    ├── url_data_small.csv  # smaller sample for testing
    └── product_data.csv    # columns: id, title, product_type, mrp, price_display_amount, ...
```

Place input videos at:
```
data/instagram_reels/
    1.mp4, 2.mp4, ...
```

Place YOLO weights at:
```
weights/
    best.pt
```

## Quick Start

```bash
# Run the full pipeline on a video
python scripts/run_pipeline.py --video data/instagram_reels/1.mp4

# Or run steps separately:
python scripts/detect.py --video data/instagram_reels/1.mp4
python scripts/match.py  --crops_dir outputs/video_crops/crops_<timestamp>

# Build / refresh the ChromaDB product index (one-time)
python scripts/build_db.py

# Pre-download product images for offline FAISS matching
python scripts/download_images.py
```

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```
