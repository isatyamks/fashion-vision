# Flickd - Fashion Video Analysis

This project analyzes fashion videos to identify outfits, match them with products from a catalog, and determine the overall vibe of the content.

## Key Features

- Video analysis for outfit detection using YOLOv8n
- Product matching from provided catalog.csv only
- Vibe tagging (1-3 vibes per video)
- Multi-product detection (2-4 products per video)
- Confidence threshold of 0.75 for product matches
- Reused embeddings for catalog matching
- Audio transcription using Whisper (medium variant)

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required models:
```bash
python scripts/download_models.py
```

## Project Structure

```
flickd/
├── videos/              # Input videos for analysis
├── catalog.csv         # Product catalog (DO NOT use external data)
├── vibes_list.json     # Predefined vibes
├── outputs/            # Generated JSON outputs
├── models/            # Downloaded ML models
│   ├── yolov8n.pt    # YOLOv8n model
│   └── clip/         # CLIP ViT-B/32 model
├── src/               # Source code
│   ├── video_processor.py    # Video processing and frame extraction
│   ├── product_matcher.py    # Product matching using CLIP
│   ├── vibe_classifier.py    # Vibe classification (hybrid approach)
│   ├── audio_processor.py    # Audio transcription using Whisper
│   └── main.py              # Main pipeline
├── scripts/           # Utility scripts
│   └── download_models.py
├── README.md
└── requirements.txt
```

## Usage

1. Place your videos in the `videos/` directory
2. Ensure `catalog.csv` and `vibes_list.json` are in the root directory
3. Run the analysis:
```bash
python src/main.py
```

## Technical Details

### Models Used
- Object Detection: YOLOv8n
- Product Matching: CLIP ViT-B/32
- Audio Transcription: Whisper (medium variant)
- Vibe Classification: Hybrid approach (CLIP + rule-based)

### Key Specifications
- Catalog: Uses only provided catalog.csv
- Product Matching: 
  - Minimum similarity threshold: 0.75
  - Target products per video: 2-4
  - Embeddings are cached and reused
- Vibe Tagging:
  - Returns 1-3 relevant vibes per video
  - Uses hybrid approach (ML + rule-based)
- Audio:
  - Uses Whisper medium variant
  - High-quality transcript generation

## Output Format

Each video generates a JSON file in the `outputs/` directory with:
```json
{
    "video_id": "string",
    "detected_vibes": ["string"],
    "matched_products": [
        {
            "product_id": "string",
            "confidence": float,
            "timestamp": float
        }
    ],
    "transcript": "string"
}
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- FFmpeg (for video processing)

## Demo

A Loom demo (max 5 minutes) is available that demonstrates:
1. Pipeline workflow
2. Product detection and matching
3. Vibe classification
4. Audio transcription
5. Output generation

## License

MIT License 