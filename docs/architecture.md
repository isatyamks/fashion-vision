# Fashion Vision — Architecture

## System Overview

Fashion Vision is an AI-powered fashion visual search system that detects clothing items in videos (e.g. Instagram Reels) and matches them against a Shopify product catalog using CLIP embeddings.

## Pipeline

```
┌─────────────────────┐
│   Instagram Reel    │
│   / Video file      │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐      ┌──────────────────────┐
│   FashionDetector   │      │    DuplicateFilter   │
│  (YOLOv8 model)     │─────▶│  (histogram dedup)   │
└────────┬────────────┘      └──────────────────────┘
         │  unique fashion crops (JPEGs)
         ▼
┌─────────────────────┐
│    CLIPEncoder      │  ← shared by both matchers
│  (ViT-B-32 model)   │
└────────┬────────────┘
         │  512-dim embeddings
         ▼
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐  ┌────────┐
│ FAISS │  │Chroma  │  ← pluggable backends
│Matcher│  │Matcher │
└───┬───┘  └───┬────┘
    │           │
    └─────┬─────┘
          │  matched product metadata
          ▼
┌─────────────────────┐
│    results/*.csv    │
└─────────────────────┘
```

## Module Map

| Module | Responsibility |
|---|---|
| `fashion_vision.config` | All paths, thresholds, device |
| `fashion_vision.detection.detector` | YOLO video processing |
| `fashion_vision.detection.dedup` | Histogram-based duplicate rejection |
| `fashion_vision.embeddings.clip_encoder` | CLIP image encoding |
| `fashion_vision.matching.chroma_matcher` | ChromaDB similarity search |
| `fashion_vision.matching.faiss_matcher` | FAISS similarity search |
| `fashion_vision.data.db_builder` | Populate ChromaDB from Shopify CSVs |
| `fashion_vision.data.downloader` | Download product images locally |
| `fashion_vision.data.validators` | Validate image URLs |

## Design Decisions

### Why two matching backends?

- **ChromaDB** (`chroma_matcher`): persistent, no image pre-download, incremental — best for production pipelines where the catalog updates frequently.
- **FAISS** (`faiss_matcher`): in-memory, faster at query time, supports batching — best for batch offline experiments. Requires local images (via `download_images.py`).

### Why DuplicateFilter uses histograms?

CLIP embeddings would be more accurate but much slower for every frame pair. Histogram correlation is a fast perceptual proxy that works well for "same outfit, slightly different angle" deduplication before expensive embedding.

### GPU / CPU

The `config.DEVICE` auto-detects CUDA. Both the YOLO model and CLIP encoder respect this device setting. FAISS itself always runs on CPU (faiss-cpu package); install faiss-gpu for GPU FAISS.
