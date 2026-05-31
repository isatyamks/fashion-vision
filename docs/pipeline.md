# Fashion Vision — Pipeline Reference

## End-to-End Pipeline

```
scripts/run_pipeline.py
    │
    ├─ Step 1: FashionDetector.process_video()
    │       Input  : video file
    │       Output : outputs/video_crops/crops_<ts>/  (JPEG crops)
    │
    ├─ Step 2: FaissMatcher.match_crops()  (or ChromaMatcher)
    │       Input  : crops directory
    │       Output : list of match result dicts
    │
    └─ Step 3: Save CSV
            Output : outputs/results/pipeline_<backend>_<ts>.csv
```

## Script Reference

### `scripts/detect.py`
```
usage: detect.py --video VIDEO [--output_dir DIR] [--weights PT]
                 [--conf FLOAT] [--frame_skip INT] [--device STR]
                 [--visualise]
```

### `scripts/match.py`
```
usage: match.py --crops_dir DIR [--backend {faiss,chroma}]
                [--threshold FLOAT] [--device STR] [--visualise]
                [--output_csv PATH]
```

### `scripts/build_db.py`
```
usage: build_db.py [--url_csv PATH] [--product_csv PATH]
```

### `scripts/download_images.py`
```
usage: download_images.py [--url_csv PATH] [--output_dir DIR]
                          [--timeout INT]
```

### `scripts/run_pipeline.py`
```
usage: run_pipeline.py --video VIDEO [--weights PT] [--conf FLOAT]
                       [--frame_skip INT] [--crops_dir DIR]
                       [--backend {faiss,chroma}] [--threshold FLOAT]
                       [--device STR] [--visualise] [--output_csv PATH]
```

## Output Format

### `outputs/results/*.csv`

| Column | Type | Description |
|---|---|---|
| `crop_filename` | str | Source crop image filename |
| `matched_id` | str | Shopify product ID |
| `similarity` | float | Cosine similarity (0–1) |
| `title` | str | Product name |
| `product_type` | str | Category |
| `mrp` | float | Original price |
| `price` | float | Display price |
| `tags` | str | Product tags |
| `collections` | str | Collections |
| `top_matches` | list | Top-K match dicts (FAISS only) |

## Tuning Tips

- **Lower `--conf`** (e.g. `0.15`) → more detections, more false positives
- **Higher `--threshold`** (e.g. `0.85`) → fewer but more confident matches
- **Lower `--frame_skip`** (e.g. `1`) → denser sampling, slower processing
- **Increase `BATCH_SIZE`** in `config.py` if you have VRAM to spare
