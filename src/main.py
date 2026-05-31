"""
scripts/run_pipeline.py
------------------------
CLI entry point: run the full end-to-end Fashion Vision pipeline.

Steps:
  1. Detect and crop fashion items from a video  (FashionDetector)
  2. Match crops against the product catalog     (FaissService)
  3. Save results to a timestamped CSV

Replaces the old run_all.py.

Usage:
    python scripts/run_pipeline.py --video data/instagram_reels/1.mp4
    python scripts/run_pipeline.py \\
        --video data/instagram_reels/1.mp4 \\
        --crops_dir outputs/video_crops/my_run \\
        --conf 0.25 \\
        --threshold 0.75 \\
        --visualise
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.utils.config import (
    CONF_THRESHOLD,
    DEVICE,
    FRAME_SKIP,
    MATCH_THRESHOLD,
    RESULTS_DIR,
    VIDEO_CROPS_DIR,
)
from src.models.yolo_detector import FashionDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end Fashion Vision pipeline: detect → match → report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Detection
    parser.add_argument("--video", required=True, help="Input video file path.")
    parser.add_argument("--weights", default=None, help="YOLO weights path.")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--frame_skip", type=int, default=FRAME_SKIP)
    parser.add_argument("--crops_dir", default=None, help="Directory for crops (auto if omitted).")


    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--visualise", action="store_true")
    parser.add_argument("--output_csv", default=None, help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Step 1: Detection ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1 — Fashion Detection")
    print("=" * 60)

    detector = FashionDetector(
        weights_path=args.weights,
        conf_threshold=args.conf,
        frame_skip=args.frame_skip,
        device=args.device,
    )
    crops, crops_dir = detector.process_video(
        video_path=args.video,
        output_dir=args.crops_dir,
    )

    if not crops:
        print("\n⚠️  No crops detected. Exiting.")
        return

    # ── Step 2: Matching ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"STEP 2 — Product Matching  [FAISS backend]")
    print("=" * 60)

    from src.services.retrieval_service import FaissService
    matcher = FaissService(threshold=args.threshold, device=args.device)
    results = matcher.match_crops(crops_dir=crops_dir, visualise=args.visualise)

    # ── Step 3: Save Results ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — Saving Results")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = args.output_csv or str(RESULTS_DIR / f"pipeline_faiss_{ts}.csv")

    if results:
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"\n✅ {len(results)} matches saved → '{out_csv}'")
    else:
        print("\n❌ No matches found above threshold.")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
