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
    parser.add_argument("--video", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--frame_skip", type=int, default=FRAME_SKIP)
    parser.add_argument("--crops_dir", default=None)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--output_csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        print("\nNo crops detected. Exiting.")
        return
    print("\n" + "=" * 60)
    print(f"STEP 2 — Product Matching  [FAISS backend]")
    print("=" * 60)
    from src.services.retrieval_service import FaissService

    matcher = FaissService(threshold=args.threshold, device=args.device)
    results = matcher.match_crops(crops_dir=crops_dir)
    print("\n" + "=" * 60)
    print("STEP 3 — Saving Results")
    print("=" * 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = args.output_csv or str(RESULTS_DIR / f"pipeline_faiss_{ts}.csv")
    if results:
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"\n{len(results)} matches saved -> '{out_csv}'")
    else:
        print("\nNo matches found above threshold.")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
