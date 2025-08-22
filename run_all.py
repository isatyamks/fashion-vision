import subprocess
import sys
import os
from datetime import datetime
import argparse

def run_script(script, args):
    print(f"\nRunning {script} {' '.join(args)}...")
    result = subprocess.run([sys.executable, script] + args, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("[stderr]", result.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model.py and cuda_matching.py in sequence.")
    parser.add_argument('--video', type=str, required=True, help='Input video file path')
    parser.add_argument('--crops_dir', type=str, default=None, help='Directory to save and use cropped images')
    args = parser.parse_args()

    # Use a timestamped crops dir if not provided
    if args.crops_dir:
        crops_dir = args.crops_dir
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        crops_dir = f"video_crops/crops_{timestamp}"
    os.makedirs(crops_dir, exist_ok=True)

    run_script("model.py", ["--video", args.video, "--output_dir", crops_dir])
    run_script("cuda_matching.py", ["--crops_dir", crops_dir])
