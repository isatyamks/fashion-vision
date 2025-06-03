import os
import json
from pathlib import Path
from typing import Dict, List
import argparse

from video_processor import VideoProcessor
from product_matcher import ProductMatcher
from vibe_classifier import VibeClassifier

def process_video(
    video_path: str,
    video_processor: VideoProcessor,
    product_matcher: ProductMatcher,
    vibe_classifier: VibeClassifier
) -> Dict:
    """Process a single video and return the analysis results."""
    # Extract video ID from filename
    video_id = Path(video_path).stem
    
    # Process video frames
    frames = video_processor.extract_frames(video_path)
    detections = video_processor.process_video(video_path)
    
    # Find matching products
    products = product_matcher.process_detections(detections, frames)
    
    # Classify vibes
    vibes = vibe_classifier.process_frames(frames)
    
    # Prepare output
    output = {
        "video_id": video_id,
        "vibes": vibes,
        "products": products
    }
    
    return output

def main():
    parser = argparse.ArgumentParser(description="Process fashion videos and generate analysis")
    parser.add_argument("--videos_dir", default="videos", help="Directory containing input videos")
    parser.add_argument("--catalog_path", default="catalog.csv", help="Path to product catalog")
    parser.add_argument("--vibes_path", default="vibes_list.json", help="Path to vibes list")
    parser.add_argument("--output_dir", default="outputs", help="Directory for output files")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize components
    video_processor = VideoProcessor()
    product_matcher = ProductMatcher(args.catalog_path)
    vibe_classifier = VibeClassifier(args.vibes_path)
    
    # Process all videos in the directory
    video_files = [f for f in os.listdir(args.videos_dir) if f.endswith(('.mp4', '.MP4'))]
    
    for video_file in video_files:
        video_path = os.path.join(args.videos_dir, video_file)
        print(f"Processing {video_file}...")
        
        try:
            # Process video
            result = process_video(
                video_path,
                video_processor,
                product_matcher,
                vibe_classifier
            )
            
            # Save results
            output_path = os.path.join(args.output_dir, f"{result['video_id']}.json")
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
                
            print(f"Results saved to {output_path}")
            
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 