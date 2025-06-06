import os
import json
import logging
from pathlib import Path
from typing import Dict, List
import torch
from faster_whisper import WhisperModel
from video_processor import VideoProcessor
from product_matcher import ProductMatcher
from vibe_classifier import VibeClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlickdPipeline:
    def __init__(
        self,
        catalog_path: str = "catalog.csv",
        vibes_path: str = "vibes_list.json",
        output_dir: str = "outputs"
    ):
        """
        Initialize the Flickd pipeline.
        
        Args:
            catalog_path: Path to catalog CSV
            vibes_path: Path to vibes list JSON
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        logger.info("Initializing pipeline components...")
        self.video_processor = VideoProcessor()
        self.product_matcher = ProductMatcher(catalog_path=catalog_path)
        self.vibe_classifier = VibeClassifier(vibes_path=vibes_path)
        
        # Initialize Whisper
        logger.info("Loading Whisper model...")
        self.whisper_model = WhisperModel(
            "medium",
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "float32"
        )
        
    def transcribe_audio(self, video_path: str) -> str:
        """
        Transcribe audio from video using Whisper.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Transcribed text
        """
        logger.info(f"Transcribing audio from {video_path}")
        segments, _ = self.whisper_model.transcribe(
            video_path,
            beam_size=5,
            word_timestamps=True
        )
        
        # Combine all segments into a single text
        transcript = " ".join([segment.text for segment in segments])
        return transcript
        
    def process_video(
        self,
        video_path: str,
        video_id: str,
        caption: str = ""
    ) -> Dict:
        """
        Process a video through the entire pipeline.
        
        Args:
            video_path: Path to video file
            video_id: Unique identifier for the video
            caption: Optional video caption
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing video {video_id}")
        
        # 1. Process video frames and detect objects
        detections = self.video_processor.process_video(video_path)
        
        # 2. Match products
        matched_products = []
        for frame, timestamp in self.video_processor.extract_frames(video_path):
            frame_detections = [d for d in detections if d["timestamp"] == timestamp]
            products = self.product_matcher.process_detections(frame, frame_detections)
            matched_products.extend(products)
            
        # 3. Transcribe audio
        transcript = self.transcribe_audio(video_path)
        
        # 4. Classify vibes
        vibe_results = self.vibe_classifier.process_video(
            caption=caption,
            transcript=transcript
        )
        
        # 5. Prepare output
        output = {
            "video_id": video_id,
            "vibes": vibe_results["vibes"],
            "products": matched_products,
            "transcript": transcript
        }
        
        # 6. Save output
        output_path = self.output_dir / f"{video_id}.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
            
        logger.info(f"Results saved to {output_path}")
        return output
        
def main():
    """Main entry point."""
    # Initialize pipeline
    pipeline = FlickdPipeline()
    
    # Process all videos in the videos directory
    video_dir = Path("videos")
    for video_path in video_dir.glob("*.mp4"):
        video_id = video_path.stem
        pipeline.process_video(
            video_path=str(video_path),
            video_id=video_id
        )
        
if __name__ == "__main__":
    main() 