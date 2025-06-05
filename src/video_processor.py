import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Generator
import logging
from ultralytics import YOLO
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        frame_interval: int = 30,  # Extract 1 frame per second for 30fps video
        min_confidence: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the video processor.
        
        Args:
            model_path: Path to YOLOv8n model
            frame_interval: Number of frames to skip between extractions
            min_confidence: Minimum confidence threshold for detections
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.frame_interval = frame_interval
        self.min_confidence = min_confidence
        self.device = device
        
        # Load YOLOv8n model
        logger.info(f"Loading YOLOv8n model from {model_path}")
        self.model = YOLO(model_path)
        
    def extract_frames(
        self,
        video_path: str,
        output_dir: str = None
    ) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Extract frames from video at specified intervals.
        
        Args:
            video_path: Path to input video
            output_dir: Optional directory to save extracted frames
            
        Yields:
            Tuple of (frame, timestamp)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"FPS: {fps}, Duration: {duration:.2f}s, Total frames: {frame_count}")
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % self.frame_interval == 0:
                timestamp = frame_idx / fps
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if output_dir:
                    output_path = Path(output_dir) / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(output_path), frame)
                
                yield frame_rgb, timestamp
                
            frame_idx += 1
            
        cap.release()
        
    def detect_objects(
        self,
        frame: np.ndarray
    ) -> List[dict]:
        """
        Detect objects in a frame using YOLOv8n.
        
        Args:
            frame: RGB image as numpy array
            
        Returns:
            List of detected objects with bounding boxes and confidence scores
        """
        results = self.model(frame, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            confidence = float(box.conf[0])
            if confidence < self.min_confidence:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence,
                "class": class_name,
                "class_id": class_id
            })
            
        return detections
        
    def process_video(
        self,
        video_path: str,
        output_dir: str = None
    ) -> List[dict]:
        """
        Process video and return detected objects with timestamps.
        
        Args:
            video_path: Path to input video
            output_dir: Optional directory to save extracted frames
            
        Returns:
            List of detections with timestamps
        """
        all_detections = []
        
        for frame, timestamp in self.extract_frames(video_path, output_dir):
            detections = self.detect_objects(frame)
            
            for detection in detections:
                detection["timestamp"] = timestamp
                all_detections.append(detection)
                
        return all_detections
        
    def preprocess_frame(
        self,
        frame: np.ndarray,
        target_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """
        Preprocess frame for model input.
        
        Args:
            frame: RGB image as numpy array
            target_size: Target size for resizing
            
        Returns:
            Preprocessed frame
        """
        # Resize
        frame = cv2.resize(frame, target_size)
        
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        return frame

    def get_cropped_regions(self, frame: np.ndarray, detections: List[dict]) -> List[np.ndarray]:
        """Extract cropped regions based on detections."""
        regions = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            region = frame[y1:y2, x1:x2]
            if region.size > 0:  # Check if region is valid
                regions.append(region)
        return regions 