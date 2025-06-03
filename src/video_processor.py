import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from ultralytics import YOLO
from PIL import Image

class VideoProcessor:
    def __init__(self, model_path: str = "models/yolov8n.pt"):
        """Initialize the video processor with YOLOv8 model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        
    def extract_frames(self, video_path: str, num_frames: int = 5) -> List[np.ndarray]:
        """Extract evenly spaced frames from the video."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to extract
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                
        cap.release()
        return frames
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects in a frame using YOLOv8."""
        results = self.model(frame, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence,
                'class': class_name
            })
            
        return detections
    
    def process_video(self, video_path: str) -> List[Dict]:
        """Process a video and return detections from key frames."""
        frames = self.extract_frames(video_path)
        all_detections = []
        
        for frame in frames:
            detections = self.detect_objects(frame)
            all_detections.extend(detections)
            
        return all_detections
    
    def get_cropped_regions(self, frame: np.ndarray, detections: List[Dict]) -> List[np.ndarray]:
        """Extract cropped regions based on detections."""
        regions = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            region = frame[y1:y2, x1:x2]
            if region.size > 0:  # Check if region is valid
                regions.append(region)
        return regions 