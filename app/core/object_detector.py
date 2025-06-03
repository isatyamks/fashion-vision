from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from ..config import YOLO_MODEL_PATH

class ObjectDetector:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL_PATH)
        self.fashion_classes = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            8: "boat",
            9: "traffic light",
            10: "fire hydrant",
            11: "stop sign",
            12: "parking meter",
            13: "bench",
            14: "bird",
            15: "cat",
            16: "dog",
            17: "horse",
            18: "sheep",
            19: "cow",
            20: "elephant",
            21: "bear",
            22: "zebra",
            23: "giraffe",
            24: "backpack",
            25: "umbrella",
            26: "handbag",
            27: "tie",
            28: "suitcase",
            29: "frisbee",
            30: "skis",
            31: "snowboard",
            32: "sports ball",
            33: "kite",
            34: "baseball bat",
            35: "baseball glove",
            36: "skateboard",
            37: "surfboard",
            38: "tennis racket",
            39: "bottle",
            40: "wine glass",
            41: "cup",
            42: "fork",
            43: "knife",
            44: "spoon",
            45: "bowl",
            46: "banana",
            47: "apple",
            48: "sandwich",
            49: "orange",
            50: "broccoli",
            51: "carrot",
            52: "hot dog",
            53: "pizza",
            54: "donut",
            55: "cake",
            56: "chair",
            57: "couch",
            58: "potted plant",
            59: "bed",
            60: "dining table",
            61: "toilet",
            62: "tv",
            63: "laptop",
            64: "mouse",
            65: "remote",
            66: "keyboard",
            67: "cell phone",
            68: "microwave",
            69: "oven",
            70: "toaster",
            71: "sink",
            72: "refrigerator",
            73: "book",
            74: "clock",
            75: "vase",
            76: "scissors",
            77: "teddy bear",
            78: "hair drier",
            79: "toothbrush"
        }
        
        # Map of relevant fashion classes
        self.fashion_items = {
            "tops": ["t-shirt", "shirt", "blouse", "sweater"],
            "bottoms": ["pants", "jeans", "shorts", "skirt"],
            "dresses": ["dress"],
            "jackets": ["jacket", "coat"],
            "accessories": ["handbag", "backpack", "purse", "watch", "glasses", "hat"]
        }

    def detect_objects(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect objects in a frame using YOLOv8.
        
        Args:
            frame: Input frame (RGB)
            confidence_threshold: Minimum confidence score for detections
            
        Returns:
            List of detections with class, confidence, and bounding box
        """
        results = self.model(frame, conf=confidence_threshold)[0]
        detections = []
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xywh[0].tolist()  # x, y, w, h format
            
            # Normalize bbox coordinates
            h, w = frame.shape[:2]
            bbox = [bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h]
            
            class_name = self.fashion_classes.get(class_id, "unknown")
            
            # Check if the detected object is a fashion item
            if self._is_fashion_item(class_name):
                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": bbox
                })
        
        return detections

    def _is_fashion_item(self, class_name: str) -> bool:
        """
        Check if a detected class is a fashion item.
        
        Args:
            class_name: Name of the detected class
            
        Returns:
            Boolean indicating if the class is a fashion item
        """
        class_name = class_name.lower()
        for category, items in self.fashion_items.items():
            if any(item in class_name for item in items):
                return True
        return False 