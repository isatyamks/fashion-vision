import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
from PIL import Image
from ultralytics import YOLO
from src.utils.config import (
    CONF_THRESHOLD,
    DEVICE,
    FRAME_SKIP,
    VIDEO_CROPS_DIR,
    YOLO_WEIGHTS,
)
from src.preprocessing.image_processing import DuplicateFilter
CropInfo = Tuple[Image.Image, str, float]
class FashionDetector:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        conf_threshold: float = CONF_THRESHOLD,
        frame_skip: int = FRAME_SKIP,
        device: str = DEVICE,
    ):
        weights_path = weights_path or str(YOLO_WEIGHTS)
        self.conf_threshold = conf_threshold
        self.frame_skip = frame_skip
        self.device = device
        self.model = YOLO(weights_path)
        self.model.to(device)
        print(f"[FashionDetector] Loaded '{weights_path}' on {device}")
    def process_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
    ) -> Tuple[List[CropInfo], str]:
        output_dir = output_dir or self._default_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        names = self.model.names
        dedup = DuplicateFilter()
        frame_count = 0
        crop_infos: List[CropInfo] = []
        print(f"[FashionDetector] Processing '{video_path}' → '{output_dir}'")
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % self.frame_skip != 0:
                    frame_count += 1
                    continue
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                results = self.model(frame, device=self.device)[0]
                for box, conf, cls in zip(
                    results.boxes.xyxy,
                    results.boxes.conf,
                    results.boxes.cls,
                ):
                    conf_val = float(conf)
                    if conf_val < self.conf_threshold:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    cropped = pil_image.crop((x1, y1, x2, y2))
                    class_name = names[int(cls.item())]
                    if dedup.is_duplicate(cropped):
                        continue
                    dedup.add(cropped)
                    save_path = os.path.join(output_dir, f"{class_name}__{conf_val:.2f}.jpg")
                    cropped.save(save_path)
                    crop_infos.append((cropped, class_name, conf_val))
                    print(f"  Saved: {save_path}")
                frame_count += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()
        print(f"[FashionDetector] Done — {len(crop_infos)} unique crops saved.")
        return crop_infos, output_dir
    @staticmethod
    def _default_output_dir() -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return str(VIDEO_CROPS_DIR / f"crops_{timestamp}")
