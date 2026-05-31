"""
fashion_vision/detection/detector.py
--------------------------------------
FashionDetector: YOLO-based clothing detection from video.

Merges model.py (CPU) and cuda_model.py (GPU) into a single configurable
class. Device selection is automatic via `configs.config.DEVICE`.
"""
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
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

# Type alias
CropInfo = Tuple[Image.Image, str, float]  # (image, class_name, confidence)


class FashionDetector:
    """
    Detects and crops unique fashion items from a video using a YOLO model.

    Usage::

        detector = FashionDetector(weights_path="weights/best.pt")
        crops, output_dir = detector.process_video("data/instagram_reels/1.mp4")
        detector.visualise(crops)
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        conf_threshold: float = CONF_THRESHOLD,
        frame_skip: int = FRAME_SKIP,
        device: str = DEVICE,
    ):
        """
        Args:
            weights_path: Path to YOLO .pt weights. Defaults to config.YOLO_WEIGHTS.
            conf_threshold: Minimum confidence to accept a detection.
            frame_skip: Process every Nth frame (2 = every other frame).
            device: "cuda" or "cpu".
        """
        weights_path = weights_path or str(YOLO_WEIGHTS)
        self.conf_threshold = conf_threshold
        self.frame_skip = frame_skip
        self.device = device

        self.model = YOLO(weights_path)
        self.model.to(device)
        print(f"[FashionDetector] Loaded '{weights_path}' on {device}")

    # ── Public API ────────────────────────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
    ) -> Tuple[List[CropInfo], str]:
        """
        Run detection on *video_path*, deduplicate crops, and save to disk.

        Args:
            video_path: Path to input video file.
            output_dir: Directory for saved crops. Auto-timestamped if None.

        Returns:
            Tuple of (list of CropInfo, output directory path).

        Raises:
            FileNotFoundError: If the video cannot be opened.
        """
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

    def visualise(self, crop_infos: List[CropInfo], cols: int = 4) -> None:
        """
        Display a grid of cropped detections with their class and confidence.

        Args:
            crop_infos: List of (image, class_name, confidence) from :meth:`process_video`.
            cols: Number of columns in the grid.
        """
        if not crop_infos:
            print("[FashionDetector] No crops to display.")
            return

        rows = (len(crop_infos) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        axes = axes.flatten() if rows > 1 else [axes] * cols

        for ax, (img, cls, conf) in zip(axes, crop_infos):
            ax.imshow(img)
            ax.set_title(f"{cls}\n({conf:.2f})", fontsize=9)
            ax.axis("off")

        # Hide unused subplots
        for ax in axes[len(crop_infos):]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _default_output_dir() -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return str(VIDEO_CROPS_DIR / f"crops_{timestamp}")
