import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional

try:
    from ultralytics import YOLO
except ImportError:
    pass


class VideoProcessor:
    def __init__(
        self, fps_sample_rate: float = 1.0, similarity_threshold: float = 0.45
    ) -> None:
        self.fps_sample_rate: float = fps_sample_rate
        self.similarity_threshold: float = similarity_threshold
        self.model: YOLO = YOLO("yolov8n.pt")

    def _is_duplicate(
        self, new_crop_bgr: np.ndarray, existing_hists: List[np.ndarray]
    ) -> bool:
        if not existing_hists:
            return False
        hsv: np.ndarray = cv2.cvtColor(new_crop_bgr, cv2.COLOR_BGR2HSV)
        hist: np.ndarray = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        for saved_hist in existing_hists:
            similarity: float = cv2.compareHist(hist, saved_hist, cv2.HISTCMP_CORREL)
            if similarity > self.similarity_threshold:
                return True
        return False

    def extract_dresses(
        self, video_path: str, save_dir: Optional[str] = None
    ) -> List[Image.Image]:
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} not found.")
            return []
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        cap: cv2.VideoCapture = cv2.VideoCapture(video_path)
        fps: float = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or not fps:
            fps = 30.0
        frame_skip: int = max(1, int(fps / self.fps_sample_rate))
        crops: List[Image.Image] = []
        saved_hists: List[np.ndarray] = []
        frame_idx: int = 0
        print(f"Analyzing video at {self.fps_sample_rate} frames per second...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                results = self.model.predict(source=frame, classes=[0], verbose=False)
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        h, w = frame.shape[:2]
                        x1 = max(0, x1 - 10)
                        y1 = max(0, y1 - 10)
                        x2 = min(w, x2 + 10)
                        y2 = min(h, y2 + 10)
                        cropped_frame: np.ndarray = frame[y1:y2, x1:x2]
                        if cropped_frame.size > 0:
                            hsv: np.ndarray = cv2.cvtColor(
                                cropped_frame, cv2.COLOR_BGR2HSV
                            )
                            hist: np.ndarray = cv2.calcHist(
                                [hsv], [0, 1], None, [50, 60], [0, 180, 0, 256]
                            )
                            cv2.normalize(
                                hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
                            )
                            is_dup: bool = False
                            for saved_hist in saved_hists:
                                sim: float = cv2.compareHist(
                                    hist, saved_hist, cv2.HISTCMP_CORREL
                                )
                                if sim > self.similarity_threshold:
                                    is_dup = True
                                    break
                            if not is_dup:
                                saved_hists.append(hist)
                                rgb_crop: np.ndarray = cv2.cvtColor(
                                    cropped_frame, cv2.COLOR_BGR2RGB
                                )
                                pil_img: Image.Image = Image.fromarray(rgb_crop)
                                crops.append(pil_img)
                                if save_dir:
                                    save_path: str = os.path.join(
                                        save_dir,
                                        f"frame_{frame_idx:04d}_crop_{len(crops)}.jpg",
                                    )
                                    pil_img.save(save_path, quality=90)
            frame_idx += 1
        cap.release()
        print(
            f"Extracted {len(crops)} highly unique outfit crops from the reel (discarded duplicates)."
        )
        return crops
