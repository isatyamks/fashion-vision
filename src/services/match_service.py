import os
import uuid
import time
from typing import Dict, Any, Tuple
from collections import defaultdict
from fastapi import UploadFile
from src.utils.config import VECTOR_DB_DIR, DEVICE
from src.preprocessing.video_processor import VideoProcessor
from src.models.embeddings import SigLIPEncoder
from src.retrieval.indexing import FaissIndexer
from src.models.fashion_color_classifier import FashionColorClassifier
from src.data.loader import (
    load_product_images,
    load_product_metadata,
    load_product_colors,
)


class MatchService:
    def __init__(self) -> None:
        self.processor: VideoProcessor = VideoProcessor(fps_sample_rate=1.0)
        self.encoder: SigLIPEncoder = SigLIPEncoder(device=DEVICE)
        self.color_analyzer: FashionColorClassifier = FashionColorClassifier(
            device=DEVICE
        )
        main_index_path: str = os.path.join(VECTOR_DB_DIR, "index.faiss")
        main_ids_path: str = os.path.join(VECTOR_DB_DIR, "index_ids.json")
        self.indexer: FaissIndexer = FaissIndexer(
            index_path=main_index_path, ids_path=main_ids_path, encoder=self.encoder
        )
        self.product_images_map: Dict[str, str] = load_product_images()
        self.product_metadata_map: Dict[str, Dict[str, str]] = load_product_metadata()
        self.product_colors_map, _ = load_product_colors()

    def is_ready(self) -> bool:
        return all([self.processor, self.encoder, self.indexer, self.color_analyzer])

    async def process_match(self, video: UploadFile) -> Dict[str, Any]:
        if not self.indexer or not self.indexer.index:
            raise ValueError("Vector DB is not initialized.")
        t0: float = time.time()
        temp_filepath: str = os.path.join(os.getcwd(), f"temp_{uuid.uuid4().hex}.mp4")
        try:
            with open(temp_filepath, "wb") as f:
                content: bytes = await video.read()
                f.write(content)
            t1: float = time.time()
            save_dir: str = os.path.join(os.getcwd(), "outputs", "video_crops")
            crops = self.processor.extract_dresses(temp_filepath, save_dir=save_dir)
            t2: float = time.time()
            if not crops:
                return {"matches": [], "extracted_attributes": {}, "latency": {}}
            mid_idx: int = len(crops) // 2
            predicted_color, confidence = self.color_analyzer.predict_color(
                crops[mid_idx]
            )
            t3: float = time.time()
            embeddings = self.encoder.encode(crops)
            t4: float = time.time()
            distances, indices = self.indexer.search(embeddings, search_k=15)
            t5: float = time.time()
            product_scores: Dict[str, float] = defaultdict(float)
            for i in range(len(crops)):
                for j in range(len(indices[i])):
                    idx: int = indices[i][j]
                    dist: float = distances[i][j]
                    if idx != -1 and idx < len(self.indexer.product_ids):
                        full_id: str = self.indexer.product_ids[idx]
                        base_id: str = (
                            full_id.split("_")[0] if "_" in full_id else full_id
                        )
                        visual_score: float = (dist + 1.0) / 2.0
                        color_score: float = 0.0
                        if (
                            base_id in self.product_colors_map
                            and predicted_color != "unknown"
                        ):
                            candidate_color: str = self.product_colors_map[base_id]
                            if candidate_color == predicted_color:
                                color_score = 1.0
                        final_score: float = (0.8 * visual_score) + (0.2 * color_score)
                        if final_score > product_scores[base_id]:
                            product_scores[base_id] = float(final_score)
            sorted_matches = sorted(
                list(product_scores.items()), key=lambda x: x[1], reverse=True
            )
            t6: float = time.time()
            results = []
            for pid, score in sorted_matches[:5]:
                meta = self.product_metadata_map.get(pid, {})
                results.append(
                    {
                        "product_id": pid,
                        "confidence": score,
                        "image_url": self.product_images_map.get(
                            pid,
                            "https://via.placeholder.com/400x600?text=Image+Not+Found",
                        ),
                        "title": meta.get("title", f"Product #{pid}"),
                        "price": meta.get("price", "N/A"),
                        "mrp": meta.get("mrp", "N/A"),
                    }
                )
            return {
                "matches": results,
                "extracted_attributes": {
                    "color": predicted_color,
                    "predicted_color": predicted_color,
                    "category": "unknown",
                    "style": "pure visual",
                    "confidence": round(confidence, 2),
                },
                "latency": {
                    "video_io": f"{(t1 - t0):.2f}s",
                    "yolo_cropping": f"{(t2 - t1):.2f}s",
                    "color_extraction": f"{(t3 - t2):.2f}s",
                    "siglip_encoding": f"{(t4 - t3):.2f}s",
                    "faiss_search": f"{(t5 - t4):.2f}s",
                    "color_ranking": f"{(t6 - t5):.2f}s",
                    "total_time": f"{(t6 - t0):.2f}s",
                },
            }
        finally:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
