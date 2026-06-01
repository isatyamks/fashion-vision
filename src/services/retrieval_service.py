import os
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from tqdm import tqdm
import logging
from src.utils.config import (
    BATCH_SIZE,
    DEVICE,
    VECTOR_DB_DIR,
    MATCH_THRESHOLD,
    PRODUCT_IMAGES_DIR,
    SHOPIFY_PRODUCT_CSV,
    TOP_K,
)
from src.models.embeddings import SigLIPEncoder
from src.retrieval.indexing import FaissIndexer
from src.retrieval.ranking import CropMatcher
from src.database.models import MetadataManager
logger = logging.getLogger(__name__)
class FaissService:
    def __init__(
        self,
        product_images_dir: str = str(PRODUCT_IMAGES_DIR),
        product_csv: str = str(SHOPIFY_PRODUCT_CSV),
        threshold: float = MATCH_THRESHOLD,
        batch_size: int = BATCH_SIZE,
        top_k: int = TOP_K,
        device: str = DEVICE,
        encoder: Optional[SigLIPEncoder] = None,
    ):
        self.batch_size = batch_size
        self.encoder = encoder or SigLIPEncoder(device=device)
        self.metadata_mgr = MetadataManager(product_csv=product_csv)
        index_path = os.path.join(VECTOR_DB_DIR, "index.faiss")
        ids_path = os.path.join(VECTOR_DB_DIR, "index_ids.json")
        self.indexer = FaissIndexer(
            index_path=index_path,
            ids_path=ids_path,
            encoder=self.encoder,
            default_images_dir=product_images_dir,
        )
        if self.indexer.index is None:
            self.indexer.update_index()
        self.matcher = CropMatcher(
            indexer=self.indexer,
            metadata=self.metadata_mgr,
            threshold=threshold,
            top_k=top_k,
        )
    def update_index(self, directory: Optional[str] = None) -> None:
        self.indexer.update_index(directory=directory)
    def match_crops(
        self,
        crops_dir: str,
    ) -> List[Dict[str, Any]]:
        crop_files = sorted(
            f for f in os.listdir(crops_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        )
        results: List[Dict[str, Any]] = []
        for i in tqdm(range(0, len(crop_files), self.batch_size), desc="Matching crops"):
            batch_files = crop_files[i : i + self.batch_size]
            images, valid_names = self._load_batch(crops_dir, batch_files)
            if not images:
                continue
            embs = self.encoder.encode(images, normalize=True).astype("float32")
            search_k = self.matcher.top_k * 5
            distances, indices = self.indexer.search(embs, search_k)
            for j, fname in enumerate(valid_names):
                detected_class = fname.split("__")[0] if "__" in fname else ""
                top_matches = self.matcher.top_matches(distances[j], indices[j], detected_class)
                if top_matches:
                    best = top_matches[0]
                    details = self.metadata_mgr.get_details(best["product_id"])
                    row: Dict[str, Any] = {
                        "crop_filename": fname,
                        "matched_id": best["product_id"],
                        "similarity": best["score"],
                        "vector_similarity": best["vector_sim"],
                        "top_matches": top_matches,
                        **details,
                    }
                    results.append(row)
                    print(f"  {fname} -> {best['product_id']} ({best['score']:.3f})")
                else:
                    scores = [distances[j][k] for k in range(self.matcher.top_k)]
                    print(f"  {fname} - no match (top scores: {scores})")
        return results
    def _load_batch(
        self, crops_dir: str, filenames: List[str]
    ) -> Tuple[List[Image.Image], List[str]]:
        images, valid = [], []
        for fname in filenames:
            try:
                img = Image.open(os.path.join(crops_dir, fname)).convert("RGB")
                images.append(img)
                valid.append(fname)
            except Exception as e:
                logger.error(f"Cannot open {fname}: {e}")
        return images, valid
