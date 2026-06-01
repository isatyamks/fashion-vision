import json
import os
import faiss
import numpy as np
from PIL import Image
from typing import Tuple, List, Set, Any
class FaissIndexer:
    def __init__(self, index_path: str, ids_path: str, encoder: Any) -> None:
        self.index_path: str = index_path
        self.ids_path: str = ids_path
        self.encoder: Any = encoder
        self.index: Any = None
        self.product_ids: List[str] = []
        self._load_or_init()
    def _load_or_init(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.ids_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.ids_path, "r", encoding="utf-8") as f:
                self.product_ids = json.load(f)
        else:
            self.product_ids = []
            self.index = None
    def update_index(self, directory: str) -> Tuple[List[np.ndarray], List[str]]:
        image_files: List[str] = []
        for root, _, files in os.walk(directory):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_files.append(os.path.join(root, f))
        image_files.sort()
        print(f"Found {len(image_files)} images in {os.path.basename(directory)}")
        existing_ids: Set[str] = set(self.product_ids)
        new_embeddings: List[np.ndarray] = []
        new_ids: List[str] = []
        skipped: int = 0
        print("Checking for already indexed images...")
        for path in image_files:
            fname: str = os.path.basename(path)
            img_id: str = os.path.splitext(fname)[0]
            if img_id in existing_ids:
                skipped += 1
                continue
            try:
                img: Image.Image = Image.open(path).convert("RGB")
                emb: np.ndarray = self.encoder.encode(img, normalize=True)
                new_embeddings.append(emb)
                new_ids.append(img_id)
            except Exception:
                pass
        if skipped > 0:
            print(f"Skipped {skipped} images that were already indexed.")
        if not new_embeddings:
            print("No new images to index.")
            return new_embeddings, new_ids
        self.add_embeddings(new_embeddings, new_ids)
        return new_embeddings, new_ids
    def add_embeddings(self, embeddings: List[np.ndarray], ids: List[str]) -> None:
        if not embeddings:
            return
        existing_ids: Set[str] = set(self.product_ids)
        filtered_embs: List[np.ndarray] = []
        filtered_ids: List[str] = []
        skipped: int = 0
        for emb, img_id in zip(embeddings, ids):
            if img_id in existing_ids:
                skipped += 1
                continue
            filtered_embs.append(emb)
            filtered_ids.append(img_id)
        if skipped > 0:
            print(f"Skipped {skipped} embeddings already present in this index.")
        if not filtered_embs:
            return
        print(f"Adding {len(filtered_embs)} embeddings to FAISS index at {self.index_path}...")
        arr: np.ndarray = np.array(filtered_embs, dtype="float32")
        dim: int = arr.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)
        self.index.add(arr)
        self.product_ids.extend(ids)
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.ids_path, "w", encoding="utf-8") as f:
            json.dump(self.product_ids, f)
        print(f"Successfully updated index! Total images in DB: {len(self.product_ids)}")
    def search(self, embeddings: np.ndarray, search_k: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.index.search(embeddings, min(search_k, len(self.product_ids)))
