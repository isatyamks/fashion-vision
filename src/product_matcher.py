import torch
import numpy as np
import pandas as pd
from pathlib import Path
import faiss
from PIL import Image
import requests
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import logging
from typing import List, Dict, Tuple, Optional
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductMatcher:
    def __init__(
        self,
        catalog_path: str = "catalog.csv",
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        similarity_threshold: float = 0.75
    ):
        self.device = device
        self.similarity_threshold = similarity_threshold
        logger.info(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.catalog = self._load_catalog(catalog_path)
        self.catalog_embeddings = self._compute_catalog_embeddings()
        self.index = self._build_faiss_index()

    def save_response():

        
        pass
    
    


    def _load_catalog(self, catalog_path: str) -> pd.DataFrame:
        catalog = pd.read_csv(catalog_path)
        required_columns = ['id', 'title', 'description', 'product_type']
        if not all(col in catalog.columns for col in required_columns):
            raise ValueError(f"Catalog must contain columns: {required_columns}")
        return catalog
        
    def _download_image(self, url: str) -> Optional[Image.Image]:
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            logger.warning(f"Failed to download image from {url}: {e}")
            return None
            
    def _compute_catalog_embeddings(self) -> np.ndarray:
        logger.info("Computing catalog embeddings...")
        embeddings = []
        batch_size = 32
        max_length = 77
        
        for i in range(0, len(self.catalog), batch_size):
            batch = self.catalog.iloc[i:i+batch_size]
            texts = batch['description'].tolist()
            texts = [text[:max_length] if len(text) > max_length else text for text in texts]
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                embeddings.append(text_features.cpu().numpy())
                
        return np.vstack(embeddings)
        
    def _build_faiss_index(self) -> faiss.Index:
        dimension = self.catalog_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(self.catalog_embeddings)
        return index
        
    def match_product(
        self,
        image: np.ndarray,
        detection_type: str
    ) -> Optional[Dict]:
        image = Image.fromarray(image)
        inputs = self.processor(
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            query_embedding = image_features.cpu().numpy()
        k = 5
        similarities, indices = self.index.search(query_embedding, k)
        best_similarity = similarities[0][0]
        best_idx = indices[0][0]
        if best_similarity < self.similarity_threshold:
            return None
        if best_similarity > 0.9:
            match_type = "exact"
        else:
            match_type = "similar"
        matched_product = self.catalog.iloc[best_idx]
        if matched_product["product_type"].lower() != detection_type.lower():
            return None
        return {
            "type": detection_type,
            "match_type": match_type,
            "matched_product_id": matched_product["id"],
            "confidence": float(best_similarity),
            "product_name": matched_product["title"],
            "product_type": matched_product["product_type"],
            "description": matched_product["description"]
        }
        
    def process_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict]
    ) -> List[Dict]:
        matched_products = []
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            match = self.match_product(roi, detection["class"])
            if match:
                match["timestamp"] = detection["timestamp"] 


                # match["timestamp"] = detection["timestamp"]
                matched_products.append(match)
        return matched_products
