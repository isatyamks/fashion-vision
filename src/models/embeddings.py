import logging
import numpy as np
import torch
from typing import List, Union
from PIL import Image
from transformers import AutoModel, AutoProcessor
from src.utils.config import BATCH_SIZE, DEVICE
class SigLIPEncoder:
    def __init__(self, model_name: str = "google/siglip-base-patch16-224", device: str = DEVICE) -> None:
        logger = logging.getLogger(__name__)
        logger.info(f"Loading '{model_name}' on {device}...")
        self.model_name: str = model_name
        self.device: str = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
    def encode(self, images: Union[Image.Image, List[Image.Image]], normalize: bool = True, batch_size: int = BATCH_SIZE) -> np.ndarray:
        single: bool = isinstance(images, Image.Image)
        imgs: List[Image.Image] = [images] if single else images
        all_embs: List[np.ndarray] = []
        for i in range(0, len(imgs), batch_size):
            batch: List[Image.Image] = imgs[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                if normalize:
                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            all_embs.append(image_features.cpu().numpy())
        embeddings: np.ndarray = np.vstack(all_embs)
        return embeddings[0] if single else embeddings
    def encode_to_list(self, images: Union[Image.Image, List[Image.Image]], normalize: bool = True) -> List[List[float]]:
        arr: np.ndarray = self.encode(images, normalize=normalize)
        if arr.ndim == 1:
            return [arr.tolist()]
        return [row.tolist() for row in arr]
