import logging
from typing import Tuple, List
from PIL import Image
from transformers import pipeline

from src.data.loader import load_product_colors

class FashionColorClassifier:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu") -> None:
        logger = logging.getLogger(__name__)
        logger.info(f"Initializing FashionColorClassifier: {model_name} on {device}")
        
        device_id: int = 0 if device != "cpu" else -1
        
        self.classifier = pipeline(
            "zero-shot-image-classification", 
            model=model_name,
            device=device_id
        )
        
        _, self.labels = load_product_colors()
        self.candidate_labels: List[str] = [f"a photo of a {color} garment" for color in self.labels]

    def predict_color(self, image: Image.Image) -> Tuple[str, float]:
        try:
            results = self.classifier(image, candidate_labels=self.candidate_labels)
            best_match = results[0]
            
            extracted_color: str = best_match["label"].replace("a photo of a ", "").replace(" garment", "")
            return extracted_color, float(best_match["score"])
            
        except Exception as e:
            print(f"FashionColorClassifier error: {e}")
            return "unknown", 0.0
