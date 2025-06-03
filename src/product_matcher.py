import torch
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
import json

class ProductMatcher:
    def __init__(self, catalog_path: str, model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize the product matcher with CLIP model and catalog."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.catalog = self._load_catalog(catalog_path)
        self._precompute_catalog_embeddings()
        
    def _load_catalog(self, catalog_path: str) -> pd.DataFrame:
        """Load and preprocess the product catalog."""
        df = pd.read_csv(catalog_path)
        return df
    
    def _precompute_catalog_embeddings(self):
        """Precompute embeddings for all catalog items."""
        self.catalog_embeddings = []
        self.catalog_texts = []
        
        for _, row in self.catalog.iterrows():
            # Combine relevant text fields for richer embeddings
            text = f"{row['title']} {row['product_type']} {row['description']}"
            
            # Attempt to extract color from product_tags
            color = ""
            if 'product_tags' in row and isinstance(row['product_tags'], str):
                tags = row['product_tags'].split(', ')
                for tag in tags:
                    if tag.startswith('Colour:'):
                        color = tag.replace('Colour:', '')
                        break
            
            if color:
                text += f" {color}"
                
            # Tokenize and truncate text to fit CLIP's max length (77 tokens)
            # Use the processor for accurate tokenization and truncation
            inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77)
            
            # Move inputs to the correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Use the tokenized inputs directly
                text_features = self.model.get_text_features(**inputs).cpu().numpy()
            self.catalog_embeddings.extend(text_features)
            
        self.catalog_embeddings = np.array(self.catalog_embeddings)
        
    def compute_image_embedding(self, image: np.ndarray) -> np.ndarray:
        """Compute CLIP embedding for an image."""
        image = Image.fromarray(image)
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Move inputs to the correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs).cpu().numpy()
        return image_features[0]
    
    def find_matches(self, image: np.ndarray, threshold: float = 0.75) -> List[Dict]:
        """Find matching products for an image."""
        image_embedding = self.compute_image_embedding(image)
        
        # Compute similarities
        similarities = np.dot(self.catalog_embeddings, image_embedding) / (
            np.linalg.norm(self.catalog_embeddings, axis=1) * np.linalg.norm(image_embedding)
        )
        
        # Get top matches regardless of threshold, sorted by confidence
        matches = []
        
        # Print top similarity scores for debugging
        print(f"Top similarity scores: {similarities[np.argsort(similarities)[::-1][:5]]}")

        # Iterate through all products sorted by similarity
        for idx in np.argsort(similarities)[::-1]:
            product = self.catalog.iloc[idx]
            confidence = float(similarities[idx])

            # Determine match_type based on original thresholds (even if not filtering by 0.75)
            match_type = "exact" if confidence > 0.9 else ("similar" if confidence >= 0.75 else "low_confidence")

            # Attempt to extract color again for the output dictionary
            output_color = ""
            if 'product_tags' in product and isinstance(product['product_tags'], str):
                 tags = product['product_tags'].split(', ')
                 for tag in tags:
                     if tag.startswith('Colour:'):
                         output_color = tag.replace('Colour:', '')
                         break

            matches.append({
                "type": product["product_type"],
                "color": output_color,
                "matched_product_id": int(product["id"]),
                "match_type": match_type,
                "confidence": confidence
            })
            
        return matches[:4]  # Return top 4 matches by similarity
    
    def process_detections(self, detections: List[Dict], frames: List[np.ndarray]) -> List[Dict]:
        """Process all detections and find matching products."""
        all_matches = []
        
        for frame, detection in zip(frames, detections):
            if detection["class"] in ["person", "clothing"]:  # Filter relevant detections
                x1, y1, x2, y2 = map(int, detection["bbox"])
                region = frame[y1:y2, x1:x2]
                if region.size > 0:
                    matches = self.find_matches(region)
                    all_matches.extend(matches)
                    
        return all_matches 