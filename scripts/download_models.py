import os
from pathlib import Path
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

def download_models():
    """Download and save required ML models."""
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("Downloading YOLOv8 model...")
    yolo_model = YOLO("yolov8n.pt")
    yolo_model.save(models_dir / "yolov8n.pt")
    
    print("Downloading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Save CLIP model and processor
    clip_model.save_pretrained(models_dir / "clip")
    clip_processor.save_pretrained(models_dir / "clip")
    
    print("All models downloaded successfully!")

if __name__ == "__main__":
    download_models() 