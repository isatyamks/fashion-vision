import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Core Configuration Settings for the Fashion Vision System.
    Uses pydantic_settings for strict type validation of environment variables.
    """
    
    # Model Settings
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
    SIGLIP_MODEL_NAME: str = os.getenv("SIGLIP_MODEL_NAME", "google/siglip-base-patch16-224")
    
    # FAISS Retrieval Settings
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "vectordb/index.faiss")
    FAISS_IDS_PATH: str = os.getenv("FAISS_IDS_PATH", "vectordb/index_ids.json")
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "10"))
    
    # Ranking Weights
    VISUAL_WEIGHT: float = float(os.getenv("VISUAL_WEIGHT", "0.8"))
    SEMANTIC_WEIGHT: float = float(os.getenv("SEMANTIC_WEIGHT", "0.2"))
    
    # Inference Optimization
    DEVICE: str = os.getenv("DEVICE", "cuda") # Should ideally detect cuda if available

settings = Settings()
