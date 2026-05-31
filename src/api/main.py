"""
api/main.py
-----------
FastAPI entry point for the Visual Retrieval pipeline.
Provides endpoints for retrieving matching products given an image crop.
"""
import logging
import os
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from src.services.retrieval_service import FaissService

# Setup minimal logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Fashion Vision Retrieval API", version="1.0.0")

# Lazy-loaded FAISS matcher
matcher = None

class MatchResponse(BaseModel):
    product_id: str
    similarity: float
    vector_similarity: float
    details: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    global matcher
    logger.info("Initializing FAISS Matcher (this may take a moment to build the index)...")
    try:
        matcher = FaissService()
        logger.info("Initialized FAISS matching service.")
    except Exception as e:
        logger.error(f"Failed to initialize FaissService: {e}")


@app.post("/match", response_model=List[MatchResponse])
async def match_crop(file: UploadFile = File(...)):
    """
    Accepts an image crop and returns the top matching products from the FAISS index.
    """
    if matcher is None:
        raise HTTPException(status_code=503, detail="Matcher service is not available.")
        
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # We mock 'detected_class' as empty for raw image uploads,
    # or it could be passed as a form parameter in the future.
    try:
        embs = matcher.encoder.encode([image], normalize=True).astype("float32")
        search_k = matcher.top_k * 5
        distances, indices = matcher._index.search(embs, min(search_k, len(matcher._product_ids)))
        
        top_matches = matcher._top_matches(distances[0], indices[0], detected_class="")
        
        response = []
        for best in top_matches:
            details = matcher.product_details.get(best["product_id"], {})
            response.append(MatchResponse(
                product_id=best["product_id"],
                similarity=best["score"],
                vector_similarity=best["vector_sim"],
                details=details
            ))
            
        return response
    except Exception as e:
        logger.error(f"Error during matching: {e}")
        raise HTTPException(status_code=500, detail="Internal matching error.")


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "fashion-vision-api"}
