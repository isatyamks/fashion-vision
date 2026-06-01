from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any
from src.services.match_service import MatchService

router: APIRouter = APIRouter()
match_service: MatchService = None


def init_match_service() -> None:
    global match_service
    if match_service is None:
        match_service = MatchService()


@router.on_event("startup")
def startup_event() -> None:
    init_match_service()


@router.get("/health")
async def health_check() -> Dict[str, str]:
    if match_service and match_service.is_ready():
        return {"status": "ready", "message": "All AI Vision models loaded."}
    return {"status": "loading", "message": "Waking up AI models..."}


@router.post("/match")
async def match_reel(video: UploadFile = File(...)) -> Dict[str, Any]:
    if not match_service or not match_service.is_ready():
        raise HTTPException(status_code=503, detail="Models are still loading.")
    try:
        return await match_service.process_match(video)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
