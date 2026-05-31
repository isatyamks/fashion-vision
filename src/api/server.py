import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.routes import match_router

app: FastAPI = FastAPI(title="Fashion Vision AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(match_router.router, prefix="/api", tags=["Match"])
