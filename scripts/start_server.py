import uvicorn
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    print("Starting Fashion Vision API Server...")
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
