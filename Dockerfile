FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

ENV HF_HOME=/tmp/hf_home
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
ENV HF_HUB_DISABLE_TELEMETRY=1

ENV YOLO_CONFIG_DIR=/tmp/ultralytics
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Upgrade pip first
RUN pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Install pinned dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/docs')" || exit 1

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]