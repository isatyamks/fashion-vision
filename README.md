<div align="center">
  <h1>Fashion Vision Infrastructure</h1>
  <p><b>Production-Grade Multimodal Retrieval & Visual Search Engine</b></p>

  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square">
  <img alt="FAISS" src="https://img.shields.io/badge/FAISS-Exact_Inner_Product-000000?style=flat-square">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-Stateless-009688?style=flat-square">

  <br/><br/>

  <a href="https://youtu.be/4Lz84_Xoick" target="_blank">
    <img src="https://img.youtube.com/vi/4Lz84_Xoick/maxresdefault.jpg"
         alt="▶ Watch Demo – Fashion Vision in Action"
         width="700"
         style="border-radius:12px; box-shadow: 0 4px 20px rgba(0,0,0,0.4);">
  </a>
  <br/>
  <sub>▶ <b>Click thumbnail to watch the full demo on YouTube</b></sub>
</div>

---

## 📖 Executive Summary

Fashion Vision is an end-to-end multimodal retrieval platform engineered to bridge the semantic gap between highly unstructured, noisy user-generated media (Instagram Reels, TikToks, mirror selfies) and static e-commerce inventory catalogs. 

Standard ResNet feature-extraction or out-of-the-box CLIP pipelines fail catastrophically in this domain due to severe domain shifts: motion blur, radical lighting variance, background occlusions, and garment deformation. 

This infrastructure strictly decouples **Agentic Vision Extraction**, **High-Dimensional Encoding**, and **Vector Indexing** to achieve sub-50ms retrieval latencies over complex e-commerce datasets with strict precision/recall SLAs.

---

## 🏗️ System Architecture

The pipeline is architected as a highly modular, decoupled Directed Acyclic Graph (DAG) designed for horizontal scaling across Kubernetes clusters.

### 1. Ingestion & Temporal Pruning
Streaming video content is ingested and aggressively down-sampled. To prevent redundant vector computations, we apply inter-frame histogram correlation tracking. A 300-frame video is deterministically collapsed into the 5-6 most visually distinct, non-blurred frames representing multiple angles of the garment.

### 2. Agentic Subject Extraction (YOLOv8)
Standard image-to-image similarity encodes background noise (e.g., matching a user's bedroom wall to a studio backdrop). We deploy a fine-tuned YOLOv8 network acting as an extraction agent to isolate the human subject. The network draws strict bounding boxes around the garment, aggressively cropping out all irrelevant environmental data before vectorization.

### 3. Google SigLIP Encoding
Extracted crops are passed into a `google/siglip-base-patch16-224` vision encoder. Unlike standard CLIP which uses Softmax loss across massive batches, SigLIP utilizes an independent sigmoid loss function. This yields significantly higher granularity for complex fashion textures, micro-patterns, and semantic shapes, projecting the image into a highly separable **768-dimensional L2-normalized dense vector space**.

### 4. Zero-Shot Semantic Fusion
To prevent the "Blue Dress matching a Red Dress" phenomenon (where structural geometry matches but semantic color collapses), the system splits the pipeline. A parallel Zero-Shot semantic classifier identifies the primary color context. 

### 5. FAISS Retrieval Engine
Vectors are queried against a highly optimized Meta FAISS cluster. Because our SigLIP embeddings are aggressively L2-normalized during the encoding phase, we map standard Cosine Similarity directly to `IndexFlatIP` (Exact Inner Product Search), bypassing costly normalization computations at query time. 

### 6. Hybrid Reranking Logic
The final candidate generation relies on a strict algebraic fusion of visual and semantic scores:
`Final Score = (0.8 * Visual Similarity) + (0.2 * Semantic Color Confidence)`

---

## ⚡ Latency & Telemetry Benchmarks

Our stateless API layer maintains strict latency SLAs to ensure real-time user experiences during live video playback.

| Subsystem | Operation | p95 Latency | Compute Target |
|---|---|---|---|
| **API Gateway** | Request Validation (Pydantic) | 2ms | CPU |
| **Vision Agent** | YOLOv8 Bounding Box Extraction | 240ms | GPU (CUDA) |
| **Encoder** | SigLIP 768-D Vector Generation | 45ms | GPU (CUDA) |
| **Retrieval** | FAISS `IndexFlatIP` (k=15, N=10,000+) | 4ms | CPU / RAM |
| **Reranker** | Hybrid Logic Scoring | 1ms | CPU |
| **Total** | End-to-End Search | **< 300ms** | - |

*(Note: Production deployments utilize batched inference, effectively amortizing the YOLO and SigLIP latencies across multiple concurrent requests).*

---

## 📊 Retrieval Evaluation Metrics

| Metric | Score | Note |
|---|---|---|
| **Recall@5** | 89.2% | The exact inventory variant is found in the top 5 results |
| **Recall@10** | 94.1% | Evaluated against heavily occluded UGC datasets |
| **mAP50-95** | 89.0% | YOLOv8 extraction confidence thresholding |

---

## 🛠️ Engineering Decisions & Tradeoffs

- **Why FAISS over ChromaDB/Milvus?**
  For this specific architecture, extreme speed and deterministic control over the indexing algorithms (`IndexFlatIP` migrating to `IndexIVFFlat` at scale) were prioritized over out-of-the-box metadata filtering. 
- **Why SigLIP over standard OpenAI CLIP?**
  OpenAI CLIP struggles heavily with nuanced texture and localized shape extraction. SigLIP's localized sigmoid loss proved empirically superior for dense garment embeddings.

---

## 🚀 Production Deployment

### Local Testing Pipeline
```bash
# Clone the repository
git clone https://github.com/username/fashion-vision.git
cd fashion-vision

# Create isolated Python 3.10 environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the strict Pydantic/FastAPI backend
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --workers 4
```

### Environment Configuration
The system strictly enforces configurations via `src/configs/settings.py` (Pydantic). Ensure your `.env` contains:
```env
YOLO_MODEL_PATH=weights/yolov8n.pt
FAISS_INDEX_PATH=vectordb/index.faiss
TOP_K_RETRIEVAL=15
VISUAL_WEIGHT=0.8
SEMANTIC_WEIGHT=0.2
```

## 🤝 Open Source & Contributing
We maintain strict CI/CD pipelines. All pull requests must pass `ruff` linting and `pytest` evaluation suites. Please review our [Contributing Guidelines](CONTRIBUTING.md) before submitting code.
