# Fashion Vision

<p align="center">
  <b>Turn Fashion Content Into Structured Commerce Intelligence</b>
</p>

<p align="center">
  <img alt="GitHub Workflow Status" src="https://img.shields.io/github/actions/workflow/status/username/fashion-vision/python-app.yml?style=flat-square">
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green?style=flat-square">
</p>

---

## 📖 Project Overview

Fashion Vision transforms social media reels, videos, and unstructured images into searchable inventory using multimodal retrieval systems and production-scale AI infrastructure. 

Matching user-generated video content to static e-commerce catalogs is a mathematically ill-posed problem. Videos contain motion blur, compression artifacts, diverse lighting, and severe occlusions, whereas studio catalogs are perfectly lit and posed. **Fashion Vision** solves this by strictly decoupling Agentic Computer Vision (YOLOv8) from Multimodal Encoding (Google SigLIP) and Exact Inner-Product Vector Search (FAISS).

---

## 🚀 System Architecture

The pipeline is engineered as a highly granular DAG (Directed Acyclic Graph) of specialized neural networks and deterministic ranking algorithms.

1. **Ingestion Layer**: Sanitizes raw video frames and prunes redundant/motion-blurred content.
2. **Garment Detection (YOLOv8)**: An agentic vision model isolates subjects, drawing strict bounding boxes around garments to exclude background noise.
3. **Embedding Generation (SigLIP)**: Crops are projected into a 768-dimensional L2-normalized dense vector space using Google SigLIP's independent sigmoid loss architecture.
4. **FAISS Retrieval**: Sub-5ms Exact Inner-Product (IndexFlatIP) lookup across 10,000+ variants.
5. **Hybrid Reranking**: The final result is scored using a strict mathematical formula: `(0.8 * Visual Sim) + (0.2 * Semantic Color Context)`.

## 🛠 Tech Stack

- **Inference & Deep Learning**: PyTorch, Ultralytics YOLOv8, Google SigLIP
- **Retrieval Infrastructure**: Meta FAISS (IndexFlatIP / IndexIVFFlat)
- **API & Core Backend**: FastAPI, Pydantic, Python `logging`
- **Frontend Dashboard**: Next.js (React), Tailwind CSS, Framer Motion, Recharts

---

## ⚡ Installation & Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/username/fashion-vision.git
cd fashion-vision

# 2. Set up the environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Start the FastAPI Inference Server
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

## 📊 Benchmarks

| Metric | Score | Note |
|---|---|---|
| **Recall@5** | 89.2% | Exact inner-product exact match |
| **Recall@10** | 94.1% | |
| **FAISS Query Latency** | < 5ms | Over 10k items |
| **YOLO Extraction Latency** | 240ms | Frame-by-frame processing |

*(For detailed PR curves and latency breakdowns, view the interactive dashboard).*

## 🤝 Contributing
Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to set up the repository for development, run the test suite, and submit Pull Requests.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
