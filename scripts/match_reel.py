import argparse
import sys
import os
import time
from pathlib import Path

# Fix for OpenMP conflict between FAISS and YOLO/PyTorch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import VECTOR_DB_DIR, DEVICE
from src.preprocessing.video_processor import VideoProcessor
from src.models.embeddings import SigLIPEncoder
from src.retrieval.indexing import FaissIndexer

def main():
    parser = argparse.ArgumentParser(description="High-Speed Reel Matcher")
    parser.add_argument("--video", type=str, required=True, help="Path to the reel video file")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top matches to return per crop")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found.")
        return

    print("\n" + "="*50)
    print(" HIGH-SPEED REEL MATCHER ".center(50))
    print("="*50 + "\n")

    t0 = time.time()

    try:
        import ultralytics
    except ImportError:
        print("Error: The 'ultralytics' library is required for ultra-fast dress tracking.")
        print("Please install it by running: python -m pip install ultralytics")
        return

    # 1. Process Video -> Rips video at 1 FPS and detects person/dresses instantly
    processor = VideoProcessor(fps_sample_rate=1.0)
    
    reel_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(Path(__file__).parent.parent, "data", "crops", reel_name)
    
    print(f"Cropped dresses will be automatically saved to {save_dir}")
    crops = processor.extract_dresses(video_path, save_dir=save_dir)

    if not crops:
        print("No dresses found in the video.")
        return

    # 2. Batched Encoding -> Embed all crops simultaneously on GPU
    print("\nEncoding dresses in a single parallel batch...")
    encoder = SigLIPEncoder(device=DEVICE)
    # The batch size parameter in SigLIPEncoder natively handles chunking
    embeddings = encoder.encode(crops)

    # 3. FAISS Matrix Search -> Instantly searches all vectors
    print("\nSearching master vector database...")
    main_index_path = os.path.join(VECTOR_DB_DIR, "index.faiss")
    main_ids_path = os.path.join(VECTOR_DB_DIR, "index_ids.json")
    
    if not os.path.exists(main_index_path):
        print("Error: Master database not found. Please run the indexer first.")
        return

    indexer = FaissIndexer(
        index_path=main_index_path,
        ids_path=main_ids_path,
        encoder=encoder
    )

    # Search the entire batch at once
    distances, indices = indexer.search(embeddings, search_k=args.top_k)

    # 4. Aggregate and Display Results
    # We will tally up the most frequently matched product IDs across all frames
    from collections import defaultdict
    product_scores = defaultdict(float)

    for i in range(len(crops)):
        for j in range(len(indices[i])):
            idx = indices[i][j]
            dist = distances[i][j]
            if idx != -1 and idx < len(indexer.product_ids):
                full_id = indexer.product_ids[idx]
                base_id = full_id.split('_')[0] if '_' in full_id else full_id
                # Add score (closer distance is usually better, but for IP, higher is better)
                # Since FaissIndexer uses IndexFlatIP (Inner Product), higher distance = higher similarity
                product_scores[base_id] += float(dist)

    print("\n" + "="*50)
    print(" TOP MATCHES FOUND IN REEL ".center(50))
    print("="*50 + "\n")

    sorted_matches = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)
    
    for i, (pid, score) in enumerate(sorted_matches[:5]):
        print(f"{i+1}. Product ID: {pid:<15} (Confidence Score: {score:.2f})")
        
    t1 = time.time()
    print(f"\nTotal Pipeline Execution Time: {t1 - t0:.2f} seconds")

if __name__ == "__main__":
    main()
