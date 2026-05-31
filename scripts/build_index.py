import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import VECTOR_DB_DIR, PRODUCT_IMAGES_DIR, DEVICE
from src.retrieval.indexing import FaissIndexer
from src.models.embeddings import SigLIPEncoder

def process_batch(batch_name: str, encoder: SigLIPEncoder, main_indexer: FaissIndexer):
    batch_dir = os.path.join(PRODUCT_IMAGES_DIR, batch_name)
    if not os.path.exists(batch_dir):
        print(f"Skipping {batch_name}: Directory {batch_dir} not found.")
        return

    print(f"\n=== Processing {batch_name} ===")
    print("--- [1/2] Building Batch-Wise Index ---")
    batch_db_dir = os.path.join(VECTOR_DB_DIR, batch_name)
    os.makedirs(batch_db_dir, exist_ok=True)
    batch_index_path = os.path.join(batch_db_dir, "index.faiss")
    batch_ids_path = os.path.join(batch_db_dir, "index_ids.json")
    
    batch_indexer = FaissIndexer(
        index_path=batch_index_path,
        ids_path=batch_ids_path,
        encoder=encoder
    )
    new_embs, new_ids = batch_indexer.update_index(directory=batch_dir)
    
    print("--- [2/2] Updating Main Master Index ---")
    if new_embs:
        main_indexer.add_embeddings(embeddings=new_embs, ids=new_ids)
    else:
        print("No new embeddings to push to main index.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=str, help="Single batch name (e.g. batch_001)")
    parser.add_argument("--start_batch", type=int, help="Start index for batch range")
    parser.add_argument("--end_batch", type=int, help="End index for batch range (inclusive)")
    args = parser.parse_args()

    if not args.batch and (args.start_batch is None or args.end_batch is None):
        print("Error: Must provide either --batch OR both --start_batch and --end_batch")
        return

    encoder = SigLIPEncoder(device=DEVICE)
    
    main_index_path = os.path.join(VECTOR_DB_DIR, "index.faiss")
    main_ids_path = os.path.join(VECTOR_DB_DIR, "index_ids.json")
    main_indexer = FaissIndexer(
        index_path=main_index_path,
        ids_path=main_ids_path,
        encoder=encoder
    )

    if args.batch:
        process_batch(args.batch, encoder, main_indexer)
    else:
        for i in range(args.start_batch, args.end_batch + 1):
            batch_name = f"batch_{i:03d}"
            process_batch(batch_name, encoder, main_indexer)

    print("\nAll indexing operations complete!")

if __name__ == "__main__":
    main()
