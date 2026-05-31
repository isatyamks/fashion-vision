import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import VECTOR_DB_DIR
from src.retrieval.index_analyzer import IndexAnalyzer

def main():
    index_path = os.path.join(VECTOR_DB_DIR, "index.faiss")
    ids_path = os.path.join(VECTOR_DB_DIR, "index_ids.json")
    
    try:
        analyzer = IndexAnalyzer(index_path, ids_path)
        analyzer.analyze()
    except FileNotFoundError:
        print(f"\nError: Could not find index files at {VECTOR_DB_DIR}")
        print("Make sure you have run the indexer first!\n")

if __name__ == "__main__":
    main()
