import os
import json
import faiss
from collections import defaultdict

class IndexAnalyzer:
    def __init__(self, index_path: str, ids_path: str):
        self.index_path = index_path
        self.ids_path = ids_path
        self.index = None
        self.product_ids = []
        self._load()

    def _load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.ids_path):
            raise FileNotFoundError("Index or IDs file not found.")
        self.index = faiss.read_index(self.index_path)
        with open(self.ids_path, "r", encoding="utf-8") as f:
            self.product_ids = json.load(f)

    def analyze(self):
        total_vectors = self.index.ntotal
        dimensions = self.index.d
        metric_type = self.index.metric_type
        
        index_size_mb = os.path.getsize(self.index_path) / (1024 * 1024)
        ids_size_kb = os.path.getsize(self.ids_path) / 1024
        
        base_product_counts = defaultdict(int)
        for full_id in self.product_ids:
            base_id = full_id.split('_')[0] if '_' in full_id else full_id
            base_product_counts[base_id] += 1
            
        unique_products = len(base_product_counts)
        avg_per_product = total_vectors / max(unique_products, 1)
        
        sorted_products = sorted(base_product_counts.items(), key=lambda x: x[1], reverse=True)
        top_5 = sorted_products[:5]
        
        print("\n" + "="*50)
        print(" FAISS VECTOR DATABASE ANALYSIS ".center(50))
        print("="*50 + "\n")
        
        print("[ Storage Details ]")
        print(f"Index File: {self.index_path}")
        print(f"Index Size: {index_size_mb:.2f} MB")
        print(f"IDs Size  : {ids_size_kb:.2f} KB\n")
        
        print("[ Architecture ]")
        print(f"Metric Type: {metric_type}")
        print(f"Dimensions : {dimensions}\n")
        
        print("[ Data Distribution ]")
        print(f"Total Vectors (Images) : {total_vectors}")
        print(f"Unique Products        : {unique_products}")
        print(f"Avg Images / Product   : {avg_per_product:.2f}\n")
        
        if top_5:
            print("[ Top Products by Image Count ]")
            for pid, count in top_5:
                print(f"Product ID {pid:<15} : {count} images")
            print()
        
        print("="*50 + "\n")
