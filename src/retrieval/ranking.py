from typing import Any, Dict, List
import numpy as np
from src.retrieval.indexing import FaissIndexer
from src.database.models import MetadataManager


class CropMatcher:
    def __init__(
        self,
        indexer: FaissIndexer,
        metadata: MetadataManager,
        threshold: float,
        top_k: int,
    ):
        self.indexer = indexer
        self.metadata = metadata
        self.threshold = threshold
        self.top_k = top_k

    def top_matches(
        self, distances: np.ndarray, indices: np.ndarray, detected_class: str
    ) -> List[Dict[str, Any]]:
        product_best_sim: Dict[str, float] = {}
        for k in range(len(indices)):
            idx = int(indices[k])
            if idx < 0:
                continue
            score = float(distances[k])
            raw_id = self.indexer.product_ids[idx]
            base_id = raw_id.split("_")[0] if "_" in raw_id else raw_id
            if base_id not in product_best_sim or score > product_best_sim[base_id]:
                product_best_sim[base_id] = score
        matches = []
        for pid, emb_sim in product_best_sim.items():
            if emb_sim < self.threshold:
                continue
            details = self.metadata.get_details(pid)
            prod_type = details.get("product_type", "").lower()
            cat_match = 0.0
            if detected_class:
                det_lower = detected_class.lower()
                if (
                    ("top" in det_lower and "top" in prod_type)
                    or ("jeans" in det_lower and "jeans" in prod_type)
                    or ("dress" in det_lower and "dress" in prod_type)
                    or ("shirt" in det_lower and "shirt" in prod_type)
                ):
                    cat_match = 1.0
            attr_match = 0.5
            final_score = (0.7 * emb_sim) + (0.2 * attr_match) + (0.1 * cat_match)
            matches.append(
                {
                    "product_id": pid,
                    "score": round(final_score, 4),
                    "vector_sim": round(emb_sim, 4),
                }
            )
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[: self.top_k]
