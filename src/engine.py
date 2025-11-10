
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any
import pandas as pd

class VibeMatcherEngine:
    def __init__(self, products_df: pd.DataFrame, similarity_threshold: float = 0.7):
        self.products_df = products_df.copy().reset_index(drop=True)
        self.similarity_threshold = similarity_threshold
        # product_embeddings must be numeric arrays
        # If embeddings column contains python lists, convert to np arrays
        self.product_embeddings = np.vstack(
            [np.array(e, dtype=np.float32) for e in self.products_df["embedding"].values]
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> Dict[str, Any]:
        start = time.time()
        q = query_embedding.reshape(1, -1).astype(np.float32)
        sims = cosine_similarity(q, self.product_embeddings)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        top_scores = sims[top_idx]

        results = []
        for idx, score in zip(top_idx, top_scores):
            row = self.products_df.iloc[idx]
            results.append({
                "product_id": row["product_id"],
                "name": row["name"],
                "description": row["description"],
                "price": float(row["price"]),
                "category": row["category"],
                "vibe_tags": row["vibe_tags"],
                "score": float(score),
                "quality": "Excellent" if score >= 0.8 else ("Good" if score >= self.similarity_threshold else "Fair")
            })

        latency_ms = (time.time() - start) * 1000.0
        return {
            "query_embedding_shape": q.shape,
            "results": results,
            "latency_ms": latency_ms,
            "avg_score": float(np.mean(top_scores)) if len(top_scores) > 0 else 0.0,
            "max_score": float(np.max(top_scores)) if len(top_scores) > 0 else 0.0,
            "has_good_match": any(s >= self.similarity_threshold for s in top_scores)
        }

    def fallback(self):

        top = self.products_df.nlargest(3, "price")[["product_id","name","category","price"]]
        return top.to_dict("records")
