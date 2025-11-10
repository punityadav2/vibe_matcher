#python -m src.main
import time
import numpy as np
import pandas as pd
from src.data import get_mock_products
from src.embeddings import EmbeddingService, get_embeddings_batch
from src.engine import VibeMatcherEngine

def pretty_print_result(res):
    print("\nTop matches:")
    for i, r in enumerate(res["results"], 1):
        print(f"\n#{i}: {r['name']} ({r['product_id']})")
        print(f"    Score: {r['score']:.3f} | Quality: {r['quality']}")
        print(f"    Category: {r['category']} | Price: ${r['price']:.2f}")
        print(f"    Tags: {', '.join(r['vibe_tags'])}")

def run_demo():
    # Load data
    df = get_mock_products()

    # Embedding service (will use mock if no API key)
    svc = EmbeddingService(dim=384)
    # Generate embeddings (for all products)
    print("Generating product embeddings...")
    start = time.time()
    embeddings = get_embeddings_batch(df["description"].tolist(), svc)
    df["embedding"] = embeddings
    print(f"Embeddings created in {(time.time()-start):.3f}s")

    # Initialize engine
    engine = VibeMatcherEngine(df, similarity_threshold=0.7)

    # Demo queries
    queries = [
        "energetic urban chic",
        "soft cozy winter",
        "luxury elegant formal"
    ]

    all_stats = []
    for q in queries:
        print("\n" + "="*60)
        print(f"Query: {q}")
        q_emb = svc.embed(q)
        result = engine.search(q_emb, top_k=3)
        pretty_print_result(result)
        print(f"\nLatency: {result['latency_ms']:.2f} ms | Avg score: {result['avg_score']:.3f}")
        if not result["has_good_match"]:
            print("No strong match — showing fallback products:")
            print(engine.fallback())
        all_stats.append(result)

    # Simple metrics
    good_q = sum(1 for r in all_stats if r["has_good_match"])
    print("\n" + "="*60)
    print(f"Summary: {good_q}/{len(all_stats)} queries had good matches (threshold 0.7).")

if __name__ == "__main__":
    run_demo()
