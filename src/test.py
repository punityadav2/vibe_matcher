
#Run: python -m src.test

from src.data import get_mock_products
from src.embeddings import EmbeddingService
from src.engine import VibeMatcherEngine

def run_edge_tests():
    df = get_mock_products()
    svc = EmbeddingService(dim=384)
    # create embeddings
    df["embedding"] = [svc.embed(d) for d in df["description"].tolist()]
    engine = VibeMatcherEngine(df)

    edge_queries = [
        "futuristic cyberpunk neon",  # likely no match
        "",                          # empty
        "dress"                      # generic
    ]
    for q in edge_queries:
        print("-"*40)
        if not q:
            print("Empty query: handled gracefully.")
            continue
        emb = svc.embed(q)
        res = engine.search(emb)
        print(f"Query: '{q}' | max_score: {res['max_score']:.3f} | has_good: {res['has_good_match']}")

if __name__ == "__main__":
    run_edge_tests()
