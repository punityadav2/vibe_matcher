
import os
import hashlib
import math
from typing import List
import numpy as np

# Try to import OpenAI if user has it installed & set up. This is optional.
try:
    from openai import OpenAI as OpenAIClient
    OPENAI_SDK_TYPE = "modern"
except Exception:
    try:
        import openai  # type: ignore
        OPENAI_SDK_TYPE = "legacy"
    except Exception:
        OPENAI_SDK_TYPE = None

# Deterministic hash embedding as safe fallback
def deterministic_hash_vector(text: str, dim: int = 384) -> np.ndarray:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    nums = [int(h[i:i+8], 16) for i in range(0, min(len(h), 64), 8)]
    base = np.array(nums, dtype=np.float32)
    if base.size == 0:
        base = np.ones(1, dtype=np.float32)
    vec = np.tile(base, int(math.ceil(dim / base.size)))[:dim].astype(np.float32)
    norm = np.linalg.norm(vec) + 1e-10
    return vec / norm

# A small semantic-ish mock embedding (random seeded by text) to simulate similarity
def mock_semantic_embedding(text: str, dim: int = 384) -> np.ndarray:
    # reproducible pseudo-random vector influenced by text
    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    # simple keyword nudges for common vibes
    keywords = ["boho","urban","cozy","professional","energetic","edgy","romantic","minimalist","vintage","athletic"]
    text_low = text.lower()
    for kw in keywords:
        if kw in text_low:
            vec += rng.normal(scale=0.3, size=dim)
    return vec / (np.linalg.norm(vec) + 1e-10)

class EmbeddingService:
    def __init__(self, model_name: str = "text-embedding-3-small", dim: int = 384):
        self.model_name = model_name
        self.dim = dim
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.client = None
        self.available = False
        if self.api_key and OPENAI_SDK_TYPE:
            try:
                if OPENAI_SDK_TYPE == "modern":
                    self.client = OpenAIClient(api_key=self.api_key)
                else:
                    import openai
                    openai.api_key = self.api_key
                    self.client = openai
                self.available = True
            except Exception:
                self.available = False

    def embed(self, text: str) -> np.ndarray:
        if self.available:
            try:
                if OPENAI_SDK_TYPE == "modern":
                    resp = self.client.embeddings.create(model=self.model_name, input=text)
                    vec = np.array(resp.data[0].embedding, dtype=np.float32)
                else:
                    resp = self.client.Embedding.create(model=self.model_name, input=text)
                    vec = np.array(resp["data"][0]["embedding"], dtype=np.float32)
                return vec / (np.linalg.norm(vec) + 1e-10)
            except Exception:
                # if API call fails, fallback to deterministic
                return deterministic_hash_vector(text, self.dim)
        else:
            # use mock semantic embedding for demo
            return mock_semantic_embedding(text, self.dim)

def get_embeddings_batch(texts: List[str], service: EmbeddingService) -> List[np.ndarray]:
    return [service.embed(t) for t in texts]


# Quick CLI test
if __name__ == "__main__":
    svc = EmbeddingService()
    print("Embedding dim:", svc.embed("energetic urban chic").shape)
