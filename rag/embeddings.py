"""Vector embedding backends.

SentenceTransformerEmbedder runs a real embedding model locally (no API key).
HashingEmbedder is a dependency-free fallback used for tests and quick demos.
"""
from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np


class Embedder:
    name = "base"

    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # imported lazily

        self.name = f"sentence-transformers/{model_name}"
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


class HashingEmbedder(Embedder):
    """Bag-of-words feature hashing. Weaker than a neural model, but fast and offline."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def embed(self, texts: List[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = int(hashlib.md5(token.encode()).hexdigest(), 16)
                matrix[row, digest % self.dim] += 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


def get_embedder(kind: str = "sentence-transformers") -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    try:
        return SentenceTransformerEmbedder()
    except ImportError:
        print("sentence-transformers not installed; falling back to HashingEmbedder.")
        return HashingEmbedder()
