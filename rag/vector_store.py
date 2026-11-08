"""A small in-memory vector store with cosine-similarity search and disk persistence."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

from rag.chunking import Chunk


class VectorStore:
    def __init__(self, embedder_name: str):
        self.embedder_name = embedder_name
        self.chunks: List[Chunk] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)

    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        self.chunks.extend(chunks)
        self.vectors = vectors if self.vectors.size == 0 else np.vstack([self.vectors, vectors])

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        if not self.chunks:
            return []
        # Vectors are L2-normalized, so the dot product equals cosine similarity.
        scores = self.vectors @ query_vector.reshape(-1)
        best = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in best]

    def save(self, folder: str) -> None:
        out = Path(folder)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "vectors.npy", self.vectors)
        meta = {"embedder": self.embedder_name, "chunks": [asdict(c) for c in self.chunks]}
        (out / "chunks.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, folder: str) -> "VectorStore":
        src = Path(folder)
        meta = json.loads((src / "chunks.json").read_text(encoding="utf-8"))
        store = cls(meta["embedder"])
        store.chunks = [Chunk(**c) for c in meta["chunks"]]
        store.vectors = np.load(src / "vectors.npy")
        return store
