"""End-to-end pipeline: ingest documents into a vector index, then retrieve and answer."""
from __future__ import annotations

from typing import List, Tuple

from rag.chunking import Chunk, chunk_documents, load_documents
from rag.embeddings import Embedder, get_embedder
from rag.vector_store import VectorStore


def ingest(docs_folder: str, index_folder: str, embedder: Embedder,
           max_chars: int = 800, overlap_chars: int = 150) -> VectorStore:
    docs = load_documents(docs_folder)
    chunks = chunk_documents(docs, max_chars, overlap_chars)
    vectors = embedder.embed([c.text for c in chunks])
    store = VectorStore(embedder.name)
    store.add(chunks, vectors)
    store.save(index_folder)
    print(f"Indexed {len(chunks)} chunks from {len(docs)} documents using {embedder.name}.")
    return store


def retrieve(question: str, store: VectorStore, embedder: Embedder,
             top_k: int = 4) -> List[Tuple[Chunk, float]]:
    query_vector = embedder.embed([question])[0]
    return store.search(query_vector, top_k=top_k)


def embedder_for_store(store: VectorStore) -> Embedder:
    kind = "hashing" if store.embedder_name.startswith("hashing") else "sentence-transformers"
    return get_embedder(kind)
