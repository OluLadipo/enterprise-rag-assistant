from rag.embeddings import HashingEmbedder
from rag.pipeline import ingest, retrieve
from rag.vector_store import VectorStore


def test_retrieval_finds_relevant_doc(tmp_path):
    embedder = HashingEmbedder()
    store = ingest("data/sample_docs", str(tmp_path), embedder, max_chars=400, overlap_chars=80)
    results = retrieve("How many days is a specialist referral valid?", store, embedder, top_k=2)
    assert results[0][0].source == "specialist_referral_policy.md"


def test_store_round_trip(tmp_path):
    embedder = HashingEmbedder()
    store = ingest("data/sample_docs", str(tmp_path), embedder)
    loaded = VectorStore.load(str(tmp_path))
    assert len(loaded.chunks) == len(store.chunks)
    assert loaded.vectors.shape == store.vectors.shape
