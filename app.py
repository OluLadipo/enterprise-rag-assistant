"""Command-line entry point.

    python app.py ingest
    python app.py ask "How long is a specialist referral valid?"
    python app.py ask "..." --retrieve-only     # no API key needed
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

from rag.embeddings import get_embedder
from rag.pipeline import embedder_for_store, ingest, retrieve
from rag.vector_store import VectorStore

DEFAULT_DOCS = "data/sample_docs"
DEFAULT_INDEX = "index"


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Enterprise RAG Assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Chunk and embed documents into a local index")
    p_ingest.add_argument("--docs", default=DEFAULT_DOCS)
    p_ingest.add_argument("--index", default=DEFAULT_INDEX)
    p_ingest.add_argument("--embedder", choices=["sentence-transformers", "hashing"],
                          default="sentence-transformers")
    p_ingest.add_argument("--chunk-size", type=int, default=800)
    p_ingest.add_argument("--overlap", type=int, default=150)

    p_ask = sub.add_parser("ask", help="Ask a question against the index")
    p_ask.add_argument("question")
    p_ask.add_argument("--index", default=DEFAULT_INDEX)
    p_ask.add_argument("--top-k", type=int, default=4)
    p_ask.add_argument("--retrieve-only", action="store_true",
                       help="Show retrieved chunks without calling Claude")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(args.docs, args.index, get_embedder(args.embedder), args.chunk_size, args.overlap)
        return

    store = VectorStore.load(args.index)
    embedder = embedder_for_store(store)
    results = retrieve(args.question, store, embedder, args.top_k)

    print("\nRetrieved context:")
    for i, (chunk, score) in enumerate(results, start=1):
        preview = chunk.text[:120].replace("\n", " ")
        print(f"  [{i}] {chunk.source} (score {score:.3f}): {preview}...")

    if args.retrieve_only:
        return

    from rag.llm import ClaudeClient

    print("\nAnswer:\n")
    print(ClaudeClient().answer(args.question, results))


if __name__ == "__main__":
    main()
