# Enterprise RAG Assistant

This repository contains Python scripts for building RAG applications, including text chunking, vector embedding, and LLM API orchestration using Claude.

The demo use case is a healthcare operations knowledge assistant. A customer service rep or analyst asks a question in plain language ("Does a behavioral health visit need a referral?"), the system finds the relevant policy passages, and Claude answers using only those passages, with citations. The sample documents describe a fictional health plan.

## Why this project

Frontline staff in healthcare spend a lot of time searching across policy documents, directories, and scheduling rules while a member waits on the line. Retrieval-augmented generation (RAG) lets an LLM answer from an organization's own documents instead of its general training, which makes answers more accurate and traceable. This project is a small, readable version of that architecture.

## How it works

```
documents ──> chunking ──> embeddings ──> vector index
                                              │
question ──> embed question ──> top-k search ─┘
                                    │
                       retrieved passages + question ──> Claude ──> cited answer
```

1. **Chunking** (`rag/chunking.py`): splits documents on paragraph and sentence boundaries into chunks of a set size, with overlap so context isn't cut off between chunks.
2. **Embedding** (`rag/embeddings.py`): converts chunks into vectors. The default is a local `sentence-transformers` model (`all-MiniLM-L6-v2`, no API key needed). A lightweight hashing embedder is included for offline testing.
3. **Vector store** (`rag/vector_store.py`): stores vectors in NumPy, runs cosine-similarity search, and saves the index to disk.
4. **LLM orchestration** (`rag/llm.py`): builds a grounded prompt from the retrieved passages and calls the Claude API. The system prompt tells Claude to cite passages and to say so when the answer isn't in the documents.
5. **Pipeline + CLI** (`rag/pipeline.py`, `app.py`): ties the steps together.

## Quick start

Requires Python 3.9+.

```bash
git clone https://github.com/OluLadipo/enterprise-rag-assistant.git
cd enterprise-rag-assistant
pip install -r requirements.txt

cp .env.example .env        # then paste your Anthropic API key into .env

python app.py ingest
python app.py ask "How long is a specialist referral valid?"
```

To test retrieval without an API key:

```bash
python app.py ask "What happens when a provider record is flagged?" --retrieve-only
```

To use your own documents, put `.txt` or `.md` files in a folder and run:

```bash
python app.py ingest --docs path/to/your/docs
```

## Running tests

```bash
pytest
```

## Project structure

```
enterprise-rag-assistant/
├── app.py                  # CLI: ingest and ask
├── rag/
│   ├── chunking.py         # document loading and overlapping chunking
│   ├── embeddings.py       # sentence-transformers and hashing embedders
│   ├── vector_store.py     # NumPy vector index with cosine search
│   ├── llm.py              # Claude API prompt building and calls
│   └── pipeline.py         # ingest and retrieve
├── data/sample_docs/       # fictional health plan policy documents
├── tests/                  # chunking and retrieval tests
├── requirements.txt
└── .env.example
```

## Design choices

- **Grounded answers over fluent ones.** Claude is instructed to answer only from retrieved context and to admit when it can't find something. In healthcare workflows, a confident wrong answer costs more than "I couldn't find that."
- **Citations.** Every answer points back to numbered source passages so a human can verify it.
- **Swappable components.** The embedder and vector store sit behind simple interfaces, so the NumPy store could be replaced with a hosted vector database (Pinecone, pgvector, Chroma) without changing the pipeline.

## Next steps

- Add an evaluation set of question/answer pairs to measure retrieval accuracy
- Hybrid search (keyword + vector) for exact codes like `NETWORK_INACTIVE`
- A simple web UI
- PDF ingestion

## Author

Olu Ladipo · AI UX & Product Designer · [LinkedIn](https://linkedin.com/in/oluladipoinsights)
