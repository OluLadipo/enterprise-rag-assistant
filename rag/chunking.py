"""Text loading and chunking.

Splits documents into overlapping chunks along paragraph and sentence
boundaries so each chunk stays readable and fits inside an embedding model's
input window.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int
    metadata: Dict[str, str] = field(default_factory=dict)


def load_documents(folder: str) -> Dict[str, str]:
    """Read every .txt and .md file in a folder into {filename: text}."""
    docs = {}
    for path in sorted(Path(folder).glob("*")):
        if path.suffix.lower() in {".txt", ".md"}:
            docs[path.name] = path.read_text(encoding="utf-8")
    if not docs:
        raise FileNotFoundError(f"No .txt or .md files found in {folder}")
    return docs


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, source: str, max_chars: int = 800, overlap_chars: int = 150) -> List[Chunk]:
    """Pack sentences into chunks up to max_chars, carrying a tail of the
    previous chunk forward as overlap so context isn't lost at the edges."""
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")

    sentences = []
    for paragraph in re.split(r"\n\s*\n", text):
        sentences.extend(_split_sentences(paragraph))

    chunks: List[Chunk] = []
    current = ""
    for sentence in sentences:
        # A single very long sentence gets hard-split.
        while len(sentence) > max_chars:
            piece, sentence = sentence[:max_chars], sentence[max_chars:]
            if current:
                chunks.append(Chunk(current.strip(), source, len(chunks)))
                current = ""
            chunks.append(Chunk(piece.strip(), source, len(chunks)))

        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            chunks.append(Chunk(current.strip(), source, len(chunks)))
            tail = current[-overlap_chars:]
            # Start the overlap at a word boundary.
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            current = f"{tail} {sentence}".strip()

    if current.strip():
        chunks.append(Chunk(current.strip(), source, len(chunks)))
    return chunks


def chunk_documents(docs: Dict[str, str], max_chars: int = 800, overlap_chars: int = 150) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for name, text in docs.items():
        all_chunks.extend(chunk_text(text, name, max_chars, overlap_chars))
    return all_chunks
