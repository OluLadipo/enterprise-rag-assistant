"""Claude API orchestration: builds a grounded prompt from retrieved chunks
and asks Claude to answer with citations, or to say it doesn't know."""
from __future__ import annotations

import os
from typing import List, Tuple

from rag.chunking import Chunk

SYSTEM_PROMPT = (
    "You are an internal knowledge assistant. Answer only from the numbered context "
    "passages provided. Cite passages inline like [1] or [2]. If the context does not "
    "contain the answer, say you could not find it in the documents. Do not guess."
)


def build_context(results: List[Tuple[Chunk, float]]) -> str:
    blocks = []
    for i, (chunk, _score) in enumerate(results, start=1):
        blocks.append(f"[{i}] (source: {chunk.source})\n{chunk.text}")
    return "\n\n".join(blocks)


class ClaudeClient:
    def __init__(self, model: str = None, max_tokens: int = 800):
        import anthropic  # imported lazily so retrieval works without the SDK

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise EnvironmentError("Set ANTHROPIC_API_KEY in your environment or .env file.")
        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens

    def answer(self, question: str, results: List[Tuple[Chunk, float]]) -> str:
        user_message = f"Context:\n{build_context(results)}\n\nQuestion: {question}"
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
