from rag.chunking import chunk_text


def test_chunks_respect_max_size():
    text = " ".join(f"Sentence number {i} talks about referrals." for i in range(100))
    chunks = chunk_text(text, "doc.md", max_chars=300, overlap_chars=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)


def test_chunks_overlap():
    text = " ".join(f"Fact {i} is here." for i in range(60))
    chunks = chunk_text(text, "doc.md", max_chars=200, overlap_chars=60)
    first_tail = chunks[0].text.split()[-2:]
    assert " ".join(first_tail) in chunks[1].text


def test_short_text_single_chunk():
    chunks = chunk_text("Just one short line.", "doc.md")
    assert len(chunks) == 1
    assert chunks[0].source == "doc.md"
