import hashlib

import pytest

from neurograph.models.document import Document
from neurograph.retrieval.chunking import chunk_document


def _make_document(content: str, doc_id: str = "doc-id") -> Document:
    return Document(
        id=doc_id,
        source="test",
        uri="file:///notes/doc.md",
        content=content,
        metadata={"title": "Original"},
    )


def test_empty_document_produces_no_chunks() -> None:
    doc = _make_document("")

    assert chunk_document(doc) == []


def test_document_smaller_than_target_produces_single_chunk() -> None:
    content = "Hello world, this is a short document."
    doc = _make_document(content)

    chunks = chunk_document(doc, target_size=1000)

    assert len(chunks) == 1
    assert chunks[0].text == content
    assert chunks[0].chunk_index == 0
    assert chunks[0].document_id == doc.id


def test_blocks_are_grouped_up_to_target_size() -> None:
    block_a = "A" * 400
    block_b = "B" * 400
    block_c = "C" * 400
    content = f"{block_a}\n\n{block_b}\n\n{block_c}"
    doc = _make_document(content)

    chunks = chunk_document(doc, target_size=1000)

    assert len(chunks) == 2
    assert block_a in chunks[0].text
    assert block_b in chunks[0].text
    assert block_c not in chunks[0].text
    assert block_c in chunks[1].text
    assert block_a not in chunks[1].text
    assert "".join(chunk.text for chunk in chunks) == content


def test_exact_content_preservation() -> None:
    content = (
        "# Title\n\n"
        "Some intro paragraph with details.\n\n"
        "* item one\n* item two\n\n"
        "Another paragraph that continues the discussion further.\n\n"
        "Final paragraph."
    )
    doc = _make_document(content)

    chunks = chunk_document(doc, target_size=50)

    assert "".join(chunk.text for chunk in chunks) == content


def test_determinism() -> None:
    content = "Paragraph one.\n\nParagraph two.\n\nParagraph three with more text."
    doc = _make_document(content)

    chunks_1 = chunk_document(doc, target_size=30)
    chunks_2 = chunk_document(doc, target_size=30)

    assert chunks_1 == chunks_2
    assert [c.id for c in chunks_1] == [c.id for c in chunks_2]
    assert [c.chunk_index for c in chunks_1] == [c.chunk_index for c in chunks_2]
    assert [c.text for c in chunks_1] == [c.text for c in chunks_2]


def test_chunk_identity_matches_sha256_contract() -> None:
    content = "First block.\n\nSecond block with more content to force a split here."
    doc = _make_document(content, doc_id="identity-doc")

    chunks = chunk_document(doc, target_size=20)

    assert len(chunks) > 1
    for chunk in chunks:
        expected_id = hashlib.sha256(
            f"{doc.id}::{chunk.chunk_index}".encode("utf-8")
        ).hexdigest()
        assert chunk.id == expected_id


def test_giant_paragraph_uses_fragmentation_cascade() -> None:
    sentences = [f"This is sentence number {i} in the document" for i in range(200)]
    content = ". ".join(sentences) + "."
    doc = _make_document(content)
    target_size = 1000

    chunks = chunk_document(doc, target_size=target_size)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= target_size for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == content


def test_large_markdown_code_block_is_fragmented() -> None:
    code_lines = "\n".join(f"line_{i} = {i}" for i in range(200))
    code_block = f"```python\n{code_lines}\n```"
    content = f"Intro paragraph.\n\n{code_block}\n\nOutro paragraph."
    doc = _make_document(content)
    target_size = 300

    chunks = chunk_document(doc, target_size=target_size)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= target_size for chunk in chunks)
    reconstructed = "".join(chunk.text for chunk in chunks)
    assert reconstructed == content
    assert reconstructed.count(code_lines) == 1


def test_character_level_fallback_for_undelimited_text() -> None:
    content = "a" * 3000
    doc = _make_document(content)
    target_size = 500

    chunks = chunk_document(doc, target_size=target_size)

    assert len(chunks) == 6
    assert all(len(chunk.text) <= target_size for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == content


def test_invalid_target_size_raises_value_error() -> None:
    doc = _make_document("some content")

    with pytest.raises(ValueError):
        chunk_document(doc, target_size=0)

    with pytest.raises(ValueError):
        chunk_document(doc, target_size=-10)


def test_cascade_prioritizes_double_newline_over_single_newline() -> None:
    paragraph_a = "\n".join(f"a_line_{i}" for i in range(30))
    paragraph_b = "\n".join(f"b_line_{i}" for i in range(30))
    code_block = f"```\n{paragraph_a}\n\n{paragraph_b}\n```"
    doc = _make_document(code_block)
    prefix = f"```\n{paragraph_a}\n\n"
    target_size = len(prefix) + 5

    chunks = chunk_document(doc, target_size=target_size)

    assert "".join(chunk.text for chunk in chunks) == code_block
    assert chunks[0].text == prefix
    assert all(len(chunk.text) <= target_size for chunk in chunks)


def test_multiple_consecutive_separators_are_preserved() -> None:
    content = "First paragraph.\n\n\n\nSecond paragraph.\n\n\nThird paragraph."
    doc = _make_document(content)

    chunks = chunk_document(doc, target_size=15)

    assert "".join(chunk.text for chunk in chunks) == content


def test_multiple_code_blocks_preserved_independently() -> None:
    content = (
        "Intro text.\n\n"
        "```\ncode block one\n```\n\n"
        "Middle text.\n\n"
        "```\ncode block two\n```\n\n"
        "Outro text."
    )
    doc = _make_document(content)

    chunks = chunk_document(doc, target_size=25)

    joined = "".join(chunk.text for chunk in chunks)
    assert joined == content
    assert joined.count("code block one") == 1
    assert joined.count("code block two") == 1


def test_leading_and_trailing_whitespace_is_preserved() -> None:
    content = "  \n\nHello world.\n\n  "
    doc = _make_document(content)

    chunks = chunk_document(doc, target_size=1000)

    assert "".join(chunk.text for chunk in chunks) == content
    assert chunks[0].text.startswith("  ")
    assert chunks[-1].text.endswith("  ")


def test_target_size_boundary_exact_and_off_by_one() -> None:
    exact_content = "X" * 100
    doc_exact = _make_document(exact_content)
    chunks_exact = chunk_document(doc_exact, target_size=100)
    assert len(chunks_exact) == 1
    assert chunks_exact[0].text == exact_content

    over_content = "X" * 101
    doc_over = _make_document(over_content, doc_id="over-doc")
    chunks_over = chunk_document(doc_over, target_size=100)
    assert len(chunks_over) == 2
    assert "".join(chunk.text for chunk in chunks_over) == over_content
    assert all(len(chunk.text) <= 100 for chunk in chunks_over)


def test_chunks_preserve_document_provenance() -> None:
    content = "Paragraph one.\n\nParagraph two.\n\nParagraph three with extra content to split."
    doc = _make_document(content, doc_id="provenance-doc")

    chunks = chunk_document(doc, target_size=20)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.source == doc.source
        assert chunk.uri == doc.uri
    assert "".join(chunk.text for chunk in chunks) == doc.content


def test_chunking_does_not_mutate_document() -> None:
    content = "Paragraph one.\n\nParagraph two."
    doc = _make_document(content)
    original_id = doc.id
    original_source = doc.source
    original_uri = doc.uri
    original_content = doc.content
    original_metadata = dict(doc.metadata)

    chunk_document(doc, target_size=10)

    assert doc.id == original_id
    assert doc.source == original_source
    assert doc.uri == original_uri
    assert doc.content == original_content
    assert doc.metadata == original_metadata
