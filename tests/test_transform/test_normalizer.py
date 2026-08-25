import pytest

from neurograph.models.document import Document, RawData
from neurograph.transform.normalizer import normalize


def test_normalize_is_deterministic() -> None:
    raw = RawData(
        source="markdown_local",
        uri="file:///notes/a.md",
        raw_content="# Title\n\nSome content.",
        extracted_metadata={"Creation Date": "2026-01-01"},
    )

    result_1 = normalize(raw)
    result_2 = normalize(raw)

    assert result_1 == result_2
    assert result_1.id == result_2.id
    assert result_1.content == result_2.content
    assert result_1.metadata == result_2.metadata


def test_normalize_standardizes_line_endings_and_strips_without_touching_markdown() -> None:
    raw = RawData(
        source="markdown_local",
        uri="file:///notes/b.md",
        raw_content="  \r\n# Heading\r\n\r\n* item one\r\n* item two\r\n\r\n> quote\r\n\r\n[link](https://example.com)\r\n  ",
        extracted_metadata={},
    )

    result = normalize(raw)

    assert result.content == (
        "# Heading\n\n* item one\n* item two\n\n> quote\n\n[link](https://example.com)"
    )
    assert "\r" not in result.content
    assert result.content == result.content.strip()


def test_normalize_metadata_snake_case() -> None:
    raw = RawData(
        source="markdown_local",
        uri="file:///notes/c.md",
        raw_content="content",
        extracted_metadata={"Creation Date": "2026-01-01", "Author-Name": "Ada"},
    )

    result = normalize(raw)

    assert result.metadata == {
        "creation_date": "2026-01-01",
        "author_name": "Ada",
    }


def test_normalize_metadata_collision_raises_value_error() -> None:
    raw = RawData(
        source="markdown_local",
        uri="file:///notes/d.md",
        raw_content="content",
        extracted_metadata={"Tag Name": "x", "tag-name": "y"},
    )

    with pytest.raises(ValueError) as excinfo:
        normalize(raw)

    message = str(excinfo.value)
    assert raw.uri in message
    assert "Tag Name" in message
    assert "tag-name" in message


def test_document_id_depends_on_raw_content_but_content_is_normalized() -> None:
    raw_a = RawData(
        source="markdown_local",
        uri="file:///notes/e.md",
        raw_content="Texto",
        extracted_metadata={},
    )
    raw_b = RawData(
        source="markdown_local",
        uri="file:///notes/e.md",
        raw_content=" Texto ",
        extracted_metadata={},
    )

    doc_a = normalize(raw_a)
    doc_b = normalize(raw_b)

    assert isinstance(doc_a, Document)
    assert isinstance(doc_b, Document)
    assert doc_a.id != doc_b.id
    assert doc_a.content == doc_b.content
