from pathlib import Path

from neurograph.models.document import RawData
from neurograph.parsing.markdown import parse_markdown_file

FRONTMATTER_CONTENT = """---
title: Test Note
tags:
  - alpha
  - beta
---

# Heading

Body text here.
"""

NO_FRONTMATTER_CONTENT = """# Heading

Body text without frontmatter.
"""


def test_parse_markdown_with_frontmatter(tmp_path: Path) -> None:
    file_path = tmp_path / "note.md"
    file_path.write_text(FRONTMATTER_CONTENT, encoding="utf-8")

    result = parse_markdown_file(file_path)

    assert isinstance(result, RawData)
    assert result.extracted_metadata == {
        "title": "Test Note",
        "tags": ["alpha", "beta"],
    }
    assert "# Heading" in result.raw_content
    assert "Body text here." in result.raw_content
    assert "title" not in result.raw_content
    assert "Test Note" not in result.raw_content
    assert "---" not in result.raw_content
    assert "Heading" not in str(result.extracted_metadata)
    assert "Body text here." not in str(result.extracted_metadata)


def test_parse_markdown_without_frontmatter(tmp_path: Path) -> None:
    file_path = tmp_path / "plain.md"
    file_path.write_text(NO_FRONTMATTER_CONTENT, encoding="utf-8")

    result = parse_markdown_file(file_path)

    assert isinstance(result, RawData)
    assert result.extracted_metadata == {}
    assert result.raw_content == NO_FRONTMATTER_CONTENT
