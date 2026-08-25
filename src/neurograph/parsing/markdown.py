from pathlib import Path
from typing import Any, Dict, Tuple, Union

import yaml

from neurograph.models.document import RawData

FRONTMATTER_DELIMITER = "---"


def parse_markdown_file(file_path: Union[str, Path]) -> RawData:
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    extracted_metadata, raw_content = _split_frontmatter(text)
    return RawData(
        source="markdown_local",
        uri=str(path),
        raw_content=raw_content,
        extracted_metadata=extracted_metadata,
    )


def _split_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != FRONTMATTER_DELIMITER:
        return {}, text
    for index in range(1, len(lines)):
        if lines[index].strip() == FRONTMATTER_DELIMITER:
            yaml_block = "".join(lines[1:index])
            body = "".join(lines[index + 1 :])
            metadata = yaml.safe_load(yaml_block)
            if metadata is None:
                metadata = {}
            return metadata, body
    return {}, text
