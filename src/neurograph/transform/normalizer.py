import hashlib
import re
from typing import Any, Dict

from neurograph.models.document import Document, RawData


def normalize(raw: RawData) -> Document:
    document_id = hashlib.sha256(
        f"{raw.uri}::{raw.raw_content}".encode("utf-8")
    ).hexdigest()
    content = raw.raw_content.replace("\r\n", "\n").replace("\r", "\n").strip()
    metadata = _normalize_metadata(raw.extracted_metadata, raw.uri)
    return Document(
        id=document_id,
        source=raw.source,
        uri=raw.uri,
        content=content,
        metadata=metadata,
    )


def _normalize_metadata(extracted_metadata: Dict[str, Any], uri: str) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    origin_of: Dict[str, str] = {}
    for original_key, value in extracted_metadata.items():
        normalized_key = _to_snake_case(original_key)
        if normalized_key in origin_of:
            raise ValueError(
                f"Colisión de metadata en '{uri}': las claves "
                f"'{origin_of[normalized_key]}' y '{original_key}' "
                f"normalizan ambas a '{normalized_key}'."
            )
        origin_of[normalized_key] = original_key
        normalized[normalized_key] = value
    return normalized


def _to_snake_case(key: str) -> str:
    lowered = key.lower()
    underscored = lowered.replace(" ", "_").replace("-", "_")
    return re.sub(r"[^a-z0-9_]", "", underscored)
