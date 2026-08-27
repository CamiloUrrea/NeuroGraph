import hashlib
import re

from neurograph.models.document import Chunk, Document

_CODE_FENCE_PATTERN = re.compile(r"```.*?```", re.DOTALL)
_CASCADE_DELIMITERS = ["\n\n", "\n", ". ", "? ", "! "]


def chunk_document(doc: Document, target_size: int = 1000) -> list[Chunk]:
    if target_size <= 0:
        raise ValueError("target_size debe ser un entero estrictamente positivo.")
    if doc.content == "":
        return []
    blocks = _split_into_blocks(doc.content)
    atomic_pieces: list[str] = []
    for block in blocks:
        if len(block) > target_size:
            atomic_pieces.extend(_fragment_oversized(block, target_size))
        else:
            atomic_pieces.append(block)
    chunk_texts = _group_pieces(atomic_pieces, target_size)
    return [
        Chunk(
            id=_chunk_id(doc.id, index),
            document_id=doc.id,
            chunk_index=index,
            text=text,
            source=doc.source,
            uri=doc.uri,
        )
        for index, text in enumerate(chunk_texts)
    ]


def _chunk_id(document_id: str, chunk_index: int) -> str:
    return hashlib.sha256(f"{document_id}::{chunk_index}".encode("utf-8")).hexdigest()


def _split_into_blocks(content: str) -> list[str]:
    fence_spans = [match.span() for match in _CODE_FENCE_PATTERN.finditer(content)]
    boundaries = {0, len(content)}
    for start, end in fence_spans:
        boundaries.add(start)
        boundaries.add(end)
    for match in re.finditer(r"\n\n", content):
        if not any(start <= match.start() < end for start, end in fence_spans):
            boundaries.add(match.end())
    sorted_boundaries = sorted(boundaries)
    return [
        content[start:end]
        for start, end in zip(sorted_boundaries, sorted_boundaries[1:])
        if content[start:end] != ""
    ]


def _group_pieces(pieces: list[str], target_size: int) -> list[str]:
    groups: list[str] = []
    current = ""
    for piece in pieces:
        if current == "":
            current = piece
        elif len(current) + len(piece) <= target_size:
            current += piece
        else:
            groups.append(current)
            current = piece
    if current != "":
        groups.append(current)
    return groups


def _fragment_oversized(text: str, target_size: int) -> list[str]:
    if len(text) <= target_size:
        return [text]
    for delimiter in _CASCADE_DELIMITERS:
        pieces = [
            piece for piece in _split_keep_delimiter(text, delimiter) if piece != ""
        ]
        if len(pieces) > 1:
            grouped = _group_pieces(pieces, target_size)
            result: list[str] = []
            for piece in grouped:
                if len(piece) > target_size:
                    result.extend(_fragment_oversized(piece, target_size))
                else:
                    result.append(piece)
            return result
    return _split_by_chars(text, target_size)


def _split_keep_delimiter(text: str, delimiter: str) -> list[str]:
    boundaries = {0, len(text)}
    search_start = 0
    while True:
        index = text.find(delimiter, search_start)
        if index == -1:
            break
        boundary = index + len(delimiter)
        boundaries.add(boundary)
        search_start = boundary
    sorted_boundaries = sorted(boundaries)
    return [
        text[start:end] for start, end in zip(sorted_boundaries, sorted_boundaries[1:])
    ]


def _split_by_chars(text: str, target_size: int) -> list[str]:
    return [text[i : i + target_size] for i in range(0, len(text), target_size)]
