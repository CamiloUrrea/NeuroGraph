from pathlib import Path
from typing import Any

import chromadb
from fastembed import TextEmbedding

from neurograph.models.document import Chunk, Document

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "neurograph_chunks"
DEFAULT_PERSIST_DIRECTORY = ".chroma_db"

_STRUCTURAL_METADATA_KEYS = ("document_id", "source", "uri", "chunk_index")
_ALLOWED_METADATA_SCALAR_TYPES = (str, int, float, bool)


class LocalVectorStore:
    def __init__(self, persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY) -> None:
        self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._collection = self._client.get_or_create_collection(name=COLLECTION_NAME)
        self._embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)

    def upsert_document(self, doc: Document, chunks: list[Chunk]) -> None:
        # PREPARE
        existing_ids = self._get_existing_ids(doc.id)
        metadatas = [self._build_metadata(doc, chunk) for chunk in chunks]

        # EMBED
        embeddings = self._embed([chunk.text for chunk in chunks])

        # UPSERT
        if chunks:
            self._collection.upsert(
                ids=[chunk.id for chunk in chunks],
                embeddings=embeddings,
                documents=[chunk.text for chunk in chunks],
                metadatas=metadatas,
            )

        # CLEANUP
        new_ids = {chunk.id for chunk in chunks}
        stale_ids = existing_ids - new_ids
        if stale_ids:
            self._collection.delete(ids=list(stale_ids))

    def _get_existing_ids(self, document_id: str) -> set[str]:
        result = self._collection.get(where={"document_id": document_id}, include=[])
        return set(result["ids"])

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [embedding.tolist() for embedding in self._embedding_model.embed(texts)]

    def _build_metadata(self, doc: Document, chunk: Chunk) -> dict[str, Any]:
        metadata = dict(doc.metadata)
        for key in _STRUCTURAL_METADATA_KEYS:
            metadata.pop(key, None)
        _validate_metadata(metadata)
        metadata["document_id"] = doc.id
        metadata["source"] = doc.source
        metadata["uri"] = doc.uri
        metadata["chunk_index"] = chunk.chunk_index
        return metadata


def _validate_metadata(metadata: dict[str, Any]) -> None:
    for key, value in metadata.items():
        _validate_metadata_value(key, value)


def _validate_metadata_value(key: str, value: Any) -> None:
    if value is None or isinstance(value, _ALLOWED_METADATA_SCALAR_TYPES):
        return
    if isinstance(value, list):
        if value:
            first_type = type(value[0])
            if first_type in _ALLOWED_METADATA_SCALAR_TYPES and all(
                type(item) is first_type for item in value
            ):
                return
        raise ValueError(
            f"Metadata inválida para la clave '{key}': la lista {value!r} debe ser "
            "no vacía y contener elementos homogéneos de tipo str, int, float o bool."
        )
    raise ValueError(
        f"Metadata inválida para la clave '{key}': valor {value!r} de tipo "
        f"'{type(value).__name__}' no es compatible con ChromaDB "
        "(se admiten str, int, float, bool, None o listas homogéneas de esos tipos)."
    )
