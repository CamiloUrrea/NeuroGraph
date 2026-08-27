from pathlib import Path
from typing import Optional

import chromadb
from fastembed import TextEmbedding

from neurograph.models.document import Chunk
from neurograph.models.search import SearchFilters
from neurograph.retrieval.vector_store import (
    COLLECTION_NAME,
    DEFAULT_PERSIST_DIRECTORY,
    EMBEDDING_MODEL,
)


class SemanticSearch:
    def __init__(self, persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY) -> None:
        self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._collection = self._client.get_or_create_collection(name=COLLECTION_NAME)
        self._embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[SearchFilters] = None,
    ) -> list[tuple[Chunk, float]]:
        _validate_query(query)
        _validate_top_k(top_k)
        where = _build_where(filters)

        query_embedding = next(iter(self._embedding_model.embed([query]))).tolist()

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        return [
            (
                Chunk(
                    id=chunk_id,
                    document_id=metadata["document_id"],
                    chunk_index=metadata["chunk_index"],
                    text=text,
                ),
                distance,
            )
            for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances)
        ]


def _validate_query(query: str) -> None:
    if not isinstance(query, str) or query.strip() == "":
        raise ValueError("query no puede estar vacío ni contener únicamente espacios.")


def _validate_top_k(top_k: int) -> None:
    if top_k <= 0:
        raise ValueError("top_k debe ser un entero estrictamente positivo.")


def _build_where(filters: Optional[SearchFilters]) -> Optional[dict]:
    if filters is None:
        return None
    if not isinstance(filters, SearchFilters):
        raise TypeError(
            "filters debe ser una instancia de SearchFilters, no un diccionario arbitrario."
        )

    conditions = []
    if filters.source is not None:
        conditions.append({"source": filters.source})
    if filters.document_id is not None:
        conditions.append({"document_id": filters.document_id})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
