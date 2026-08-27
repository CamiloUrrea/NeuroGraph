import copy
from unittest.mock import patch

import pytest

from neurograph.models.document import Chunk, Document
from neurograph.retrieval.vector_store import LocalVectorStore


def make_document(doc_id: str, metadata: dict | None = None) -> Document:
    return Document(
        id=doc_id,
        source="test-source",
        uri=f"file:///{doc_id}",
        content="contenido irrelevante para estos tests",
        metadata=metadata if metadata is not None else {},
    )


def make_chunks(doc_id: str, texts: list[str]) -> list[Chunk]:
    return [
        Chunk(id=f"{doc_id}::chunk::{i}", document_id=doc_id, chunk_index=i, text=text)
        for i, text in enumerate(texts)
    ]


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> LocalVectorStore:
    persist_dir = tmp_path_factory.mktemp("chroma_db_shared")
    return LocalVectorStore(persist_directory=persist_dir)


def test_initial_ingestion_creates_one_record_per_chunk(store: LocalVectorStore) -> None:
    doc = make_document("doc-a")
    chunks = make_chunks("doc-a", ["primer chunk", "segundo chunk", "tercer chunk"])

    store.upsert_document(doc, chunks)

    result = store._collection.get(where={"document_id": "doc-a"})
    assert sorted(result["ids"]) == sorted(c.id for c in chunks)
    assert len(result["ids"]) == len(chunks)


def test_idempotent_upsert_produces_no_duplicates(store: LocalVectorStore) -> None:
    doc = make_document("doc-b")
    chunks = make_chunks("doc-b", ["texto a", "texto b"])

    store.upsert_document(doc, chunks)
    store.upsert_document(doc, chunks)

    result = store._collection.get(where={"document_id": "doc-b"})
    assert sorted(result["ids"]) == sorted(c.id for c in chunks)


def test_update_removes_stale_chunks_keeps_new_ones(store: LocalVectorStore) -> None:
    doc = make_document("doc-c")
    v1_chunks = make_chunks("doc-c", ["v1 a", "v1 b", "v1 c"])
    store.upsert_document(doc, v1_chunks)

    v2_chunks = [
        Chunk(id="doc-c::chunk::0", document_id="doc-c", chunk_index=0, text="v2 a"),
        Chunk(id="doc-c::chunk::new", document_id="doc-c", chunk_index=1, text="v2 nuevo"),
    ]
    store.upsert_document(doc, v2_chunks)

    result = store._collection.get(where={"document_id": "doc-c"})
    assert sorted(result["ids"]) == sorted(c.id for c in v2_chunks)
    assert "doc-c::chunk::1" not in result["ids"]
    assert "doc-c::chunk::2" not in result["ids"]


def test_embedding_failure_leaves_previous_state_intact(store: LocalVectorStore) -> None:
    doc = make_document("doc-d")
    chunks_v1 = make_chunks("doc-d", ["chunk ok 1", "chunk ok 2"])
    store.upsert_document(doc, chunks_v1)

    chunks_v2 = make_chunks("doc-d", ["chunk roto"])
    with patch.object(store._embedding_model, "embed", side_effect=RuntimeError("fastembed boom")):
        with pytest.raises(RuntimeError):
            store.upsert_document(doc, chunks_v2)

    result = store._collection.get(where={"document_id": "doc-d"})
    assert sorted(result["ids"]) == sorted(c.id for c in chunks_v1)


def test_stored_ids_match_chunk_ids_exactly(store: LocalVectorStore) -> None:
    doc = make_document("doc-e")
    chunks = make_chunks("doc-e", ["uno", "dos"])

    store.upsert_document(doc, chunks)

    result = store._collection.get(where={"document_id": "doc-e"})
    assert set(result["ids"]) == {chunk.id for chunk in chunks}


def test_structural_metadata_overrides_doc_metadata(store: LocalVectorStore) -> None:
    doc = make_document(
        "doc-f",
        metadata={
            "document_id": "WRONG-ID",
            "source": "WRONG-SOURCE",
            "uri": "WRONG-URI",
            "chunk_index": -999,
            "custom_field": "mantener-este-valor",
        },
    )
    chunks = make_chunks("doc-f", ["texto de prueba"])

    store.upsert_document(doc, chunks)

    result = store._collection.get(ids=[chunks[0].id], include=["metadatas"])
    meta = result["metadatas"][0]
    assert meta["document_id"] == "doc-f"
    assert meta["source"] == doc.source
    assert meta["uri"] == doc.uri
    assert meta["chunk_index"] == 0
    assert meta["custom_field"] == "mantener-este-valor"


def test_invalid_metadata_raises_and_preserves_previous_state(store: LocalVectorStore) -> None:
    doc_v1 = make_document("doc-g")
    chunks_v1 = make_chunks("doc-g", ["texto valido"])
    store.upsert_document(doc_v1, chunks_v1)

    doc_v2 = make_document("doc-g", metadata={"anidado": {"a": 1}})
    chunks_v2 = make_chunks("doc-g", ["texto nuevo"])
    with pytest.raises(ValueError):
        store.upsert_document(doc_v2, chunks_v2)

    result = store._collection.get(where={"document_id": "doc-g"})
    assert sorted(result["ids"]) == sorted(c.id for c in chunks_v1)


def test_persistence_across_store_instances(tmp_path_factory) -> None:
    persist_dir = tmp_path_factory.mktemp("chroma_db_persistence")
    doc = make_document("doc-h")
    chunks = make_chunks("doc-h", ["chunk persistente"])

    store1 = LocalVectorStore(persist_directory=persist_dir)
    store1.upsert_document(doc, chunks)
    del store1

    store2 = LocalVectorStore(persist_directory=persist_dir)
    result = store2._collection.get(where={"document_id": "doc-h"})
    assert sorted(result["ids"]) == sorted(c.id for c in chunks)


def test_upsert_does_not_mutate_inputs(store: LocalVectorStore) -> None:
    doc = make_document("doc-i", metadata={"key": "value"})
    chunks = make_chunks("doc-i", ["chunk de texto a", "chunk de texto b"])

    doc_copy = copy.deepcopy(doc)
    chunks_copy = copy.deepcopy(chunks)
    metadata_copy = copy.deepcopy(doc.metadata)

    store.upsert_document(doc, chunks)

    assert doc == doc_copy
    assert chunks == chunks_copy
    assert doc.metadata == metadata_copy


def test_isolation_between_documents(store: LocalVectorStore) -> None:
    doc_x = make_document("doc-j-x")
    doc_y = make_document("doc-j-y")
    chunks_x = make_chunks("doc-j-x", ["x1", "x2"])
    chunks_y = make_chunks("doc-j-y", ["y1", "y2"])

    store.upsert_document(doc_x, chunks_x)
    store.upsert_document(doc_y, chunks_y)

    store.upsert_document(doc_x, [chunks_x[0]])

    result_x = store._collection.get(where={"document_id": "doc-j-x"})
    result_y = store._collection.get(where={"document_id": "doc-j-y"})

    assert set(result_x["ids"]) == {chunks_x[0].id}
    assert set(result_y["ids"]) == {c.id for c in chunks_y}
