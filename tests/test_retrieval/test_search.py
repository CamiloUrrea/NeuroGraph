import pydantic
import pytest

from neurograph.models.document import Chunk, Document
from neurograph.models.search import SearchFilters
from neurograph.retrieval import search as search_module
from neurograph.retrieval import vector_store as vector_store_module
from neurograph.retrieval.search import SemanticSearch
from neurograph.retrieval.vector_store import LocalVectorStore


def make_document(doc_id: str, source: str) -> Document:
    return Document(
        id=doc_id,
        source=source,
        uri=f"file:///{doc_id}",
        content="contenido irrelevante para estos tests",
        metadata={},
    )


@pytest.fixture(scope="module")
def seeded_index(tmp_path_factory):
    persist_dir = tmp_path_factory.mktemp("chroma_search_index")
    store = LocalVectorStore(persist_directory=persist_dir)

    doc1 = make_document("doc-1", source="wiki")
    chunks1 = [
        Chunk(id="doc-1::0", document_id="doc-1", chunk_index=0, text="El gato duerme en el sofa."),
        Chunk(id="doc-1::1", document_id="doc-1", chunk_index=1, text="Los perros ladran en el parque."),
    ]
    store.upsert_document(doc1, chunks1)

    doc2 = make_document("doc-2", source="notes")
    chunks2 = [
        Chunk(id="doc-2::0", document_id="doc-2", chunk_index=0, text="Python es un lenguaje de programacion."),
    ]
    store.upsert_document(doc2, chunks2)

    searcher = SemanticSearch(persist_directory=persist_dir)
    all_chunks_by_id = {c.id: c for c in chunks1 + chunks2}
    return searcher, all_chunks_by_id


def test_basic_semantic_search_returns_results(seeded_index) -> None:
    searcher, _ = seeded_index

    results = searcher.search("animales domesticos", top_k=5)

    assert len(results) > 0
    for chunk, distance in results:
        assert isinstance(chunk, Chunk)
        assert isinstance(distance, float)


def test_returned_ids_match_stored_ids(seeded_index) -> None:
    searcher, all_chunks_by_id = seeded_index

    results = searcher.search("animales domesticos", top_k=5)

    for chunk, _ in results:
        assert chunk.id in all_chunks_by_id


def test_reconstructed_text_matches_stored_chunk(seeded_index) -> None:
    searcher, all_chunks_by_id = seeded_index

    results = searcher.search("animales domesticos", top_k=5)

    for chunk, _ in results:
        assert chunk.text == all_chunks_by_id[chunk.id].text
        assert chunk.document_id == all_chunks_by_id[chunk.id].document_id
        assert chunk.chunk_index == all_chunks_by_id[chunk.id].chunk_index


def test_results_preserve_relevance_order(seeded_index) -> None:
    searcher, _ = seeded_index

    results = searcher.search("animales domesticos", top_k=5)

    distances = [distance for _, distance in results]
    assert distances == sorted(distances)


def test_top_k_limits_number_of_results(seeded_index) -> None:
    searcher, _ = seeded_index

    results = searcher.search("animales domesticos", top_k=1)

    assert len(results) == 1


def test_query_with_no_matches_returns_empty_list(seeded_index) -> None:
    searcher, _ = seeded_index

    results = searcher.search(
        "cualquier consulta",
        top_k=5,
        filters=SearchFilters(source="fuente-inexistente"),
    )

    assert results == []


def test_filter_by_source(seeded_index) -> None:
    searcher, _ = seeded_index

    results = searcher.search("texto", top_k=10, filters=SearchFilters(source="wiki"))

    assert len(results) > 0
    for chunk, _ in results:
        assert chunk.document_id == "doc-1"


def test_filter_by_document_id(seeded_index) -> None:
    searcher, _ = seeded_index

    results = searcher.search("texto", top_k=10, filters=SearchFilters(document_id="doc-2"))

    assert len(results) > 0
    for chunk, _ in results:
        assert chunk.document_id == "doc-2"


def test_invalid_filters_raise_before_querying_chromadb(seeded_index) -> None:
    searcher, _ = seeded_index

    with pytest.raises(pydantic.ValidationError):
        SearchFilters(unexpected_field="x")

    with pytest.raises(TypeError):
        searcher.search("texto", top_k=5, filters={"source": "wiki"})


def test_empty_query_raises_explicit_error(seeded_index) -> None:
    searcher, _ = seeded_index

    with pytest.raises(ValueError):
        searcher.search("", top_k=5)

    with pytest.raises(ValueError):
        searcher.search("   ", top_k=5)


def test_non_positive_top_k_raises_explicit_error(seeded_index) -> None:
    searcher, _ = seeded_index

    with pytest.raises(ValueError):
        searcher.search("texto", top_k=0)

    with pytest.raises(ValueError):
        searcher.search("texto", top_k=-1)


def test_query_embedding_model_matches_vector_store_embedding_model(seeded_index) -> None:
    searcher, _ = seeded_index

    assert search_module.EMBEDDING_MODEL == vector_store_module.EMBEDDING_MODEL
    assert searcher._embedding_model.model_name == vector_store_module.EMBEDDING_MODEL
