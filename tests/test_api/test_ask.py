import pytest
from fastapi.testclient import TestClient

from neurograph.models.document import Chunk
from neurograph.models.inference import Answer, InferenceError, Source

AUTH_HEADERS = {"X-NeuroGraph-Key": "secret-key"}


class FakeSemanticSearch:
    def __init__(self):
        self.results: list[tuple[Chunk, float]] = []
        self.calls: list[dict] = []

    def search(self, query, top_k=5, filters=None):
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return self.results


class FakeVectorStore:
    def upsert_document(self, doc, chunks):
        raise AssertionError("upsert_document no debe invocarse desde /v1/ask")


def make_chunk(index: int) -> Chunk:
    return Chunk(
        id=f"chunk-{index}",
        document_id="doc-1",
        chunk_index=index,
        text=f"texto del chunk {index}",
        source="test-source",
        uri="file:///doc-1",
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("NEUROGRAPH_API_KEY", "secret-key")
    fake_search = FakeSemanticSearch()
    fake_store = FakeVectorStore()
    monkeypatch.setattr("neurograph.api.main.SemanticSearch", lambda *a, **kw: fake_search)
    monkeypatch.setattr("neurograph.api.main.LocalVectorStore", lambda *a, **kw: fake_store)

    from neurograph.api.main import app

    with TestClient(app) as test_client:
        yield test_client, fake_search


def test_ask_valid_request_returns_answer(client, monkeypatch):
    test_client, fake_search = client
    fake_search.results = [(make_chunk(0), 0.5)]
    expected_answer = Answer(
        content="respuesta generada",
        sources=[Source(document_id="doc-1", uri="file:///doc-1", source="test-source", chunk_index=0)],
    )
    monkeypatch.setattr(
        "neurograph.api.routes.ask.generate_answer",
        lambda query, context: expected_answer,
    )

    response = test_client.post("/v1/ask", json={"query": "hola", "top_k": 3}, headers=AUTH_HEADERS)

    assert response.status_code == 200
    assert response.json() == expected_answer.model_dump()
    assert fake_search.calls[0]["query"] == "hola"
    assert fake_search.calls[0]["top_k"] == 3


def test_ask_missing_auth_header_returns_401(client):
    test_client, _ = client

    response = test_client.post("/v1/ask", json={"query": "hola"})

    assert response.status_code == 401


def test_ask_wrong_auth_header_returns_401(client):
    test_client, _ = client

    response = test_client.post("/v1/ask", json={"query": "hola"}, headers={"X-NeuroGraph-Key": "wrong"})

    assert response.status_code == 401


def test_ask_unconfigured_api_key_denies_all_requests(client, monkeypatch):
    test_client, _ = client
    monkeypatch.delenv("NEUROGRAPH_API_KEY", raising=False)

    response = test_client.post("/v1/ask", json={"query": "hola"}, headers=AUTH_HEADERS)

    assert response.status_code == 401


def test_ask_empty_query_returns_422(client):
    test_client, _ = client

    response = test_client.post("/v1/ask", json={"query": ""}, headers=AUTH_HEADERS)
    assert response.status_code == 422

    response = test_client.post("/v1/ask", json={"query": "   "}, headers=AUTH_HEADERS)
    assert response.status_code == 422


def test_ask_invalid_top_k_returns_422(client):
    test_client, _ = client

    response = test_client.post("/v1/ask", json={"query": "hola", "top_k": 0}, headers=AUTH_HEADERS)
    assert response.status_code == 422

    response = test_client.post("/v1/ask", json={"query": "hola", "top_k": -1}, headers=AUTH_HEADERS)
    assert response.status_code == 422

    response = test_client.post("/v1/ask", json={"query": "hola", "top_k": 999}, headers=AUTH_HEADERS)
    assert response.status_code == 422


def test_ask_propagates_answer_with_sources(client, monkeypatch):
    test_client, fake_search = client
    fake_search.results = [(make_chunk(1), 0.1)]
    expected = Answer(
        content="contenido final",
        sources=[Source(document_id="doc-1", uri="file:///doc-1", source="test-source", chunk_index=1)],
    )
    monkeypatch.setattr(
        "neurograph.api.routes.ask.generate_answer",
        lambda query, context: expected,
    )

    response = test_client.post("/v1/ask", json={"query": "pregunta"}, headers=AUTH_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["content"] == "contenido final"
    assert body["sources"][0]["document_id"] == "doc-1"
    assert body["sources"][0]["chunk_index"] == 1


def test_ask_inference_error_returns_502_without_leaking_details(client, monkeypatch):
    test_client, fake_search = client
    fake_search.results = [(make_chunk(0), 0.2)]

    def raise_inference_error(query, context):
        raise InferenceError("clave de Gemini invalida: sk-xyz")

    monkeypatch.setattr("neurograph.api.routes.ask.generate_answer", raise_inference_error)

    response = test_client.post("/v1/ask", json={"query": "pregunta"}, headers=AUTH_HEADERS)

    assert response.status_code == 502
    assert "sk-xyz" not in response.text
