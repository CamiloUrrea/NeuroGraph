import pytest
from fastapi.testclient import TestClient

AUTH_HEADERS = {"X-NeuroGraph-Key": "secret-key"}


class FakeSemanticSearch:
    def search(self, query, top_k=5, filters=None):
        return []


class FakeVectorStore:
    def __init__(self):
        self.upserted: list[tuple] = []

    def upsert_document(self, doc, chunks):
        self.upserted.append((doc, chunks))


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv("NEUROGRAPH_API_KEY", "secret-key")
    fake_search = FakeSemanticSearch()
    fake_store = FakeVectorStore()
    monkeypatch.setattr("neurograph.api.main.SemanticSearch", lambda *a, **kw: fake_search)
    monkeypatch.setattr("neurograph.api.main.LocalVectorStore", lambda *a, **kw: fake_store)

    from neurograph.api.main import app

    with TestClient(app) as test_client:
        yield test_client, fake_store


def write_markdown(path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_ingest_valid_directory_processes_files(tmp_path, api_client, monkeypatch):
    monkeypatch.setenv("NEUROGRAPH_VAULT_ROOT", str(tmp_path))
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    write_markdown(notes_dir / "a.md", "# Nota A\n\nContenido de la nota A.")
    write_markdown(notes_dir / "b.md", "# Nota B\n\nContenido de la nota B.")

    test_client, fake_store = api_client
    response = test_client.post("/v1/ingest/directory", json={"path": "notes"}, headers=AUTH_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["documents_processed"] == 2
    assert body["documents_failed"] == 0
    assert body["failed_files"] == []
    assert body["chunks_created"] > 0
    assert len(fake_store.upserted) == 2


def test_ingest_missing_auth_header_returns_401(tmp_path, api_client, monkeypatch):
    monkeypatch.setenv("NEUROGRAPH_VAULT_ROOT", str(tmp_path))

    test_client, _ = api_client
    response = test_client.post("/v1/ingest/directory", json={"path": "notes"})

    assert response.status_code == 401


def test_ingest_path_traversal_returns_403(tmp_path, api_client, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("NEUROGRAPH_VAULT_ROOT", str(vault))

    test_client, _ = api_client
    response = test_client.post(
        "/v1/ingest/directory", json={"path": "../outside"}, headers=AUTH_HEADERS
    )

    assert response.status_code == 403


def test_ingest_absolute_path_returns_422(tmp_path, api_client, monkeypatch):
    monkeypatch.setenv("NEUROGRAPH_VAULT_ROOT", str(tmp_path))

    test_client, _ = api_client
    response = test_client.post(
        "/v1/ingest/directory", json={"path": "/etc"}, headers=AUTH_HEADERS
    )

    assert response.status_code == 422


def test_ingest_empty_path_returns_422(tmp_path, api_client, monkeypatch):
    monkeypatch.setenv("NEUROGRAPH_VAULT_ROOT", str(tmp_path))

    test_client, _ = api_client
    response = test_client.post(
        "/v1/ingest/directory", json={"path": ""}, headers=AUTH_HEADERS
    )

    assert response.status_code == 422


def test_ingest_missing_vault_root_returns_500(api_client, monkeypatch):
    monkeypatch.delenv("NEUROGRAPH_VAULT_ROOT", raising=False)

    test_client, _ = api_client
    response = test_client.post(
        "/v1/ingest/directory", json={"path": "notes"}, headers=AUTH_HEADERS
    )

    assert response.status_code == 500


def test_ingest_discovers_markdown_recursively(tmp_path, api_client, monkeypatch):
    monkeypatch.setenv("NEUROGRAPH_VAULT_ROOT", str(tmp_path))
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    write_markdown(nested / "deep.md", "# Profundo\n\nTexto profundo.")
    (tmp_path / "ignore.txt").write_text("no debe procesarse", encoding="utf-8")

    test_client, fake_store = api_client
    response = test_client.post("/v1/ingest/directory", json={"path": "."}, headers=AUTH_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["documents_processed"] == 1
    assert len(fake_store.upserted) == 1


def test_ingest_failed_file_does_not_abort_others(tmp_path, api_client, monkeypatch):
    monkeypatch.setenv("NEUROGRAPH_VAULT_ROOT", str(tmp_path))
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    # 'Author' y 'author' normalizan al mismo key snake_case -> normalize()
    # lanza ValueError (colision de metadata), forzando un fallo real de pipeline.
    write_markdown(
        notes_dir / "bad.md",
        "---\nAuthor: uno\nauthor: dos\n---\n\nContenido malo.",
    )
    write_markdown(notes_dir / "good.md", "# Buena\n\nContenido bueno.")

    test_client, fake_store = api_client
    response = test_client.post("/v1/ingest/directory", json={"path": "notes"}, headers=AUTH_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["documents_processed"] == 1
    assert body["documents_failed"] == 1
    assert body["failed_files"] == ["notes/bad.md"]
    assert body["status"] == "partial_success"
    assert len(fake_store.upserted) == 1


def test_ingest_status_ok_when_no_failures(tmp_path, api_client, monkeypatch):
    monkeypatch.setenv("NEUROGRAPH_VAULT_ROOT", str(tmp_path))
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    write_markdown(notes_dir / "a.md", "# A\n\nTexto.")

    test_client, _ = api_client
    response = test_client.post("/v1/ingest/directory", json={"path": "notes"}, headers=AUTH_HEADERS)

    assert response.json()["status"] == "ok"


def test_ingest_status_failed_when_all_files_fail(tmp_path, api_client, monkeypatch):
    monkeypatch.setenv("NEUROGRAPH_VAULT_ROOT", str(tmp_path))
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    write_markdown(
        notes_dir / "bad.md", "---\nAuthor: uno\nauthor: dos\n---\n\nContenido malo."
    )

    test_client, fake_store = api_client
    response = test_client.post("/v1/ingest/directory", json={"path": "notes"}, headers=AUTH_HEADERS)

    body = response.json()
    assert body["status"] == "failed"
    assert body["documents_processed"] == 0
    assert body["documents_failed"] == 1
    assert len(fake_store.upserted) == 0


def test_ingest_empty_directory_without_markdown_returns_ok(tmp_path, api_client, monkeypatch):
    monkeypatch.setenv("NEUROGRAPH_VAULT_ROOT", str(tmp_path))
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    test_client, _ = api_client
    response = test_client.post("/v1/ingest/directory", json={"path": "empty"}, headers=AUTH_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["documents_processed"] == 0
    assert body["documents_failed"] == 0
    assert body["chunks_created"] == 0
    assert body["failed_files"] == []


def test_ingest_counters_reflect_real_processing(tmp_path, api_client, monkeypatch):
    monkeypatch.setenv("NEUROGRAPH_VAULT_ROOT", str(tmp_path))
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    write_markdown(notes_dir / "a.md", "# A\n\n" + ("x" * 2500))
    write_markdown(notes_dir / "bad.md", "---\nAuthor: uno\nauthor: dos\n---\n\nMalo.")

    test_client, fake_store = api_client
    response = test_client.post("/v1/ingest/directory", json={"path": "notes"}, headers=AUTH_HEADERS)

    body = response.json()
    total_chunks = sum(len(chunks) for _, chunks in fake_store.upserted)
    assert body["chunks_created"] == total_chunks
    assert body["documents_processed"] == 1
    assert body["documents_failed"] == 1


def test_ingest_symlink_outside_vault_is_excluded(tmp_path, api_client, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    write_markdown(outside / "secret.md", "# Secreto\n\nNo debe procesarse.")
    monkeypatch.setenv("NEUROGRAPH_VAULT_ROOT", str(vault))

    link_path = vault / "link.md"
    try:
        link_path.symlink_to(outside / "secret.md")
    except (OSError, NotImplementedError):
        pytest.skip("El entorno no permite crear symlinks.")

    test_client, fake_store = api_client
    response = test_client.post("/v1/ingest/directory", json={"path": "."}, headers=AUTH_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["documents_processed"] == 0
    assert len(fake_store.upserted) == 0
