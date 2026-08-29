import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from neurograph.api.schemas import IngestRequest, IngestResponse
from neurograph.parsing.markdown import parse_markdown_file
from neurograph.retrieval.chunking import chunk_document
from neurograph.transform.normalizer import normalize

router = APIRouter(prefix="/v1", tags=["ingest"])

VAULT_ROOT_ENV_VAR = "NEUROGRAPH_VAULT_ROOT"

logger = logging.getLogger(__name__)


@router.post("/ingest/directory", response_model=IngestResponse)
def ingest_directory(payload: IngestRequest, request: Request) -> IngestResponse:
    vault_root = _resolve_vault_root()
    target_dir = _resolve_target_directory(vault_root, payload.path)
    markdown_files = _discover_markdown_files(target_dir, vault_root)

    vector_store = request.app.state.vector_store

    documents_processed = 0
    documents_failed = 0
    chunks_created = 0
    failed_files: list[str] = []

    for file_path in markdown_files:
        relative_path = file_path.relative_to(vault_root).as_posix()
        try:
            raw = parse_markdown_file(file_path)
            document = normalize(raw)
            chunks = chunk_document(document)
            vector_store.upsert_document(document, chunks)
        except Exception:
            # El error real se registra internamente (logs) pero no se
            # expone al cliente HTTP: solo se reporta la ruta relativa.
            logger.exception("Fallo al ingerir el archivo '%s'", relative_path)
            documents_failed += 1
            failed_files.append(relative_path)
            continue

        documents_processed += 1
        chunks_created += len(chunks)

    return IngestResponse(
        status=_resolve_status(documents_processed, documents_failed),
        documents_processed=documents_processed,
        documents_failed=documents_failed,
        chunks_created=chunks_created,
        failed_files=failed_files,
    )


def _resolve_vault_root() -> Path:
    raw_root = os.environ.get(VAULT_ROOT_ENV_VAR, "")
    if not raw_root.strip():
        raise HTTPException(
            status_code=500,
            detail="NEUROGRAPH_VAULT_ROOT no está configurada.",
        )
    return Path(raw_root).resolve()


def _resolve_target_directory(vault_root: Path, relative_path: str) -> Path:
    candidate = (vault_root / relative_path).resolve()
    if not candidate.is_relative_to(vault_root):
        raise HTTPException(status_code=403, detail="Ruta fuera del vault.")
    if not candidate.is_dir():
        raise HTTPException(
            status_code=422,
            detail="La ruta indicada no representa un directorio válido dentro del vault.",
        )
    return candidate


def _discover_markdown_files(target_dir: Path, vault_root: Path) -> list[Path]:
    discovered: list[Path] = []
    for candidate in target_dir.rglob("*.md"):
        resolved = candidate.resolve()
        # Re-verificación tras resolver symlinks: un enlace dentro del
        # vault podría apuntar fuera de él.
        if resolved.is_relative_to(vault_root):
            discovered.append(resolved)
    return discovered


def _resolve_status(documents_processed: int, documents_failed: int) -> str:
    if documents_failed == 0:
        return "ok"
    if documents_processed > 0:
        return "partial_success"
    return "failed"
