from typing import Literal
from pathlib import PurePosixPath, PureWindowsPath

from pydantic import BaseModel, Field, field_validator

# Límite superior de top_k para evitar requests abusivos (búsquedas costosas).
MAX_TOP_K = 20


class AskRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, gt=0, le=MAX_TOP_K)

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        if value.strip() == "":
            raise ValueError("query no puede estar vacío ni contener únicamente espacios.")
        return value


class IngestRequest(BaseModel):
    path: str

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        if value.strip() == "":
            raise ValueError("path no puede estar vacío.")
        # Chequeo sintáctico temprano (no reemplaza la validación de
        # pertenencia al vault basada en Path.is_relative_to() del route).
        if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
            raise ValueError("path debe ser una ruta relativa al vault, no absoluta.")
        return value


class IngestResponse(BaseModel):
    status: Literal["ok", "partial_success", "failed"]
    documents_processed: int
    documents_failed: int
    chunks_created: int
    failed_files: list[str]
