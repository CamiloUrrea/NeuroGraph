from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Optional[str] = Field(
        default=None, description="Filtra resultados por fuente/proveedor del documento."
    )
    document_id: Optional[str] = Field(
        default=None, description="Filtra resultados por identificador de documento."
    )
