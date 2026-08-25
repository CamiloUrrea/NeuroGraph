from typing import Any, Dict

from pydantic import BaseModel, Field


class RawData(BaseModel):
    source: str = Field(description="Identificador de la fuente/proveedor.")
    uri: str = Field(description="Ubicación exacta del recurso original.")
    raw_content: str = Field(description="Contenido textual extraído del recurso.")
    extracted_metadata: Dict[str, Any] = Field(
        description="Metadata extraída directamente de la fuente."
    )


class Document(BaseModel):
    id: str = Field(description="Identificador único del documento.")
    source: str = Field(description="Identificador de la fuente/proveedor.")
    uri: str = Field(description="Ubicación exacta del recurso original.")
    content: str = Field(description="Contenido normalizado del documento.")
    metadata: Dict[str, Any] = Field(description="Metadata asociada al documento.")


class Chunk(BaseModel):
    id: str = Field(description="Identificador único del chunk.")
    document_id: str = Field(description="Identificador del documento de origen.")
    chunk_index: int = Field(description="Posición del chunk dentro del documento.")
    text: str = Field(description="Texto del chunk.")
