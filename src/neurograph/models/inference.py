from pydantic import BaseModel, Field

INSUFFICIENT_CONTEXT_MESSAGE = "No tengo información suficiente en tus notas para responder esto."


class Source(BaseModel):
    document_id: str = Field(description="Identificador del documento de origen.")
    uri: str = Field(description="Ubicación exacta del recurso original.")
    source: str = Field(description="Identificador de la fuente/proveedor.")
    chunk_index: int = Field(description="Posición del chunk dentro del documento.")


class Answer(BaseModel):
    content: str = Field(description="Contenido de la respuesta generada.")
    sources: list[Source] = Field(description="Fuentes que respaldan la respuesta.")


class GenerationResult(BaseModel):
    answer: str = Field(description="Texto de la respuesta generada por el LLM.")
    citations: list[int] = Field(
        description="Índices 1-based de los contextos citados por el LLM."
    )


class InferenceError(Exception):
    pass
