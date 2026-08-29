from fastapi import APIRouter, HTTPException, Request

from neurograph.api.schemas import AskRequest
from neurograph.inference.engine import generate_answer
from neurograph.models.inference import Answer, InferenceError
from neurograph.retrieval.context import select_context

router = APIRouter(prefix="/v1", tags=["ask"])

# MAX_DISTANCE: límite PROVISIONAL de seguridad/compatibilidad, no un
# umbral semántico validado. La calibración empírica sobre ChromaDB no
# encontró separación limpia entre distancias relevantes e irrelevantes
# (ver reporte de calibración); 2.0 solo evita recuperar resultados
# arbitrariamente lejanos mientras el Arquitecto define una política
# definitiva con datos de retrieval reales.
MAX_DISTANCE = 2.0
MAX_CHARS = 4000


@router.post("/ask", response_model=Answer)
def ask(payload: AskRequest, request: Request) -> Answer:
    semantic_search = request.app.state.semantic_search

    results = semantic_search.search(payload.query, top_k=payload.top_k)
    context = select_context(results, max_distance=MAX_DISTANCE, max_chars=MAX_CHARS)

    try:
        return generate_answer(payload.query, context)
    except InferenceError as exc:
        raise HTTPException(
            status_code=502,
            detail="El proveedor de inferencia no pudo generar una respuesta en este momento.",
        ) from exc
