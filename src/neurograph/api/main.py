import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException

from neurograph.retrieval.search import SemanticSearch
from neurograph.retrieval.vector_store import LocalVectorStore

API_KEY_ENV_VAR = "NEUROGRAPH_API_KEY"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # SemanticSearch/LocalVectorStore cargan ChromaDB y el modelo de
    # embeddings: se instancian una única vez en startup, nunca por request.
    app.state.semantic_search = SemanticSearch()
    app.state.vector_store = LocalVectorStore()
    yield
    # Ni LocalVectorStore ni SemanticSearch exponen un mecanismo público de
    # cierre; no se inventa un protocolo de shutdown para ellos.


def verify_api_key(
    x_neurograph_key: str | None = Header(default=None, alias="X-NeuroGraph-Key"),
) -> None:
    expected = os.environ.get(API_KEY_ENV_VAR)
    is_valid = (
        bool(expected)
        and x_neurograph_key is not None
        and secrets.compare_digest(x_neurograph_key, expected)
    )
    if not is_valid:
        # Mensaje genérico: no revela si falta el header, si la clave no
        # coincide, o si NEUROGRAPH_API_KEY no está configurada.
        raise HTTPException(status_code=401, detail="No autorizado.")


# Los routers no dependen de main.py (evita import circular); la
# autenticación se aplica aquí, al incluirlos.
from neurograph.api.routes.ask import router as ask_router  # noqa: E402
from neurograph.api.routes.ingest import router as ingest_router  # noqa: E402

app = FastAPI(title="NeuroGraph API", lifespan=lifespan)
app.include_router(ask_router, dependencies=[Depends(verify_api_key)])
app.include_router(ingest_router, dependencies=[Depends(verify_api_key)])
