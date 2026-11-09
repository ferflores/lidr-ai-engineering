"""Punto de entrada de la aplicación FastAPI.

Arranque en desarrollo:
    uv run uvicorn app.main:app --reload
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import structlog
from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.embedding_pipeline.retrieval import RerankerUnavailable
from app.embedding_pipeline.router import get_reranker
from app.embedding_pipeline.router import router as embeddings_router
from app.embedding_pipeline.router import search_router
from app.routers import estimations, sessions
from app.services.observability import configure_logging
from app.services.sessions import get_session_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
configure_logging(get_settings())  # structlog: un evento `turn_observed` por turno (JSON en stdout)

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Startup: pre-load the cross-encoder when reranking is on by default, so the first `POST /search` does
    not pay the model's cold start (seconds). A failed load is logged, not fatal: vector search keeps working
    and a request that asks for reranking gets a 500 with the reason."""
    settings = get_settings()
    if settings.rerank_enabled and settings.rerank_preload:
        try:
            await run_in_threadpool(get_reranker(settings).load)
        except RerankerUnavailable as exc:
            log.error("reranker_preload_failed", model=settings.rerank_model, detail=str(exc))
    yield

DESCRIPTION = """
Servicio que recibe la **transcripción de una reunión** con un cliente más unos
**parámetros tipados** (tarifa, tipo de proyecto, plazo, enfoque, nivel de detalle) y devuelve
una **estimación de software** (desglose de tareas, horas, coste, equipo y duración)
generada por un LLM.

Arquitectura **CAG**: el contexto que necesita el modelo (estimaciones de ejemplo)
viaja íntegro dentro del prompt en cada llamada. No hay base de datos ni retrieval.

Los prompts son **plantillas versionadas** en `prompts/estimacion/<versión>/`, fuera del código.
`GET /api/v1/prompts` lista las versiones; `prompt_version` en la petición elige una concreta.

Flujo: `parámetros + transcripción → plantilla vN (system + user) → LLM → estimación en Markdown`.

**Sesiones conversacionales** (sesión 05): `POST /api/v1/sessions` crea una sesión y
`POST /api/v1/sessions/{session_id}/estimate` (multipart/form-data) encadena turnos sobre el mismo proyecto,
con **adjuntos** (PDF, Word, texto) cuyo texto se extrae localmente, un **historial con ventana deslizante**
(`SESSION_MAX_TURNS`) y una **memoria del proyecto** (`project_metadata`) que se inyecta en el system prompt.
Las sesiones viven en memoria del proceso.

**Compresión y observación** (sesión 05 directo / pre-work 06): los turnos que salen de la ventana se pliegan en un
**resumen acumulativo**, los hechos del cliente se conservan como **anclas**, cada turno resuelve un **tier** y pasa
por una **caché** (exacta y semántica), y todo se agrega en un único evento `turn_observed` que
`GET /api/v1/sessions/{session_id}` expone como `last_turn_observed`.

**Embeddings y búsqueda semántica** (pre-work sesiones 07 y 08): `POST /embeddings/ingest` recibe un presupuesto
histórico en JSON, lo parte con un **chunker estructural** (un componente = un chunk, con cabecera de contexto
del presupuesto padre), lo vectoriza con `text-embedding-3-small` por lotes y **lo persiste en PostgreSQL +
pgvector** (`documents` + `chunks`) en una sola transacción. `POST /search` vectoriza una consulta y devuelve
los **k chunks más cercanos por distancia coseno**, resuelto en SQL con el operador `<=>`.

**Búsqueda híbrida y reranking** (pre-work sesión 10): `POST /search` admite `mode=hybrid` (la rama vectorial y
una rama **full-text de PostgreSQL** sobre la columna generada `content_tsv` se fusionan con **Reciprocal Rank
Fusion**) y `rerank=true` (**recall-then-rerank**: los 50 candidatos recuperados se reordenan con un
**cross-encoder** local y se devuelven los k mejores). Cada etapa viene cronometrada en `timings`.

Todavía **sin índice vectorial** (sequential scan, el baseline del directo) y sin filtros por metadata.
"""

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=DESCRIPTION,
    version=settings.app_version,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Estimaciones", "description": "Generación de estimaciones de software con un LLM (una petición, una estimación)."},
        {"name": "Sesiones", "description": "Conversaciones multi-turno con memoria del proyecto y adjuntos."},
        {"name": "Embeddings", "description": "Chunking estructural de presupuestos JSON, embeddings y persistencia en PostgreSQL + pgvector."},
        {"name": "Búsqueda semántica", "description": "Recuperación de los k chunks más relevantes: vectorial (coseno, pgvector) o híbrida (+ full-text, RRF), con reranking opcional por cross-encoder."},
        {"name": "Sistema", "description": "Estado y diagnóstico del servicio."},
    ],
)

app.include_router(estimations.router, prefix="/api/v1")
app.include_router(sessions.router, prefix="/api/v1")
app.include_router(embeddings_router, prefix="/embeddings")  # POST /embeddings/ingest (ruta fijada por el enunciado de la sesión 07)
app.include_router(search_router)  # POST /search, en la raíz (ruta fijada por el enunciado de la sesión 08)


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    provider: str
    model: str
    llm_configured: bool
    prompt_template: str
    session_max_turns: int
    active_sessions: int
    embedding_model: str
    embeddings_configured: bool
    database: str = Field(description="URL de PostgreSQL sin la contraseña. No se comprueba la conexión: /health no hace E/S.")
    search_mode: str = Field(description="Modo de búsqueda por defecto de POST /search (vector | hybrid).")
    rerank_enabled: bool = Field(description="Si POST /search reordena con el cross-encoder por defecto.")
    rerank_model: str
    reranker_loaded: bool = Field(description="Si el modelo del reranker ya está cargado en memoria en este proceso.")
    timestamp: datetime


@app.get("/health", response_model=HealthResponse, tags=["Sistema"], summary="Estado del servicio")
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Devuelve 200 si el servicio está levantado, con el proveedor, el modelo y la plantilla de prompt configurados."""
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        provider=settings.llm_provider,
        model=settings.model,
        llm_configured=settings.llm_configured,
        prompt_template=f"estimacion/{settings.prompt_version}",
        session_max_turns=settings.session_max_turns,
        active_sessions=len(get_session_store()),
        embedding_model=settings.embedding_model,
        embeddings_configured=bool((settings.openai_api_key or "").strip()),
        database=settings.database_dsn_safe,
        search_mode=settings.search_mode,
        rerank_enabled=settings.rerank_enabled,
        rerank_model=settings.rerank_model,
        reranker_loaded=get_reranker(settings).is_loaded,
        timestamp=datetime.now(timezone.utc),
    )
