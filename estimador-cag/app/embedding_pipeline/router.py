"""Los dos endpoints del pipeline vectorial (sesión 08) y los dos interruptores de recuperación (sesión 10).

- `POST /embeddings/ingest`: un presupuesto → `documents` + `chunks` con sus vectores, en una transacción.
  200 con los identificadores y las métricas; 409 si ese `source_path` ya está ingestado; 422 si el body no
  valida; 500 si falta la API key o falla el embedder; 503 si Postgres no responde.
- `POST /search`: una consulta en lenguaje natural → los k chunks más relevantes. `mode` elige entre
  búsqueda vectorial y búsqueda híbrida (vectorial + full-text fusionadas con RRF) y `rerank` activa el patrón
  recall-then-rerank con el cross-encoder; sin indicarlos, mandan `SEARCH_MODE` y `RERANK_ENABLED`.
  Mismos códigos de error, más 500 si el modelo del reranker no se puede cargar.

Los routers se registran en `app/main.py`: el de ingesta con prefijo `/embeddings`, el de búsqueda en la raíz
(las dos rutas las fija el enunciado). El proveedor de embeddings es siempre OpenAI (`OPENAI_API_KEY`),
independientemente de `LLM_PROVIDER`: la consulta tiene que vectorizarse con el mismo modelo que la ingesta
o las distancias no significan nada.
"""

from __future__ import annotations

from functools import lru_cache

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAIError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.db.session import get_db_session
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.repository import (
    DocumentAlreadyIngested,
    ingest_budget_document,
)
from app.embedding_pipeline.retrieval import CrossEncoderReranker, RerankerUnavailable, retrieve
from app.embedding_pipeline.schemas import (
    IngestConflictResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    SearchTimings,
)

log = structlog.get_logger(__name__)

router = APIRouter(tags=["Embeddings"])
search_router = APIRouter(tags=["Búsqueda semántica"])

MISSING_KEY_DETAIL = (
    "El pipeline de embeddings necesita OPENAI_API_KEY en el archivo .env "
    "(los embeddings se generan siempre con OpenAI)."
)
DATABASE_UNAVAILABLE_DETAIL = (
    "No se puede hablar con PostgreSQL. Levanta la base (`docker compose up postgres -d`), aplica las "
    "migraciones (`alembic upgrade head`) y revisa DATABASE_URL."
)
EMBEDDING_FAILED_DETAIL = "La API de embeddings ha fallado; revisa los logs del servicio."
RERANKER_UNAVAILABLE_DETAIL = (
    "The reranker model could not be loaded (RERANK_MODEL); see the service logs. Run "
    "`python -m app.embedding_pipeline.retrieval.verify_reranker`, or disable reranking with "
    "RERANK_ENABLED=false (or `rerank: false` in the request)."
)


@lru_cache
def get_chunker() -> JSONStructuralChunker:
    """El chunker carga el tokenizador de tiktoken una vez por proceso."""
    return JSONStructuralChunker()


@lru_cache
def _reranker_for(model: str, batch_size: int, max_length: int, device: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(model, batch_size=batch_size, max_length=max_length, device=device)


def get_reranker(settings: Settings = Depends(get_settings)) -> CrossEncoderReranker:
    """The process-wide reranker wrapper. Cheap to build: the model itself loads lazily on first use, or at
    startup when RERANK_ENABLED=true (see the lifespan in `app/main.py`). Tests override this dependency with
    a wrapper that carries a fake scorer."""
    return _reranker_for(settings.rerank_model, settings.rerank_batch_size, settings.rerank_max_length, settings.rerank_device)


@lru_cache
def _embedder_for(api_key: str, model: str, batch_size: int) -> OpenAIEmbedder:
    return OpenAIEmbedder(api_key=api_key, model=model, batch_size=batch_size)


def get_embedder(settings: Settings = Depends(get_settings)) -> OpenAIEmbedder | None:
    """El embedder configurado, o `None` si no hay `OPENAI_API_KEY`.

    Devuelve `None` en vez de lanzar: FastAPI resuelve las dependencias **antes** de validar el body, así que
    un `HTTPException` aquí convertiría en 500 cualquier petición mal formada en un entorno sin clave (es lo
    que pasaba en el CI, que no tiene secrets). El contrato es 422 para la validación de Pydantic y 500 para
    los fallos de configuración, y ese orden solo se respeta comprobando la clave dentro del handler.
    """
    api_key = (settings.openai_api_key or "").strip()
    if not api_key:
        return None
    return _embedder_for(api_key, settings.embedding_model, settings.embedding_batch_size)


def _require_embedder(embedder: OpenAIEmbedder | None) -> OpenAIEmbedder:
    if embedder is None:
        raise HTTPException(status_code=500, detail=MISSING_KEY_DETAIL)
    return embedder


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingestar un presupuesto: trocear, vectorizar y persistir",
    responses={
        409: {"model": IngestConflictResponse, "description": "Ya existe un documento con ese source_path."},
        422: {"description": "El body no cumple el esquema del documento."},
        500: {"description": "Falta OPENAI_API_KEY o la API de embeddings ha fallado (detalle en los logs)."},
        503: {"description": "PostgreSQL no está disponible."},
    },
)
async def ingest(
    request: IngestRequest,
    chunker: JSONStructuralChunker = Depends(get_chunker),
    embedder: OpenAIEmbedder | None = Depends(get_embedder),
    session: AsyncSession = Depends(get_db_session),
) -> IngestResponse | JSONResponse:
    """Un chunk por componente (con cabecera de contexto del presupuesto padre), vectorizado por lotes y
    persistido junto al documento en **una sola transacción**: si el embedder falla, el rollback no deja
    documentos huérfanos sin chunks.

    Devuelve solo identificadores y métricas; los vectores quedan en `chunks.embedding` y se consultan con
    `POST /search`. Tokens y coste de la ingesta van al evento structlog `embedding_ingest_completed`.
    """
    active_embedder = _require_embedder(embedder)
    try:
        outcome = await ingest_budget_document(
            session,
            source_path=request.source_path,
            document_type=request.document_type,
            budget=request.content,
            chunker=chunker,
            embedder=active_embedder,
        )
    except DocumentAlreadyIngested as exc:
        log.info("embedding_ingest_duplicate", source_path=request.source_path, document_id=exc.document_id)
        # JSONResponse y no HTTPException: el enunciado pone `detail` y `document_id` al mismo nivel, y
        # `HTTPException` siempre anida lo que se le pase bajo la clave `detail`.
        return JSONResponse(
            status_code=409,
            content=IngestConflictResponse(detail="Document already ingested", document_id=exc.document_id).model_dump(),
        )
    except OpenAIError as exc:
        await session.rollback()
        log.error(
            "embedding_ingest_failed",
            error=type(exc).__name__,
            detail=str(exc),
            source_path=request.source_path,
            budget_id=request.content.budget_id,
        )
        raise HTTPException(status_code=500, detail=EMBEDDING_FAILED_DETAIL) from exc
    except SQLAlchemyError as exc:
        await session.rollback()
        log.error("embedding_ingest_db_error", error=type(exc).__name__, detail=str(exc), source_path=request.source_path)
        raise HTTPException(status_code=503, detail=DATABASE_UNAVAILABLE_DETAIL) from exc

    log.info(
        "embedding_ingest_completed",
        document_id=outcome.document_id,
        source_path=request.source_path,
        document_type=request.document_type,
        budget_id=request.content.budget_id,
        chunks_created=outcome.chunks_created,
        embedding_dimension=outcome.embedding_dimension,
        ingestion_time_ms=outcome.ingestion_time_ms,
        embedding_model=outcome.usage.model,
        batches=outcome.usage.batches,
        total_tokens=outcome.usage.prompt_tokens,
        counted_tokens=outcome.counted_tokens,
        estimated_cost_usd=outcome.usage.estimated_cost_usd,
    )
    return IngestResponse(
        document_id=outcome.document_id,
        chunks_created=outcome.chunks_created,
        embedding_dimension=outcome.embedding_dimension,
        ingestion_time_ms=outcome.ingestion_time_ms,
    )


@search_router.post(
    "/search",
    response_model=SearchResponse,
    summary="Buscar los k chunks más relevantes para una consulta (vectorial o híbrida, con o sin reranking)",
    responses={
        422: {"description": "El body no cumple el esquema de la consulta."},
        500: {"description": "Falta OPENAI_API_KEY, la API de embeddings ha fallado o el modelo del reranker no se puede cargar (detalle en los logs)."},
        503: {"description": "PostgreSQL no está disponible."},
    },
)
async def search(
    request: SearchRequest,
    settings: Settings = Depends(get_settings),
    embedder: OpenAIEmbedder | None = Depends(get_embedder),
    reranker: CrossEncoderReranker = Depends(get_reranker),
    session: AsyncSession = Depends(get_db_session),
) -> SearchResponse:
    """Vectoriza la consulta con el mismo modelo de la ingesta y recupera los k chunks más relevantes.

    - `mode=vector`: top-k por distancia coseno (`<=>`) en SQL, el baseline de la sesión 08.
    - `mode=hybrid`: la rama vectorial y la rama full-text (`content_tsv @@ query`) recuperan cada una
      `SEARCH_CANDIDATES` chunks y se fusionan con Reciprocal Rank Fusion.
    - `rerank=true`: recall-then-rerank. Los candidatos recuperados (50 por defecto) se reordenan con el
      cross-encoder y se devuelven los k mejores.

    Cada etapa viene cronometrada en `timings`, y cada resultado dice en `ranking` en qué posición lo puso
    cada rama y qué puntuación le dio el reranker. Sin índice vectorial: Postgres recorre la tabla entera, que
    con el corpus del curso son milisegundos."""
    active_embedder = _require_embedder(embedder)
    mode = request.mode or settings.search_mode
    rerank = settings.rerank_enabled if request.rerank is None else request.rerank
    try:
        outcome = await retrieve(
            session,
            query=request.query,
            embedder=active_embedder,
            mode=mode,
            k=request.k,
            rerank=rerank,
            reranker=reranker,
            candidates=settings.search_candidates,
            rrf_k=settings.rrf_k,
        )
    except OpenAIError as exc:
        log.error("search_embedding_failed", error=type(exc).__name__, detail=str(exc))
        raise HTTPException(status_code=500, detail=EMBEDDING_FAILED_DETAIL) from exc
    except SQLAlchemyError as exc:
        log.error("search_db_error", error=type(exc).__name__, detail=str(exc), mode=mode)
        raise HTTPException(status_code=503, detail=DATABASE_UNAVAILABLE_DETAIL) from exc
    except RerankerUnavailable as exc:
        raise HTTPException(status_code=500, detail=RERANKER_UNAVAILABLE_DETAIL) from exc

    results = outcome.results
    log.info(
        "search_completed",
        query=request.query,
        k=request.k,
        mode=mode,
        reranked=rerank,
        candidates=outcome.candidates,
        results=len(results),
        best_distance=results[0].distance if results else None,
        worst_distance=results[-1].distance if results else None,
        search_time_ms=outcome.timings.total_ms,
        embedding_ms=outcome.timings.embedding_ms,
        vector_ms=outcome.timings.vector_ms,
        lexical_ms=outcome.timings.lexical_ms,
        fusion_ms=outcome.timings.fusion_ms,
        rerank_ms=outcome.timings.rerank_ms,
        embedding_model=active_embedder.model,
        rerank_model=reranker.model_name if rerank else None,
    )
    return SearchResponse(
        query=request.query,
        k=request.k,
        mode=mode,
        reranked=rerank,
        candidates=outcome.candidates,
        search_time_ms=outcome.timings.total_ms,
        timings=SearchTimings(**outcome.timings.__dict__),
        results=results,
    )
