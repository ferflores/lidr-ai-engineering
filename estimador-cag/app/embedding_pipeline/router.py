"""`POST /embeddings/ingest`: presupuestos JSON → chunks vectorizados (en memoria, sin persistir).

Contrato: 200 con `IngestResponse`; 422 si el body no valida (lo gestiona FastAPI); 500 si falta la API key
o la API de embeddings falla (mensaje genérico al cliente, detalle en los logs).

El router se registra en `app/main.py` con el prefijo `/embeddings`. El proveedor de embeddings es siempre
OpenAI (independiente de `LLM_PROVIDER`): se usa `OPENAI_API_KEY`.
"""

from __future__ import annotations

import time
from functools import lru_cache

import structlog
from fastapi import APIRouter, Depends, HTTPException
from openai import OpenAIError

from app.config import Settings, get_settings
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.schemas import IngestRequest, IngestResponse, IngestStats

log = structlog.get_logger(__name__)

router = APIRouter(tags=["Embeddings"])


@lru_cache
def get_chunker() -> JSONStructuralChunker:
    """El chunker carga el tokenizador de tiktoken una vez por proceso."""
    return JSONStructuralChunker()


@lru_cache
def _embedder_for(api_key: str, model: str, batch_size: int) -> OpenAIEmbedder:
    return OpenAIEmbedder(api_key=api_key, model=model, batch_size=batch_size)


def get_embedder(settings: Settings = Depends(get_settings)) -> OpenAIEmbedder:
    api_key = (settings.openai_api_key or "").strip()
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="El pipeline de embeddings necesita OPENAI_API_KEY en el archivo .env (los embeddings se generan siempre con OpenAI).",
        )
    return _embedder_for(api_key, settings.embedding_model, settings.embedding_batch_size)


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Trocear presupuestos JSON y generar sus embeddings",
    responses={
        422: {"description": "El body no cumple el esquema de presupuestos."},
        500: {"description": "Falta OPENAI_API_KEY o la API de embeddings ha fallado (detalle en los logs)."},
    },
)
def ingest(
    request: IngestRequest,
    chunker: JSONStructuralChunker = Depends(get_chunker),
    embedder: OpenAIEmbedder = Depends(get_embedder),
) -> IngestResponse:
    """Un chunk por componente (con cabecera de contexto del presupuesto padre), vectorizado con
    `text-embedding-3-small` en lotes. Devuelve los chunks con su vector y las estadísticas de la ingesta.
    Los vectores no se persisten todavía: eso llega con PostgreSQL + pgvector en la sesión 08."""
    started = time.perf_counter()
    chunks = chunker.chunk(request.budgets)
    try:
        embedded, usage = embedder.embed_many_with_usage(chunks)
    except OpenAIError as exc:
        log.error(
            "embedding_ingest_failed",
            error=type(exc).__name__,
            detail=str(exc),
            budgets=len(request.budgets),
            chunks=len(chunks),
        )
        raise HTTPException(status_code=500, detail="La API de embeddings ha fallado; revisa los logs del servicio.") from exc

    stats = IngestStats(
        total_budgets=len(request.budgets),
        total_chunks=len(embedded),
        total_tokens=usage.prompt_tokens,
        estimated_cost_usd=usage.estimated_cost_usd,
        counted_tokens=sum(chunk.token_count for chunk in chunks),
        embedding_model=usage.model,
        dimensions=usage.dimensions,
        batches=usage.batches,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    log.info("embedding_ingest_completed", **stats.model_dump())
    return IngestResponse(chunks=embedded, stats=stats)
