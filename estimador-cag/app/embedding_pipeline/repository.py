"""Persistencia y búsqueda semántica sobre PostgreSQL + pgvector (sesión 08).

Dos operaciones, las dos sobre una `AsyncSession` que abre y cierra el router:

- `ingest_budget_document`: un presupuesto → una fila en `documents` + N filas en `chunks` con su vector,
  **en una sola transacción**. Si el embedder falla, el rollback deja la base como estaba: nunca queda un
  documento huérfano sin chunks.
- `search_chunks`: los k chunks más cercanos a un vector por **distancia coseno** (operador `<=>` de
  pgvector), resuelto en SQL. Sin índice vectorial todavía: Postgres hace sequential scan, que es el
  baseline que mide el directo.

El embedder del proyecto es sincrónico (SDK de OpenAI). Para no bloquear el event loop mientras se vectoriza
un lote, la llamada se delega a un hilo con `run_in_threadpool`, que es lo que hace FastAPI por debajo con
los handlers `def`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.db.models import CHUNK_TYPE_BUDGET_COMPONENT, Chunk, Document
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import EmbeddingUsage, OpenAIEmbedder
from app.embedding_pipeline.schemas import Budget, SearchResult

log = structlog.get_logger(__name__)


class DocumentAlreadyIngested(Exception):
    """Ya hay un documento con ese `source_path`. El router lo traduce a 409."""

    def __init__(self, document_id: int) -> None:
        super().__init__(f"document already ingested (id={document_id})")
        self.document_id = document_id


@dataclass
class IngestOutcome:
    """Lo que hay que saber de una ingesta: el contrato del endpoint más los datos de coste para el log."""

    document_id: int
    chunks_created: int
    embedding_dimension: int
    ingestion_time_ms: int
    usage: EmbeddingUsage
    counted_tokens: int


def build_document_metadata(budget: Budget) -> dict:
    """Metadata del presupuesto (nivel documento): lo que describe al presupuesto entero, no a un componente."""
    return {
        "budget_id": budget.budget_id,
        "client_name": budget.client_metadata.name,
        "client_sector": budget.client_metadata.sector,
        "client_country": budget.client_metadata.country,
        "project_summary": budget.project_summary,
        "main_technology": budget.main_technology,
        "year": budget.year,
        "total_estimated_hours": budget.total_estimated_hours,
        "components": len(budget.components),
    }


async def find_document_id_by_source_path(session: AsyncSession, source_path: str) -> int | None:
    return await session.scalar(select(Document.id).where(Document.source_path == source_path))


async def ingest_budget_document(
    session: AsyncSession,
    *,
    source_path: str,
    document_type: str,
    budget: Budget,
    chunker: JSONStructuralChunker,
    embedder: OpenAIEmbedder,
) -> IngestOutcome:
    """Persiste el presupuesto y sus chunks vectorizados en una transacción. Lanza `DocumentAlreadyIngested`.

    Orden del enunciado: comprobar duplicado → crear el documento → trocear → embeber por lotes →
    `add_all` de los chunks → commit.
    """
    started = time.perf_counter()

    existing_id = await find_document_id_by_source_path(session, source_path)
    if existing_id is not None:
        raise DocumentAlreadyIngested(existing_id)

    document = Document(
        source_path=source_path,
        document_type=document_type,
        doc_metadata=build_document_metadata(budget),
    )
    session.add(document)
    # `flush` manda el INSERT y recupera el id generado (BIGSERIAL) sin cerrar la transacción: los chunks
    # necesitan el document_id y todo tiene que seguir siendo reversible con un rollback.
    await session.flush()

    chunks = chunker.chunk([budget])
    # El SDK de OpenAI es sincrónico: va a un hilo para no bloquear el event loop del servidor.
    embedded, usage = await run_in_threadpool(embedder.embed_many_with_usage, chunks)

    session.add_all(
        [
            Chunk(
                document_id=document.id,
                chunk_type=CHUNK_TYPE_BUDGET_COMPONENT,
                content=item.text,
                embedding=item.embedding,
                chunk_metadata={**item.metadata, "chunk_ref": item.chunk_id, "token_count": item.token_count},
            )
            for item in embedded
        ]
    )
    await session.commit()

    return IngestOutcome(
        document_id=document.id,
        chunks_created=len(embedded),
        embedding_dimension=usage.dimensions,
        ingestion_time_ms=int((time.perf_counter() - started) * 1000),
        usage=usage,
        counted_tokens=sum(chunk.token_count for chunk in chunks),
    )


def build_search_statement(query_vector: list[float], k: int):
    """El `SELECT` de la búsqueda semántica: k chunks ordenados por distancia coseno ascendente.

    `cosine_distance` es el operador `<=>` de pgvector. Se usa dos veces (en la proyección y en el
    `ORDER BY`) a propósito: Postgres evalúa la expresión una sola vez por fila, y tener la distancia en el
    resultado es lo que permite leer el ranking en vez de confiar en él a ciegas.

    Está separado del `execute` para poder comprobar el SQL generado en los tests sin una base de datos.
    """
    distance = Chunk.embedding.cosine_distance(query_vector)
    return (
        select(
            Chunk.id,
            Chunk.document_id,
            Chunk.chunk_type,
            Chunk.content,
            Chunk.chunk_metadata,
            distance.label("distance"),
        )
        .order_by(distance)
        .limit(k)
    )


async def search_chunks(session: AsyncSession, query_vector: list[float], k: int) -> list[SearchResult]:
    """Ejecuta la búsqueda y devuelve los resultados ya tipados."""
    rows = (await session.execute(build_search_statement(query_vector, k))).all()
    return [
        SearchResult(
            chunk_id=row.id,
            document_id=row.document_id,
            chunk_type=row.chunk_type,
            content=row.content,
            distance=float(row.distance),
            metadata=row.chunk_metadata or {},
        )
        for row in rows
    ]
