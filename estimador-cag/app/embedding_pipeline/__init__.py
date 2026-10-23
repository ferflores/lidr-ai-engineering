"""Pipeline de embeddings con persistencia en PostgreSQL + pgvector (sesiones 07 y 08).

    presupuesto JSON → JSONStructuralChunker (un componente = un chunk) → OpenAIEmbedder
    (text-embedding-3-small, por lotes) → documents + chunks en PostgreSQL (una transacción)
    → POST /search resuelve la consulta por distancia coseno en SQL

La sesión 07 montó el chunker y el embedder y devolvía los vectores por HTTP. La 08 los persiste:
`POST /embeddings/ingest` guarda un documento con sus chunks y devuelve solo identificadores y métricas, y
`POST /search` recupera los k chunks más cercanos. Todavía **sin índice vectorial** (sequential scan), sin
filtros por metadata y sin búsqueda híbrida: los añade la sesión en directo.

El código de este módulo usa nombres en inglés por convención del curso.
"""

from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.repository import (
    DocumentAlreadyIngested,
    IngestOutcome,
    build_search_statement,
    ingest_budget_document,
    search_chunks,
)
from app.embedding_pipeline.schemas import (
    Budget,
    BudgetComponent,
    Chunk,
    EmbeddedChunk,
    IngestConflictResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.embedding_pipeline.similarity import cosine_similarity

__all__ = [
    "Budget",
    "BudgetComponent",
    "Chunk",
    "DocumentAlreadyIngested",
    "EmbeddedChunk",
    "IngestConflictResponse",
    "IngestOutcome",
    "IngestRequest",
    "IngestResponse",
    "JSONStructuralChunker",
    "OpenAIEmbedder",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "build_search_statement",
    "cosine_similarity",
    "ingest_budget_document",
    "search_chunks",
]
