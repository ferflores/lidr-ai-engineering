"""Pipeline de embeddings con persistencia en PostgreSQL + pgvector (sesiones 07 y 08) y recuperación
híbrida con reranking (sesión 10).

    presupuesto JSON → JSONStructuralChunker (un componente = un chunk) → OpenAIEmbedder
    (text-embedding-3-small, por lotes) → documents + chunks en PostgreSQL (una transacción)
    → POST /search: rama vectorial (coseno) [+ rama full-text, fusión RRF] [+ rerank con cross-encoder] → top-k

La sesión 07 montó el chunker y el embedder y devolvía los vectores por HTTP. La 08 los persiste:
`POST /embeddings/ingest` guarda un documento con sus chunks y devuelve solo identificadores y métricas, y
`POST /search` recupera los k chunks más cercanos. La 10 añade en `retrieval/` la rama léxica sobre la columna
generada `content_tsv`, la fusión Reciprocal Rank Fusion y el patrón recall-then-rerank con un cross-encoder
local, los dos como interruptores (`mode`, `rerank`). Todavía **sin índice vectorial** (sequential scan) y sin
filtros por metadata: los añade la sesión en directo.

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
