"""Pipeline mínimo de embeddings y chunking (pre-work de la sesión 07).

Convierte presupuestos históricos en JSON (salida normalizada de la sesión 06) en chunks vectorizados:

    presupuestos JSON → JSONStructuralChunker (un componente = un chunk) → OpenAIEmbedder
    (text-embedding-3-small, por lotes) → chunks con su vector, listos para persistir en la sesión 08.

Los vectores se generan en memoria y se devuelven por HTTP (`POST /embeddings/ingest`); todavía no hay
base de datos vectorial ni búsqueda. El código de este módulo usa nombres en inglés por convención del curso.
"""

from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.schemas import (
    Budget,
    BudgetComponent,
    Chunk,
    EmbeddedChunk,
    IngestRequest,
    IngestResponse,
    IngestStats,
)
from app.embedding_pipeline.similarity import cosine_similarity

__all__ = [
    "Budget",
    "BudgetComponent",
    "Chunk",
    "EmbeddedChunk",
    "IngestRequest",
    "IngestResponse",
    "IngestStats",
    "JSONStructuralChunker",
    "OpenAIEmbedder",
    "cosine_similarity",
]
