"""Modelos Pydantic v2 del pipeline de embeddings.

Entrada: presupuestos históricos con el esquema simplificado del curso (sesión 06). Salida: chunks listos
para embeber y chunks ya vectorizados. Decisiones:

- `Budget` y `BudgetComponent` validan lo que el chunker necesita (campos obligatorios, tipos, `complexity`
  y `sector` como vocabularios cerrados, ids sin espacios, ids de componente únicos dentro del presupuesto)
  e **ignoran** campos extra: el dataset histórico puede traer columnas que este pipeline no usa y no queremos
  rechazarlo por eso. `IngestRequest`, en cambio, es el contrato del endpoint y **prohíbe** extras, como el
  resto de peticiones del proyecto (sesión 04).
- `sector` es un `Literal` cerrado (sugerencia del enunciado). Si un dataset propio trae un sector nuevo,
  se añade a `Sector`: es un cambio de vocabulario explícito, no un dato que se cuela sin querer.
- `Chunk.metadata` guarda los campos filtrables, que viajan con el chunk pero **no** se embeben.
- `IngestRequest` / `SearchRequest` son los contratos HTTP de la sesión 08: un documento por ingesta y
  una consulta por búsqueda. `Chunk` y `EmbeddedChunk` siguen siendo los objetos internos del pipeline
  (lo que produce el chunker y lo que devuelve el embedder antes de persistirse).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import SearchMode

Sector = Literal[
    "finance",
    "ecommerce",
    "healthcare",
    "industrial",
    "logistics",
    "education",
    "energy",
    "media",
    "telecom",
    "insurance",
    "public_sector",
    "real_estate",
    "travel",
    "other",
]
Complexity = Literal["low", "medium", "high"]

# Identificadores sin espacios: el chunk_id se construye como "{budget_id}::{component_id}" y debe seguir siendo
# parseable (y trazable hasta el documento original) sin ambigüedad.
ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


class ClientMetadata(BaseModel):
    """Quién encargó el presupuesto. El sector también viaja en el texto del chunk (pesa semánticamente)."""

    name: str = Field(min_length=1, description="Client name.")
    sector: Sector = Field(description="Closed vocabulary; the course sample uses finance, ecommerce, healthcare, industrial.")
    country: str = Field(pattern=r"^[A-Z]{2}$", description="ISO 3166-1 alpha-2 country code (e.g. ES).")


class BudgetComponent(BaseModel):
    """Una unidad de negocio del presupuesto: la granularidad del chunk."""

    component_id: str = Field(pattern=ID_PATTERN, description="Unique within the budget, e.g. AUTH-001.")
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    tech_stack: list[str] = Field(default_factory=list)
    estimated_hours: int = Field(ge=0)
    complexity: Complexity
    dependencies: list[str] = Field(default_factory=list, description="component_ids this component depends on.")

    @field_validator("tech_stack", "dependencies")
    @classmethod
    def _clean_items(cls, items: list[str]) -> list[str]:
        """Quita espacios y entradas vacías; el texto del chunk se construye con estas listas."""
        return [item.strip() for item in items if item and item.strip()]


class Budget(BaseModel):
    """Un presupuesto histórico normalizado (esquema simplificado del curso)."""

    budget_id: str = Field(pattern=ID_PATTERN, description="e.g. BUD-2024-014.")
    client_metadata: ClientMetadata
    project_summary: str = Field(min_length=1)
    main_technology: str = Field(min_length=1)
    year: int = Field(ge=2000, le=2100)
    total_estimated_hours: int = Field(ge=0)
    components: list[BudgetComponent] = Field(min_length=1, description="At least one component: a budget with none produces no chunks.")

    @model_validator(mode="after")
    def _component_ids_unique(self) -> "Budget":
        seen: set[str] = set()
        duplicates = sorted({c.component_id for c in self.components if c.component_id in seen or seen.add(c.component_id)})
        if duplicates:
            raise ValueError(f"duplicate component_id in budget {self.budget_id}: {', '.join(duplicates)}")
        return self

    @property
    def components_hours(self) -> int:
        return sum(c.estimated_hours for c in self.components)


class Chunk(BaseModel):
    """Fragmento listo para embeber. `text` es lo que se vectoriza; `metadata` acompaña al chunk sin embeberse."""

    chunk_id: str = Field(min_length=1, description="{budget_id}::{component_id}")
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict, description="Filterable fields (not embedded).")
    token_count: int = Field(ge=0, description="Tokens of `text` with the embedding model's tokenizer (tiktoken).")


class EmbeddedChunk(Chunk):
    embedding: list[float] = Field(min_length=1, description="Vector returned by the embedding model.")


# --- Contrato de POST /embeddings/ingest (sesión 08: un documento por petición, persistido) --------------
#
# La sesión 07 recibía `{"budgets": [...]}` y devolvía los chunks con sus vectores en la respuesta. Ahora la
# unidad es **un documento**: un presupuesto con su ruta de origen, que se persiste en PostgreSQL y del que
# solo vuelven identificadores y métricas. Los vectores ya no viajan por HTTP: viven en la base.


class IngestRequest(BaseModel):
    """Un documento a ingestar. `content` es el presupuesto completo, validado como `Budget`."""

    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(
        min_length=1,
        max_length=500,
        description="Where the document comes from; unique per document (a second ingest of the same path is a 409).",
    )
    document_type: str = Field(min_length=1, max_length=50, description="e.g. historical_budget.")
    content: Budget = Field(description="The full budget JSON, the same schema the chunker consumes.")


class IngestResponse(BaseModel):
    """Lo que devuelve una ingesta correcta (200). Exactamente los cuatro campos del enunciado.

    Los tokens y el coste **no** están aquí porque el contrato no los incluye: se registran en el evento
    structlog `embedding_ingest_completed`, que es donde el proyecto mide el gasto desde la sesión 03.
    """

    document_id: int
    chunks_created: int
    embedding_dimension: int
    ingestion_time_ms: int


class IngestConflictResponse(BaseModel):
    """Respuesta 409: ya hay un documento con ese `source_path`."""

    detail: str = Field(examples=["Document already ingested"])
    document_id: int


# --- Contrato de POST /search ----------------------------------------------------------------------------


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000, description="Natural-language query; embedded with the same model used at ingest time.")
    # El tope de 100 no es del enunciado: evita que una petición pida medio corpus en una sola respuesta.
    k: int = Field(default=5, ge=1, le=100, description="How many chunks to return.")
    # Session 10: the two switches of the exercise. `None` means "the service default" (SEARCH_MODE and
    # RERANK_ENABLED), so one running service can be measured in all four configurations without restarts.
    mode: SearchMode | None = Field(
        default=None,
        description="vector: cosine distance only. hybrid: cosine + PostgreSQL full-text, fused with Reciprocal Rank Fusion. Default: SEARCH_MODE.",
    )
    rerank: bool | None = Field(
        default=None,
        description="Recall-then-rerank: recall SEARCH_CANDIDATES chunks, reorder them with the cross-encoder, return the top-k. Default: RERANK_ENABLED.",
    )


class RankSignals(BaseModel):
    """Where a result came from and why it ended at its position: the 1-based rank in each recall branch, the
    RRF score (hybrid) and the cross-encoder score (reranked). A field is None when that stage did not run or
    did not return the chunk."""

    vector_rank: int | None = None
    lexical_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class SearchResult(BaseModel):
    """Un chunk recuperado.

    `distance` es la distancia coseno (0 = idéntico, cuanto menor, más cercano) y es `None` cuando el chunk
    solo lo encontró la rama léxica. `score` es lo que ordenó el resultado en la última etapa (similitud
    coseno, puntuación RRF o puntuación del cross-encoder): mayor es mejor, pero las escalas no son comparables
    entre configuraciones; `ranking` dice de qué etapa viene cada número.
    """

    chunk_id: int
    document_id: int
    chunk_type: str
    content: str
    distance: float | None = Field(default=None, description="Cosine distance; None when only the lexical branch returned the chunk.")
    score: float = Field(description="Ordering score of the final stage: cosine similarity, RRF score or cross-encoder score.")
    ranking: RankSignals = Field(default_factory=RankSignals)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchTimings(BaseModel):
    """Milliseconds per stage of the request. Stages that did not run report 0."""

    embedding_ms: int = 0
    vector_ms: int = 0
    lexical_ms: int = 0
    fusion_ms: int = 0
    rerank_ms: int = 0
    total_ms: int = 0


class SearchResponse(BaseModel):
    query: str
    k: int
    mode: SearchMode = Field(description="Search mode that produced the results.")
    reranked: bool = Field(description="Whether the cross-encoder reordered the candidates.")
    candidates: int = Field(description="How many chunks reached the final stage (the recall width when reranking, k otherwise).")
    search_time_ms: int = Field(description="Whole pipeline: query embedding + SQL (+ fusion + rerank).")
    timings: SearchTimings = Field(default_factory=SearchTimings)
    results: list[SearchResult]
