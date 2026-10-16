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
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    budgets: list[Budget] = Field(min_length=1)


class IngestStats(BaseModel):
    """Agregados de una ingesta. Los cuatro primeros son los que pide el enunciado."""

    total_budgets: int
    total_chunks: int
    total_tokens: int = Field(description="Input tokens billed by the API (sum of usage.prompt_tokens over batches).")
    estimated_cost_usd: float = Field(description="total_tokens × list price of the embedding model.")
    counted_tokens: int = Field(description="Tokens counted locally with tiktoken before calling the API (sum of chunk.token_count).")
    embedding_model: str
    dimensions: int
    batches: int
    latency_ms: int = Field(description="Chunking + embedding wall time.")


class IngestResponse(BaseModel):
    chunks: list[EmbeddedChunk]
    stats: IngestStats
