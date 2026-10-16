"""Chunker estructural para presupuestos JSON: un componente del presupuesto = un chunk.

Por qué no un splitter genérico (artículo 4 de la sesión 07): partir el JSON por tamaño rompe claves a la
mitad, pierde la jerarquía padre-hijo y convierte el presupuesto en prosa plana. La unidad lógica del dominio
es el componente (nombre, descripción, stack, horas, complejidad), así que esa es la granularidad.

Tres decisiones explícitas:

1. **Granularidad**: un `BudgetComponent` → un `Chunk`. Sin overlap y sin trocear descripciones largas:
   si un componente se sale de lo normal se avisa en el log (`chunk_oversized`) y se discute en el directo.
2. **Texto del chunk**: representación legible con *contextual chunk headers* del presupuesto padre
   (proyecto, sector, año, tecnología principal) antes de los datos del componente. Sin esa cabecera un
   "Authentication backend" no sabría de qué cliente ni sector es. Es la versión estática y gratuita de
   Contextual Retrieval: el contexto sale del propio JSON, sin llamar a un LLM.
3. **Metadata**: los campos filtrables (budget_id, component_id, sector, tecnología, año, complejidad,
   horas) viajan en `metadata` y no se embeben. El sector va en los dos sitios a propósito: en el texto pesa
   semánticamente, en metadata sirve para filtrar.
"""

from __future__ import annotations

import structlog
import tiktoken

from app.embedding_pipeline.schemas import Budget, BudgetComponent, Chunk

log = structlog.get_logger(__name__)

# tiktoken resuelve text-embedding-3-small a la codificación cl100k_base: el mismo tokenizador que usa la API.
TOKEN_COUNT_MODEL = "text-embedding-3-small"

# El modelo admite 8.191 tokens por entrada; un componente que pase de aquí no es un error, es un dato raro
# (descripción anormalmente larga) que conviene mirar antes de decidir si se parte. Solo se registra.
OVERSIZED_CHUNK_TOKENS = 512

PARENT_CONTEXT_TEMPLATE = (
    "[Project: {project_summary}]\n"
    "[Client sector: {sector} | Year: {year} | Main tech: {main_technology}]"
)
COMPONENT_TEMPLATE = (
    "Component: {name}\n"
    "Description: {description}\n"
    "Tech stack: {tech_stack}\n"
    "Complexity: {complexity}\n"
    "Estimated hours: {estimated_hours}"
)


class JSONStructuralChunker:
    """Chunks budget JSON documents by business unit (component), with parent-budget headers."""

    def __init__(self, token_count_model: str = TOKEN_COUNT_MODEL, oversized_tokens: int = OVERSIZED_CHUNK_TOKENS) -> None:
        self._encoding = tiktoken.encoding_for_model(token_count_model)
        self._oversized_tokens = oversized_tokens

    # --- API pública -----------------------------------------------------------------------------

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        """Un chunk por componente, en el orden de los presupuestos y de sus componentes."""
        chunks: list[Chunk] = []
        for budget in budgets:
            budget_chunks = [self._build_chunk(budget, component) for component in budget.components]
            log.info(
                "budget_chunked",
                budget_id=budget.budget_id,
                chunks=len(budget_chunks),
                tokens=sum(c.token_count for c in budget_chunks),
                max_chunk_tokens=max((c.token_count for c in budget_chunks), default=0),
            )
            chunks.extend(budget_chunks)
        return chunks

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    @staticmethod
    def render_text(budget: Budget, component: BudgetComponent) -> str:
        """El texto que se embebe: cabecera del presupuesto padre + detalles del componente."""
        parent_context = PARENT_CONTEXT_TEMPLATE.format(
            project_summary=budget.project_summary,
            sector=budget.client_metadata.sector,
            year=budget.year,
            main_technology=budget.main_technology,
        )
        component_text = COMPONENT_TEMPLATE.format(
            name=component.name,
            description=component.description,
            tech_stack=", ".join(component.tech_stack) or "n/a",
            complexity=component.complexity,
            estimated_hours=component.estimated_hours,
        )
        return f"{parent_context}\n\n{component_text}"

    @staticmethod
    def build_metadata(budget: Budget, component: BudgetComponent) -> dict:
        """Campos filtrables; no se embeben. Pensados para `WHERE client_sector = ... AND year >= ...` en la sesión 08."""
        return {
            "budget_id": budget.budget_id,
            "component_id": component.component_id,
            "component_name": component.name,
            "client_sector": budget.client_metadata.sector,
            "client_country": budget.client_metadata.country,
            "main_technology": budget.main_technology,
            "year": budget.year,
            "complexity": component.complexity,
            "estimated_hours": component.estimated_hours,
            "tech_stack": list(component.tech_stack),
        }

    # --- Internos -----------------------------------------------------------------------------------

    def _build_chunk(self, budget: Budget, component: BudgetComponent) -> Chunk:
        text = self.render_text(budget, component)
        token_count = self.count_tokens(text)
        chunk_id = f"{budget.budget_id}::{component.component_id}"
        if token_count > self._oversized_tokens:
            log.warning("chunk_oversized", chunk_id=chunk_id, token_count=token_count, threshold=self._oversized_tokens)
        return Chunk(chunk_id=chunk_id, text=text, metadata=self.build_metadata(budget, component), token_count=token_count)
