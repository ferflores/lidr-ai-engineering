"""Pipeline de embeddings (sesión 07) sin gastar créditos: esquemas, chunker, embedder con un doble de la API
y métricas de similitud. La persistencia, los endpoints y los scripts están en `tests/test_vector_store.py`.

El enunciado no exige tests; se añaden porque el proyecto los tiene desde la sesión 02 y porque las reglas
(lotes de 100, reintentos 1 s/2 s/4 s, formato del chunk, coste) son fáciles de romper sin darse cuenta.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import tiktoken
from openai import RateLimitError
from pydantic import ValidationError

from app.embedding_pipeline.chunker import OVERSIZED_CHUNK_TOKENS, JSONStructuralChunker
from app.embedding_pipeline.embedder import (
    EMBEDDING_PRICE_USD_PER_MTOK,
    RETRY_DELAYS_SECONDS,
    OpenAIEmbedder,
    estimate_cost_usd,
)
from app.embedding_pipeline.schemas import Budget, Chunk, IngestRequest
from app.embedding_pipeline.similarity import cosine_similarity, dot_product, euclidean_distance, l2_norm

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "budgets_sample.json"

# El presupuesto de ejemplo del enunciado, tal cual.
STATEMENT_BUDGET = {
    "budget_id": "BUD-2024-014",
    "client_metadata": {"name": "FintechCorp", "sector": "finance", "country": "ES"},
    "project_summary": "Mobile banking API with OAuth 2.0 authentication and PSD2 compliance",
    "main_technology": "ruby_on_rails",
    "year": 2024,
    "total_estimated_hours": 480,
    "components": [
        {
            "component_id": "AUTH-001",
            "name": "OAuth 2.0 authentication backend",
            "description": "Implementation of OAuth 2.0 flows (authorization code, refresh token) with JWT-based session management, multi-tenant token isolation, and rate limiting per client.",
            "tech_stack": ["ruby_on_rails", "postgresql", "redis"],
            "estimated_hours": 120,
            "complexity": "high",
            "dependencies": [],
        }
    ],
}

EXPECTED_CHUNK_TEXT = (
    "[Project: Mobile banking API with OAuth 2.0 authentication and PSD2 compliance]\n"
    "[Client sector: finance | Year: 2024 | Main tech: ruby_on_rails]\n"
    "\n"
    "Component: OAuth 2.0 authentication backend\n"
    "Description: Implementation of OAuth 2.0 flows (authorization code, refresh token) with JWT-based session management, multi-tenant token isolation, and rate limiting per client.\n"
    "Tech stack: ruby_on_rails, postgresql, redis\n"
    "Complexity: high\n"
    "Estimated hours: 120"
)


# --- Doble de la API de embeddings de OpenAI ---------------------------------------------------------------


def fake_vector(text: str, dims: int = 8) -> list[float]:
    """Vector determinista y normalizado a partir del texto (sirve para comprobar orden y similitud)."""
    raw = [((hash((text, i)) % 1000) / 1000.0) + 0.01 for i in range(dims)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


class FakeEmbeddingsAPI:
    def __init__(self, rate_limit_failures: int = 0) -> None:
        self.calls: list[list[str]] = []
        self._failures_left = rate_limit_failures

    def create(self, *, model: str, input: list[str], **_: object):
        self.calls.append(list(input))
        if self._failures_left > 0:
            self._failures_left -= 1
            request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
            raise RateLimitError("rate limited", response=httpx.Response(429, request=request), body=None)
        # Los índices llegan desordenados a propósito: el embedder debe reordenar por `index`.
        data = [SimpleNamespace(index=i, embedding=fake_vector(text), object="embedding") for i, text in enumerate(input)]
        tokens = sum(len(text.split()) for text in input)
        return SimpleNamespace(data=list(reversed(data)), model=model, usage=SimpleNamespace(prompt_tokens=tokens, total_tokens=tokens))


class FakeOpenAI:
    def __init__(self, rate_limit_failures: int = 0) -> None:
        self.embeddings = FakeEmbeddingsAPI(rate_limit_failures)


def make_embedder(rate_limit_failures: int = 0, **kwargs) -> tuple[OpenAIEmbedder, FakeEmbeddingsAPI, list[float]]:
    """Devuelve el embedder, el doble de `client.embeddings` (con `calls`) y la lista de esperas del backoff."""
    client = FakeOpenAI(rate_limit_failures)
    sleeps: list[float] = []
    embedder = OpenAIEmbedder(client=client, sleep=sleeps.append, **kwargs)
    return embedder, client.embeddings, sleeps


def make_chunks(n: int) -> list[Chunk]:
    return [Chunk(chunk_id=f"B::{i}", text=f"component number {i}", metadata={"i": i}, token_count=3) for i in range(n)]


# --- Esquemas ---------------------------------------------------------------------------------------------


def test_sample_dataset_validates_and_is_consistent():
    budgets = [Budget.model_validate(item) for item in json.loads(DATA_FILE.read_text())]
    assert len(budgets) == 15
    assert {b.client_metadata.sector for b in budgets} >= {"finance", "ecommerce", "healthcare", "industrial"}
    assert len({b.budget_id for b in budgets}) == 15, "budget_id repetido"
    for budget in budgets:
        assert budget.components_hours == budget.total_estimated_hours, budget.budget_id
        ids = {c.component_id for c in budget.components}
        for component in budget.components:
            assert set(component.dependencies) <= ids, f"{budget.budget_id}: dependencia desconocida en {component.component_id}"


def test_budget_rejects_unknown_sector_and_duplicate_components():
    bad_sector = {**STATEMENT_BUDGET, "client_metadata": {**STATEMENT_BUDGET["client_metadata"], "sector": "banking"}}
    with pytest.raises(ValidationError):
        Budget.model_validate(bad_sector)
    duplicated = {**STATEMENT_BUDGET, "components": STATEMENT_BUDGET["components"] * 2}
    with pytest.raises(ValidationError, match="duplicate component_id"):
        Budget.model_validate(duplicated)
    with pytest.raises(ValidationError):
        Budget.model_validate({**STATEMENT_BUDGET, "components": []})


def test_budget_ignores_extra_fields_but_ingest_request_forbids_them():
    budget = Budget.model_validate({**STATEMENT_BUDGET, "currency": "EUR"})
    assert budget.budget_id == "BUD-2024-014"
    with pytest.raises(ValidationError):
        IngestRequest.model_validate({"budgets": [STATEMENT_BUDGET], "document_type": "budget"})


# --- Chunker ------------------------------------------------------------------------------------------------


def test_chunker_renders_the_statement_format_exactly():
    budget = Budget.model_validate(STATEMENT_BUDGET)
    [chunk] = JSONStructuralChunker().chunk([budget])
    assert chunk.chunk_id == "BUD-2024-014::AUTH-001"
    assert chunk.text == EXPECTED_CHUNK_TEXT
    assert chunk.token_count == len(tiktoken.encoding_for_model("text-embedding-3-small").encode(EXPECTED_CHUNK_TEXT))
    assert chunk.token_count > 0
    for key in ("budget_id", "component_id", "client_sector", "main_technology", "year", "complexity", "estimated_hours"):
        assert key in chunk.metadata
    assert chunk.metadata["client_sector"] == "finance" and chunk.metadata["estimated_hours"] == 120


def test_chunker_one_chunk_per_component_in_order():
    budgets = [Budget.model_validate(item) for item in json.loads(DATA_FILE.read_text())]
    chunks = JSONStructuralChunker().chunk(budgets)
    assert len(chunks) == sum(len(b.components) for b in budgets)
    expected_ids = [f"{b.budget_id}::{c.component_id}" for b in budgets for c in b.components]
    assert [c.chunk_id for c in chunks] == expected_ids
    assert len(set(expected_ids)) == len(expected_ids)
    assert all(c.text.startswith("[Project: ") for c in chunks)
    # Sin overlap ni troceado. El sample tiene un componente con descripción anormalmente larga (PLC-001):
    # dobla al resto pero sigue cabiendo como un único chunk y por debajo del umbral de aviso. Es el caso
    # límite que el enunciado pide no trocear y llevar al directo.
    tokens = sorted(c.token_count for c in chunks)
    longest = max(chunks, key=lambda c: c.token_count)
    median = tokens[len(tokens) // 2]
    assert longest.chunk_id == "BUD-2024-030::PLC-001"
    assert longest.token_count > 2 * median
    assert longest.token_count < OVERSIZED_CHUNK_TOKENS


# --- Embedder ----------------------------------------------------------------------------------------------


def test_embed_many_batches_of_100_and_preserves_order():
    embedder, client, sleeps = make_embedder()
    chunks = make_chunks(250)
    embedded, usage = embedder.embed_many_with_usage(chunks)
    assert [len(call) for call in client.calls] == [100, 100, 50]
    assert [e.chunk_id for e in embedded] == [c.chunk_id for c in chunks]
    assert all(e.embedding == fake_vector(e.text) for e in embedded), "los vectores deben ir con su chunk (orden por index)"
    assert usage.batches == 3 and usage.prompt_tokens == 250 * 3 and usage.dimensions == 8
    assert usage.estimated_cost_usd == estimate_cost_usd(750)
    assert sleeps == []
    assert embedder.embed_many([]) == []


def test_embed_many_retries_rate_limit_with_exponential_backoff():
    embedder, client, sleeps = make_embedder(rate_limit_failures=3)
    embedded = embedder.embed_many(make_chunks(2))
    assert len(embedded) == 2
    assert sleeps == list(RETRY_DELAYS_SECONDS) == [1.0, 2.0, 4.0]
    assert len(client.calls) == 4  # 3 fallos + 1 acierto


def test_embed_many_gives_up_after_three_retries():
    embedder, client, sleeps = make_embedder(rate_limit_failures=4)
    with pytest.raises(RateLimitError):
        embedder.embed_many(make_chunks(1))
    assert sleeps == [1.0, 2.0, 4.0] and len(client.calls) == 4


def test_embed_one_and_cost_constant():
    embedder, client, _ = make_embedder()
    vector = embedder.embed_one("OAuth 2.0 authentication backend")
    assert vector == fake_vector("OAuth 2.0 authentication backend") and client.calls == [["OAuth 2.0 authentication backend"]]
    assert EMBEDDING_PRICE_USD_PER_MTOK == 0.02
    assert estimate_cost_usd(1_000_000) == 0.02
    assert estimate_cost_usd(15_000) == pytest.approx(0.0003)


def test_embedder_requires_key_or_client():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbedder(api_key=None)
    with pytest.raises(ValueError, match="batch_size"):
        OpenAIEmbedder(client=FakeOpenAI(), batch_size=0)


# --- Similitud ------------------------------------------------------------------------------------------------


def test_similarity_functions():
    a, b = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    assert cosine_similarity(a, a) == pytest.approx(1.0)
    assert cosine_similarity(a, b) == pytest.approx(0.0)
    assert cosine_similarity([2.0, 0.0], [5.0, 0.0]) == pytest.approx(1.0), "el coseno ignora la magnitud"
    assert dot_product(a, b) == 0.0 and euclidean_distance(a, b) == pytest.approx(math.sqrt(2))
    assert l2_norm(fake_vector("x")) == pytest.approx(1.0)
    # Para vectores normalizados: euclid² = 2 − 2·cos
    u, v = fake_vector("u"), fake_vector("v")
    assert euclidean_distance(u, v) ** 2 == pytest.approx(2 - 2 * cosine_similarity(u, v))
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        cosine_similarity([0.0, 0.0], [1.0, 1.0])


# Los tests del endpoint `POST /embeddings/ingest` y del script de comparación viven ahora en
# `tests/test_vector_store.py`: en la sesión 08 el endpoint persiste en PostgreSQL en vez de devolver los
# vectores, y `scripts/compare.py` lo sustituye `query_examples.py`. Este archivo se queda con lo que no
# cambió: esquemas, chunker, embedder y métricas de similitud.
