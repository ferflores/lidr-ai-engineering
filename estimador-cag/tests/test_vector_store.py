"""Persistencia y búsqueda semántica (sesión 08) sin PostgreSQL y sin gastar créditos.

Tres bloques:

1. **Esquema** (`app/db/models.py` + la migración 0001): nombres de columna, dimensión del vector, cascada,
   índices presentes y —sobre todo— la **ausencia** del índice vectorial, que es una decisión del enunciado
   y no un olvido. Si alguien añade un HNSW "para ir más rápido", este test lo caza.
2. **Repositorio y endpoints** con un doble de `AsyncSession`: el orden de las operaciones de la transacción,
   el 409, el rollback cuando falla el embedder y el SQL que se genera para la búsqueda (compilado al
   dialecto de Postgres, sin base de datos).
3. **Scripts** (`query_examples.py`, `scripts/ingest_corpus.py`) contra un transporte HTTP simulado.

La comprobación de que el esquema real coincide con los modelos (`alembic check`) no se hace aquí porque
necesita una base viva: la hacen `scripts/verificar.sh` (render offline) y el CI con compose.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import OperationalError

from app.config import get_settings
from app.db.models import (
    CHUNK_TYPE_BUDGET_COMPONENT,
    EMBEDDING_DIMENSIONS,
    Chunk,
    Document,
)
from app.db.session import get_db_session
from app.embedding_pipeline.repository import (
    DocumentAlreadyIngested,
    build_search_statement,
    ingest_budget_document,
)
from app.embedding_pipeline.router import get_embedder
from app.embedding_pipeline.schemas import Budget, IngestResponse, SearchResponse
from app.main import app
from tests.conftest import make_settings
from tests.test_embedding_pipeline import STATEMENT_BUDGET, make_embedder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "budgets_sample.json"
MIGRATION_FILE = PROJECT_ROOT / "alembic" / "versions" / "0001_initial_schema.py"

SOURCE_PATH = "data/budgets_sample.json#BUD-2024-014"
INGEST_BODY = {"source_path": SOURCE_PATH, "document_type": "historical_budget", "content": STATEMENT_BUDGET}


# --- Doble de AsyncSession ---------------------------------------------------------------------------------


class FakeAsyncSession:
    """Lo mínimo de `AsyncSession` que usa el repositorio, registrando el orden de las operaciones.

    `events` es lo que de verdad se comprueba: el enunciado fija una secuencia (comprobar duplicado → crear
    el documento → trocear → embeber → `add_all` → commit) y lo que garantiza la atomicidad es que el commit
    sea el último paso, después de que el embedder haya respondido.
    """

    def __init__(self, existing_id: int | None = None, rows: list | None = None, next_id: int = 42) -> None:
        self.existing_id = existing_id
        self.rows = rows or []
        self.next_id = next_id
        self.events: list[str] = []
        self.added: list[object] = []
        self.added_all: list[object] = []
        self.statements: list[object] = []

    async def scalar(self, statement):
        self.events.append("scalar")
        self.statements.append(statement)
        return self.existing_id

    def add(self, instance) -> None:
        self.events.append("add")
        self.added.append(instance)

    def add_all(self, instances) -> None:
        self.events.append("add_all")
        self.added_all.extend(instances)

    async def flush(self) -> None:
        self.events.append("flush")
        for instance in self.added:
            if getattr(instance, "id", None) is None:
                instance.id = self.next_id

    async def execute(self, statement):
        self.events.append("execute")
        self.statements.append(statement)
        return SimpleNamespace(all=lambda: list(self.rows))

    async def commit(self) -> None:
        self.events.append("commit")

    async def rollback(self) -> None:
        self.events.append("rollback")


def make_row(chunk_id: int, distance: float, *, document_id: int = 7, name: str = "OAuth 2.0 authentication backend"):
    return SimpleNamespace(
        id=chunk_id,
        document_id=document_id,
        chunk_type=CHUNK_TYPE_BUDGET_COMPONENT,
        content=f"[Project: Mobile banking API] Component: {name}",
        chunk_metadata={"component_name": name, "client_sector": "finance", "year": 2024, "estimated_hours": 120, "complexity": "high"},
        distance=distance,
    )


@pytest.fixture
def api(request):
    """Cliente de pruebas con el embedder y la sesión de base de datos sustituidos por dobles.

    Se parametriza con `@pytest.mark.parametrize('api', [...], indirect=True)` pasando el `FakeAsyncSession`
    que toque; por defecto, una base vacía.
    """
    session = getattr(request, "param", None) or FakeAsyncSession()
    embedder, fake_api, _ = make_embedder()
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_db_session] = lambda: session
    app.dependency_overrides[get_settings] = lambda: make_settings(openai_api_key="sk-test")
    try:
        yield SimpleNamespace(client=TestClient(app), session=session, embedder=embedder, embeddings_api=fake_api)
    finally:
        app.dependency_overrides.clear()


# --- 1. Esquema --------------------------------------------------------------------------------------------


def test_metadata_columns_are_named_metadata_despite_the_python_attribute():
    """El enunciado fija la columna `metadata`; en SQLAlchemy ese nombre está reservado en las clases
    declarativas (`Base.metadata`), así que el atributo Python es otro y el mapeo es explícito."""
    assert Document.__table__.c.keys() == ["id", "source_path", "document_type", "ingested_at", "metadata"]
    assert Document.doc_metadata.property.columns[0].name == "metadata"
    assert Chunk.__table__.c.keys() == ["id", "document_id", "chunk_type", "content", "embedding", "metadata", "created_at"]
    assert Chunk.chunk_metadata.property.columns[0].name == "metadata"


def test_embedding_column_is_a_nullable_1536_vector():
    column = Chunk.__table__.c.embedding
    assert isinstance(column.type, Vector)
    assert column.type.dim == EMBEDDING_DIMENSIONS == 1536
    # nullable a propósito: deja insertar el chunk y rellenar el vector después (ingesta asíncrona).
    assert column.nullable is True


def test_migration_dimension_matches_the_models():
    """La migración duplica la constante (un histórico no se reescribe); aquí se comprueba que no divergen."""
    source = MIGRATION_FILE.read_text(encoding="utf-8")
    assert f"EMBEDDING_DIMENSIONS = {EMBEDDING_DIMENSIONS}" in source
    assert "CREATE EXTENSION IF NOT EXISTS vector" in source


def test_chunks_cascade_when_the_document_is_deleted():
    """Borrar un presupuesto borra sus chunks, y lo hace Postgres (la FK), no el ORM fila a fila."""
    foreign_key = next(iter(Chunk.__table__.c.document_id.foreign_keys))
    assert foreign_key.column is Document.__table__.c.id
    assert foreign_key.ondelete == "CASCADE"
    assert Document.chunks.property.passive_deletes is True


def test_expected_indexes_exist_and_metadata_uses_gin():
    assert {index.name for index in Document.__table__.indexes} == {"ix_documents_source_path"}
    assert {index.name for index in Chunk.__table__.indexes} == {
        "ix_chunks_document_id",
        "ix_chunks_chunk_type",
        "ix_chunks_metadata_gin",
    }
    gin = next(i for i in Chunk.__table__.indexes if i.name == "ix_chunks_metadata_gin")
    assert gin.dialect_options["postgresql"]["using"] == "gin"


def test_there_is_no_vector_index_on_purpose():
    """Decisión explícita del enunciado: sin índice, Postgres hace sequential scan y ese es el baseline
    contra el que la sesión en directo medirá un HNSW. Añadir el índice aquí sería perder la medida."""
    indexed_columns = {column.name for index in Chunk.__table__.indexes for column in index.columns}
    assert "embedding" not in indexed_columns
    # En la migración se busca la *creación* del índice, no la palabra: su docstring menciona HNSW para
    # explicar por qué no lo crea (el mismo patrón que usa scripts/verificar.sh).
    creations = re.findall(r"using\s*=?\s*[\"']?(hnsw|ivfflat)", MIGRATION_FILE.read_text(encoding="utf-8"), re.IGNORECASE)
    assert creations == []


# --- 2. Repositorio y endpoints ----------------------------------------------------------------------------


@pytest.mark.anyio
async def test_ingest_persists_document_and_chunks_in_one_transaction():
    session = FakeAsyncSession(next_id=42)
    embedder, fake_api, _ = make_embedder()
    budget = Budget.model_validate(json.loads(DATA_FILE.read_text())[0])
    from app.embedding_pipeline.chunker import JSONStructuralChunker

    outcome = await ingest_budget_document(
        session,
        source_path=SOURCE_PATH,
        document_type="historical_budget",
        budget=budget,
        chunker=JSONStructuralChunker(),
        embedder=embedder,
    )

    # El orden importa: el commit es lo último, después de que el embedder haya respondido.
    assert session.events == ["scalar", "add", "flush", "add_all", "commit"]
    assert outcome.document_id == 42
    assert outcome.chunks_created == len(budget.components) == len(session.added_all)
    # Un solo `embeddings.create` con todos los chunks, no uno por chunk.
    assert len(fake_api.calls) == 1 and len(fake_api.calls[0]) == outcome.chunks_created

    document = session.added[0]
    assert isinstance(document, Document)
    assert document.source_path == SOURCE_PATH and document.document_type == "historical_budget"
    assert document.doc_metadata["budget_id"] == budget.budget_id
    assert document.doc_metadata["client_sector"] == "finance"
    assert document.doc_metadata["components"] == len(budget.components)

    chunk = session.added_all[0]
    assert isinstance(chunk, Chunk)
    assert chunk.document_id == 42 and chunk.chunk_type == CHUNK_TYPE_BUDGET_COMPONENT
    assert chunk.content.startswith("[Project: ")
    assert len(chunk.embedding) == outcome.embedding_dimension
    # La metadata del chunk conserva su identidad lógica y el recuento de tokens.
    assert chunk.chunk_metadata["chunk_ref"] == f"{budget.budget_id}::{budget.components[0].component_id}"
    assert chunk.chunk_metadata["component_name"] == budget.components[0].name
    assert chunk.chunk_metadata["token_count"] > 0


@pytest.mark.anyio
async def test_ingest_raises_when_the_source_path_is_already_there():
    session = FakeAsyncSession(existing_id=11)
    embedder, fake_api, _ = make_embedder()
    from app.embedding_pipeline.chunker import JSONStructuralChunker

    with pytest.raises(DocumentAlreadyIngested) as exc_info:
        await ingest_budget_document(
            session,
            source_path=SOURCE_PATH,
            document_type="historical_budget",
            budget=Budget.model_validate(STATEMENT_BUDGET),
            chunker=JSONStructuralChunker(),
            embedder=embedder,
        )
    assert exc_info.value.document_id == 11
    # Nada se escribió y, sobre todo, no se pagó ninguna llamada de embeddings.
    assert session.events == ["scalar"] and fake_api.calls == []


def test_ingest_endpoint_returns_the_four_contract_fields(api):
    response = api.client.post("/embeddings/ingest", json=INGEST_BODY)
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"document_id", "chunks_created", "embedding_dimension", "ingestion_time_ms"}
    parsed = IngestResponse.model_validate(body)
    assert parsed.document_id == 42 and parsed.chunks_created == 1
    assert parsed.embedding_dimension == 8  # el doble devuelve vectores de 8 dimensiones
    assert parsed.ingestion_time_ms >= 0
    # Los vectores ya no viajan por HTTP: viven en chunks.embedding.
    assert "chunks" not in body and "embedding" not in body


@pytest.mark.parametrize("api", [FakeAsyncSession(existing_id=42)], indirect=True)
def test_ingest_endpoint_duplicate_is_409_with_detail_and_document_id(api):
    response = api.client.post("/embeddings/ingest", json=INGEST_BODY)
    assert response.status_code == 409
    # Los dos campos al mismo nivel, como el enunciado (por eso no se usa HTTPException, que anida en `detail`).
    assert response.json() == {"detail": "Document already ingested", "document_id": 42}


def test_ingest_endpoint_rolls_back_and_is_500_when_the_embedder_fails(api):
    def explode(*_args, **_kwargs):
        from openai import APIError

        raise APIError("boom", request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"), body=None)

    api.embedder.embed_many_with_usage = explode
    response = api.client.post("/embeddings/ingest", json=INGEST_BODY)
    assert response.status_code == 500
    # Lo que garantiza que no queda un documento huérfano: rollback y ningún commit.
    assert "rollback" in api.session.events and "commit" not in api.session.events


def test_ingest_endpoint_is_503_when_postgres_is_unreachable(api):
    async def unreachable(_statement):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    api.session.scalar = unreachable
    response = api.client.post("/embeddings/ingest", json=INGEST_BODY)
    assert response.status_code == 503
    assert "DATABASE_URL" in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="vacío"),
        pytest.param({"document_type": "historical_budget", "content": STATEMENT_BUDGET}, id="sin source_path"),
        pytest.param({"source_path": SOURCE_PATH, "content": STATEMENT_BUDGET}, id="sin document_type"),
        pytest.param({**INGEST_BODY, "budgets": [STATEMENT_BUDGET]}, id="campo desconocido"),
        pytest.param({**INGEST_BODY, "content": {**STATEMENT_BUDGET, "client_metadata": {"name": "n", "sector": "banking", "country": "ES"}}}, id="sector fuera del vocabulario"),
        pytest.param({**INGEST_BODY, "content": {**STATEMENT_BUDGET, "components": []}}, id="sin componentes"),
        pytest.param({**INGEST_BODY, "source_path": ""}, id="source_path vacío"),
    ],
)
def test_ingest_endpoint_validation_errors_are_422(api, body):
    assert api.client.post("/embeddings/ingest", json=body).status_code == 422


def test_ingest_endpoint_without_api_key_is_500_and_a_bad_body_is_still_422():
    """El mismo orden que se fijó en la sesión 07: Pydantic antes que la configuración."""
    session = FakeAsyncSession()
    app.dependency_overrides[get_settings] = lambda: make_settings(openai_api_key=None)
    app.dependency_overrides[get_db_session] = lambda: session
    try:
        client = TestClient(app)
        without_key = client.post("/embeddings/ingest", json=INGEST_BODY)
        invalid = client.post("/embeddings/ingest", json={})
    finally:
        app.dependency_overrides.clear()
    assert without_key.status_code == 500 and "OPENAI_API_KEY" in without_key.json()["detail"]
    assert invalid.status_code == 422
    assert session.events == [], "sin clave no se toca la base de datos"


# --- Búsqueda ----------------------------------------------------------------------------------------------


def test_search_statement_uses_the_cosine_operator_ordered_and_limited():
    """El SQL se compila al dialecto de Postgres y se inspecciona sin base de datos.

    `<=>` es el operador de distancia coseno de pgvector. Que aparezca en el ORDER BY es lo que, cuando el
    directo añada un índice HNSW con `vector_cosine_ops`, permitirá que Postgres lo use: si la consulta
    usara otro operador, ignoraría el índice en silencio y seguiría haciendo sequential scan.
    """
    sql = str(build_search_statement([0.1] * EMBEDDING_DIMENSIONS, 5).compile(dialect=postgresql.dialect()))
    assert sql.count("<=>") == 2  # proyección y ORDER BY
    assert "ORDER BY" in sql and "LIMIT" in sql
    assert "AS distance" in sql
    # Sin WHERE: los filtros por metadata son del directo, no de este ejercicio.
    assert "WHERE" not in sql.upper()


@pytest.mark.parametrize("api", [FakeAsyncSession(rows=[make_row(6, 0.4109), make_row(8, 0.4582, name="Accounts API")])], indirect=True)
def test_search_endpoint_returns_results_with_distances(api):
    response = api.client.post("/search", json={"query": "JWT authentication for fintech", "k": 2})
    assert response.status_code == 200, response.text
    body = SearchResponse.model_validate(response.json())
    assert body.query == "JWT authentication for fintech" and body.k == 2
    assert body.search_time_ms >= 0
    assert [r.chunk_id for r in body.results] == [6, 8]
    assert [round(r.distance, 4) for r in body.results] == [0.4109, 0.4582]
    assert body.results[0].chunk_type == CHUNK_TYPE_BUDGET_COMPONENT
    assert body.results[0].metadata["component_name"] == "OAuth 2.0 authentication backend"
    # Una sola llamada de embeddings, con la consulta como único input.
    assert api.embeddings_api.calls == [["JWT authentication for fintech"]]


def test_search_endpoint_defaults_to_k_5_and_returns_an_empty_list_on_an_empty_corpus(api):
    response = api.client.post("/search", json={"query": "cualquier cosa"})
    assert response.status_code == 200
    body = response.json()
    assert body["k"] == 5 and body["results"] == []


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="sin query"),
        pytest.param({"query": ""}, id="query vacía"),
        pytest.param({"query": "algo", "k": 0}, id="k=0"),
        pytest.param({"query": "algo", "k": 101}, id="k por encima del tope"),
        pytest.param({"query": "algo", "top_k": 5}, id="campo desconocido"),
    ],
)
def test_search_endpoint_validation_errors_are_422(api, body):
    assert api.client.post("/search", json=body).status_code == 422


def test_search_endpoint_is_503_when_postgres_is_unreachable(api):
    async def unreachable(_statement):
        raise OperationalError("SELECT ...", {}, Exception("connection refused"))

    api.session.execute = unreachable
    response = api.client.post("/search", json={"query": "algo"})
    assert response.status_code == 503


def test_both_endpoints_are_documented_in_openapi():
    schema = TestClient(app).get("/openapi.json").json()
    assert "post" in schema["paths"]["/embeddings/ingest"]
    assert "409" in schema["paths"]["/embeddings/ingest"]["post"]["responses"]
    assert "post" in schema["paths"]["/search"]
    assert "database" in schema["components"]["schemas"]["HealthResponse"]["properties"]


# --- 3. Scripts --------------------------------------------------------------------------------------------


def load_script(path: Path, module_name: str):
    """Carga un script suelto como módulo (necesario para `query_examples.py`, que está en la raíz)."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # `@dataclass` resuelve sus anotaciones mirando sys.modules
    try:
        spec.loader.exec_module(module)
        return module
    except Exception:
        sys.modules.pop(module_name, None)
        raise


@pytest.fixture
def query_examples():
    module = load_script(PROJECT_ROOT / "query_examples.py", "query_examples")
    yield module
    sys.modules.pop("query_examples", None)


@pytest.fixture
def ingest_corpus():
    module = load_script(PROJECT_ROOT / "scripts" / "ingest_corpus.py", "ingest_corpus")
    yield module
    sys.modules.pop("ingest_corpus", None)


def test_query_examples_covers_the_five_angles_of_the_statement(query_examples):
    angles = [q.angle for q in query_examples.EXAMPLE_QUERIES]
    assert angles == [
        "Componente directo conocido",
        "Reformulación semántica",
        "Dominio distinto",
        "Consulta ambigua",
        "Consulta muy específica",
    ]
    # Los textos literales que propone el enunciado.
    assert query_examples.EXAMPLE_QUERIES[0].text == "REST API development with JWT authentication for financial sector"
    assert query_examples.EXAMPLE_QUERIES[2].text == "mobile application for restaurant reservations"
    assert all(q.expectation for q in query_examples.EXAMPLE_QUERIES), "cada consulta dice qué habría que observar"


def test_query_examples_runs_one_search_per_query_and_formats_the_top_k(query_examples, monkeypatch):
    posted: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posted.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "query": json.loads(request.content)["query"],
                "k": 5,
                "search_time_ms": 87,
                "results": [
                    {
                        "chunk_id": 156,
                        "document_id": 12,
                        "chunk_type": "budget_component",
                        "content": "[Project: Mobile banking API]\n\nComponent: OAuth 2.0 authentication backend\nDescription: " + "x" * 300,
                        "distance": 0.2311,
                        "metadata": {"component_name": "OAuth 2.0 authentication backend", "client_sector": "finance", "year": 2024},
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    original_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original_client(**{**kwargs, "transport": transport}))

    report, failures = query_examples.run("http://test", k=5, timeout=5.0)

    assert failures == 0
    assert [p["query"] for p in posted] == [q.text for q in query_examples.EXAMPLE_QUERIES]
    assert all(p["k"] == 5 for p in posted)
    # Formato: chunk_id, distancia con 4 decimales, chunk_type y el contenido recortado en una línea.
    assert "#156" in report and "d=0.2311" in report and "budget_component" in report
    assert "…" in report, "el contenido largo se recorta"
    assert "\n" not in query_examples.preview("una\nlínea\npartida"), "el preview es una sola línea"
    assert len(query_examples.preview("x" * 500)) == query_examples.CONTENT_PREVIEW_CHARS
    # El cierre compara las cinco consultas entre sí.
    assert "LECTURA COMPARADA" in report
    for angle in (q.angle for q in query_examples.EXAMPLE_QUERIES):
        assert angle in report


def test_query_examples_reports_a_failed_search_without_crashing(query_examples, monkeypatch):
    transport = httpx.MockTransport(lambda request: httpx.Response(503, json={"detail": "sin base de datos"}))
    original_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original_client(**{**kwargs, "transport": transport}))

    report, failures = query_examples.run("http://test", k=5, timeout=5.0)
    assert failures == len(query_examples.EXAMPLE_QUERIES)
    assert "HTTP 503" in report


def test_ingest_corpus_posts_one_document_per_budget_and_counts_409s(ingest_corpus, monkeypatch, capsys):
    budgets = json.loads(DATA_FILE.read_text())
    posted: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        posted.append(body)
        if len(posted) == 2:  # el segundo ya estaba ingestado
            return httpx.Response(409, json={"detail": "Document already ingested", "document_id": 2})
        return httpx.Response(200, json={"document_id": len(posted), "chunks_created": 5, "embedding_dimension": 1536, "ingestion_time_ms": 300})

    transport = httpx.MockTransport(handler)
    original_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original_client(**{**kwargs, "transport": transport}))

    exit_code = ingest_corpus.main(["--api-base-url", "http://test"])
    assert exit_code == 0  # un 409 no es un fallo: el documento ya está

    assert len(posted) == len(budgets) == 15
    assert all(p["document_type"] == "historical_budget" for p in posted)
    # Un documento por presupuesto, con el source_path que lo identifica dentro del corpus.
    assert posted[0]["source_path"] == f"data/budgets_sample.json#{budgets[0]['budget_id']}"
    assert len({p["source_path"] for p in posted}) == 15
    assert posted[0]["content"]["budget_id"] == budgets[0]["budget_id"]
    out = capsys.readouterr().out
    assert "14 ingestado(s) · 1 ya estaba(n) · 0 fallo(s)" in out
