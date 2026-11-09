"""Fixtures compartidas: configuración aislada (sin .env ni API keys), clientes de pruebas (sync y async) y almacén de sesiones limpio."""

import os

# The app's lifespan pre-loads the cross-encoder when RERANK_ENABLED=true in the real settings (.env), and
# every `TestClient(app)` context runs the lifespan. The tests never use the real model (they inject a fake
# scorer), so the preload is switched off before `app.main` is imported and reads its settings.
os.environ.setdefault("RERANK_PRELOAD", "false")

import httpx  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings, get_settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services.sessions import SessionStore, get_session_store  # noqa: E402
from tests.documentos import make_docx, make_pdf  # noqa: E402,F401  (reexportados para los tests)


def make_settings(**overrides) -> Settings:
    """Settings de prueba: no lee `.env` y no tiene API keys salvo que se indiquen."""
    values = dict(
        _env_file=None,
        llm_provider="openai",
        openai_api_key=None,
        anthropic_api_key=None,
        llm_temperature=0.2,
        llm_max_tokens=1234,
        prompt_version="v3",
        # La caché global del proceso y el summarizer por LLM se activan solo en los tests que los prueban:
        # así el resto sigue viendo exactamente una llamada al LLM por turno.
        response_cache_enabled=False,
        summary_mode="heuristic",
        # Session 10: the service defaults are hybrid + rerank; the tests pin the session 08 baseline and
        # switch each feature on explicitly where it is under test (tests/test_retrieval.py).
        search_mode="vector",
        rerank_enabled=False,
        rerank_preload=False,
    )
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def store() -> SessionStore:
    """Almacén de sesiones vacío para cada test (el real es un diccionario global del proceso)."""
    fresh = SessionStore()
    app.dependency_overrides[get_session_store] = lambda: fresh
    yield fresh
    app.dependency_overrides.pop(get_session_store, None)


@pytest.fixture
def client(settings: Settings, store: SessionStore):
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def anyio_backend():
    """Los tests `async` corren solo con asyncio (el plugin de anyio probaría también trio, que no está instalado)."""
    return "asyncio"


@pytest.fixture
async def async_client(settings: Settings, store: SessionStore):
    """`httpx.AsyncClient` contra la app ASGI en el mismo proceso, sin servidor."""
    app.dependency_overrides[get_settings] = lambda: settings
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as test_client:
        yield test_client
    app.dependency_overrides.clear()
