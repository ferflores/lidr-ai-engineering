"""Fixtures compartidas: configuración aislada (sin .env ni API keys), clientes de pruebas (sync y async) y almacén de sesiones limpio."""

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.services.sessions import SessionStore, get_session_store
from tests.documentos import make_docx, make_pdf  # noqa: F401  (reexportados para los tests)


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
