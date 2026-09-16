"""Fixtures compartidas: configuración aislada (sin .env ni API keys) y cliente de pruebas."""

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app


def make_settings(**overrides) -> Settings:
    """Settings de prueba: no lee `.env` y no tiene API keys salvo que se indiquen."""
    values = dict(
        _env_file=None,
        llm_provider="openai",
        openai_api_key=None,
        anthropic_api_key=None,
        llm_temperature=0.2,
        llm_max_tokens=1234,
    )
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def client(settings: Settings):
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
