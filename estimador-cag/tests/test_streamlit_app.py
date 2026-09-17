"""Tests de la interfaz de chat (streamlit_app.py) con AppTest, sin llamadas reales al LLM."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.config import get_settings
from app.services import llm_service
from tests.test_streaming import TRANSCRIPCION, _fake_openai

APP = Path(__file__).resolve().parent.parent / "streamlit_app.py"


@pytest.fixture
def entorno_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_abre_una_interfaz_de_chat(entorno_openai):
    at = AppTest.from_file(str(APP), default_timeout=30).run()

    assert not at.exception
    assert at.title[0].value == "🧮 Estimador CAG"
    assert len(at.chat_input) == 1
    assert at.session_state["messages"] == []


def test_conversacion_con_streaming_y_historial(entorno_openai, monkeypatch):
    captured = {}
    monkeypatch.setattr(llm_service.openai, "OpenAI", _fake_openai(captured))
    at = AppTest.from_file(str(APP), default_timeout=30).run()

    at.chat_input[0].set_value(TRANSCRIPCION).run()

    assert not at.exception
    mensajes = at.session_state["messages"]
    assert [m["role"] for m in mensajes] == ["user", "assistant"]
    assert mensajes[0]["content"] == TRANSCRIPCION
    assert mensajes[1]["content"].startswith("## Estimación")
    assert mensajes[1]["meta"]["input_tokens"] == 100
    assert mensajes[1]["meta"]["output_tokens"] == 40
    assert mensajes[1]["meta"]["model"] == "gpt-4o-mini"
    # Lo que se envió al LLM: mismo system prompt (con los ejemplos) que el endpoint
    assert captured["request"]["stream"] is True
    assert "estimador de software" in captured["request"]["messages"][0]["content"].lower()
    # El historial se mantiene en pantalla: dos burbujas de chat
    assert len(at.chat_message) == 2
    assert [m.name for m in at.chat_message] == ["user", "assistant"]

    # Segundo turno: la conversación anterior viaja al modelo (historial activado por defecto)
    at.chat_input[0].set_value("Reduce el alcance a la mitad").run()

    assert not at.exception
    assert [m["role"] for m in at.session_state["messages"]] == ["user", "assistant", "user", "assistant"]
    assert [m["role"] for m in captured["request"]["messages"]] == ["system", "user", "assistant", "user"]
    assert len(at.chat_message) == 4


def test_sin_api_key_muestra_error_y_desactiva_el_chat(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    get_settings.cache_clear()
    try:
        at = AppTest.from_file(str(APP), default_timeout=30).run()
    finally:
        get_settings.cache_clear()

    assert not at.exception
    assert any("OPENAI_API_KEY" in e.value for e in at.error)
    assert at.chat_input[0].disabled
