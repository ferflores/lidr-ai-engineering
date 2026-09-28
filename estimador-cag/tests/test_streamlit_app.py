"""Tests del formulario (streamlit_app.py) con AppTest, sin llamadas reales al LLM."""

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
    monkeypatch.delenv("PROMPT_VERSION", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _submit(at: AppTest):
    """El botón de envío del formulario."""
    return next(b for b in at.button if b.label == "Generar estimación")


def test_abre_un_formulario_en_vez_de_un_chat(entorno_openai):
    at = AppTest.from_file(str(APP), default_timeout=30).run()

    assert not at.exception
    assert at.title[0].value == "🧮 Estimador CAG"
    assert len(at.chat_input) == 0
    assert len(at.chat_message) == 0
    assert at.text_area(key="transcription").value == ""
    assert not _submit(at).disabled
    assert at.session_state["last"] is None
    # La plantilla por defecto aparece seleccionada en la barra lateral
    assert at.selectbox(key="prompt_version").value == "v2"


def test_envia_el_formulario_con_parametros_y_streaming(entorno_openai, monkeypatch):
    captured = {}
    monkeypatch.setattr(llm_service.openai, "OpenAI", _fake_openai(captured))
    at = AppTest.from_file(str(APP), default_timeout=30).run()

    at.text_area(key="transcription").set_value(TRANSCRIPCION)
    at.text_input[0].set_value("Landing HubSpot")
    at.number_input[0].set_value(60.0)  # tarifa
    at.number_input[1].set_value(4)     # plazo
    _submit(at).click().run()

    assert not at.exception
    last = at.session_state["last"]
    assert last["request"]["transcription"] == TRANSCRIPCION
    assert last["request"]["project_name"] == "Landing HubSpot"
    assert last["request"]["hourly_rate_eur"] == 60
    assert last["request"]["deadline_weeks"] == 4
    assert last["request"]["estimation_mode"] == "conservadora"
    assert last["request"]["detail_level"] == "completa"
    assert last["request"]["prompt_version"] == "v2"
    assert last["estimation"].startswith("## Estimación")
    assert last["meta"]["input_tokens"] == 100
    assert last["meta"]["output_tokens"] == 40
    assert last["meta"]["model"] == "gpt-4o-mini"
    assert last["meta"]["template"] == "estimacion/v2"

    # Lo que se envió al LLM: la plantilla v2 con los parámetros del formulario
    llamada = captured["request"]
    assert llamada["stream"] is True
    assert [m["role"] for m in llamada["messages"]] == ["system", "user"]
    assert "estimador de software" in llamada["messages"][0]["content"].lower()
    user = llamada["messages"][1]["content"]
    assert "Nombre del proyecto: Landing HubSpot" in user
    assert "Tarifa: 60 €/hora" in user
    assert "Plazo objetivo: 4 semanas" in user
    assert TRANSCRIPCION in user

    # Métricas en la barra lateral
    etiquetas = [m.label for m in at.metric]
    assert "Tokens de entrada" in etiquetas
    assert "Coste" in etiquetas


def test_transcripcion_corta_muestra_error_y_no_llama_al_llm(entorno_openai, monkeypatch):
    captured = {}
    monkeypatch.setattr(llm_service.openai, "OpenAI", _fake_openai(captured))
    at = AppTest.from_file(str(APP), default_timeout=30).run()

    at.text_area(key="transcription").set_value("muy corta")
    _submit(at).click().run()

    assert not at.exception
    assert any("Transcripción" in e.value for e in at.error)
    assert "request" not in captured
    assert at.session_state["last"] is None


def test_cargar_un_ejemplo_rellena_la_transcripcion(entorno_openai):
    at = AppTest.from_file(str(APP), default_timeout=30).run()

    at.selectbox(key="sample").select("Panadería: descripción pobre").run()

    assert not at.exception
    assert len(at.session_state["transcription"]) >= 20
    assert at.text_area(key="transcription").value == at.session_state["transcription"]


def test_sin_api_key_muestra_error_y_desactiva_el_envio(monkeypatch):
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
    assert _submit(at).disabled
