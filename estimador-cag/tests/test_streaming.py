"""Tests del streaming (EstimationStream) con clientes simulados."""

from types import SimpleNamespace

import pytest

from app.context.examples import ESTIMATION_EXAMPLES
from app.schemas.estimation import EstimationRequest
from app.services import llm_service
from app.services.llm_service import (
    EstimationStream,
    LLMConfigurationError,
    LLMProviderError,
    build_user_prompt,
)
from app.services.prompts import PromptTemplateNotFound
from tests.conftest import make_settings

TRANSCRIPCION = "El cliente quiere una landing page con formulario de contacto e integración con HubSpot."
REQUEST = EstimationRequest(transcription=TRANSCRIPCION)
FRAGMENTOS = ["## Estimación: ", "Landing page\n\n", "**Total estimado: 80 horas**"]


class _FakeSyncClient:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _fake_openai(captured: dict, fragmentos=FRAGMENTOS):
    class FakeOpenAI(_FakeSyncClient):
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            captured["request"] = kwargs
            for texto in fragmentos:
                yield SimpleNamespace(
                    usage=None,
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=texto), finish_reason=None)],
                )
            yield SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason="stop")])
            yield SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=40), choices=[])

    return FakeOpenAI


def _fake_anthropic(captured: dict, fragmentos=FRAGMENTOS, stop_reason="end_turn"):
    class FakeStream(_FakeSyncClient):
        text_stream = iter(fragmentos)

        def get_final_message(self):
            return SimpleNamespace(stop_reason=stop_reason, usage=SimpleNamespace(input_tokens=100, output_tokens=40))

    class FakeAnthropic(_FakeSyncClient):
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.messages = SimpleNamespace(stream=self._stream)

        def _stream(self, **kwargs):
            captured["request"] = kwargs
            return FakeStream()

    return FakeAnthropic


# --- OpenAI ----------------------------------------------------------------------


def test_streaming_openai_devuelve_fragmentos_y_resultado(monkeypatch):
    captured = {}
    monkeypatch.setattr(llm_service.openai, "OpenAI", _fake_openai(captured))
    request = EstimationRequest(transcription=TRANSCRIPCION, deadline_weeks=4, estimation_mode="equilibrada")

    stream = EstimationStream(request, make_settings(openai_api_key="sk-test"))
    assert stream.result is None
    assert stream.prompt is None
    fragmentos = list(stream)

    assert fragmentos == FRAGMENTOS
    llamada = captured["request"]
    assert llamada["stream"] is True
    assert llamada["stream_options"] == {"include_usage": True}
    assert llamada["model"] == "gpt-4o-mini"
    assert [m["role"] for m in llamada["messages"]] == ["system", "user"]
    assert ESTIMATION_EXAMPLES[0]["meeting_summary"] in llamada["messages"][0]["content"]
    assert TRANSCRIPCION in llamada["messages"][1]["content"]
    assert "Plazo objetivo: 4 semanas" in llamada["messages"][1]["content"]
    assert "Enfoque: equilibrada" in llamada["messages"][1]["content"]

    assert stream.prompt.template.id == "estimacion/v2"
    result = stream.result
    assert result.estimation == "".join(FRAGMENTOS)
    assert result.provider == "openai"
    assert result.usage.total_tokens == 140
    assert result.cost.total_usd == 0.000039
    assert result.prompt.version == "v2"
    assert result.elapsed_seconds is not None and result.elapsed_seconds >= 0


def test_streaming_respeta_la_version_de_plantilla(monkeypatch):
    captured = {}
    monkeypatch.setattr(llm_service.openai, "OpenAI", _fake_openai(captured))
    request = EstimationRequest(transcription=TRANSCRIPCION, prompt_version="v1")

    stream = EstimationStream(request, make_settings(openai_api_key="sk-test"))
    list(stream)

    assert stream.prompt.template.id == "estimacion/v1"
    assert "Tarifa de referencia para el coste: 50 €/hora." in captured["request"]["messages"][0]["content"]
    assert stream.result.prompt.version == "v1"


def test_streaming_respuesta_vacia_lanza_error(monkeypatch):
    monkeypatch.setattr(llm_service.openai, "OpenAI", _fake_openai({}, fragmentos=[]))

    with pytest.raises(LLMProviderError, match="vacía"):
        list(EstimationStream(REQUEST, make_settings(openai_api_key="sk-test")))


def test_streaming_sin_api_key_lanza_error_al_iterar():
    stream = EstimationStream(REQUEST, make_settings())

    with pytest.raises(LLMConfigurationError, match="OPENAI_API_KEY"):
        list(stream)


def test_streaming_version_inexistente_lanza_error_al_iterar():
    stream = EstimationStream(EstimationRequest(transcription=TRANSCRIPCION, prompt_version="v99"), make_settings())

    with pytest.raises(PromptTemplateNotFound, match="v99"):
        list(stream)


# --- Anthropic -------------------------------------------------------------------


def test_streaming_anthropic_devuelve_fragmentos_y_resultado(monkeypatch):
    captured = {}
    monkeypatch.setattr(llm_service.anthropic, "Anthropic", _fake_anthropic(captured))
    settings = make_settings(llm_provider="anthropic", anthropic_api_key="sk-ant-test")

    stream = EstimationStream(REQUEST, settings)
    fragmentos = list(stream)

    assert fragmentos == FRAGMENTOS
    llamada = captured["request"]
    assert llamada["model"] == "claude-haiku-4-5"
    assert llamada["max_tokens"] == 1234
    assert llamada["extra_body"] == {"temperature": 0.2}
    assert ESTIMATION_EXAMPLES[0]["meeting_summary"] in llamada["system"]
    assert llamada["messages"] == [{"role": "user", "content": build_user_prompt(REQUEST, settings)}]

    assert stream.result.provider == "anthropic"
    assert stream.result.usage.total_tokens == 140
    assert stream.result.cost.total_usd == 0.0003


def test_streaming_anthropic_rechazo_lanza_error(monkeypatch):
    monkeypatch.setattr(llm_service.anthropic, "Anthropic", _fake_anthropic({}, stop_reason="refusal"))
    settings = make_settings(llm_provider="anthropic", anthropic_api_key="sk-ant-test")

    with pytest.raises(LLMProviderError, match="rechazado"):
        list(EstimationStream(REQUEST, settings))
