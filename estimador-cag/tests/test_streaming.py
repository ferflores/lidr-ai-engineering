"""Tests del streaming (EstimationStream) y del historial de conversación, con clientes simulados."""

from types import SimpleNamespace

import pytest

from app.context.examples import ESTIMATION_EXAMPLES
from app.services import llm_service
from app.services.llm_service import (
    ChatTurn,
    EstimationStream,
    LLMConfigurationError,
    LLMProviderError,
    build_messages,
    build_user_prompt,
)
from tests.conftest import make_settings

TRANSCRIPCION = "El cliente quiere una landing page con formulario de contacto e integración con HubSpot."
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


# --- Historial -------------------------------------------------------------------


def test_build_messages_envuelve_solo_la_primera_transcripcion():
    mensajes = build_messages(
        "¿Y si quitamos el blog?",
        history=[ChatTurn("user", TRANSCRIPCION), ChatTurn("assistant", "## Estimación: Landing page")],
    )

    assert [m["role"] for m in mensajes] == ["user", "assistant", "user"]
    assert mensajes[0]["content"] == build_user_prompt(TRANSCRIPCION)
    assert mensajes[1]["content"] == "## Estimación: Landing page"
    assert mensajes[2]["content"] == "¿Y si quitamos el blog?"


def test_build_messages_sin_historial():
    assert build_messages(TRANSCRIPCION) == [{"role": "user", "content": build_user_prompt(TRANSCRIPCION)}]


# --- OpenAI ----------------------------------------------------------------------


def test_streaming_openai_devuelve_fragmentos_y_resultado(monkeypatch):
    captured = {}
    monkeypatch.setattr(llm_service.openai, "OpenAI", _fake_openai(captured))

    stream = EstimationStream(TRANSCRIPCION, make_settings(openai_api_key="sk-test"))
    assert stream.result is None
    fragmentos = list(stream)

    assert fragmentos == FRAGMENTOS
    request = captured["request"]
    assert request["stream"] is True
    assert request["stream_options"] == {"include_usage": True}
    assert request["model"] == "gpt-4o-mini"
    assert [m["role"] for m in request["messages"]] == ["system", "user"]
    assert ESTIMATION_EXAMPLES[0]["meeting_summary"] in request["messages"][0]["content"]
    assert TRANSCRIPCION in request["messages"][1]["content"]

    result = stream.result
    assert result.estimation == "".join(FRAGMENTOS)
    assert result.provider == "openai"
    assert result.usage.total_tokens == 140
    assert result.cost.total_usd == 0.000039
    assert result.elapsed_seconds is not None and result.elapsed_seconds >= 0


def test_streaming_con_historial_envia_la_conversacion(monkeypatch):
    captured = {}
    monkeypatch.setattr(llm_service.openai, "OpenAI", _fake_openai(captured))
    history = [ChatTurn("user", TRANSCRIPCION), ChatTurn("assistant", "## Estimación: Landing page")]

    list(EstimationStream("Reduce el alcance a 60 horas", make_settings(openai_api_key="sk-test"), history=history))

    roles = [m["role"] for m in captured["request"]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert "<transcripcion>" in captured["request"]["messages"][1]["content"]
    assert captured["request"]["messages"][3]["content"] == "Reduce el alcance a 60 horas"


def test_streaming_respuesta_vacia_lanza_error(monkeypatch):
    monkeypatch.setattr(llm_service.openai, "OpenAI", _fake_openai({}, fragmentos=[]))

    with pytest.raises(LLMProviderError, match="vacía"):
        list(EstimationStream(TRANSCRIPCION, make_settings(openai_api_key="sk-test")))


def test_streaming_sin_api_key_lanza_error_al_iterar():
    stream = EstimationStream(TRANSCRIPCION, make_settings())

    with pytest.raises(LLMConfigurationError, match="OPENAI_API_KEY"):
        list(stream)


# --- Anthropic -------------------------------------------------------------------


def test_streaming_anthropic_devuelve_fragmentos_y_resultado(monkeypatch):
    captured = {}
    monkeypatch.setattr(llm_service.anthropic, "Anthropic", _fake_anthropic(captured))
    settings = make_settings(llm_provider="anthropic", anthropic_api_key="sk-ant-test")

    stream = EstimationStream(TRANSCRIPCION, settings)
    fragmentos = list(stream)

    assert fragmentos == FRAGMENTOS
    request = captured["request"]
    assert request["model"] == "claude-haiku-4-5"
    assert request["max_tokens"] == 1234
    assert request["extra_body"] == {"temperature": 0.2}
    assert ESTIMATION_EXAMPLES[0]["meeting_summary"] in request["system"]
    assert request["messages"] == [{"role": "user", "content": build_user_prompt(TRANSCRIPCION)}]

    assert stream.result.provider == "anthropic"
    assert stream.result.usage.total_tokens == 140
    assert stream.result.cost.total_usd == 0.0003


def test_streaming_anthropic_rechazo_lanza_error(monkeypatch):
    monkeypatch.setattr(llm_service.anthropic, "Anthropic", _fake_anthropic({}, stop_reason="refusal"))
    settings = make_settings(llm_provider="anthropic", anthropic_api_key="sk-ant-test")

    with pytest.raises(LLMProviderError, match="rechazado"):
        list(EstimationStream(TRANSCRIPCION, settings))
