"""Tests del servicio LLM: inyección de contexto (CAG) y estructura de mensajes por proveedor.

Los clientes de OpenAI y Anthropic se sustituyen por dobles: se comprueba qué
se les envía, sin hacer llamadas reales.
"""

import asyncio
from types import SimpleNamespace

import pytest

from app.context.examples import ESTIMATION_EXAMPLES
from app.services import llm_service
from app.services.llm_service import (
    LLMConfigurationError,
    build_system_prompt,
    build_user_prompt,
    generate_estimation,
)
from tests.conftest import make_settings

TRANSCRIPCION = "El cliente quiere una landing page con formulario de contacto e integración con HubSpot."


class _FakeAsyncClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


# --- Contexto estático ---------------------------------------------------------------


def test_hay_al_menos_dos_ejemplos_de_contexto():
    assert len(ESTIMATION_EXAMPLES) >= 2
    for ejemplo in ESTIMATION_EXAMPLES:
        assert ejemplo["meeting_summary"].strip()
        assert ejemplo["estimation"].strip().startswith("## Estimación:")
        assert "Total estimado" in ejemplo["estimation"]


def test_system_prompt_inyecta_todos_los_ejemplos():
    prompt = build_system_prompt()

    assert "estimador de software" in prompt.lower()
    for ejemplo in ESTIMATION_EXAMPLES:
        titulo = ejemplo["estimation"].strip().splitlines()[0]
        assert titulo in prompt
        assert ejemplo["meeting_summary"] in prompt


def test_user_prompt_contiene_la_transcripcion():
    prompt = build_user_prompt("   El cliente quiere una app.   ")

    assert "<transcripcion>\nEl cliente quiere una app.\n</transcripcion>" in prompt


# --- Configuración ----------------------------------------------------------------------


def test_sin_api_key_lanza_error_de_configuracion():
    with pytest.raises(LLMConfigurationError, match="OPENAI_API_KEY"):
        asyncio.run(generate_estimation(TRANSCRIPCION, make_settings()))

    with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
        asyncio.run(generate_estimation(TRANSCRIPCION, make_settings(llm_provider="anthropic")))


def test_api_key_en_blanco_cuenta_como_ausente():
    assert make_settings(openai_api_key="   ").llm_configured is False
    assert make_settings(openai_api_key="sk-test").llm_configured is True


# --- OpenAI --------------------------------------------------------------------------------


def test_openai_recibe_system_con_ejemplos_y_user_con_transcripcion(monkeypatch):
    captured = {}

    class FakeAsyncOpenAI(_FakeAsyncClient):
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        async def _create(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="## Estimación: Landing page\n**Total estimado: 80 horas**"),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=40),
            )

    monkeypatch.setattr(llm_service.openai, "AsyncOpenAI", FakeAsyncOpenAI)

    result = asyncio.run(generate_estimation(TRANSCRIPCION, make_settings(openai_api_key="sk-test")))

    assert captured["client"]["api_key"] == "sk-test"
    request = captured["request"]
    assert request["model"] == "gpt-4o-mini"
    assert request["temperature"] == 0.2
    assert request["max_completion_tokens"] == 1234
    assert [m["role"] for m in request["messages"]] == ["system", "user"]
    assert ESTIMATION_EXAMPLES[0]["meeting_summary"] in request["messages"][0]["content"]
    assert TRANSCRIPCION in request["messages"][1]["content"]

    assert result.provider == "openai"
    assert result.model == "gpt-4o-mini"
    assert result.estimation.startswith("## Estimación")
    assert result.usage.total_tokens == 140
    # 100 tokens a 0.15 $/M + 40 tokens a 0.60 $/M
    assert (result.cost.input_usd, result.cost.output_usd, result.cost.total_usd) == (0.000015, 0.000024, 0.000039)
    assert result.prompt.transcription_chars == len(TRANSCRIPCION)
    assert result.prompt.system_prompt_chars == len(request["messages"][0]["content"])


# --- Anthropic ------------------------------------------------------------------------------


def test_anthropic_recibe_system_con_ejemplos_y_user_con_transcripcion(monkeypatch):
    captured = {}

    class FakeAsyncAnthropic(_FakeAsyncClient):
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.messages = SimpleNamespace(create=self._create)

        async def _create(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="## Estimación: Landing page\n**Total estimado: 80 horas**")],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=100, output_tokens=40),
            )

    monkeypatch.setattr(llm_service.anthropic, "AsyncAnthropic", FakeAsyncAnthropic)

    result = asyncio.run(
        generate_estimation(TRANSCRIPCION, make_settings(llm_provider="anthropic", anthropic_api_key="sk-ant-test"))
    )

    assert captured["client"]["api_key"] == "sk-ant-test"
    request = captured["request"]
    assert request["model"] == "claude-haiku-4-5"
    assert request["max_tokens"] == 1234
    assert request["extra_body"] == {"temperature": 0.2}
    assert ESTIMATION_EXAMPLES[0]["meeting_summary"] in request["system"]
    assert request["messages"] == [{"role": "user", "content": build_user_prompt(TRANSCRIPCION)}]

    assert result.provider == "anthropic"
    assert result.model == "claude-haiku-4-5"
    assert result.estimation.startswith("## Estimación")
    assert result.usage.total_tokens == 140
    # 100 tokens a 1 $/M + 40 tokens a 5 $/M
    assert (result.cost.input_usd, result.cost.output_usd, result.cost.total_usd) == (0.0001, 0.0002, 0.0003)


def test_precio_sobrescrito_desde_configuracion(monkeypatch):
    class FakeAsyncOpenAI(_FakeAsyncClient):
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        async def _create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="## Estimación: X"), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=1_000_000),
            )

    monkeypatch.setattr(llm_service.openai, "AsyncOpenAI", FakeAsyncOpenAI)
    settings = make_settings(openai_api_key="sk-test", llm_input_price_per_mtok=2.0, llm_output_price_per_mtok=8.0)

    result = asyncio.run(generate_estimation(TRANSCRIPCION, settings))

    assert result.cost.total_usd == 10.0
    assert result.cost.pricing.input_per_mtok == 2.0
