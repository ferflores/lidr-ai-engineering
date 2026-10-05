"""Tests del servicio LLM: plantilla renderizada, inyección de contexto (CAG) y estructura de mensajes por proveedor.

Los clientes de OpenAI y Anthropic se sustituyen por dobles: se comprueba qué
se les envía, sin hacer llamadas reales.
"""

import asyncio
from types import SimpleNamespace

import pytest

from app.context.examples import ESTIMATION_EXAMPLES
from app.schemas.estimation import EstimationRequest
from app.services import llm_service
from app.services.llm_service import (
    LLMConfigurationError,
    build_prompt,
    build_system_prompt,
    build_user_prompt,
    generate_estimation,
)
from app.services.prompts import PromptTemplateNotFound
from tests.conftest import make_settings

TRANSCRIPCION = "El cliente quiere una landing page con formulario de contacto e integración con HubSpot."
REQUEST = EstimationRequest(transcription=TRANSCRIPCION)


class _FakeAsyncClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


def _fake_async_openai(captured: dict, content: str = "## Estimación: Landing page\n**Total estimado: 80 horas**"):
    class FakeAsyncOpenAI(_FakeAsyncClient):
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        async def _create(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=40),
            )

    return FakeAsyncOpenAI


# --- Contexto estático y plantilla -----------------------------------------------------


def test_hay_al_menos_dos_ejemplos_de_contexto():
    assert len(ESTIMATION_EXAMPLES) >= 2
    for ejemplo in ESTIMATION_EXAMPLES:
        assert ejemplo["meeting_summary"].strip()
        assert ejemplo["estimation"].strip().startswith("## Estimación:")
        assert "Total estimado" in ejemplo["estimation"]


def test_system_prompt_inyecta_todos_los_ejemplos():
    prompt = build_system_prompt(settings=make_settings())

    assert "estimador de software" in prompt.lower()
    for ejemplo in ESTIMATION_EXAMPLES:
        titulo = ejemplo["estimation"].strip().splitlines()[0]
        assert titulo in prompt
        assert ejemplo["meeting_summary"] in prompt


def test_user_prompt_contiene_la_transcripcion_y_los_parametros():
    request = EstimationRequest(transcription="   El cliente quiere una app nueva para su negocio.   ", deadline_weeks=6)

    prompt = build_user_prompt(request, make_settings())

    assert "<transcripcion>\nEl cliente quiere una app nueva para su negocio.\n</transcripcion>" in prompt
    assert "Plazo objetivo: 6 semanas" in prompt


def test_el_prompt_no_esta_en_el_codigo():
    """El texto del prompt vive en prompts/, no en llm_service.py."""
    import inspect

    fuente = inspect.getsource(llm_service)
    assert "Eres un estimador" not in fuente
    assert "Formato de salida" not in fuente


def test_build_prompt_usa_la_version_pedida_o_la_configurada():
    settings = make_settings(prompt_version="v1")

    assert build_prompt(REQUEST, settings).template.id == "estimacion/v1"
    assert build_prompt(EstimationRequest(transcription=TRANSCRIPCION, prompt_version="v2"), settings).template.id == "estimacion/v2"


# --- Configuración ----------------------------------------------------------------------


def test_sin_api_key_lanza_error_de_configuracion():
    with pytest.raises(LLMConfigurationError, match="OPENAI_API_KEY"):
        asyncio.run(generate_estimation(REQUEST, make_settings()))

    with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
        asyncio.run(generate_estimation(REQUEST, make_settings(llm_provider="anthropic")))


def test_version_de_plantilla_inexistente_falla_antes_que_la_api_key():
    request = EstimationRequest(transcription=TRANSCRIPCION, prompt_version="v99")

    with pytest.raises(PromptTemplateNotFound, match="v99"):
        asyncio.run(generate_estimation(request, make_settings()))


def test_api_key_en_blanco_cuenta_como_ausente():
    assert make_settings(openai_api_key="   ").llm_configured is False
    assert make_settings(openai_api_key="sk-test").llm_configured is True


# --- OpenAI --------------------------------------------------------------------------------


def test_openai_recibe_system_con_ejemplos_y_user_con_parametros_y_transcripcion(monkeypatch):
    captured = {}
    monkeypatch.setattr(llm_service.openai, "AsyncOpenAI", _fake_async_openai(captured))
    request = EstimationRequest(transcription=TRANSCRIPCION, project_name="Landing HubSpot", hourly_rate_eur=60)

    result = asyncio.run(generate_estimation(request, make_settings(openai_api_key="sk-test")))

    assert captured["client"]["api_key"] == "sk-test"
    llamada = captured["request"]
    assert llamada["model"] == "gpt-4o-mini"
    assert llamada["temperature"] == 0.2
    assert llamada["max_completion_tokens"] == 1234
    assert [m["role"] for m in llamada["messages"]] == ["system", "user"]
    system, user = (m["content"] for m in llamada["messages"])
    assert ESTIMATION_EXAMPLES[0]["meeting_summary"] in system
    assert "Landing HubSpot" not in system  # los datos de la petición no van en el contexto fijo
    assert TRANSCRIPCION in user
    assert "Nombre del proyecto: Landing HubSpot" in user
    assert "Tarifa: 60 €/hora" in user

    assert result.provider == "openai"
    assert result.model == "gpt-4o-mini"
    assert result.estimation.startswith("## Estimación")
    assert result.usage.total_tokens == 140
    # 100 tokens a 0.15 $/M + 40 tokens a 0.60 $/M
    assert (result.cost.input_usd, result.cost.output_usd, result.cost.total_usd) == (0.000015, 0.000024, 0.000039)
    assert result.prompt.template == "estimacion"
    assert result.prompt.version == "v3"
    assert result.prompt.transcription_chars == len(TRANSCRIPCION)
    assert result.prompt.system_prompt_chars == len(system)
    assert result.prompt.user_prompt_chars == len(user)


def test_prompt_version_v1_envia_el_prompt_original(monkeypatch):
    captured = {}
    monkeypatch.setattr(llm_service.openai, "AsyncOpenAI", _fake_async_openai(captured))
    request = EstimationRequest(transcription=TRANSCRIPCION, prompt_version="v1")

    result = asyncio.run(generate_estimation(request, make_settings(openai_api_key="sk-test")))

    system, user = (m["content"] for m in captured["request"]["messages"])
    assert "Tarifa de referencia para el coste: 50 €/hora." in system  # v1 lleva la tarifa en el system prompt
    assert "## Parámetros" not in user
    assert user.startswith("Esta es la transcripción de la reunión con el cliente.")
    assert result.prompt.version == "v1"


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
    settings = make_settings(llm_provider="anthropic", anthropic_api_key="sk-ant-test")

    result = asyncio.run(generate_estimation(REQUEST, settings))

    assert captured["client"]["api_key"] == "sk-ant-test"
    llamada = captured["request"]
    assert llamada["model"] == "claude-haiku-4-5"
    assert llamada["max_tokens"] == 1234
    assert llamada["extra_body"] == {"temperature": 0.2}
    assert ESTIMATION_EXAMPLES[0]["meeting_summary"] in llamada["system"]
    assert llamada["messages"] == [{"role": "user", "content": build_user_prompt(REQUEST, settings)}]

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

    result = asyncio.run(generate_estimation(REQUEST, settings))

    assert result.cost.total_usd == 10.0
    assert result.cost.pricing.input_per_mtok == 2.0
