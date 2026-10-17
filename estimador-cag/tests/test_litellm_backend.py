"""El backend de LiteLLM: que el desvío funcione y que normalice igual que los SDK, sin llamar a nadie."""

from __future__ import annotations

import pytest

from app.schemas.estimation import EstimationRequest
from app.services import litellm_backend
from app.services.litellm_backend import model_id
from app.services.llm_service import LLMProviderError, build_prompt, call_llm
from tests.conftest import make_settings

pytestmark = pytest.mark.anyio

TRANSCRIPCION = "El cliente quiere una landing page con formulario de contacto e integración con HubSpot."


def test_el_identificador_de_modelo_lleva_prefijo_solo_donde_hace_falta():
    assert model_id(make_settings(llm_provider="openai")) == "gpt-4o-mini"
    assert model_id(make_settings(llm_provider="anthropic")) == "anthropic/claude-haiku-4-5"
    # Si ya viene con proveedor, no se duplica
    assert model_id(make_settings(llm_provider="anthropic"), "anthropic/claude-haiku-4-5") == "anthropic/claude-haiku-4-5"
    # El tier puede pedir otro modelo y el prefijo se respeta
    assert model_id(make_settings(llm_provider="openai"), "gpt-4o") == "gpt-4o"


class FakeResponse:
    def __init__(self, texto="## Estimación: X", finish="stop", tokens=(1000, 200)):
        self.choices = [type("C", (), {"message": type("M", (), {"content": texto})(), "finish_reason": finish})()]
        self.usage = type("U", (), {"prompt_tokens": tokens[0], "completion_tokens": tokens[1]})()


async def test_el_backend_litellm_recibe_el_prompt_completo_y_normaliza_la_respuesta(monkeypatch):
    capturado: dict = {}

    async def fake_acompletion(**kwargs):
        capturado.update(kwargs)
        return FakeResponse()

    import litellm

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    settings = make_settings(llm_backend="litellm", openai_api_key="sk-test", prompt_version="v2")
    prompt = build_prompt(EstimationRequest(transcription=TRANSCRIPCION), settings)

    texto, usage = await call_llm(prompt, settings, "sk-test")

    assert texto == "## Estimación: X"
    assert (usage.input_tokens, usage.output_tokens) == (1000, 200)
    # El system prompt viaja como un mensaje más: es la convención de OpenAI para todos los proveedores
    assert capturado["messages"][0]["role"] == "system"
    assert capturado["messages"][-1]["role"] == "user"
    assert TRANSCRIPCION in capturado["messages"][-1]["content"]
    assert capturado["model"] == "gpt-4o-mini"
    assert capturado["num_retries"] == settings.litellm_retries


async def test_una_respuesta_vacia_se_traduce_a_error_del_proveedor(monkeypatch):
    import litellm

    async def fake_acompletion(**kwargs):
        return FakeResponse(texto="")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    settings = make_settings(llm_backend="litellm", openai_api_key="sk-test", prompt_version="v2")
    prompt = build_prompt(EstimationRequest(transcription=TRANSCRIPCION), settings)

    with pytest.raises(LLMProviderError, match="respuesta vacía"):
        await call_llm(prompt, settings, "sk-test")


async def test_cualquier_error_del_agregador_se_normaliza(monkeypatch):
    import litellm

    async def fake_acompletion(**kwargs):
        raise RuntimeError("se cayó el proveedor")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    settings = make_settings(llm_backend="litellm", openai_api_key="sk-test", prompt_version="v2")
    prompt = build_prompt(EstimationRequest(transcription=TRANSCRIPCION), settings)

    with pytest.raises(LLMProviderError, match="se cayó el proveedor"):
        await call_llm(prompt, settings, "sk-test")


async def test_con_el_backend_sdk_litellm_ni_se_importa(monkeypatch):
    """El backend por defecto no debe cargar las dependencias del agregador."""
    import litellm

    async def no_deberia_llamarse(**kwargs):
        raise AssertionError("con LLM_BACKEND=sdk no se debe pasar por LiteLLM")

    monkeypatch.setattr(litellm, "acompletion", no_deberia_llamarse)
    llamadas = []

    async def fake_openai(prompt, settings, api_key, model, max_tokens):
        llamadas.append(model)
        return "## Estimación", None

    monkeypatch.setattr("app.services.llm_service._call_openai", fake_openai)
    settings = make_settings(llm_backend="sdk", openai_api_key="sk-test", prompt_version="v2")
    prompt = build_prompt(EstimationRequest(transcription=TRANSCRIPCION), settings)

    await call_llm(prompt, settings, "sk-test")

    assert llamadas == ["gpt-4o-mini"]
