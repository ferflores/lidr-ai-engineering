"""Tests de la API HTTP (el LLM se sustituye por un doble para no gastar créditos)."""

import pytest

from app.routers import estimations
from app.services.llm_service import EstimationResult, LLMProviderError, PromptInfo, TokenUsage
from app.services.pricing import ModelPricing, estimate_cost

TRANSCRIPCION = (
    "En la reunión con el equipo de marketing, el cliente explicó que necesita una landing page "
    "con formulario de contacto e integración con HubSpot."
)


def test_health_responde_200(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["provider"] == "openai"
    assert body["model"] == "gpt-4o-mini"
    assert body["llm_configured"] is False


def test_swagger_y_openapi_disponibles(client):
    assert client.get("/docs").status_code == 200

    spec = client.get("/openapi.json").json()
    assert "post" in spec["paths"]["/api/v1/estimate"]
    assert "get" in spec["paths"]["/health"]


def test_estimate_devuelve_la_estimacion(client, monkeypatch):
    async def fake_generate_estimation(transcription, settings=None):
        assert "landing page" in transcription
        return EstimationResult(
            estimation="## Estimación: Landing page\n\n**Total estimado: 80 horas**",
            model="gpt-4o-mini",
            provider="openai",
            usage=TokenUsage(input_tokens=1200, output_tokens=350),
            cost=estimate_cost(1200, 350, ModelPricing(0.15, 0.60)),
            prompt=PromptInfo(system_prompt_chars=6000, transcription_chars=len(transcription)),
        )

    monkeypatch.setattr(estimations, "generate_estimation", fake_generate_estimation)

    response = client.post("/api/v1/estimate", json={"transcription": TRANSCRIPCION})

    assert response.status_code == 200
    body = response.json()
    assert body["estimation"].startswith("## Estimación")
    assert body["model"] == "gpt-4o-mini"
    assert body["provider"] == "openai"
    assert body["usage"] == {
        "input_tokens": 1200,
        "output_tokens": 350,
        "total_tokens": 1550,
        "cost": {
            "input_usd": 0.00018,
            "output_usd": 0.00021,
            "total_usd": 0.00039,
            "input_price_per_mtok": 0.15,
            "output_price_per_mtok": 0.60,
            "currency": "USD",
        },
    }
    assert body["prompt"] == {"system_prompt_chars": 6000, "transcription_chars": len(TRANSCRIPCION)}
    assert body["generated_at"]


@pytest.mark.parametrize(
    "payload",
    [{}, {"transcription": ""}, {"transcription": "     corta     "}, {"transcripcion": TRANSCRIPCION}],
)
def test_estimate_valida_el_body(client, payload):
    assert client.post("/api/v1/estimate", json=payload).status_code == 422


def test_estimate_sin_api_key_devuelve_500(client):
    response = client.post("/api/v1/estimate", json={"transcription": TRANSCRIPCION})

    assert response.status_code == 500
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_estimate_error_del_proveedor_devuelve_502(client, monkeypatch):
    async def fake_generate_estimation(transcription, settings=None):
        raise LLMProviderError("OpenAI ha devuelto un error de cuota")

    monkeypatch.setattr(estimations, "generate_estimation", fake_generate_estimation)

    response = client.post("/api/v1/estimate", json={"transcription": TRANSCRIPCION})

    assert response.status_code == 502
    assert "cuota" in response.json()["detail"]
