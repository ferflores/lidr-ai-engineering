"""Tests de la API HTTP (el LLM se sustituye por un doble para no gastar créditos)."""

import pytest

from app.routers import estimations
from app.schemas.estimation import EstimationRequest
from app.services.llm_service import EstimationResult, LLMProviderError, PromptInfo, TokenUsage
from app.services.pricing import ModelPricing, estimate_cost

TRANSCRIPCION = (
    "En la reunión con el equipo de marketing, el cliente explicó que necesita una landing page "
    "con formulario de contacto e integración con HubSpot."
)


def _fake_generate(captured: dict):
    async def fake_generate_estimation(request, settings=None):
        captured["request"] = request
        return EstimationResult(
            estimation="## Estimación: Landing page\n\n**Total estimado: 80 horas**",
            model="gpt-4o-mini",
            provider="openai",
            usage=TokenUsage(input_tokens=1200, output_tokens=350),
            cost=estimate_cost(1200, 350, ModelPricing(0.15, 0.60)),
            prompt=PromptInfo(
                template="estimacion",
                version=request.prompt_version or "v3",
                system_prompt_chars=6000,
                user_prompt_chars=500,
                transcription_chars=len(request.transcription),
            ),
        )

    return fake_generate_estimation


def test_health_responde_200(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["provider"] == "openai"
    assert body["model"] == "gpt-4o-mini"
    assert body["llm_configured"] is False
    assert body["prompt_template"] == "estimacion/v3"
    assert body["session_max_turns"] == 6
    assert body["active_sessions"] == 0


def test_swagger_y_openapi_disponibles(client):
    assert client.get("/docs").status_code == 200

    spec = client.get("/openapi.json").json()
    assert "post" in spec["paths"]["/api/v1/estimate"]
    assert "get" in spec["paths"]["/api/v1/prompts"]
    assert "get" in spec["paths"]["/health"]
    assert "post" in spec["paths"]["/api/v1/sessions"]
    assert "post" in spec["paths"]["/api/v1/sessions/{session_id}/estimate"]

    # El contrato de entrada está documentado: parámetros tipados, solo la transcripción obligatoria
    schema = spec["components"]["schemas"]["EstimationRequest"]
    assert set(schema["properties"]) == {
        "transcription", "project_name", "project_type", "hourly_rate_eur",
        "deadline_weeks", "estimation_mode", "detail_level", "prompt_version",
        "response_format",  # markdown | structured | both (sesión 04: salida estructurada)
    }
    assert schema["required"] == ["transcription"]
    assert schema["additionalProperties"] is False


def test_estimate_devuelve_la_estimacion(client, monkeypatch):
    captured = {}
    monkeypatch.setattr(estimations, "generate_estimation", _fake_generate(captured))

    response = client.post(
        "/api/v1/estimate",
        json={
            "transcription": TRANSCRIPCION,
            "project_name": "  Landing HubSpot  ",
            "project_type": "web",
            "hourly_rate_eur": 60,
            "deadline_weeks": 4,
            "estimation_mode": "equilibrada",
            "detail_level": "resumen",
        },
    )

    assert response.status_code == 200
    request = captured["request"]
    assert isinstance(request, EstimationRequest)
    assert request.transcription == TRANSCRIPCION
    assert request.project_name == "Landing HubSpot"
    assert request.project_type == "web"
    assert request.hourly_rate_eur == 60
    assert request.deadline_weeks == 4
    assert request.estimation_mode == "equilibrada"
    assert request.detail_level == "resumen"
    assert request.prompt_version is None

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
    assert body["prompt"] == {
        "template": "estimacion",
        "version": "v3",
        "system_prompt_chars": 6000,
        "user_prompt_chars": 500,
        "transcription_chars": len(TRANSCRIPCION),
    }
    assert body["generated_at"]


def test_estimate_solo_con_transcripcion_usa_los_valores_por_defecto(client, monkeypatch):
    captured = {}
    monkeypatch.setattr(estimations, "generate_estimation", _fake_generate(captured))

    response = client.post("/api/v1/estimate", json={"transcription": TRANSCRIPCION})

    assert response.status_code == 200
    request = captured["request"]
    assert request.project_name is None
    assert request.project_type is None
    assert request.hourly_rate_eur == 50
    assert request.deadline_weeks is None
    assert request.estimation_mode == "conservadora"
    assert request.detail_level == "completa"
    assert request.prompt_version is None


def test_estimate_respeta_la_version_de_plantilla_pedida(client, monkeypatch):
    captured = {}
    monkeypatch.setattr(estimations, "generate_estimation", _fake_generate(captured))

    response = client.post("/api/v1/estimate", json={"transcription": TRANSCRIPCION, "prompt_version": "v1"})

    assert response.status_code == 200
    assert captured["request"].prompt_version == "v1"
    assert response.json()["prompt"]["version"] == "v1"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"transcription": ""},
        {"transcription": "     corta     "},
        {"transcripcion": TRANSCRIPCION},                            # campo mal escrito
        {"transcription": TRANSCRIPCION, "extra": 1},                 # campo desconocido: se rechaza, no se ignora
        {"transcription": TRANSCRIPCION, "hourly_rate_eur": 0},
        {"transcription": TRANSCRIPCION, "hourly_rate_eur": 5000},
        {"transcription": TRANSCRIPCION, "deadline_weeks": 0},
        {"transcription": TRANSCRIPCION, "project_type": "videojuego"},
        {"transcription": TRANSCRIPCION, "estimation_mode": "agresiva"},
        {"transcription": TRANSCRIPCION, "detail_level": "larga"},
        {"transcription": TRANSCRIPCION, "prompt_version": "2"},
        {"transcription": TRANSCRIPCION, "project_name": "x" * 101},
    ],
)
def test_estimate_valida_el_body(client, payload):
    assert client.post("/api/v1/estimate", json=payload).status_code == 422


def test_estimate_version_de_plantilla_inexistente_devuelve_422(client):
    # Se comprueba antes que la API key: es un error de la petición, no de configuración
    response = client.post("/api/v1/estimate", json={"transcription": TRANSCRIPCION, "prompt_version": "v99"})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "v99" in detail
    assert "v1, v2" in detail


def test_estimate_sin_api_key_devuelve_500(client):
    response = client.post("/api/v1/estimate", json={"transcription": TRANSCRIPCION})

    assert response.status_code == 500
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_estimate_error_del_proveedor_devuelve_502(client, monkeypatch):
    async def fake_generate_estimation(request, settings=None):
        raise LLMProviderError("OpenAI ha devuelto un error de cuota")

    monkeypatch.setattr(estimations, "generate_estimation", fake_generate_estimation)

    response = client.post("/api/v1/estimate", json={"transcription": TRANSCRIPCION})

    assert response.status_code == 502
    assert "cuota" in response.json()["detail"]


def test_prompts_lista_las_versiones(client):
    response = client.get("/api/v1/prompts")

    assert response.status_code == 200
    body = response.json()
    assert body["default_version"] == "v3"
    por_version = {t["version"]: t for t in body["templates"]}
    assert list(por_version)[:3] == ["v1", "v2", "v3"]
    assert por_version["v1"]["is_default"] is False
    assert por_version["v2"]["is_default"] is False
    assert por_version["v3"]["is_default"] is True
    assert por_version["v3"]["created"] == "2026-10-05"
    assert por_version["v2"]["id"] == "estimacion/v2"
    assert por_version["v2"]["name"] == "estimacion"
    assert por_version["v2"]["created"] == "2026-09-28"
    assert por_version["v2"]["description"]
    assert len(por_version["v2"]["changelog"]) >= 1
