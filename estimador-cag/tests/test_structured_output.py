"""Salida estructurada: el contrato de datos y el bucle de reparación, sin gastar créditos.

Lo que se prueba es lo que el material de la sesión 04 señala como el valor real de la técnica: que el
esquema garantiza la forma, que los validadores garantizan la aritmética, y que cuando algo de eso falla
el error vuelve al modelo en vez de llegar al cliente.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.estimation import EstimationRequest
from app.schemas.structured import EstimationItem, StructuredEstimation
from app.services import structured_output
from app.services.structured_output import MAX_ATTEMPTS, StructuredOutputError, generate_structured_estimation
from tests.conftest import make_settings

TRANSCRIPCION = (
    "El cliente quiere una landing page con formulario de contacto e integración con HubSpot. "
    "El diseño ya existe en Figma."
)

# Igual que en tests/test_sessions_api.py: los tests `async` de este módulo corren con anyio sobre asyncio.
pytestmark = pytest.mark.anyio


def estimacion_valida(**overrides) -> dict:
    base = {
        "title": "Landing page Acme",
        "scope_summary": "Landing con formulario e integración con HubSpot.",
        "items": [
            {"name": "Análisis y diseño", "hours": 10},
            {"name": "Formulario e integración", "hours": 40},
            {"name": "QA y despliegue", "hours": 10},
        ],
        "subtotal_hours": 60,
        "contingency_pct": 20,
        "total_hours": 72,
        "hourly_rate_eur": 50,
        "total_cost_eur": 3600,
        "team": "1 desarrollador full-stack",
        "duration_weeks_min": 3,
        "duration_weeks_max": 4,
        "deadline_weeks": 4,
        "deadline_feasible": True,
        "assumptions": ["El diseño en Figma es definitivo."],
        "risks": ["La API de HubSpot puede requerir ajustes."],
        "open_questions": ["¿Qué campos lleva el formulario?"],
        "confidence": "media",
    }
    base.update(overrides)
    return base


# --- El contrato de datos --------------------------------------------------------------------


def test_una_estimacion_coherente_valida():
    e = StructuredEstimation.model_validate(estimacion_valida())
    assert e.total_hours == 72
    assert e.items[1].hours == 40


@pytest.mark.parametrize(
    "overrides, mensaje",
    [
        ({"subtotal_hours": 100}, "subtotal_hours"),                      # los bloques no suman el subtotal
        ({"total_hours": 60}, "contingencia"),                             # el subtotal + contingencia no da el total
        ({"total_cost_eur": 9999}, "€/h"),                                 # el coste no es horas × tarifa
        ({"duration_weeks_min": 9, "duration_weeks_max": 2}, "menor que"),  # rango invertido
        ({"deadline_feasible": None}, "deadline_feasible"),                # hay plazo pero no se valora
    ],
)
def test_la_incoherencia_aritmetica_se_rechaza(overrides, mensaje):
    with pytest.raises(ValidationError) as exc:
        StructuredEstimation.model_validate(estimacion_valida(**overrides))
    assert mensaje in str(exc.value)


def test_campos_desconocidos_y_vacios_se_rechazan():
    with pytest.raises(ValidationError):
        StructuredEstimation.model_validate(estimacion_valida(sobra="x"))
    with pytest.raises(ValidationError):
        StructuredEstimation.model_validate(estimacion_valida(items=[]))
    with pytest.raises(ValidationError):
        StructuredEstimation.model_validate(estimacion_valida(confidence="altísima"))


def test_el_markdown_se_deriva_del_objeto():
    md = StructuredEstimation.model_validate(estimacion_valida()).to_markdown()
    assert md.startswith("## Estimación: Landing page Acme")
    assert "**Total estimado: 72 horas**" in md
    assert "3.600 €" in md.replace(",", ".")
    assert "4 semanas — viable" in md
    assert "¿Qué campos lleva el formulario?" in md


def test_sin_plazo_el_markdown_no_habla_de_plazo():
    md = StructuredEstimation.model_validate(
        estimacion_valida(deadline_weeks=None, deadline_feasible=None)
    ).to_markdown()
    assert "Plazo objetivo" not in md


def test_el_esquema_que_viaja_al_modelo_describe_los_campos():
    schema = StructuredEstimation.model_json_schema()
    assert schema["required"] and "total_hours" in schema["required"]
    assert schema["properties"]["confidence"]["enum"] == ["baja", "media", "alta"]


# --- El bucle de reparación -------------------------------------------------------------------


class FakeProvider:
    """Doble del proveedor: devuelve las respuestas que se le den, en orden, y recuerda los mensajes recibidos."""

    def __init__(self, *respuestas: dict):
        self.respuestas = list(respuestas)
        self.llamadas: list[list[dict]] = []

    async def __call__(self, system, mensajes, settings, api_key):
        self.llamadas.append([dict(m) for m in mensajes])
        return self.respuestas[min(len(self.llamadas) - 1, len(self.respuestas) - 1)], None


@pytest.fixture
def peticion() -> EstimationRequest:
    return EstimationRequest(transcription=TRANSCRIPCION, response_format="structured")


@pytest.fixture
def settings_con_key():
    return make_settings(openai_api_key="sk-test", prompt_version="v2")


async def test_valida_a_la_primera_no_reintenta(monkeypatch, peticion, settings_con_key):
    fake = FakeProvider(estimacion_valida())
    monkeypatch.setattr(structured_output, "_parse_openai", fake)

    result = await generate_structured_estimation(peticion, settings_con_key)

    assert result.attempts == 1
    assert result.repair_errors == ()
    assert result.estimation.total_hours == 72
    assert len(fake.llamadas) == 1


async def test_una_salida_incoherente_se_devuelve_al_modelo_para_que_la_corrija(
    monkeypatch, peticion, settings_con_key
):
    # El primer intento declara un coste que no cuadra con horas × tarifa; el segundo ya está bien.
    fake = FakeProvider(estimacion_valida(total_cost_eur=9999), estimacion_valida())
    monkeypatch.setattr(structured_output, "_parse_openai", fake)

    result = await generate_structured_estimation(peticion, settings_con_key)

    assert result.attempts == 2
    assert len(result.repair_errors) == 1
    assert "€" in result.repair_errors[0] or "total_cost_eur" in result.repair_errors[0]

    # Al reintentar se le enseña su propia respuesta y el error concreto, no se repite el prompt a secas.
    segunda = fake.llamadas[1]
    assert segunda[-2]["role"] == "assistant"
    assert segunda[-1]["role"] == "user"
    assert "no es válido" in segunda[-1]["content"]


async def test_si_nunca_valida_se_agota_y_falla(monkeypatch, peticion, settings_con_key):
    fake = FakeProvider(estimacion_valida(subtotal_hours=999))
    monkeypatch.setattr(structured_output, "_parse_openai", fake)

    with pytest.raises(StructuredOutputError, match=f"{MAX_ATTEMPTS} intentos"):
        await generate_structured_estimation(peticion, settings_con_key)

    assert len(fake.llamadas) == MAX_ATTEMPTS


async def test_el_modo_structured_pide_json_sin_tocar_la_plantilla(monkeypatch, peticion, settings_con_key):
    fake = FakeProvider(estimacion_valida())
    monkeypatch.setattr(structured_output, "_parse_openai", fake)

    await generate_structured_estimation(peticion, settings_con_key)

    mensaje_usuario = fake.llamadas[0][-1]["content"]
    assert "objeto JSON" in mensaje_usuario
    assert TRANSCRIPCION in mensaje_usuario  # la plantilla versionada sigue siendo la que manda


async def test_el_formato_no_entra_en_las_variables_de_la_plantilla():
    """Pedir JSON en vez de Markdown no debe cambiar el prompt renderizado: si cambiara, se invalidaría
    la caché de prefijo del proveedor en cada cambio de formato."""
    md = EstimationRequest(transcription=TRANSCRIPCION, response_format="markdown")
    js = EstimationRequest(transcription=TRANSCRIPCION, response_format="structured")
    assert md.prompt_variables() == js.prompt_variables()


async def test_anthropic_usa_el_esquema_en_output_config(monkeypatch, peticion):
    """La vía de Anthropic es `output_config.format`, no el obsoleto `output_format`."""
    capturado: dict = {}

    async def fake_parse(system, mensajes, settings, api_key):
        capturado["provider"] = settings.llm_provider
        return estimacion_valida(), None

    monkeypatch.setattr(structured_output, "_parse_anthropic", fake_parse)
    settings = make_settings(llm_provider="anthropic", anthropic_api_key="sk-ant-test", prompt_version="v2")

    result = await generate_structured_estimation(peticion, settings)

    assert capturado["provider"] == "anthropic"
    assert result.provider == "anthropic"
