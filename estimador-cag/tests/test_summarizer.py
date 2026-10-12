"""Resumen acumulativo: extracción de lo que dijo el cliente, modo heurístico, modo LLM (con doble) y su fallback."""

import pytest

from app.services import llm_service
from app.services.sessions import Turn
from app.services.summarizer import client_text, estimation_excerpt, fold_into_summary, heuristic_summary
from tests.conftest import make_settings
from tests.fakes import FakeEstimatorLLM, estimacion_falsa

pytestmark = pytest.mark.anyio

USER_MESSAGE = (
    "## Parámetros\n- Tarifa: 50 €/hora\n\n## Transcripción de la reunión\n<transcripcion>\n"
    "El cliente fija un presupuesto de 30.000 €.\n\n--- attachment: spec.pdf ---\n" + "x" * 2000 + "\n--- fin del adjunto: spec.pdf ---\n"
    "</transcripcion>\n\n## Recordatorios para esta petición\n- Comprueba la aritmética."
)


def _turn(index: int, user: str = USER_MESSAGE) -> Turn:
    turn = Turn(user=user, assistant=estimacion_falsa("Faro", con_pagos=False))
    turn.index = index
    return turn


def test_client_text_se_queda_con_la_transcripcion_y_recorta_los_adjuntos():
    text = client_text(USER_MESSAGE)

    assert text.startswith("El cliente fija un presupuesto de 30.000 €.")
    assert "[adjunto spec.pdf, 2.000 caracteres]" in text
    assert len(text) < 1800 and "Recordatorios" not in text and "Tarifa" not in text


def test_estimation_excerpt_conserva_titulo_alcance_y_totales():
    excerpt = estimation_excerpt(estimacion_falsa("Faro", con_pagos=True))

    assert excerpt.startswith("## Estimación: Faro | Tienda online para una panadería artesanal con pasarela de pago Stripe.")
    assert "**Total estimado: 108 horas**" in excerpt and "Equipo recomendado" in excerpt
    assert "Desglose" not in excerpt
    assert estimation_excerpt(None) == "(sin estimación)"


def test_resumen_heuristico_acumula_y_pierde_lo_mas_antiguo_cuando_no_cabe():
    primero = heuristic_summary("", [_turn(1)], max_chars=1500)
    assert primero.startswith("Turno 1 — cliente: El cliente fija un presupuesto de 30.000 €.")
    assert "→ estimación: ## Estimación: Faro" in primero

    acumulado = heuristic_summary(primero, [_turn(2), _turn(3)], max_chars=5000)
    assert acumulado.count("Turno ") == 3 and acumulado.startswith(primero)

    # Con el tope por defecto (1.500) no caben tres entradas de ~900 caracteres: se pierde lo más antiguo
    recortado = heuristic_summary(acumulado, [_turn(4)], max_chars=1500)
    assert "Turno 1" not in recortado and "Turno 4" in recortado and len(recortado) <= 1500

    minimo = heuristic_summary("", [_turn(5)], max_chars=200)
    assert len(minimo) == 200 and minimo.endswith("…")


async def test_modo_off_y_heuristico_no_llaman_al_llm():
    off = await fold_into_summary("antes", [_turn(1)], make_settings(summary_mode="off"), api_key="sk")
    assert off.summary == "antes" and off.mode == "off" and off.called_llm is False

    heuristico = await fold_into_summary("", [_turn(1)], make_settings(summary_mode="heuristic"), api_key="sk")
    assert heuristico.mode == "heuristic" and heuristico.called_llm is False and heuristico.usage is None
    assert heuristico.summary.startswith("Turno 1 — cliente:")

    sin_turnos = await fold_into_summary("antes", [], make_settings(summary_mode="llm"), api_key="sk")
    assert sin_turnos.summary == "antes" and sin_turnos.mode == "off"


async def test_modo_llm_usa_la_plantilla_resumen_y_mide_tokens_coste_y_latencia(monkeypatch):
    fake = FakeEstimatorLLM()
    monkeypatch.setattr(llm_service.openai, "AsyncOpenAI", fake)

    update = await fold_into_summary("Resumen previo.", [_turn(1), _turn(2)], make_settings(summary_mode="llm", openai_api_key="sk-test"), api_key="sk-test")

    assert update.mode == "llm" and update.called_llm is True
    assert update.summary.startswith("Resumen previo. || Turno 1: El cliente fija un presupuesto de 30.000 €.")
    assert "Turno 2:" in update.summary
    assert update.usage is not None and update.usage.input_tokens > 0 and update.cost is not None and update.cost.total_usd > 0
    assert update.latency_ms >= 0
    system = fake.calls[0]["messages"][0]["content"]
    assert "resumen acumulativo" in system and "1500 caracteres" in system
    assert fake.calls[0]["max_completion_tokens"] == 1024
    assert "## Turnos que salen de la ventana" in fake.calls[0]["messages"][-1]["content"]


async def test_si_el_llm_falla_se_degrada_al_resumen_heuristico(monkeypatch):
    async def falla(*args, **kwargs):
        raise llm_service.LLMProviderError("cuota")

    monkeypatch.setattr("app.services.summarizer.call_llm", falla)
    update = await fold_into_summary("", [_turn(1)], make_settings(summary_mode="llm"), api_key="sk-test")

    assert update.mode == "llm-fallback" and update.called_llm is True and update.usage is None
    assert update.summary.startswith("Turno 1 — cliente:")
