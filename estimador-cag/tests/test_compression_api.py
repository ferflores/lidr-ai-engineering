"""Integración de la compresión y la observación por turno (sesión 05 directo / pre-work 06) sobre la API, con el LLM sustituido por un doble.

Plantilla `v4`, summarizer por LLM, caché activa y ventana de 3 turnos: resumen acumulativo, anclas, tier,
aciertos de caché, el evento `turn_observed` (13 campos) emitido por structlog y expuesto por la API.
"""

import pytest
import structlog

from app.config import get_settings
from app.main import app
from app.services import llm_service, summarizer
from app.services.observability import TURN_OBSERVED_FIELDS
from app.services.response_cache import ResponseCache
from tests.conftest import make_settings
from tests.fakes import FakeEstimatorLLM

pytestmark = pytest.mark.anyio

MAX_TURNS = 3
TURNOS = [
    "El proyecto se llama Nimbus: plataforma web para gestionar una flota de 300 vehículos. Lo harán en React con PostgreSQL.",
    "Importante: el presupuesto es cerrado, 30.000 € como máximo para la primera fase.",
    "Necesitamos autenticación con Google mediante OAuth para todos los usuarios.",
    "Hay que añadir un audit log de todas las acciones de los usuarios.",
    "Piden notificaciones por WhatsApp al conductor cuando se le asigna una ruta.",
    "Novedad: el presupuesto sube a 80.000 € y quieren meter más alcance.",
]


@pytest.fixture
def cache() -> ResponseCache:
    return ResponseCache(semantic_threshold=0.9)


@pytest.fixture
def settings(cache):
    return make_settings(openai_api_key="sk-test", session_max_turns=MAX_TURNS, prompt_version="v4", summary_mode="llm", response_cache_enabled=True)


@pytest.fixture
def llm(monkeypatch, cache) -> FakeEstimatorLLM:
    fake = FakeEstimatorLLM()
    monkeypatch.setattr(llm_service.openai, "AsyncOpenAI", fake)
    monkeypatch.setattr("app.services.conversation.get_response_cache", lambda: cache)
    return fake


async def _nueva_sesion(client) -> str:
    response = await client.post("/api/v1/sessions")
    assert response.status_code == 201
    return response.json()["session_id"]


async def _turno(client, session_id, transcript, **data):
    response = await client.post(f"/api/v1/sessions/{session_id}/estimate", data={"transcript": transcript, **data})
    assert response.status_code == 200, response.text
    return response.json()


async def test_los_turnos_que_salen_de_la_ventana_se_resumen_y_las_anclas_sobreviven(async_client, llm):
    session_id = await _nueva_sesion(async_client)
    bodies = [await _turno(async_client, session_id, t, project_name="Nimbus" if i == 0 else None) for i, t in enumerate(TURNOS)]

    # Turnos 1-3 caben en la ventana: sin resumen y sin llamada al summarizer
    assert all(b["compression"]["summary"] == "" for b in bodies[:3])
    assert all(b["turn_observed"]["summarizer_called"] is False for b in bodies[:3])
    # El turno 4 expulsa al 1 y lo pliega en el resumen antes de llamar al LLM
    t4 = bodies[3]
    assert t4["turn_observed"]["summarizer_called"] is True and t4["turn_observed"]["summarizer_mode"] == "llm"
    assert t4["compression"]["summary"].startswith("Turno 1: El proyecto se llama Nimbus")
    assert t4["turn_observed"]["summary_chars"] == len(t4["compression"]["summary"]) > 0
    assert t4["turn_observed"]["summarizer_tokens_in"] > 0 and t4["turn_observed"]["summarizer_cost_usd"] > 0
    # ...y el coste y los tokens del turno incluyen al summarizer
    assert t4["turn_observed"]["tokens_in"] == t4["usage"]["input_tokens"] + t4["turn_observed"]["summarizer_tokens_in"]
    assert t4["turn_observed"]["cost_usd"] == pytest.approx(t4["usage"]["cost"]["total_usd"] + t4["turn_observed"]["summarizer_cost_usd"], abs=2e-6)

    # Las anclas: el presupuesto del turno 2 queda sustituido por el del 6 (el dato más reciente manda)
    anclas = {(a["kind"], a["text"]) for a in bodies[-1]["compression"]["anchors"]}
    assert ("name", "Proyecto: Nimbus") in anclas
    assert ("budget", "Presupuesto: 80.000 €") in anclas and not any(t == "Presupuesto: 30.000 €" for _, t in anclas)
    assert ("stack", "Stack: React") in anclas and ("stack", "Stack: WhatsApp") in anclas
    assert any(k == "requirement" and "audit log" in t for k, t in anclas)
    # El presupuesto antiguo solo sobrevive en el resumen (es lo que mide MemoryDriftMetric)
    assert "30.000 €" in bodies[-1]["compression"]["summary"]

    # El system prompt de la última llamada de estimación lleva los tres bloques, y el prefijo cacheable no cambia
    system_ultimo = llm.estimation_calls[-1]["messages"][0]["content"]
    system_primero = llm.estimation_calls[0]["messages"][0]["content"]
    assert "<anchors>\n- [turno 1] Proyecto: Nimbus" in system_ultimo
    assert "<summary>\nTurno 1: El proyecto se llama Nimbus" in system_ultimo
    assert system_primero.split("## Proyecto en curso")[0] == system_ultimo.split("## Proyecto en curso")[0]
    # La ventana sigue siendo de 3 turnos: nunca viajan más de 3 mensajes de usuario
    assert all(sum(m["role"] == "user" for m in call["messages"]) <= MAX_TURNS for call in llm.estimation_calls)
    # Y el summarizer se ha llamado una vez por cada turno expulsado (turnos 4, 5 y 6)
    assert len(llm.calls) - len(llm.estimation_calls) == 3

    estado = (await async_client.get(f"/api/v1/sessions/{session_id}")).json()
    assert estado["message_count"] == 2 * MAX_TURNS
    assert estado["anchors_count"] == len(bodies[-1]["compression"]["anchors"]) > 0
    assert estado["summary_chars"] == len(bodies[-1]["compression"]["summary"])
    assert estado["last_resolved_tier"] == "standard" and estado["last_tier_rule"] == "default"
    assert estado["last_turn_observed"] == bodies[-1]["turn_observed"]
    assert estado["compression"]["anchors"] == bodies[-1]["compression"]["anchors"]


async def test_turn_observed_tiene_los_13_campos_y_se_emite_por_structlog(async_client, llm):
    session_id = await _nueva_sesion(async_client)
    with structlog.testing.capture_logs() as logs:
        body = await _turno(async_client, session_id, TURNOS[0], project_name="Nimbus")

    observed = body["turn_observed"]
    assert list(observed)[: len(TURN_OBSERVED_FIELDS)] == list(TURN_OBSERVED_FIELDS)
    assert observed["turn_index"] == 1 and observed["session_id"] == session_id
    assert observed["enriched_transcript_chars"] == len(TURNOS[0]) and observed["attachments_total_chars"] == 0
    assert observed["messages_in_window"] == 2 and observed["anchors_count"] == 3 and observed["summary_chars"] == 0  # nombre + 2 de stack
    assert observed["tokens_in"] == body["usage"]["input_tokens"] and observed["tokens_out"] == body["usage"]["output_tokens"]
    assert observed["cost_usd"] == body["usage"]["cost"]["total_usd"] and observed["latency_ms"] >= 0
    assert observed["cache_hit_kind"] == "none" and observed["last_resolved_tier"] == "standard"
    assert observed["model"] == "gpt-4o-mini" and observed["prompt_version"] == "v4"

    eventos = [log for log in logs if log.get("event") == "turn_observed"]
    assert len(eventos) == 1
    assert all(eventos[0][campo] == observed[campo] for campo in TURN_OBSERVED_FIELDS)


async def test_el_mismo_prompt_acierta_en_la_cache_exacta_y_uno_casi_igual_en_la_semantica(async_client, llm, cache):
    a = await _nueva_sesion(async_client)
    b = await _nueva_sesion(async_client)
    c = await _nueva_sesion(async_client)
    primero = await _turno(async_client, a, TURNOS[0], project_name="Nimbus")
    repetido = await _turno(async_client, b, TURNOS[0], project_name="Nimbus")
    casi_igual = await _turno(async_client, c, TURNOS[0].rstrip(".") + "!", project_name="Nimbus")

    assert primero["cache_hit_kind"] == "none" and primero["usage"]["input_tokens"] > 0
    assert repetido["cache_hit_kind"] == "exact" and repetido["estimation"] == primero["estimation"]
    assert repetido["usage"] == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost": repetido["usage"]["cost"]}
    assert repetido["usage"]["cost"]["total_usd"] == 0 and repetido["turn_observed"]["cost_usd"] == 0
    assert casi_igual["cache_hit_kind"] == "semantic" and casi_igual["estimation"] == primero["estimation"]
    assert len(llm.calls) == 1  # una sola llamada real para las tres sesiones
    assert cache.stats() == {
        "entries": 1, "exact_hits": 1, "semantic_hits": 1, "misses": 1,
        "semantic_backend": "jaccard", "embedding_calls": 0, "embedding_errors": 0,
    }
    # La memoria se actualiza igual con una respuesta cacheada
    assert repetido["project_metadata"]["project_name"] == "Nimbus"


async def test_un_adjunto_grande_resuelve_el_tier_extended(async_client, llm):
    session_id = await _nueva_sesion(async_client)
    grande = ("Especificación con muchos requisitos. " * 500).encode()  # > 15.000 caracteres

    body = await async_client.post(
        f"/api/v1/sessions/{session_id}/estimate",
        data={"transcript": TURNOS[0]},
        files=[("attachments", ("spec.txt", grande, "text/plain"))],
    )
    assert body.status_code == 200
    body = body.json()
    assert body["tier"] == {"tier": "extended", "rule": "attachments_large", "model": "gpt-4o-mini", "max_tokens": 1234}
    assert body["turn_observed"]["last_resolved_tier"] == "extended" and body["turn_observed"]["last_tier_rule"] == "attachments_large"
    assert body["turn_observed"]["attachments_total_chars"] == body["attachments"][0]["chars"] > 15_000
    assert body["turn_observed"]["enriched_transcript_chars"] > body["turn_observed"]["attachments_total_chars"]
    assert llm.calls[0]["model"] == "gpt-4o-mini"


async def test_si_el_summarizer_falla_el_turno_sigue_con_resumen_heuristico(async_client, llm, monkeypatch):
    session_id = await _nueva_sesion(async_client)
    for t in TURNOS[:3]:
        await _turno(async_client, session_id, t)

    async def falla(*args, **kwargs):
        raise llm_service.LLMProviderError("cuota")

    monkeypatch.setattr(summarizer, "call_llm", falla)
    body = await _turno(async_client, session_id, TURNOS[3])

    assert body["turn"] == 4
    assert body["turn_observed"]["summarizer_mode"] == "llm-fallback" and body["turn_observed"]["summarizer_called"] is True
    assert body["compression"]["summary"].startswith("Turno 1 — cliente: El proyecto se llama Nimbus")
    assert body["turn_observed"]["summarizer_cost_usd"] == 0


async def test_con_la_plantilla_v3_el_resumen_y_las_anclas_se_calculan_pero_no_viajan(async_client, llm):
    app.dependency_overrides[get_settings] = lambda: make_settings(
        openai_api_key="sk-test", session_max_turns=1, prompt_version="v3", summary_mode="heuristic"
    )
    session_id = await _nueva_sesion(async_client)
    await _turno(async_client, session_id, TURNOS[0])
    body = await _turno(async_client, session_id, TURNOS[1])

    assert body["compression"]["summary_chars"] > 0 and body["compression"]["anchors_count"] > 0
    system = llm.calls[-1]["messages"][0]["content"]
    assert "<anchors>" not in system and "<summary>" not in system
    assert body["turn_observed"]["summarizer_mode"] == "heuristic" and body["turn_observed"]["summarizer_called"] is False
