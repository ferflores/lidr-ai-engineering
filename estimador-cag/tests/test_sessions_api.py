"""Tests de integración de las sesiones con `httpx.AsyncClient` contra la app ASGI (LLM sustituido por un doble).

Los tres del enunciado (memoria entre turnos, PDF que influye en la estimación, ventana deslizante con
8 turnos) más los errores HTTP del endpoint multipart.
"""

import uuid

import pytest

from app.config import get_settings
from app.main import app
from app.services import conversation, llm_service
from app.services.llm_service import LLMProviderError
from tests.conftest import make_settings
from tests.documentos import make_pdf
from tests.fakes import FakeEstimatorLLM

pytestmark = pytest.mark.anyio

MAX_TURNS = 3
TRANSCRIPCION_1 = (
    "El cliente, una panadería artesanal, quiere vender por internet con recogida en tienda y reparto local. "
    "Tienen el catálogo en Excel."
)
TRANSCRIPCION_2 = "Confirman que el reparto será solo en el barrio y añaden pedidos recurrentes semanales. Pagarán con Stripe."


@pytest.fixture
def settings():
    """Sustituye la fixture de conftest: con API key (para llegar al LLM falso) y una ventana pequeña."""
    return make_settings(openai_api_key="sk-test", session_max_turns=MAX_TURNS)


@pytest.fixture
def llm(monkeypatch) -> FakeEstimatorLLM:
    fake = FakeEstimatorLLM()
    monkeypatch.setattr(llm_service.openai, "AsyncOpenAI", fake)
    return fake


async def _nueva_sesion(client) -> str:
    response = await client.post("/api/v1/sessions")
    assert response.status_code == 201
    return response.json()["session_id"]


def _roles(messages: list[dict]) -> list[str]:
    return [m["role"] for m in messages]


# --- Crear y consultar sesiones ----------------------------------------------------------------


async def test_post_sessions_crea_una_sesion_vacia(async_client):
    response = await async_client.post("/api/v1/sessions")

    assert response.status_code == 201
    body = response.json()
    assert uuid.UUID(body["session_id"]).version == 4
    assert body["max_turns"] == MAX_TURNS
    assert body["created_at"]

    state = await async_client.get(f"/api/v1/sessions/{body['session_id']}")
    assert state.status_code == 200
    estado = state.json()
    assert estado["turn_count"] == 0
    assert estado["window"] == {"max_turns": MAX_TURNS, "turns_in_window": 0, "dropped_turns": 0, "messages_sent": None}
    assert estado["history"] == []
    assert estado["project_metadata"]["project_name"] is None
    assert estado["project_metadata"]["mentioned_technologies"] == []


async def test_sesion_inexistente_devuelve_404(async_client):
    assert (await async_client.get("/api/v1/sessions/no-existe")).status_code == 404
    assert (await async_client.delete("/api/v1/sessions/no-existe")).status_code == 404
    response = await async_client.post("/api/v1/sessions/no-existe/estimate", data={"transcript": TRANSCRIPCION_1})
    assert response.status_code == 404
    assert "POST /api/v1/sessions" in response.json()["detail"]


async def test_borrar_una_sesion(async_client):
    session_id = await _nueva_sesion(async_client)

    assert (await async_client.delete(f"/api/v1/sessions/{session_id}")).status_code == 204
    assert (await async_client.get(f"/api/v1/sessions/{session_id}")).status_code == 404


# --- Test 1 del enunciado: dos peticiones enlazadas actualizan el project_metadata -------------------


async def test_dos_turnos_en_la_misma_sesion_actualizan_la_memoria(async_client, llm):
    session_id = await _nueva_sesion(async_client)

    # Turno 1: con nombre de proyecto en los parámetros
    r1 = await async_client.post(
        f"/api/v1/sessions/{session_id}/estimate",
        data={"transcript": TRANSCRIPCION_1, "project_name": "Panadería online", "project_type": "web", "hourly_rate_eur": "50"},
    )
    assert r1.status_code == 200
    body1 = r1.json()
    # Respeta el schema de EstimationResponse...
    assert body1["estimation"].startswith("## Estimación: Panadería online")
    assert body1["model"] == "gpt-4o-mini" and body1["provider"] == "openai"
    assert body1["usage"]["total_tokens"] > 0
    assert body1["prompt"]["version"] == "v3"
    assert body1["generated_at"]
    # ...y añade el estado de la sesión
    assert body1["session_id"] == session_id
    assert body1["turn"] == 1
    assert body1["window"] == {"max_turns": MAX_TURNS, "turns_in_window": 1, "dropped_turns": 0, "messages_sent": 2}
    memoria1 = body1["project_metadata"]
    assert memoria1["project_name"] == "Panadería online"
    assert memoria1["project_type"] == "web"
    assert memoria1["agreed_scope"] == "Tienda online para una panadería artesanal."
    assert memoria1["assumed_team"] == "1 desarrollador full-stack, 1 diseñador UX/UI"
    assert memoria1["assumed_team_size"] == 2
    assert memoria1["mentioned_technologies"] == ["Excel"]
    assert memoria1["last_total_hours"] == 84
    assert memoria1["hourly_rate_eur"] == 50
    assert memoria1["attached_documents"] == []
    assert memoria1["updated_at"]

    # En el primer turno el bloque <project_metadata> del system prompt va vacío
    system1 = llm.calls[0]["messages"][0]["content"]
    assert "<project_metadata>\n</project_metadata>" in system1

    # Turno 2: sin nombre de proyecto; el modelo lo "recuerda" porque va en la memoria del system prompt
    r2 = await async_client.post(f"/api/v1/sessions/{session_id}/estimate", data={"transcript": TRANSCRIPCION_2})
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["turn"] == 2
    assert body2["estimation"].startswith("## Estimación: Panadería online")
    memoria2 = body2["project_metadata"]
    assert memoria2["project_name"] == "Panadería online"
    assert memoria2["project_type"] == "web"  # no se borra aunque el turno 2 no lo indique
    assert memoria2["mentioned_technologies"] == ["Excel", "Stripe"]  # se acumulan
    assert memoria2["agreed_scope"] == "Tienda online para una panadería artesanal con pasarela de pago Stripe."
    assert memoria2["last_total_hours"] == 108  # cambió con la nueva información
    assert memoria2["updated_at"] > memoria1["updated_at"]

    # El system prompt del turno 2 lleva la memoria del turno 1...
    system2 = llm.calls[1]["messages"][0]["content"]
    assert "- Nombre del proyecto: Panadería online" in system2
    assert "- Tecnologías mencionadas: Excel" in system2
    assert "- Última estimación: 84 horas" in system2
    assert "- Equipo asumido: 1 desarrollador full-stack, 1 diseñador UX/UI (2 personas)" in system2
    # ...y el prefijo (rol, reglas, ejemplos CAG) es idéntico entre turnos: solo cambia la cola (cacheable)
    assert system1.split("## Proyecto en curso")[0] == system2.split("## Proyecto en curso")[0]
    # El historial del turno 1 viajó como turnos anteriores
    assert _roles(llm.calls[1]["messages"]) == ["system", "user", "assistant", "user"]
    assert llm.calls[1]["messages"][2]["content"] == body1["estimation"]

    estado = (await async_client.get(f"/api/v1/sessions/{session_id}")).json()
    assert estado["turn_count"] == 2
    assert estado["window"]["turns_in_window"] == 2
    assert _roles(estado["history"]) == ["user", "assistant", "user", "assistant"]
    assert estado["project_metadata"] == memoria2


# --- Test 2 del enunciado: un PDF adjunto influye en la estimación ------------------------------------


async def test_un_pdf_adjunto_cambia_la_estimacion(async_client, llm):
    # Misma transcripción en dos sesiones: sin adjunto y con un PDF que menciona la pasarela de pago
    sin = await _nueva_sesion(async_client)
    con = await _nueva_sesion(async_client)
    pdf = make_pdf(
        ["Especificación técnica v2", "Pasarela de pago: Stripe (tarjeta y Bizum)."],
        ["Hosting en AWS. Catálogo importado desde Excel."],
    )

    r_sin = await async_client.post(f"/api/v1/sessions/{sin}/estimate", data={"transcript": TRANSCRIPCION_1})
    r_con = await async_client.post(
        f"/api/v1/sessions/{con}/estimate",
        data={"transcript": TRANSCRIPCION_1},
        files=[("attachments", ("especificacion.pdf", pdf, "application/pdf"))],
    )
    assert r_sin.status_code == 200 and r_con.status_code == 200
    body_sin, body_con = r_sin.json(), r_con.json()

    # El documento llegó al prompt, delimitado y con su texto
    user_con = llm.calls[1]["messages"][-1]["content"]
    assert "--- attachment: especificacion.pdf ---" in user_con
    assert "Pasarela de pago: Stripe" in user_con
    assert "--- fin del adjunto: especificacion.pdf ---" in user_con
    assert "attachment" not in llm.calls[0]["messages"][-1]["content"]

    # ...y cambió la salida: aparece la tarea de pagos y el total es otro
    assert "Integración de pagos con Stripe" not in body_sin["estimation"]
    assert "Integración de pagos con Stripe" in body_con["estimation"]
    assert body_sin["project_metadata"]["last_total_hours"] == 84
    assert body_con["project_metadata"]["last_total_hours"] == 108

    # La memoria recoge lo que aportó el documento
    assert body_con["project_metadata"]["mentioned_technologies"] == ["Excel", "Stripe", "Bizum", "AWS"]
    assert body_con["project_metadata"]["attached_documents"] == ["especificacion.pdf"]
    assert body_con["attachments"] == [
        {
            "filename": "especificacion.pdf",
            "kind": "pdf",
            "size_bytes": len(pdf),
            "chars": body_con["attachments"][0]["chars"],
            "pages": 2,
            "truncated": False,
            "warning": None,
        }
    ]
    assert body_con["attachments"][0]["chars"] > 50
    # transcription_chars mide la transcripción tal cual; el adjunto se refleja en user_prompt_chars
    assert body_con["prompt"]["transcription_chars"] == len(TRANSCRIPCION_1)
    assert body_con["prompt"]["user_prompt_chars"] > body_sin["prompt"]["user_prompt_chars"]


async def test_varios_adjuntos_de_distintos_tipos(async_client, llm):
    from tests.documentos import make_docx

    session_id = await _nueva_sesion(async_client)
    response = await async_client.post(
        f"/api/v1/sessions/{session_id}/estimate",
        data={"transcript": TRANSCRIPCION_1},
        files=[
            ("attachments", ("propuesta.docx", make_docx(["Propuesta anterior con WordPress."]), None)),
            ("attachments", ("notas.md", b"# Notas\nIntegrar con HubSpot.", "text/markdown")),
        ],
    )

    assert response.status_code == 200
    body = response.json()
    assert [a["kind"] for a in body["attachments"]] == ["docx", "texto"]
    assert body["project_metadata"]["attached_documents"] == ["propuesta.docx", "notas.md"]
    assert set(body["project_metadata"]["mentioned_technologies"]) == {"Excel", "WordPress", "HubSpot"}
    user = llm.calls[0]["messages"][-1]["content"]
    assert user.index("--- attachment: propuesta.docx ---") < user.index("--- attachment: notas.md ---")


# --- Test 3 del enunciado: 8 turnos nunca superan MAX_TURNS en el historial enviado -------------------


async def test_ocho_turnos_respetan_la_ventana_deslizante(async_client, llm):
    session_id = await _nueva_sesion(async_client)

    for i in range(1, 9):
        response = await async_client.post(
            f"/api/v1/sessions/{session_id}/estimate",
            data={"transcript": f"Turno {i}: el cliente añade un detalle más sobre la tienda online de la panadería."},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["turn"] == i
        assert body["window"]["turns_in_window"] == min(i, MAX_TURNS)
        assert body["window"]["dropped_turns"] == max(0, i - MAX_TURNS)
        assert body["window"]["messages_sent"] == 1 + 2 * (min(i, MAX_TURNS) - 1) + 1

    assert len(llm.calls) == 8
    for messages in llm.messages_per_call:
        assert messages[0]["role"] == "system"
        assert sum(m["role"] == "system" for m in messages) == 1  # el system prompt, invariante, siempre presente
        assert sum(m["role"] == "user" for m in messages) <= MAX_TURNS
        assert sum(m["role"] == "assistant" for m in messages) <= MAX_TURNS - 1
        assert len(messages) <= 1 + 2 * MAX_TURNS - 1

    # En la última llamada viajan los turnos 6, 7 y 8; el 5 y anteriores ya salieron
    ultimos = [m["content"] for m in llm.calls[-1]["messages"] if m["role"] == "user"]
    assert len(ultimos) == MAX_TURNS
    assert "Turno 6:" in ultimos[0] and "Turno 7:" in ultimos[1] and "Turno 8:" in ultimos[2]
    assert not any("Turno 5:" in m["content"] for m in llm.calls[-1]["messages"])

    estado = (await async_client.get(f"/api/v1/sessions/{session_id}")).json()
    assert estado["turn_count"] == 8
    assert estado["window"] == {"max_turns": MAX_TURNS, "turns_in_window": MAX_TURNS, "dropped_turns": 5, "messages_sent": None}
    assert len(estado["history"]) == 2 * MAX_TURNS
    # La memoria sobrevive a la ventana: el nombre se fijó en el turno 1 y sigue ahí tras descartar ese turno
    assert estado["project_metadata"]["project_name"] == "Proyecto sin nombre"


async def test_el_tamano_de_la_ventana_sale_de_la_configuracion(async_client, llm):
    app.dependency_overrides[get_settings] = lambda: make_settings(openai_api_key="sk-test", session_max_turns=1)

    session_id = await _nueva_sesion(async_client)
    for i in range(3):
        assert (await async_client.post(f"/api/v1/sessions/{session_id}/estimate", data={"transcript": TRANSCRIPCION_1})).status_code == 200

    for messages in llm.messages_per_call:
        assert _roles(messages) == ["system", "user"]  # con 1 turno de ventana solo viaja el mensaje actual


# --- Errores del endpoint multipart --------------------------------------------------------------------


async def test_validacion_del_formulario(async_client, llm):
    session_id = await _nueva_sesion(async_client)

    corta = await async_client.post(f"/api/v1/sessions/{session_id}/estimate", data={"transcript": "corta"})
    assert corta.status_code == 422
    assert corta.json()["detail"][0]["loc"] == ["body", "transcript"]
    assert "20 characters" in corta.json()["detail"][0]["msg"]

    sin_transcript = await async_client.post(f"/api/v1/sessions/{session_id}/estimate", data={"project_name": "x"})
    assert sin_transcript.status_code == 422

    modo = await async_client.post(
        f"/api/v1/sessions/{session_id}/estimate", data={"transcript": TRANSCRIPCION_1, "estimation_mode": "agresiva"}
    )
    assert modo.status_code == 422

    version = await async_client.post(
        f"/api/v1/sessions/{session_id}/estimate", data={"transcript": TRANSCRIPCION_1, "prompt_version": "v99"}
    )
    assert version.status_code == 422
    assert "v99" in version.json()["detail"]

    assert llm.calls == []  # ninguna petición inválida llegó al LLM
    estado = (await async_client.get(f"/api/v1/sessions/{session_id}")).json()
    assert estado["history"] == [] and estado["turn_count"] == 0


async def test_errores_de_adjuntos(async_client, llm):
    session_id = await _nueva_sesion(async_client)
    url = f"/api/v1/sessions/{session_id}/estimate"
    data = {"transcript": TRANSCRIPCION_1}

    no_soportado = await async_client.post(url, data=data, files=[("attachments", ("programa.exe", b"MZ", "application/octet-stream"))])
    assert no_soportado.status_code == 415
    assert ".pdf, .docx, .txt, .md" in no_soportado.json()["detail"]

    corrupto = await async_client.post(url, data=data, files=[("attachments", ("roto.pdf", b"%PDF-1.4 basura", "application/pdf"))])
    assert corrupto.status_code == 422
    assert "roto.pdf" in corrupto.json()["detail"]

    demasiados = await async_client.post(
        url, data=data, files=[("attachments", (f"n{i}.txt", b"hola", "text/plain")) for i in range(6)]
    )
    assert demasiados.status_code == 422
    assert "Demasiados adjuntos" in demasiados.json()["detail"]

    app.dependency_overrides[get_settings] = lambda: make_settings(openai_api_key="sk-test", attachment_max_bytes=100)
    grande = await async_client.post(url, data=data, files=[("attachments", ("grande.txt", b"x" * 101, "text/plain"))])
    assert grande.status_code == 413

    assert llm.calls == []


async def test_sin_api_key_devuelve_500_y_no_toca_la_sesion(async_client):
    app.dependency_overrides[get_settings] = lambda: make_settings()
    session_id = await _nueva_sesion(async_client)

    response = await async_client.post(f"/api/v1/sessions/{session_id}/estimate", data={"transcript": TRANSCRIPCION_1})

    assert response.status_code == 500
    assert "OPENAI_API_KEY" in response.json()["detail"]
    assert (await async_client.get(f"/api/v1/sessions/{session_id}")).json()["history"] == []


async def test_un_fallo_del_proveedor_no_deja_un_turno_huerfano(async_client, llm, monkeypatch):
    session_id = await _nueva_sesion(async_client)

    async def falla(prompt, settings, api_key, **kwargs):
        raise LLMProviderError("OpenAI ha devuelto un error de cuota")

    monkeypatch.setattr(conversation, "call_llm", falla)
    response = await async_client.post(f"/api/v1/sessions/{session_id}/estimate", data={"transcript": TRANSCRIPCION_1})
    assert response.status_code == 502
    assert "cuota" in response.json()["detail"]
    estado = (await async_client.get(f"/api/v1/sessions/{session_id}")).json()
    assert estado["history"] == [] and estado["turn_count"] == 0 and estado["project_metadata"]["project_name"] is None

    # Al reintentar, el turno es el 1 y el LLM solo ve el mensaje actual
    monkeypatch.setattr(conversation, "call_llm", llm_service.call_llm)
    reintento = await async_client.post(f"/api/v1/sessions/{session_id}/estimate", data={"transcript": TRANSCRIPCION_1})
    assert reintento.status_code == 200
    assert reintento.json()["turn"] == 1
    assert _roles(llm.calls[-1]["messages"]) == ["system", "user"]


async def test_openapi_documenta_los_endpoints_de_sesion(async_client):
    spec = (await async_client.get("/openapi.json")).json()

    assert "post" in spec["paths"]["/api/v1/sessions"]
    assert "get" in spec["paths"]["/api/v1/sessions/{session_id}"]
    assert "delete" in spec["paths"]["/api/v1/sessions/{session_id}"]
    estimate = spec["paths"]["/api/v1/sessions/{session_id}/estimate"]["post"]
    assert "multipart/form-data" in estimate["requestBody"]["content"]
    form = estimate["requestBody"]["content"]["multipart/form-data"]["schema"]
    form_schema = spec["components"]["schemas"][form["$ref"].rsplit("/", 1)[-1]] if "$ref" in form else form
    assert "transcript" in form_schema["properties"]
    assert "attachments" in form_schema["properties"]
    assert form_schema["required"] == ["transcript"]
