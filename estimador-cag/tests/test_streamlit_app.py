"""Tests de la interfaz (streamlit_app.py) con AppTest.

La interfaz habla con la API por HTTP; aquí el cliente HTTP se sustituye por el `TestClient` de FastAPI,
así que la app de Streamlit ejercita la API real en el mismo proceso (sesiones, memoria, historial) y
solo el LLM es un doble. `st.file_uploader` no se puede manejar desde AppTest: los adjuntos se prueban
en `tests/test_sessions_api.py`.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from app.clients import estimator_api
from app.config import get_settings
from app.main import app
from app.services import llm_service
from app.services.sessions import SessionStore, get_session_store
from tests.conftest import make_settings
from tests.fakes import FakeEstimatorLLM

APP = Path(__file__).resolve().parent.parent / "streamlit_app.py"
TRANSCRIPCION = (
    "El cliente, una panadería artesanal, quiere vender por internet con recogida en tienda y reparto local."
)


@pytest.fixture
def entorno(monkeypatch):
    settings = make_settings(openai_api_key="sk-test", session_max_turns=6)
    store = SessionStore()
    llm = FakeEstimatorLLM()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_session_store] = lambda: store
    monkeypatch.setattr(estimator_api, "make_http_client", lambda base_url, timeout=None: TestClient(app))
    monkeypatch.setattr(llm_service.openai, "AsyncOpenAI", llm)
    yield SimpleNamespace(store=store, llm=llm)
    app.dependency_overrides.clear()


def _abrir() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=60).run()


def _submit(at: AppTest):
    return next(b for b in at.button if b.label.startswith(("Generar estimación", "Enviar turno")))


def _nueva_conversacion(at: AppTest):
    return next(b for b in at.button if "Nueva conversación" in b.label)


def _texto_sidebar(at: AppTest) -> str:
    return " ".join(md.value for md in at.sidebar.markdown) + " ".join(c.value for c in at.sidebar.caption)


def test_abre_un_formulario_y_crea_una_sesion_al_cargar(entorno):
    at = _abrir()

    assert not at.exception
    assert at.title[0].value == "🧮 Estimador CAG"
    assert len(at.chat_input) == 0 and len(at.chat_message) == 0  # sigue siendo un formulario, no un chat
    session_id = at.session_state["session_id"]
    assert entorno.store.get(session_id).turn_count == 0  # la sesión existe en la API
    assert at.session_state["turns"] == []
    assert at.text_area(key="transcript").value == ""
    assert at.selectbox(key="prompt_version").value == "v3"
    assert not _submit(at).disabled
    assert "Vacía" in _texto_sidebar(at)  # panel de memoria vacío


def test_tres_turnos_conservan_el_proyecto_y_la_memoria_se_ve_en_el_panel(entorno):
    at = _abrir()

    # Turno 1: con nombre de proyecto
    at.text_area(key="transcript").set_value(TRANSCRIPCION)
    at.text_input[0].set_value("Panadería online")
    _submit(at).click().run()
    assert not at.exception
    turnos = at.session_state["turns"]
    assert len(turnos) == 1
    assert turnos[0]["estimation"].startswith("## Estimación: Panadería online")
    assert turnos[0]["project_metadata"]["project_name"] == "Panadería online"
    assert turnos[0]["meta"]["messages_sent"] == 2

    # Turno 2: sin nombre; la API lo recuerda por la memoria del proyecto
    at.text_area(key="transcript").set_value("Añaden pagos con Stripe y reparto solo en el barrio.")
    at.text_input[0].set_value("")
    _submit(at).click().run()
    assert not at.exception
    turnos = at.session_state["turns"]
    assert len(turnos) == 2
    assert turnos[1]["estimation"].startswith("## Estimación: Panadería online")
    assert "Stripe" in turnos[1]["project_metadata"]["mentioned_technologies"]
    assert turnos[1]["meta"]["messages_sent"] == 4

    # Turno 3
    at.text_area(key="transcript").set_value("El cliente confirma el alcance y pregunta por el plazo de entrega.")
    _submit(at).click().run()
    assert not at.exception
    assert len(at.session_state["turns"]) == 3

    # El LLM recibió el historial de la ventana en cada turno
    assert [[m["role"] for m in call["messages"]] for call in entorno.llm.calls] == [
        ["system", "user"],
        ["system", "user", "assistant", "user"],
        ["system", "user", "assistant", "user", "assistant", "user"],
    ]
    # La API tiene el mismo estado que la interfaz
    sesion = entorno.store.get(at.session_state["session_id"])
    assert sesion.turn_count == 3
    assert sesion.metadata.project_name == "Panadería online"
    # Panel lateral: memoria del proyecto e historial efectivo visibles
    sidebar = _texto_sidebar(at)
    assert "Panadería online" in sidebar
    assert "Stripe" in sidebar
    assert "3 en ventana" in sidebar
    # Los tres turnos aparecen en la página con sus estimaciones
    assert sum(1 for md in at.main.markdown if md.value.startswith("**Turno ")) == 3


def test_nueva_conversacion_crea_otra_sesion_y_resetea(entorno):
    at = _abrir()
    primera = at.session_state["session_id"]
    at.text_area(key="transcript").set_value(TRANSCRIPCION)
    _submit(at).click().run()
    assert len(at.session_state["turns"]) == 1

    _nueva_conversacion(at).click().run()

    assert not at.exception
    assert at.session_state["session_id"] != primera
    assert at.session_state["turns"] == []
    assert len(entorno.store) == 2
    assert entorno.store.get(primera).turn_count == 1  # la anterior sigue en la API, intacta
    assert "Vacía" in _texto_sidebar(at)


def test_una_transcripcion_invalida_no_llama_al_llm(entorno):
    at = _abrir()

    at.text_area(key="transcript").set_value("corta")
    _submit(at).click().run()

    assert not at.exception
    assert any("Transcripción" in e.value for e in at.error)
    assert at.session_state["turns"] == []
    assert entorno.llm.calls == []


def test_sin_api_muestra_el_error_y_se_detiene(monkeypatch):
    import httpx

    class SinServidor(httpx.Client):
        def request(self, *args, **kwargs):
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(estimator_api, "make_http_client", lambda base_url, timeout=None: SinServidor(base_url=base_url))

    at = _abrir()

    assert not at.exception
    assert any("No se ha podido conectar con la API" in e.value for e in at.error)
    assert "session_id" not in at.session_state
