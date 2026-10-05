"""Tests unitarios del estado de sesión: ventana deslizante, memoria del proyecto y almacén en memoria."""

import uuid
from datetime import timedelta

import pytest

from app.services.sessions import ConversationHistory, ProjectMetadata, SessionNotFound, SessionStore

# --- ConversationHistory: ventana deslizante -------------------------------------------------


def _conversar(history: ConversationHistory, n: int) -> None:
    for i in range(1, n + 1):
        history.start_turn(f"u{i}")
        history.complete_turn(f"a{i}")


def test_la_ventana_descarta_los_turnos_mas_antiguos():
    history = ConversationHistory(max_turns=3)

    _conversar(history, 5)

    assert [t.user for t in history.turns] == ["u3", "u4", "u5"]
    assert history.dropped_turns == 2
    assert len(history) == 3


def test_to_messages_list_antepone_el_system_prompt_y_alterna_roles():
    history = ConversationHistory(max_turns=3)
    _conversar(history, 3)

    messages = history.to_messages_list("SYSTEM")

    assert messages[0] == {"role": "system", "content": "SYSTEM"}
    assert [m["role"] for m in messages[1:]] == ["user", "assistant"] * 3
    assert [m["content"] for m in messages[1:]] == ["u1", "a1", "u2", "a2", "u3", "a3"]


def test_el_turno_en_curso_cuenta_en_la_ventana():
    """Con max_turns=2 viajan como mucho 2 mensajes de usuario: el anterior (con respuesta) y el actual."""
    history = ConversationHistory(max_turns=2)
    _conversar(history, 2)

    history.start_turn("u3")
    messages = history.to_messages_list("S")

    assert [m["content"] for m in messages] == ["S", "u2", "a2", "u3"]
    assert sum(m["role"] == "user" for m in messages) == 2
    assert history.dropped_turns == 1


def test_el_system_prompt_se_regenera_en_cada_llamada():
    """El system prompt no se guarda en el historial: cada `to_messages_list` recibe el actual (con la memoria nueva)."""
    history = ConversationHistory(max_turns=6)
    _conversar(history, 2)

    antes = history.to_messages_list("memoria vacía")
    despues = history.to_messages_list("memoria con el nombre del proyecto")

    assert antes[0]["content"] != despues[0]["content"]
    assert antes[1:] == despues[1:]


def test_descartar_un_turno_incompleto_deja_el_historial_como_estaba():
    history = ConversationHistory(max_turns=6)
    _conversar(history, 1)

    history.start_turn("u2")
    assert history.discard_incomplete_turn() is True
    assert [t.user for t in history.turns] == ["u1"]
    assert history.completed_turns == history.turns
    assert history.discard_incomplete_turn() is False  # no había nada que descartar


def test_no_se_puede_empezar_otro_turno_con_uno_en_curso():
    history = ConversationHistory()
    history.start_turn("u1")

    with pytest.raises(RuntimeError):
        history.start_turn("u2")
    with pytest.raises(RuntimeError):
        ConversationHistory().complete_turn("sin turno en curso")


def test_max_turns_debe_ser_positivo():
    with pytest.raises(ValueError):
        ConversationHistory(max_turns=0)
    assert ConversationHistory().max_turns == 6  # MAX_TURNS por defecto


# --- ProjectMetadata: memoria del proyecto --------------------------------------------------------


def test_la_memoria_empieza_vacia_y_no_se_inyecta():
    metadata = ProjectMetadata()

    assert metadata.is_empty()
    assert metadata.prompt_variables() is None


def test_merged_el_dato_mas_reciente_manda_y_las_listas_se_acumulan():
    actual = ProjectMetadata(
        project_name="Panadería online",
        mentioned_technologies=["WordPress", "Stripe"],
        agreed_scope="Tienda online con recogida.",
        last_total_hours=200,
        attached_documents=["propuesta.docx"],
    )
    nuevo = ProjectMetadata(
        mentioned_technologies=["stripe", "AWS"],  # 'stripe' ya está (sin distinguir mayúsculas)
        agreed_scope="Tienda online con recogida y reparto.",
        last_total_hours=250,
        hourly_rate_eur=60,
    )

    resultado = actual.merged(nuevo)

    assert resultado.project_name == "Panadería online"  # ausente en el nuevo: no se borra
    assert resultado.mentioned_technologies == ["WordPress", "Stripe", "AWS"]
    assert resultado.agreed_scope == "Tienda online con recogida y reparto."  # el más reciente manda
    assert resultado.last_total_hours == 250
    assert resultado.hourly_rate_eur == 60
    assert resultado.attached_documents == ["propuesta.docx"]
    assert resultado.updated_at is not None
    assert not resultado.is_empty()
    assert "updated_at" not in resultado.prompt_variables()


def test_merged_no_modifica_la_memoria_original():
    original = ProjectMetadata(project_name="A")
    original.merged(ProjectMetadata(project_name="B"))

    assert original.project_name == "A"


# --- SessionStore: diccionario en memoria del proceso ---------------------------------------------


def test_el_store_crea_sesiones_con_uuid_v4_y_las_recupera():
    store = SessionStore()

    session = store.create(max_turns=4)

    assert uuid.UUID(session.session_id).version == 4
    assert store.get(session.session_id) is session
    assert session.history.max_turns == 4
    assert session.metadata.is_empty()
    assert session.turn_count == 0
    assert len(store) == 1


def test_sesion_inexistente():
    store = SessionStore()

    with pytest.raises(SessionNotFound, match="no-existe"):
        store.get("no-existe")
    assert store.delete("no-existe") is False


def test_borrar_y_vaciar():
    store = SessionStore()
    session = store.create()

    assert store.delete(session.session_id) is True
    assert len(store) == 0
    store.create()
    store.clear()
    assert len(store) == 0


def test_al_superar_el_tope_se_expulsa_la_sesion_menos_reciente():
    store = SessionStore(max_sessions=2)
    primera = store.create()
    segunda = store.create()
    segunda.updated_at -= timedelta(minutes=5)  # la segunda lleva más tiempo sin usarse (sin depender del reloj)

    tercera = store.create()

    assert len(store) == 2
    assert store.get(primera.session_id) is primera
    assert store.get(tercera.session_id) is tercera
    with pytest.raises(SessionNotFound):
        store.get(segunda.session_id)
