"""Interfaz web (Streamlit) del Estimador CAG: conversación multi-turno con memoria y adjuntos.

Ejecutar (con la API arrancada en API_BASE_URL, por defecto http://localhost:8000):
    uv run uvicorn app.main:app --reload
    uv run streamlit run streamlit_app.py

Sesión 05: el formulario de la sesión 04 sigue siendo la forma de pedir una estimación (parámetros tipados,
validados antes de gastar un token), pero ahora cada envío es un **turno** dentro de una **sesión**:
la API recuerda el proyecto en curso (historial con ventana deslizante + `project_metadata`) y acepta
**adjuntos** (PDF, Word, texto). La interfaz habla con la API por HTTP (`app/clients/estimator_api.py`):
el estado vive en el proceso de la API, no aquí. La barra lateral muestra la memoria del proyecto y el
historial efectivo para ver la diferencia entre lo que el sistema "sabe" y lo que viaja al LLM.
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from app.clients.estimator_api import EstimatorAPI, EstimatorAPIError
from app.schemas.estimation import (
    DETAIL_LEVEL_LABELS,
    ESTIMATION_MODE_LABELS,
    PROJECT_TYPE_LABELS,
    EstimationRequest,
)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  # API_BASE_URL puede venir del .env

SAMPLES = {
    "Landing page con HubSpot (enunciado)": BASE_DIR / "transcripciones/reunion-landing-page.txt",
    "Panadería: descripción pobre": BASE_DIR / "transcripciones/panaderia-descripcion-pobre.txt",
    "Panadería: descripción detallada": BASE_DIR / "transcripciones/panaderia-descripcion-detallada.txt",
    "Panadería: turno 2, cambios de alcance": BASE_DIR / "transcripciones/panaderia-turno-2-cambios.txt",
    "Panadería: turno 3, adjunta la especificación": BASE_DIR / "transcripciones/panaderia-turno-3-adjunto.txt",
    "Restaurantes: reservas (reunión larga)": BASE_DIR / "transcripciones/reunion-app-reservas-restaurante.txt",
}
SAMPLE_ATTACHMENTS_DIR = BASE_DIR / "transcripciones/adjuntos"
ACCEPTED_TYPES = ["pdf", "docx", "txt", "md"]
FIELD_LABELS = {
    "transcription": "Transcripción",
    "project_name": "Nombre del proyecto",
    "project_type": "Tipo de proyecto",
    "hourly_rate_eur": "Tarifa",
    "deadline_weeks": "Plazo objetivo",
    "estimation_mode": "Enfoque",
    "detail_level": "Nivel de detalle",
    "prompt_version": "Versión de plantilla",
}
DEFAULTS = {name: field.default for name, field in EstimationRequest.model_fields.items()}
PARAM_FIELDS = ("project_name", "project_type", "hourly_rate_eur", "deadline_weeks", "estimation_mode", "detail_level")


# --- Utilidades ---------------------------------------------------------------------


def api() -> EstimatorAPI:
    if "api" not in st.session_state:
        st.session_state.api = EstimatorAPI()
    return st.session_state.api


def new_conversation() -> None:
    """Crea una sesión nueva en la API y resetea el estado de la interfaz (botón "Nueva conversación")."""
    created = api().create_session()
    st.session_state.session_id = created["session_id"]
    st.session_state.max_turns = created["max_turns"]
    st.session_state.turns = []


def format_validation_errors(exc: ValidationError) -> list[str]:
    mensajes = []
    for error in exc.errors():
        campo = str(error["loc"][0]) if error["loc"] else "petición"
        mensajes.append(f"{FIELD_LABELS.get(campo, campo)}: {error['msg']}")
    return mensajes


def load_sample() -> None:
    name = st.session_state.get("sample")
    if name in SAMPLES and SAMPLES[name].is_file():
        st.session_state["transcript"] = SAMPLES[name].read_text(encoding="utf-8").strip()


def miles(n: int | float | None) -> str:
    return "?" if n is None else f"{n:,.0f}".replace(",", ".")


def call_metadata(response: dict, elapsed_seconds: float) -> dict:
    usage = response.get("usage") or {}
    cost = usage.get("cost") or {}
    prompt = response.get("prompt") or {}
    window = response.get("window") or {}
    return {
        "model": response.get("model"),
        "provider": response.get("provider"),
        "template": f"{prompt.get('template')}/{prompt.get('version')}" if prompt else None,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cost_usd": cost.get("total_usd"),
        "elapsed_seconds": elapsed_seconds,
        "system_prompt_chars": prompt.get("system_prompt_chars"),
        "user_prompt_chars": prompt.get("user_prompt_chars"),
        "messages_sent": window.get("messages_sent"),
        "turns_in_window": window.get("turns_in_window"),
        "dropped_turns": window.get("dropped_turns"),
    }


def format_metadata(meta: dict) -> str:
    partes = [f"{meta['model']} ({meta['provider']})"]
    if meta.get("template"):
        partes.append(f"plantilla {meta['template']}")
    if meta.get("messages_sent") is not None:
        partes.append(f"{meta['messages_sent']} mensajes al LLM")
    if meta.get("input_tokens") is not None:
        partes.append(f"{miles(meta['input_tokens'])} tokens de entrada · {miles(meta['output_tokens'])} de salida")
    if meta.get("cost_usd") is not None:
        partes.append(f"{meta['cost_usd']:.6f} USD")
    if meta.get("elapsed_seconds") is not None:
        partes.append(f"{meta['elapsed_seconds']:.1f} s")
    return " · ".join(partes)


def render_project_metadata(metadata: dict) -> None:
    """La memoria del proyecto tal y como se inyecta en el system prompt (vacía en el primer turno)."""
    team = metadata.get("assumed_team")
    if team and metadata.get("assumed_team_size"):
        team = f"{team} ({metadata['assumed_team_size']} personas)"
    rows = [
        ("Proyecto", metadata.get("project_name")),
        ("Tipo", PROJECT_TYPE_LABELS.get(metadata.get("project_type") or "", None)),
        ("Alcance acordado", metadata.get("agreed_scope")),
        ("Tecnologías", ", ".join(metadata.get("mentioned_technologies") or [])),
        ("Equipo asumido", team),
        ("Última estimación", f"{metadata['last_total_hours']:g} horas" if metadata.get("last_total_hours") else None),
        ("Tarifa", f"{metadata['hourly_rate_eur']:g} €/hora" if metadata.get("hourly_rate_eur") else None),
        ("Plazo objetivo", f"{metadata['deadline_weeks']} semanas" if metadata.get("deadline_weeks") else None),
        ("Documentos", ", ".join(metadata.get("attached_documents") or [])),
    ]
    present = [(label, value) for label, value in rows if value]
    if not present:
        st.caption("Vacía: se rellena con la primera estimación de la sesión.")
        return
    for label, value in present:
        st.markdown(f"**{label}:** {value}")
    with st.expander("JSON (lo que recibe la plantilla)"):
        st.json(metadata)


def render_compression(compression: dict | None, observed: dict | None) -> None:
    """Anclas y resumen acumulativo de la sesión, y lo observado en el último turno (caché, tier, summarizer)."""
    if not compression or (not compression.get("anchors") and not compression.get("summary")):
        st.caption("Nada todavía: las anclas aparecen con el primer turno y el resumen cuando un turno sale de la ventana.")
    else:
        with st.expander(f"Anclas ({compression.get('anchors_count', 0)})", expanded=False):
            for anchor in compression.get("anchors") or []:
                st.markdown(f"- `{anchor['kind']}` · turno {anchor['turn']} · {anchor['text']}")
        with st.expander(f"Resumen acumulativo ({miles(compression.get('summary_chars', 0))} caracteres)", expanded=False):
            st.write(compression.get("summary") or "(vacío)")
    if observed:
        st.caption(
            f"Último turno: caché `{observed.get('cache_hit_kind')}` · tier `{observed.get('last_resolved_tier')}` "
            f"({observed.get('last_tier_rule')}) · summarizer `{observed.get('summarizer_mode')}`"
            + (f" ({observed.get('summarizer_tokens_in')} tokens, {observed.get('summarizer_latency_ms')} ms)" if observed.get("summarizer_called") else "")
        )
        with st.expander("Evento `turn_observed`"):
            st.json(observed)


def render_turn(index: int, turn: dict) -> None:
    with st.container(border=True):
        st.markdown(f"**Turno {index}**")
        adjuntos = turn.get("attachments") or []
        resumen = f"Transcripción enviada ({miles(len(turn['transcript']))} caracteres)"
        if adjuntos:
            resumen += f" · {len(adjuntos)} adjunto(s): " + ", ".join(a["filename"] for a in adjuntos)
        with st.expander(resumen):
            st.text(turn["transcript"])
            for a in adjuntos:
                detalle = f"{a['kind']}, {miles(a['chars'])} caracteres extraídos"
                if a.get("pages"):
                    detalle += f", {a['pages']} páginas"
                if a.get("truncated"):
                    detalle += ", truncado"
                st.caption(f"📎 {a['filename']} ({detalle})")
                if a.get("warning"):
                    st.warning(a["warning"])
        st.markdown(turn["estimation"])
        if turn.get("meta"):
            st.caption(format_metadata(turn["meta"]))


# --- Estado y conexión con la API ---------------------------------------------------------

st.set_page_config(page_title="Estimador CAG", page_icon="🧮", layout="wide")

try:
    health = api().health()
    if "session_id" not in st.session_state:
        new_conversation()
    if "params" not in st.session_state:
        st.session_state.params = {name: DEFAULTS[name] for name in PARAM_FIELDS}
    if "uploader_round" not in st.session_state:
        st.session_state.uploader_round = 0
    prompts_info = api().prompts()
except EstimatorAPIError as exc:
    st.error(str(exc))
    st.stop()


# --- Conversación: turnos anteriores y formulario del turno siguiente -------------------------

st.title("🧮 Estimador CAG")
st.caption(
    "Conversación con memoria: cada envío es un turno sobre el mismo proyecto. La API conserva los últimos "
    f"{st.session_state.max_turns} turnos (ventana deslizante) y una memoria con los hechos del proyecto que "
    "inyecta en el system prompt. Los adjuntos (PDF, Word, texto) se leen en el servidor y se añaden a la transcripción."
)

turns_area = st.container()

if st.session_state.pop("clear_form", False):
    # Tras un turno con éxito se vacía la transcripción y el selector de ejemplos. No se usa
    # `clear_on_submit`: impedía cargar un ejemplo en el turno siguiente y no vacía el file_uploader.
    st.session_state["transcript"] = ""
    st.session_state["sample"] = "(ninguna)"

st.selectbox(
    "Cargar una transcripción de ejemplo",
    options=["(ninguna)", *[name for name, path in SAMPLES.items() if path.is_file()]],
    key="sample",
    on_change=load_sample,
    help="Las de 'Panadería' forman una conversación de tres turnos: detallada → cambios → especificación adjunta.",
)

params = st.session_state.params
with st.form("turno", border=True):
    transcript = st.text_area(
        "Transcripción de la reunión (o lo nuevo de este turno)",
        key="transcript",
        height=200,
        placeholder="Pega aquí la transcripción, o lo que ha cambiado desde el turno anterior (mínimo 20 caracteres)…",
    )
    uploads = st.file_uploader(
        "Adjuntos: especificaciones, propuestas anteriores, diagramas (PDF, Word .docx, .txt, .md)",
        key=f"uploads_{st.session_state.uploader_round}",  # una key nueva por turno vacía el selector de ficheros
        type=ACCEPTED_TYPES,
        accept_multiple_files=True,
        help=f"Hay ejemplos en {SAMPLE_ATTACHMENTS_DIR.relative_to(BASE_DIR)}/ (genéralos con scripts/generar-adjuntos.py).",
    )
    with st.expander("Parámetros de la estimación", expanded=not st.session_state.turns):
        col_nombre, col_tipo = st.columns(2)
        project_name = col_nombre.text_input(
            "Nombre del proyecto (opcional)",
            max_chars=100,
            placeholder="Si lo dejas vacío, el modelo lo deduce; a partir del segundo turno lo recuerda la memoria",
        )
        tipos = [None, *PROJECT_TYPE_LABELS]
        project_type = col_tipo.selectbox(
            "Tipo de proyecto",
            options=tipos,
            index=tipos.index(params.get("project_type")) if params.get("project_type") in tipos else 0,
            format_func=lambda v: "Que lo deduzca el modelo" if v is None else PROJECT_TYPE_LABELS[v],
        )
        col_tarifa, col_plazo, col_enfoque = st.columns(3)
        hourly_rate_eur = col_tarifa.number_input(
            "Tarifa (€/hora)", min_value=1.0, max_value=1000.0, value=float(params["hourly_rate_eur"]), step=5.0
        )
        deadline_weeks = col_plazo.number_input(
            "Plazo objetivo (semanas)", min_value=0, max_value=104, value=int(params.get("deadline_weeks") or 0), step=1,
            help="0 = sin plazo objetivo",
        )
        modos = list(ESTIMATION_MODE_LABELS)
        estimation_mode = col_enfoque.selectbox(
            "Enfoque", options=modos, index=modos.index(params["estimation_mode"]), format_func=ESTIMATION_MODE_LABELS.get
        )
        niveles = list(DETAIL_LEVEL_LABELS)
        detail_level = st.radio(
            "Nivel de detalle", options=niveles, index=niveles.index(params["detail_level"]),
            format_func=DETAIL_LEVEL_LABELS.get, horizontal=True,
        )
    # La etiqueta no depende del estado: si cambiara entre ejecuciones, el clic llegaría con el id del botón
    # anterior y Streamlit lo descartaría (el turno se perdería).
    submitted = st.form_submit_button(
        "Enviar turno y estimar", type="primary", use_container_width=True, disabled=not health.get("llm_configured")
    )

if submitted:
    try:
        request = EstimationRequest(
            transcription=transcript,
            project_name=project_name,
            project_type=project_type,
            hourly_rate_eur=hourly_rate_eur,
            deadline_weeks=deadline_weeks or None,
            estimation_mode=estimation_mode,
            detail_level=detail_level,
            prompt_version=st.session_state.get("prompt_version") or prompts_info["default_version"],
        )
    except ValidationError as exc:
        for mensaje in format_validation_errors(exc):
            st.error(mensaje)
    else:
        request_params = request.model_dump(exclude={"transcription"})
        st.session_state.params = {name: request_params[name] for name in PARAM_FIELDS if name != "project_name"}
        attachments = [(f.name, f.getvalue(), f.type) for f in (uploads or [])]
        with st.spinner("Estimando… (la API lee los adjuntos, regenera el system prompt con la memoria y llama al LLM)"):
            started = time.perf_counter()
            try:
                response = api().estimate(st.session_state.session_id, request.transcription, request_params, attachments)
            except EstimatorAPIError as exc:
                response = None
                if exc.status_code == 404:  # la API se reinició y perdió la sesión (vive en memoria del proceso)
                    st.warning("La sesión ya no existe en la API (¿se reinició?). Se ha creado una nueva: vuelve a enviar.")
                    new_conversation()
                else:
                    st.error(str(exc))
        if response:
            st.session_state.turns.append(
                {
                    "transcript": request.transcription,
                    "attachments": response.get("attachments") or [],
                    "estimation": response["estimation"],
                    "meta": call_metadata(response, time.perf_counter() - started),
                    "project_metadata": response.get("project_metadata") or {},
                    "params": request_params,
                }
            )
            st.session_state.clear_form = True
            st.session_state.uploader_round += 1
            st.rerun()  # vuelve a pintar con el turno nuevo, el formulario vacío y los paneles actualizados

with turns_area:
    for index, turn in enumerate(st.session_state.turns, start=1):
        render_turn(index, turn)
    if st.session_state.turns:
        last = st.session_state.turns[-1]
        st.download_button(
            "Descargar la última estimación (.md)",
            data=last["estimation"],
            file_name=f"estimacion-{(last['project_metadata'].get('project_name') or 'proyecto').lower().replace(' ', '-')}.md",
            mime="text/markdown",
        )


# --- Barra lateral (se renderiza al final para reflejar el turno recién procesado) --------------------

with st.sidebar:
    st.header("Conversación")
    st.caption(
        f"Sesión `{st.session_state.session_id[:8]}…` · {len(st.session_state.turns)} turno(s) · "
        f"ventana de {st.session_state.max_turns} turnos"
    )
    st.button(
        "🆕 Nueva conversación",
        on_click=new_conversation,
        use_container_width=True,
        help="Crea otra sesión en la API (POST /api/v1/sessions): historial y memoria vuelven a cero.",
    )

    st.divider()
    st.subheader("Servicio")
    st.caption(f"API: `{api().base_url}` · modelo `{health['model']}` ({health['provider']})")
    if not health.get("llm_configured"):
        st.error("La API no tiene API key configurada: define OPENAI_API_KEY o ANTHROPIC_API_KEY en su .env.")
    versions = [t["version"] for t in prompts_info["templates"]]
    default_version = prompts_info["default_version"] if prompts_info["default_version"] in versions else versions[-1]
    prompt_version = st.selectbox(
        "Versión de plantilla",
        key="prompt_version",
        options=versions,
        index=versions.index(default_version),
        format_func=lambda v: f"{v} (por defecto)" if v == prompts_info["default_version"] else v,
        help="v3 inyecta la memoria del proyecto en el system prompt; v1 y v2 solo usan el historial.",
    )
    template = next(t for t in prompts_info["templates"] if t["version"] == prompt_version)
    st.caption(template["description"])
    with st.expander("Changelog"):
        for line in template["changelog"]:
            st.markdown(f"- {line}")

# --- Barra lateral (final): memoria, historial efectivo y métricas de la última llamada ----------

try:
    state = api().get_session(st.session_state.session_id)
except EstimatorAPIError as exc:
    state = None
    if exc.status_code == 404:
        new_conversation()
        st.rerun()

with st.sidebar:
    st.divider()
    st.subheader("Memoria del proyecto")
    st.caption("`project_metadata`: hechos extraídos de cada turno; van en el bloque `<project_metadata>` del system prompt.")
    render_project_metadata(state["project_metadata"] if state else {})

    st.divider()
    st.subheader("Compresión del historial")
    st.caption("Lo que sobrevive a la ventana: anclas (`<anchors>`) y resumen acumulativo (`<summary>`), plantilla `v4`.")
    render_compression(state.get("compression") if state else None, state.get("last_turn_observed") if state else None)

    st.divider()
    st.subheader("Historial efectivo")
    if state:
        window = state["window"]
        st.caption(
            f"Ventana de {window['max_turns']} turnos · {window['turns_in_window']} en ventana · "
            f"{window['dropped_turns']} descartados · {state['turn_count']} turnos en total"
        )
        with st.expander(f"Mensajes que viajarían al LLM ({len(state['history'])}, sin el system prompt)"):
            if not state["history"]:
                st.caption("Vacío.")
            for message in state["history"]:
                icono = "🧑" if message["role"] == "user" else "🤖"
                preview = message["content"].strip().replace("\n", " ")
                st.markdown(f"{icono} `{message['role']}` · {miles(message['chars'])} caracteres")
                st.caption(preview[:160] + ("…" if len(preview) > 160 else ""))

    st.divider()
    st.subheader("Última llamada")
    last_meta = st.session_state.turns[-1]["meta"] if st.session_state.turns else None
    if not last_meta:
        st.caption("Aún no se ha generado ninguna estimación en esta sesión.")
    else:
        st.caption(f"Modelo: `{last_meta['model']}` ({last_meta['provider']}) · plantilla `{last_meta['template']}`")
        col1, col2 = st.columns(2)
        col1.metric("Tokens de entrada", miles(last_meta["input_tokens"]))
        col2.metric("Tokens de salida", miles(last_meta["output_tokens"]))
        col1.metric("Tiempo", f"{last_meta['elapsed_seconds']:.1f} s" if last_meta["elapsed_seconds"] is not None else "?")
        col2.metric("Coste", f"{last_meta['cost_usd']:.6f} $" if last_meta["cost_usd"] is not None else "?")
        if last_meta.get("messages_sent"):
            st.caption(
                f"{last_meta['messages_sent']} mensajes al LLM · system prompt {miles(last_meta['system_prompt_chars'])} "
                f"caracteres · mensaje de usuario {miles(last_meta['user_prompt_chars'])}"
            )
