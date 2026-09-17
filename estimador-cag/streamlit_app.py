"""Interfaz conversacional (Streamlit) para el Estimador CAG.

Ejecutar:
    uv run streamlit run streamlit_app.py

Reutiliza la lógica del proyecto: el mismo system prompt, los mismos ejemplos de contexto (CAG)
y la misma llamada al LLM que el endpoint `POST /api/v1/estimate`, pero mostrando la estimación
en streaming. La API key se lee de `.env` (via `app.config`) o, si no está ahí, de `st.secrets`.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.config import Settings, get_settings
from app.context.examples import ESTIMATION_EXAMPLES
from app.services.llm_service import (
    ChatTurn,
    EstimationResult,
    EstimationStream,
    LLMServiceError,
    build_system_prompt,
)

BASE_DIR = Path(__file__).resolve().parent
PROVIDERS = ("openai", "anthropic")
SAMPLES = {
    "Landing page con HubSpot (enunciado)": BASE_DIR / "transcripciones/reunion-landing-page.txt",
    "Panadería: descripción pobre": BASE_DIR / "transcripciones/panaderia-descripcion-pobre.txt",
    "Panadería: descripción detallada": BASE_DIR / "transcripciones/panaderia-descripcion-detallada.txt",
    "Restaurantes: reservas (reunión larga)": BASE_DIR / "transcripciones/reunion-app-reservas-restaurante.txt",
}


# --- Configuración ---------------------------------------------------------------


def _secret(name: str) -> str | None:
    """Lee un secreto de `.streamlit/secrets.toml`; None si no existe el archivo o la clave."""
    try:
        value = st.secrets.get(name)
    except Exception:  # sin secrets.toml Streamlit lanza una excepción al acceder
        return None
    return str(value).strip() or None if value else None


def resolve_settings() -> tuple[Settings, list[str]]:
    """Settings del proyecto completados con `st.secrets`, y lista de proveedores con API key."""
    base = get_settings()
    keys = {
        "openai": (base.openai_api_key or "").strip() or _secret("OPENAI_API_KEY"),
        "anthropic": (base.anthropic_api_key or "").strip() or _secret("ANTHROPIC_API_KEY"),
    }
    settings = base.model_copy(update={"openai_api_key": keys["openai"], "anthropic_api_key": keys["anthropic"]})
    return settings, [provider for provider in PROVIDERS if keys[provider]]


def call_metadata(result: EstimationResult | None) -> dict | None:
    if result is None:
        return None
    return {
        "model": result.model,
        "provider": result.provider,
        "input_tokens": result.usage.input_tokens if result.usage else None,
        "output_tokens": result.usage.output_tokens if result.usage else None,
        "cost_usd": result.cost.total_usd if result.cost else None,
        "elapsed_seconds": result.elapsed_seconds,
        "system_prompt_chars": result.prompt.system_prompt_chars if result.prompt else None,
    }


def format_metadata(meta: dict) -> str:
    partes = [f"{meta['model']} ({meta['provider']})"]
    if meta.get("input_tokens") is not None:
        partes.append(f"{meta['input_tokens']:,} tokens de entrada · {meta['output_tokens']:,} de salida".replace(",", "."))
    if meta.get("cost_usd") is not None:
        partes.append(f"{meta['cost_usd']:.6f} USD")
    if meta.get("elapsed_seconds") is not None:
        partes.append(f"{meta['elapsed_seconds']:.1f} s")
    return " · ".join(partes)


# --- Estado de la sesión ---------------------------------------------------------

st.set_page_config(page_title="Estimador CAG", page_icon="🧮", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"|"assistant", "content": str, "meta": dict|None}]
if "last_call" not in st.session_state:
    st.session_state.last_call = None

settings, available_providers = resolve_settings()


# --- Barra lateral: configuración, contexto CAG y métricas (Nivel 3) ---------------

sample_to_send: str | None = None

with st.sidebar:
    st.header("Configuración")
    if available_providers:
        default = settings.llm_provider if settings.llm_provider in available_providers else available_providers[0]
        provider = st.selectbox(
            "Proveedor",
            options=available_providers,
            index=available_providers.index(default),
            help="Solo aparecen los proveedores con API key configurada en .env o en st.secrets.",
        )
        settings = settings.model_copy(update={"llm_provider": provider})
    st.caption(f"Modelo: `{settings.model}` · temperatura {settings.llm_temperature} · máx. {settings.llm_max_tokens} tokens")

    include_history = st.toggle(
        "Enviar el historial en cada llamada",
        value=True,
        help=(
            "Activado: cada mensaje nuevo incluye la conversación anterior, así puedes pedir ajustes "
            "sobre la estimación. Desactivado: cada mensaje se trata como una transcripción nueva. "
            "Con historial, los tokens de entrada crecen en cada turno."
        ),
    )

    if st.button("Nueva conversación", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_call = None
        st.rerun()

    st.divider()
    st.subheader("Ejemplos de transcripción")
    sample_name = st.selectbox("Transcripción de ejemplo", options=list(SAMPLES), label_visibility="collapsed")
    if st.button("Enviar este ejemplo", use_container_width=True):
        sample_to_send = SAMPLES[sample_name].read_text(encoding="utf-8").strip()

    st.divider()
    st.subheader("Contexto CAG")
    system_prompt = build_system_prompt()
    with st.expander(f"System prompt activo ({len(system_prompt):,} caracteres)".replace(",", ".")):
        st.text_area("System prompt", value=system_prompt, height=320, disabled=True, label_visibility="collapsed")
    with st.expander(f"Contexto estático inyectado ({len(ESTIMATION_EXAMPLES)} estimaciones de ejemplo)"):
        for indice, ejemplo in enumerate(ESTIMATION_EXAMPLES, start=1):
            st.markdown(f"**Ejemplo {indice} · Reunión:** {ejemplo['meeting_summary']}")
            st.markdown(ejemplo["estimation"])
            st.divider()


# --- Chat (Niveles 1 y 2) ---------------------------------------------------------

st.title("🧮 Estimador CAG")
st.caption(
    "Pega la transcripción de una reunión con el cliente y recibe una estimación de software. "
    "El contexto (estimaciones de ejemplo) viaja en cada llamada: arquitectura CAG."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("meta"):
            st.caption(format_metadata(message["meta"]))

if not settings.api_key:
    st.error(
        f"No hay API key para el proveedor `{settings.llm_provider}`. Define `{settings.api_key_env_var}` "
        "en `.env` o en `.streamlit/secrets.toml` y recarga la página."
    )

prompt = st.chat_input("Pega aquí la transcripción de la reunión…", disabled=not settings.api_key)
if sample_to_send and settings.api_key:
    prompt = sample_to_send

if prompt:
    history = [ChatTurn(role=m["role"], content=m["content"]) for m in st.session_state.messages] if include_history else []

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "meta": None})

    with st.chat_message("assistant"):
        stream = EstimationStream(prompt, settings, history=history)
        try:
            estimation = st.write_stream(stream)  # se va pintando fragmento a fragmento
        except LLMServiceError as exc:
            st.error(str(exc))
            estimation = None
        if estimation:
            meta = call_metadata(stream.result)
            if meta:
                st.caption(format_metadata(meta))
            st.session_state.messages.append({"role": "assistant", "content": estimation, "meta": meta})
            st.session_state.last_call = meta


# --- Métricas de la última llamada (Nivel 3) -------------------------------------

with st.sidebar:
    st.divider()
    st.subheader("Última llamada")
    last = st.session_state.last_call
    if not last:
        st.caption("Aún no se ha generado ninguna estimación.")
    else:
        st.caption(f"Modelo: `{last['model']}` ({last['provider']})")
        col1, col2 = st.columns(2)
        col1.metric("Tokens de entrada", f"{last['input_tokens']:,}".replace(",", ".") if last["input_tokens"] is not None else "?")
        col2.metric("Tokens de salida", f"{last['output_tokens']:,}".replace(",", ".") if last["output_tokens"] is not None else "?")
        col1.metric("Tiempo", f"{last['elapsed_seconds']:.1f} s" if last["elapsed_seconds"] is not None else "?")
        col2.metric("Coste", f"{last['cost_usd']:.6f} $" if last["cost_usd"] is not None else "?")
        if last.get("system_prompt_chars"):
            st.caption(f"Contexto fijo en el prompt: {last['system_prompt_chars']:,} caracteres".replace(",", "."))
