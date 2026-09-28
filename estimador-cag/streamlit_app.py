"""Interfaz web (Streamlit) del Estimador CAG: un formulario con parámetros tipados.

Ejecutar:
    uv run streamlit run streamlit_app.py

Sesión 04: el chat deja de ser la interfaz. Pedir una estimación es rellenar un formulario cuyos campos
son exactamente los del modelo `EstimationRequest` (el mismo que valida el endpoint REST), y cada envío
es una llamada independiente y reproducible: misma petición + misma versión de plantilla = mismo prompt.
La estimación se muestra en streaming y con sus métricas de tokens, tiempo y coste.
La API key se lee de `.env` (via `app.config`) o, si no está ahí, de `st.secrets`.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.context.examples import ESTIMATION_EXAMPLES
from app.schemas.estimation import (
    DETAIL_LEVEL_LABELS,
    ESTIMATION_MODE_LABELS,
    PROJECT_TYPE_LABELS,
    EstimationRequest,
)
from app.services.llm_service import EstimationResult, EstimationStream, LLMServiceError, build_prompt
from app.services.prompts import PromptTemplateError, list_versions, load_template

BASE_DIR = Path(__file__).resolve().parent
PROVIDERS = ("openai", "anthropic")
SAMPLES = {
    "Landing page con HubSpot (enunciado)": BASE_DIR / "transcripciones/reunion-landing-page.txt",
    "Panadería: descripción pobre": BASE_DIR / "transcripciones/panaderia-descripcion-pobre.txt",
    "Panadería: descripción detallada": BASE_DIR / "transcripciones/panaderia-descripcion-detallada.txt",
    "Restaurantes: reservas (reunión larga)": BASE_DIR / "transcripciones/reunion-app-reservas-restaurante.txt",
}
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
        "template": f"{result.prompt.template}/{result.prompt.version}" if result.prompt else None,
        "input_tokens": result.usage.input_tokens if result.usage else None,
        "output_tokens": result.usage.output_tokens if result.usage else None,
        "cost_usd": result.cost.total_usd if result.cost else None,
        "elapsed_seconds": result.elapsed_seconds,
        "system_prompt_chars": result.prompt.system_prompt_chars if result.prompt else None,
        "user_prompt_chars": result.prompt.user_prompt_chars if result.prompt else None,
    }


def format_metadata(meta: dict) -> str:
    partes = [f"{meta['model']} ({meta['provider']})"]
    if meta.get("template"):
        partes.append(f"plantilla {meta['template']}")
    if meta.get("input_tokens") is not None:
        partes.append(f"{meta['input_tokens']:,} tokens de entrada · {meta['output_tokens']:,} de salida".replace(",", "."))
    if meta.get("cost_usd") is not None:
        partes.append(f"{meta['cost_usd']:.6f} USD")
    if meta.get("elapsed_seconds") is not None:
        partes.append(f"{meta['elapsed_seconds']:.1f} s")
    return " · ".join(partes)


def format_validation_errors(exc: ValidationError) -> list[str]:
    """Errores de Pydantic como frases legibles: 'Transcripción: String should have at least 20 characters'."""
    mensajes = []
    for error in exc.errors():
        campo = str(error["loc"][0]) if error["loc"] else "petición"
        mensajes.append(f"{FIELD_LABELS.get(campo, campo)}: {error['msg']}")
    return mensajes


def load_sample() -> None:
    """Callback del selector de ejemplos: vuelca el texto elegido en el campo de transcripción."""
    name = st.session_state.get("sample")
    if name in SAMPLES:
        st.session_state["transcription"] = SAMPLES[name].read_text(encoding="utf-8").strip()


# --- Estado de la sesión ---------------------------------------------------------

st.set_page_config(page_title="Estimador CAG", page_icon="🧮", layout="wide")

if "last" not in st.session_state:
    st.session_state.last = None  # {"request": dict, "estimation": str, "meta": dict}

settings, available_providers = resolve_settings()


# --- Barra lateral: proveedor, plantilla de prompt y contexto CAG --------------------

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

    st.divider()
    st.subheader("Plantilla de prompt")
    versions = list_versions()
    default_version = settings.prompt_version if settings.prompt_version in versions else versions[-1]
    prompt_version = st.selectbox(
        "Versión",
        key="prompt_version",
        options=versions,
        index=versions.index(default_version),
        format_func=lambda v: f"{v} (por defecto)" if v == settings.prompt_version else v,
        help="Cada versión es una carpeta en prompts/estimacion/. La versión por defecto se fija con PROMPT_VERSION.",
    )
    template = load_template("estimacion", prompt_version)
    st.caption(template.description)
    if template.created:
        st.caption(f"Creada el {template.created.isoformat()}")
    with st.expander("Changelog"):
        for line in template.changelog:
            st.markdown(f"- {line}")
    with st.expander("Plantilla system.md.j2"):
        st.code(template.system_source, language="jinja2", wrap_lines=True)
    with st.expander("Plantilla user.md.j2"):
        st.code(template.user_source, language="jinja2", wrap_lines=True)

    st.divider()
    st.subheader("Contexto CAG")
    with st.expander(f"Contexto estático inyectado ({len(ESTIMATION_EXAMPLES)} estimaciones de ejemplo)"):
        for indice, ejemplo in enumerate(ESTIMATION_EXAMPLES, start=1):
            st.markdown(f"**Ejemplo {indice} · Reunión:** {ejemplo['meeting_summary']}")
            st.markdown(ejemplo["estimation"])
            st.divider()


# --- Formulario -----------------------------------------------------------------

st.title("🧮 Estimador CAG")
st.caption(
    "Rellena los parámetros y pega la transcripción de la reunión: cada envío es una estimación independiente "
    "y reproducible. El contexto (estimaciones de ejemplo) viaja en cada llamada: arquitectura CAG."
)

if not settings.api_key:
    st.error(
        f"No hay API key para el proveedor `{settings.llm_provider}`. Define `{settings.api_key_env_var}` "
        "en `.env` o en `.streamlit/secrets.toml` y recarga la página."
    )

st.selectbox(
    "Cargar una transcripción de ejemplo",
    options=["(ninguna)", *SAMPLES],
    key="sample",
    on_change=load_sample,
    help="Rellena el campo de transcripción con uno de los ejemplos del proyecto. Puedes editarlo antes de enviar.",
)

with st.form("estimacion", border=True):
    transcription = st.text_area(
        "Transcripción de la reunión",
        key="transcription",
        height=220,
        placeholder="Pega aquí la transcripción de la reunión con el cliente (mínimo 20 caracteres)…",
    )
    col_nombre, col_tipo = st.columns(2)
    project_name = col_nombre.text_input(
        "Nombre del proyecto (opcional)",
        max_chars=100,
        placeholder="Si lo dejas vacío, el modelo lo deduce",
    )
    project_type = col_tipo.selectbox(
        "Tipo de proyecto",
        options=[None, *PROJECT_TYPE_LABELS],
        format_func=lambda v: "Que lo deduzca el modelo" if v is None else PROJECT_TYPE_LABELS[v],
    )
    col_tarifa, col_plazo, col_enfoque = st.columns(3)
    hourly_rate_eur = col_tarifa.number_input(
        "Tarifa (€/hora)", min_value=1.0, max_value=1000.0, value=float(DEFAULTS["hourly_rate_eur"]), step=5.0
    )
    deadline_weeks = col_plazo.number_input(
        "Plazo objetivo (semanas)", min_value=0, max_value=104, value=0, step=1, help="0 = sin plazo objetivo"
    )
    estimation_mode = col_enfoque.selectbox(
        "Enfoque",
        options=list(ESTIMATION_MODE_LABELS),
        index=list(ESTIMATION_MODE_LABELS).index(DEFAULTS["estimation_mode"]),
        format_func=ESTIMATION_MODE_LABELS.get,
    )
    detail_level = st.radio(
        "Nivel de detalle",
        options=list(DETAIL_LEVEL_LABELS),
        index=list(DETAIL_LEVEL_LABELS).index(DEFAULTS["detail_level"]),
        format_func=DETAIL_LEVEL_LABELS.get,
        horizontal=True,
    )
    submitted = st.form_submit_button(
        "Generar estimación", type="primary", use_container_width=True, disabled=not settings.api_key
    )


# --- Envío: validación tipada, streaming y métricas ----------------------------------


def render_result(request: EstimationRequest, estimation: str, meta: dict | None, settings: Settings) -> None:
    if meta:
        st.caption(format_metadata(meta))
    st.download_button(
        "Descargar estimación (.md)",
        data=estimation,
        file_name=f"estimacion-{(request.project_name or 'proyecto').lower().replace(' ', '-')}.md",
        mime="text/markdown",
    )
    try:
        prompt = build_prompt(request, settings)
    except PromptTemplateError as exc:
        st.warning(f"No se pudo reconstruir el prompt enviado: {exc}")
        return
    with st.expander(f"Prompt enviado (plantilla {prompt.template.id})"):
        st.markdown(f"**System prompt** · {len(prompt.system):,} caracteres (contexto fijo)".replace(",", "."))
        st.code(prompt.system, language="markdown", wrap_lines=True)
        st.markdown(f"**Mensaje de usuario** · {len(prompt.user):,} caracteres (parámetros + transcripción)".replace(",", "."))
        st.code(prompt.user, language="markdown", wrap_lines=True)


if submitted:
    try:
        request = EstimationRequest(
            transcription=transcription,
            project_name=project_name,
            project_type=project_type,
            hourly_rate_eur=hourly_rate_eur,
            deadline_weeks=deadline_weeks or None,
            estimation_mode=estimation_mode,
            detail_level=detail_level,
            prompt_version=prompt_version,
        )
    except ValidationError as exc:
        for mensaje in format_validation_errors(exc):
            st.error(mensaje)
    else:
        with st.container(border=True):
            st.subheader("Estimación")
            stream = EstimationStream(request, settings)
            try:
                estimation = st.write_stream(stream)  # se va pintando fragmento a fragmento
            except (LLMServiceError, PromptTemplateError) as exc:
                st.error(str(exc))
                estimation = None
            if estimation:
                meta = call_metadata(stream.result)
                st.session_state.last = {"request": request.model_dump(), "estimation": estimation, "meta": meta}
                render_result(request, estimation, meta, settings)
elif st.session_state.last:
    last = st.session_state.last
    with st.container(border=True):
        st.subheader("Última estimación")
        st.markdown(last["estimation"])
        render_result(EstimationRequest(**last["request"]), last["estimation"], last["meta"], settings)


# --- Métricas de la última llamada -------------------------------------------------

with st.sidebar:
    st.divider()
    st.subheader("Última llamada")
    last_meta = st.session_state.last["meta"] if st.session_state.last else None
    if not last_meta:
        st.caption("Aún no se ha generado ninguna estimación.")
    else:
        st.caption(f"Modelo: `{last_meta['model']}` ({last_meta['provider']}) · plantilla `{last_meta['template']}`")
        col1, col2 = st.columns(2)
        col1.metric("Tokens de entrada", f"{last_meta['input_tokens']:,}".replace(",", ".") if last_meta["input_tokens"] is not None else "?")
        col2.metric("Tokens de salida", f"{last_meta['output_tokens']:,}".replace(",", ".") if last_meta["output_tokens"] is not None else "?")
        col1.metric("Tiempo", f"{last_meta['elapsed_seconds']:.1f} s" if last_meta["elapsed_seconds"] is not None else "?")
        col2.metric("Coste", f"{last_meta['cost_usd']:.6f} $" if last_meta["cost_usd"] is not None else "?")
        if last_meta.get("system_prompt_chars"):
            st.caption(
                f"Prompt: {last_meta['system_prompt_chars']:,} caracteres fijos (system) + "
                f"{last_meta['user_prompt_chars']:,} variables (user)".replace(",", ".")
            )
