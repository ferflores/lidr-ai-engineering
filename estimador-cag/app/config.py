"""Configuración de la aplicación.

Todas las variables se cargan desde el entorno o desde el archivo `.env` de la
raíz del proyecto usando Pydantic BaseSettings. Las API keys nunca se escriben
en el código: solo viven en `.env` (que está en `.gitignore`).
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

# Exporta el contenido de `.env` a os.environ (sin sobrescribir variables ya
# definidas). pydantic-settings también lee `.env` por su cuenta; hacerlo aquí
# además deja las variables disponibles para cualquier otra librería.
load_dotenv(ENV_FILE)

Provider = Literal["openai", "anthropic"]


class Settings(BaseSettings):
    """Variables de entorno del proyecto (ver `.env.example`)."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Aplicación ---------------------------------------------------------
    app_name: str = "Estimador CAG"
    app_version: str = "0.1.0"

    # --- Proveedor LLM ------------------------------------------------------
    llm_provider: Provider = Field(
        default="openai",
        description="Proveedor a usar para generar las estimaciones: openai | anthropic",
    )
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # Modelos económicos recomendados por el curso para este ejercicio.
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-haiku-4-5"

    # --- Prompt --------------------------------------------------------------
    # Versión de la plantilla de prompt (carpeta prompts/estimacion/<versión>). Se fija de forma
    # explícita: cambiar el prompt es cambiar de versión, nunca editar la publicada.
    prompt_version: str = Field(
        default="v4",
        pattern=r"^v\d+$",
        description="Versión de la plantilla de prompt que usan la API y el formulario por defecto.",
    )

    # --- Sesiones conversacionales (sesión 05) ----------------------------------
    # Ventana deslizante del historial: cuántos turnos (par user+assistant) se conservan por sesión.
    session_max_turns: int = Field(
        default=6,
        ge=1,
        le=50,
        description="Turnos que se conservan en el historial de cada sesión (MAX_TURNS de la ventana deslizante).",
    )
    # Tope de sesiones vivas en memoria del proceso; al superarlo se expulsa la menos reciente.
    session_max_sessions: int = Field(default=500, ge=1)

    # --- Compresión del historial (sesión 05, directo): anclas y resumen acumulativo -----------
    # Anclas heurísticas: hechos cortos (nombre, presupuesto, plazo, stack, requisitos, cambios) que se
    # extraen de cada mensaje del usuario y viajan en el system prompt aunque el turno salga de la ventana.
    anchors_max: int = Field(default=24, ge=0, description="Anclas que se conservan por sesión (0 desactiva las anclas).")
    anchors_per_turn: int = Field(default=6, ge=1, description="Anclas máximas que puede aportar un solo turno.")
    # Resumen acumulativo: cuando un turno sale de la ventana se pliega en un resumen que viaja en el
    # system prompt. `llm` lo hace con una llamada extra (plantilla prompts/resumen/v1, se paga y se mide),
    # `heuristic` recorta el turno sin llamar a nadie, `off` lo descarta sin más (comportamiento de la sesión 05 pre-work).
    summary_mode: Literal["llm", "heuristic", "off"] = Field(default="llm")
    summary_max_chars: int = Field(default=1500, ge=200, description="Longitud máxima del resumen acumulado.")
    summary_prompt_version: str = Field(default="v1", pattern=r"^v\d+$")

    # --- Tier dinámico: qué modelo y cuántos tokens de salida según el contexto del turno -------------
    # Las reglas están en app/services/tiering.py; aquí solo los umbrales y a qué modelo va cada tier.
    # Por defecto los dos tiers usan el mismo modelo (el del proveedor): el tier se resuelve y se registra
    # (last_resolved_tier / last_tier_rule) sin cambiar de modelo, que es lo que mantiene comparables las medidas.
    tier_extended_min_context_chars: int = Field(default=40_000, gt=0, description="Caracteres de contexto (system + mensajes) a partir de los que el turno se considera 'extended'.")
    tier_extended_min_attachment_chars: int = Field(default=15_000, gt=0, description="Caracteres de adjuntos a partir de los que el turno se considera 'extended'.")
    tier_extended_model: str | None = Field(default=None, description="Modelo del tier 'extended' (por defecto, el mismo que el estándar).")
    tier_extended_max_tokens: int | None = Field(default=None, gt=0, description="Tokens de salida del tier 'extended' (por defecto, LLM_MAX_TOKENS).")

    # --- Caché de respuestas (exacta y semántica) ------------------------------------------------
    response_cache_enabled: bool = Field(default=True)
    response_cache_max_entries: int = Field(default=256, ge=1)
    response_cache_ttl_seconds: float = Field(default=3600.0, gt=0)
    semantic_cache_threshold: float = Field(default=0.9, ge=0.0, le=1.0, description="Similitud (Jaccard sobre palabras) mínima para un acierto semántico; 1.0 lo desactiva en la práctica.")

    # --- Observabilidad ------------------------------------------------------------------------
    log_format: Literal["json", "console"] = Field(default="json", description="Formato de los eventos structlog (turn_observed...).")

    # --- Adjuntos (PDF, Word, texto) -------------------------------------------
    attachment_max_files: int = Field(default=5, ge=1, le=20, description="Adjuntos máximos por petición.")
    attachment_max_bytes: int = Field(default=10 * 1024 * 1024, gt=0, description="Tamaño máximo de cada adjunto, en bytes.")
    # 60.000 es el tope de referencia del curso (MAX_ATTACHMENT_CHARS); el pre-work de la 05 usaba 20.000.
    attachment_max_chars: int = Field(
        default=60_000,
        gt=0,
        description="Caracteres de texto extraído que se conservan por adjunto; el resto se recorta (y se avisa).",
    )

    # --- Parámetros del modelo ---------------------------------------------
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=4096, gt=0)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)

    # --- Precios (USD por millón de tokens) para calcular el coste de cada llamada ---
    # Si no se definen, se usa la tabla de app/services/pricing.py según el modelo activo.
    llm_input_price_per_mtok: float | None = Field(default=None, ge=0)
    llm_output_price_per_mtok: float | None = Field(default=None, ge=0)

    # --- Propiedades derivadas ---------------------------------------------
    @property
    def model(self) -> str:
        """Nombre del modelo que corresponde al proveedor activo."""
        return self.openai_model if self.llm_provider == "openai" else self.anthropic_model

    @property
    def api_key(self) -> str | None:
        """API key del proveedor activo, o None si no está configurada."""
        key = self.openai_api_key if self.llm_provider == "openai" else self.anthropic_api_key
        key = (key or "").strip()
        return key or None

    @property
    def api_key_env_var(self) -> str:
        """Nombre de la variable de entorno que debería contener la API key."""
        return "OPENAI_API_KEY" if self.llm_provider == "openai" else "ANTHROPIC_API_KEY"

    @property
    def llm_configured(self) -> bool:
        return self.api_key is not None


@lru_cache
def get_settings() -> Settings:
    """Instancia única de la configuración (se usa como dependencia de FastAPI)."""
    return Settings()
