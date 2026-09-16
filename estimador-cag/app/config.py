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
