"""Parámetros tipados de una estimación.

Es el contrato de entrada del sistema, compartido por el endpoint `POST /api/v1/estimate` y por el
formulario de Streamlit: los dos construyen un `EstimationRequest`, así que la validación (rangos,
valores permitidos, campos desconocidos) es la misma se entre por donde se entre.

Los valores de los campos cerrados (`project_type`, `estimation_mode`, `detail_level`) son los que
interpretan las plantillas de `prompts/`; las etiquetas legibles de abajo son solo para la interfaz.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ProjectType = Literal["web", "movil", "api", "datos", "otro"]
EstimationMode = Literal["conservadora", "equilibrada", "optimista"]
DetailLevel = Literal["completa", "resumen"]

PROJECT_TYPE_LABELS: dict[str, str] = {
    "web": "Aplicación web",
    "movil": "Aplicación móvil",
    "api": "API / backend",
    "datos": "Datos / analítica",
    "otro": "Otro",
}
ESTIMATION_MODE_LABELS: dict[str, str] = {
    "conservadora": "Conservadora (contingencia del 20 %)",
    "equilibrada": "Equilibrada (contingencia del 10 %)",
    "optimista": "Optimista (sin contingencia)",
}
DETAIL_LEVEL_LABELS: dict[str, str] = {
    "completa": "Completa: desglose por tareas, supuestos y preguntas pendientes",
    "resumen": "Resumen: desglose por bloques, sin preguntas pendientes",
}

TRANSCRIPTION_EXAMPLE = (
    "En la reunión con el equipo de marketing, el cliente explicó que necesita una landing "
    "page con formulario de contacto, integración con su CRM actual (HubSpot), y una sección "
    "de blog con editor WYSIWYG. El plazo ideal sería tenerlo listo en 4 semanas. El diseño "
    "ya existe en Figma."
)


class EstimationRequest(BaseModel):
    """Lo que hace falta para pedir una estimación: la transcripción y los parámetros que la condicionan."""

    model_config = ConfigDict(extra="forbid")  # un campo desconocido es un error, no se ignora en silencio

    transcription: str = Field(
        ...,
        min_length=20,
        description="Texto de la transcripción de la reunión con el cliente.",
        examples=[TRANSCRIPTION_EXAMPLE],
    )
    project_name: str | None = Field(
        default=None,
        max_length=100,
        description="Nombre del proyecto para el título de la estimación. Si se omite, el modelo lo deduce.",
        examples=["Landing page corporativa"],
    )
    project_type: ProjectType | None = Field(
        default=None,
        description="Tipo de proyecto; condiciona las tareas transversales. Si se omite, el modelo lo deduce.",
        examples=["web"],
    )
    hourly_rate_eur: float = Field(
        default=50,
        gt=0,
        le=1000,
        description="Tarifa en euros por hora con la que se calcula el coste.",
        examples=[50],
    )
    deadline_weeks: int | None = Field(
        default=None,
        ge=1,
        le=104,
        description="Plazo objetivo en semanas. Si se indica, la estimación valora si es viable.",
        examples=[4],
    )
    estimation_mode: EstimationMode = Field(
        default="conservadora",
        description="Enfoque de la estimación: cuánta contingencia se añade y qué escenarios se asumen.",
    )
    detail_level: DetailLevel = Field(
        default="completa",
        description="Formato de salida: `completa` (desglose por tareas) o `resumen` (por bloques, menos tokens).",
    )
    prompt_version: str | None = Field(
        default=None,
        pattern=r"^v\d+$",
        description="Versión de la plantilla de prompt (`prompts/estimacion/<versión>`). Por defecto, PROMPT_VERSION.",
        examples=["v2"],
    )

    @field_validator("transcription", "project_name", mode="before")
    @classmethod
    def _strip_whitespace(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("project_name", mode="after")
    @classmethod
    def _empty_name_is_none(cls, value: str | None) -> str | None:
        return value or None

    def prompt_variables(self) -> dict[str, object]:
        """Variables que reciben las plantillas: todos los parámetros menos la versión, que elige la plantilla."""
        return self.model_dump(exclude={"prompt_version"})
