"""Endpoint HTTP de estimaciones: `POST /api/v1/estimate`."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from app.config import Settings, get_settings
from app.services.llm_service import LLMConfigurationError, LLMProviderError, generate_estimation

router = APIRouter(tags=["Estimaciones"])

TRANSCRIPTION_EXAMPLE = (
    "En la reunión con el equipo de marketing, el cliente explicó que necesita una landing "
    "page con formulario de contacto, integración con su CRM actual (HubSpot), y una sección "
    "de blog con editor WYSIWYG. El plazo ideal sería tenerlo listo en 4 semanas. El diseño "
    "ya existe en Figma."
)


# --- Schemas ----------------------------------------------------------------------


class EstimationRequest(BaseModel):
    transcription: str = Field(
        ...,
        min_length=20,
        description="Texto de la transcripción de la reunión con el cliente.",
        examples=[TRANSCRIPTION_EXAMPLE],
    )

    @field_validator("transcription", mode="before")
    @classmethod
    def _strip_whitespace(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class TokenUsageResponse(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class EstimationResponse(BaseModel):
    estimation: str = Field(description="Estimación generada por el LLM, en Markdown.")
    model: str = Field(description="Modelo utilizado.", examples=["gpt-4o-mini"])
    provider: str = Field(description="Proveedor del LLM.", examples=["openai"])
    usage: TokenUsageResponse | None = Field(default=None, description="Tokens consumidos en la llamada.")
    generated_at: datetime = Field(description="Momento de generación (UTC).")


# --- Endpoint ---------------------------------------------------------------------


@router.post(
    "/estimate",
    response_model=EstimationResponse,
    summary="Genera una estimación de software a partir de una transcripción",
    description=(
        "Recibe la transcripción de una reunión con el cliente, inyecta en el prompt las "
        "estimaciones de ejemplo (arquitectura CAG) y devuelve la estimación generada por el LLM."
    ),
    responses={
        500: {"description": "El servicio no está configurado (falta la API key del proveedor)."},
        502: {"description": "El proveedor del LLM ha devuelto un error."},
    },
)
async def create_estimation(
    payload: EstimationRequest,
    settings: Settings = Depends(get_settings),
) -> EstimationResponse:
    try:
        result = await generate_estimation(payload.transcription, settings)
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except LLMProviderError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    usage = None
    if result.usage is not None:
        usage = TokenUsageResponse(
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            total_tokens=result.usage.total_tokens,
        )
    return EstimationResponse(
        estimation=result.estimation,
        model=result.model,
        provider=result.provider,
        usage=usage,
        generated_at=datetime.now(timezone.utc),
    )
