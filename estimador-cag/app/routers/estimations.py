"""Endpoints HTTP: `POST /api/v1/estimate` y `GET /api/v1/prompts`."""

from __future__ import annotations

from datetime import date, datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.schemas.estimation import EstimationRequest
from app.services.llm_service import LLMConfigurationError, LLMProviderError, generate_estimation
from app.services.prompts import PromptTemplateError, PromptTemplateNotFound, list_versions, load_template

router = APIRouter(tags=["Estimaciones"])


# --- Schemas de respuesta ----------------------------------------------------------


class CostResponse(BaseModel):
    input_usd: float = Field(description="Coste de los tokens de entrada.")
    output_usd: float = Field(description="Coste de los tokens de salida.")
    total_usd: float = Field(description="Coste total de la llamada.")
    input_price_per_mtok: float = Field(description="Precio aplicado: USD por millón de tokens de entrada.")
    output_price_per_mtok: float = Field(description="Precio aplicado: USD por millón de tokens de salida.")
    currency: str = "USD"


class TokenUsageResponse(BaseModel):
    input_tokens: int = Field(description="Tokens de entrada: system prompt (contexto CAG) + parámetros + transcripción.")
    output_tokens: int = Field(description="Tokens de salida: la estimación generada.")
    total_tokens: int
    cost: CostResponse | None = Field(
        default=None,
        description="Coste según precios de lista; null si el modelo no está en la tabla de precios.",
    )


class PromptResponse(BaseModel):
    template: str = Field(description="Plantilla de prompt usada.", examples=["estimacion"])
    version: str = Field(description="Versión de la plantilla usada.", examples=["v2"])
    system_prompt_chars: int = Field(
        description="Tamaño del contexto fijo (instrucciones + ejemplos CAG) que viaja en todas las llamadas."
    )
    user_prompt_chars: int = Field(description="Tamaño de la parte variable: parámetros + transcripción.")
    transcription_chars: int = Field(description="Tamaño de la transcripción enviada.")


class EstimationResponse(BaseModel):
    estimation: str = Field(description="Estimación generada por el LLM, en Markdown.")
    model: str = Field(description="Modelo utilizado.", examples=["gpt-4o-mini"])
    provider: str = Field(description="Proveedor del LLM.", examples=["openai"])
    usage: TokenUsageResponse | None = Field(default=None, description="Tokens consumidos y coste de la llamada.")
    prompt: PromptResponse | None = Field(default=None, description="Plantilla usada y tamaño del prompt enviado.")
    generated_at: datetime = Field(description="Momento de generación (UTC).")


class PromptTemplateResponse(BaseModel):
    id: str = Field(examples=["estimacion/v2"])
    name: str
    version: str
    description: str
    created: date | None
    changelog: list[str]
    is_default: bool = Field(description="True para la versión configurada en PROMPT_VERSION.")


class PromptTemplateListResponse(BaseModel):
    default_version: str = Field(description="Versión que se usa si la petición no indica `prompt_version`.")
    templates: list[PromptTemplateResponse]


# --- Endpoints ---------------------------------------------------------------------


@router.post(
    "/estimate",
    response_model=EstimationResponse,
    summary="Genera una estimación de software a partir de una transcripción y unos parámetros",
    description=(
        "Recibe la transcripción de una reunión con el cliente y los parámetros tipados de la estimación "
        "(tarifa, tipo de proyecto, plazo, enfoque, nivel de detalle). Renderiza la plantilla de prompt "
        "versionada (`prompts/estimacion/<versión>`), inyecta las estimaciones de ejemplo (arquitectura CAG) "
        "y devuelve la estimación generada por el LLM."
    ),
    responses={
        422: {"description": "Parámetros inválidos o versión de plantilla inexistente."},
        500: {"description": "El servicio no está configurado (falta la API key del proveedor)."},
        502: {"description": "El proveedor del LLM ha devuelto un error."},
    },
)
async def create_estimation(
    payload: EstimationRequest,
    settings: Settings = Depends(get_settings),
) -> EstimationResponse:
    try:
        result = await generate_estimation(payload, settings)
    except PromptTemplateNotFound as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (LLMConfigurationError, PromptTemplateError) as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except LLMProviderError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    usage = None
    if result.usage is not None:
        cost = None
        if result.cost is not None:
            cost = CostResponse(
                input_usd=result.cost.input_usd,
                output_usd=result.cost.output_usd,
                total_usd=result.cost.total_usd,
                input_price_per_mtok=result.cost.pricing.input_per_mtok,
                output_price_per_mtok=result.cost.pricing.output_per_mtok,
            )
        usage = TokenUsageResponse(
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            total_tokens=result.usage.total_tokens,
            cost=cost,
        )
    prompt = None
    if result.prompt is not None:
        prompt = PromptResponse(
            template=result.prompt.template,
            version=result.prompt.version,
            system_prompt_chars=result.prompt.system_prompt_chars,
            user_prompt_chars=result.prompt.user_prompt_chars,
            transcription_chars=result.prompt.transcription_chars,
        )
    return EstimationResponse(
        estimation=result.estimation,
        model=result.model,
        provider=result.provider,
        usage=usage,
        prompt=prompt,
        generated_at=datetime.now(timezone.utc),
    )


@router.get(
    "/prompts",
    response_model=PromptTemplateListResponse,
    summary="Lista las versiones disponibles de la plantilla de prompt",
    description=(
        "Cada versión es una carpeta en `prompts/estimacion/` con sus metadatos y su changelog. "
        "La versión por defecto se fija con `PROMPT_VERSION`; cualquier otra se puede pedir con "
        "`prompt_version` en el cuerpo de `POST /api/v1/estimate`."
    ),
)
async def list_prompt_templates(settings: Settings = Depends(get_settings)) -> PromptTemplateListResponse:
    templates = []
    for version in list_versions():
        template = load_template("estimacion", version)
        templates.append(
            PromptTemplateResponse(
                id=template.id,
                name=template.name,
                version=template.version,
                description=template.description,
                created=template.created,
                changelog=list(template.changelog),
                is_default=template.version == settings.prompt_version,
            )
        )
    return PromptTemplateListResponse(default_version=settings.prompt_version, templates=templates)
