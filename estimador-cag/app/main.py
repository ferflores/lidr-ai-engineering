"""Punto de entrada de la aplicación FastAPI.

Arranque en desarrollo:
    uv run uvicorn app.main:app --reload
"""

import logging
from datetime import datetime, timezone

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.routers import estimations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

DESCRIPTION = """
Servicio que recibe la **transcripción de una reunión** con un cliente y devuelve una
**estimación de software** (desglose de tareas, horas, coste, equipo y duración)
generada por un LLM.

Arquitectura **CAG**: el contexto que necesita el modelo (estimaciones de ejemplo)
viaja íntegro dentro del prompt en cada llamada. No hay base de datos ni retrieval.

Flujo: `transcripción → system prompt con ejemplos → LLM → estimación en Markdown`.
"""

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=DESCRIPTION,
    version=settings.app_version,
    openapi_tags=[
        {"name": "Estimaciones", "description": "Generación de estimaciones de software con un LLM."},
        {"name": "Sistema", "description": "Estado y diagnóstico del servicio."},
    ],
)

app.include_router(estimations.router, prefix="/api/v1")


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    provider: str
    model: str
    llm_configured: bool
    timestamp: datetime


@app.get("/health", response_model=HealthResponse, tags=["Sistema"], summary="Estado del servicio")
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Devuelve 200 si el servicio está levantado, junto con el proveedor y modelo configurados."""
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        provider=settings.llm_provider,
        model=settings.model,
        llm_configured=settings.llm_configured,
        timestamp=datetime.now(timezone.utc),
    )
