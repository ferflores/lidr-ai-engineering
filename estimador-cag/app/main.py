"""Punto de entrada de la aplicación FastAPI.

Arranque en desarrollo:
    uv run uvicorn app.main:app --reload
"""

import logging
from datetime import datetime, timezone

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.routers import estimations, sessions
from app.services.sessions import get_session_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

DESCRIPTION = """
Servicio que recibe la **transcripción de una reunión** con un cliente más unos
**parámetros tipados** (tarifa, tipo de proyecto, plazo, enfoque, nivel de detalle) y devuelve
una **estimación de software** (desglose de tareas, horas, coste, equipo y duración)
generada por un LLM.

Arquitectura **CAG**: el contexto que necesita el modelo (estimaciones de ejemplo)
viaja íntegro dentro del prompt en cada llamada. No hay base de datos ni retrieval.

Los prompts son **plantillas versionadas** en `prompts/estimacion/<versión>/`, fuera del código.
`GET /api/v1/prompts` lista las versiones; `prompt_version` en la petición elige una concreta.

Flujo: `parámetros + transcripción → plantilla vN (system + user) → LLM → estimación en Markdown`.

**Sesiones conversacionales** (sesión 05): `POST /api/v1/sessions` crea una sesión y
`POST /api/v1/sessions/{session_id}/estimate` (multipart/form-data) encadena turnos sobre el mismo proyecto,
con **adjuntos** (PDF, Word, texto) cuyo texto se extrae localmente, un **historial con ventana deslizante**
(`SESSION_MAX_TURNS`) y una **memoria del proyecto** (`project_metadata`) que se inyecta en el system prompt.
Las sesiones viven en memoria del proceso.
"""

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=DESCRIPTION,
    version=settings.app_version,
    openapi_tags=[
        {"name": "Estimaciones", "description": "Generación de estimaciones de software con un LLM (una petición, una estimación)."},
        {"name": "Sesiones", "description": "Conversaciones multi-turno con memoria del proyecto y adjuntos."},
        {"name": "Sistema", "description": "Estado y diagnóstico del servicio."},
    ],
)

app.include_router(estimations.router, prefix="/api/v1")
app.include_router(sessions.router, prefix="/api/v1")


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    provider: str
    model: str
    llm_configured: bool
    prompt_template: str
    session_max_turns: int
    active_sessions: int
    timestamp: datetime


@app.get("/health", response_model=HealthResponse, tags=["Sistema"], summary="Estado del servicio")
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Devuelve 200 si el servicio está levantado, con el proveedor, el modelo y la plantilla de prompt configurados."""
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        provider=settings.llm_provider,
        model=settings.model,
        llm_configured=settings.llm_configured,
        prompt_template=f"estimacion/{settings.prompt_version}",
        session_max_turns=settings.session_max_turns,
        active_sessions=len(get_session_store()),
        timestamp=datetime.now(timezone.utc),
    )
