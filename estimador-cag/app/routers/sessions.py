"""Endpoints de sesión: `POST /api/v1/sessions`, `GET|DELETE /api/v1/sessions/{id}` y
`POST /api/v1/sessions/{id}/estimate` (multipart/form-data con transcripción, parámetros y adjuntos)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field, ValidationError

from app.config import Settings, get_settings
from app.routers.estimations import CostResponse, EstimationResponse, PromptResponse, TokenUsageResponse
from app.schemas.estimation import DetailLevel, EstimationMode, EstimationRequest, ProjectType
from app.services.attachments import (
    AttachmentTooLarge,
    AttachmentUnreadable,
    ExtractedAttachment,
    UnsupportedAttachment,
    extract_text,
)
from app.services.conversation import run_turn
from app.services.llm_service import LLMConfigurationError, LLMProviderError
from app.services.prompts import PromptTemplateError, PromptTemplateNotFound
from app.services.sessions import ProjectMetadata, Session, SessionNotFound, SessionStore, get_session_store

router = APIRouter(prefix="/sessions", tags=["Sesiones"])


# --- Schemas de respuesta ----------------------------------------------------------


class SessionCreatedResponse(BaseModel):
    session_id: str = Field(description="Identificador de la sesión (UUID v4). Viaja en cada petición posterior.")
    created_at: datetime
    max_turns: int = Field(description="Turnos (par user+assistant) que conserva la ventana deslizante del historial.")


class MemoryWindowResponse(BaseModel):
    max_turns: int = Field(description="Tamaño de la ventana deslizante (SESSION_MAX_TURNS).")
    turns_in_window: int = Field(description="Turnos que hay ahora en el historial.")
    dropped_turns: int = Field(description="Turnos que ya han salido de la ventana en esta sesión.")
    messages_sent: int | None = Field(default=None, description="Mensajes enviados al LLM en el último turno, system incluido.")


class HistoryMessageResponse(BaseModel):
    role: str
    chars: int
    content: str


class SessionStateResponse(BaseModel):
    session_id: str
    created_at: datetime
    updated_at: datetime
    turn_count: int = Field(description="Turnos completados en la sesión, hayan salido o no de la ventana.")
    window: MemoryWindowResponse
    project_metadata: ProjectMetadata = Field(description="Memoria del proyecto: los hechos que se inyectan en el system prompt.")
    history: list[HistoryMessageResponse] = Field(description="Historial efectivo (lo que viajaría al LLM, sin el system prompt).")


class AttachmentResponse(BaseModel):
    filename: str
    kind: str = Field(examples=["pdf"])
    size_bytes: int
    chars: int = Field(description="Caracteres de texto extraído que se añadieron al prompt.")
    pages: int | None = None
    truncated: bool
    warning: str | None = None


class SessionEstimationResponse(EstimationResponse):
    """La misma estimación que `POST /api/v1/estimate` más el estado de la sesión tras el turno."""

    session_id: str
    turn: int = Field(description="Número de este turno dentro de la sesión.")
    window: MemoryWindowResponse
    project_metadata: ProjectMetadata
    attachments: list[AttachmentResponse]


# --- Utilidades ---------------------------------------------------------------------


def _get_session(session_id: str, store: SessionStore) -> Session:
    try:
        return store.get(session_id)
    except SessionNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


def _state(session: Session) -> SessionStateResponse:
    return SessionStateResponse(
        session_id=session.session_id,
        created_at=session.created_at,
        updated_at=session.updated_at,
        turn_count=session.turn_count,
        window=MemoryWindowResponse(
            max_turns=session.history.max_turns,
            turns_in_window=len(session.history),
            dropped_turns=session.history.dropped_turns,
        ),
        project_metadata=session.metadata,
        history=[
            HistoryMessageResponse(role=m["role"], chars=len(m["content"]), content=m["content"])
            for m in session.history.messages()
        ],
    )


def _validation_error(errors: list[dict]) -> HTTPException:
    """422 con el mismo formato que FastAPI, renombrando `transcription` al campo del formulario (`transcript`)."""
    detail = []
    for error in errors:
        loc = ["body", *("transcript" if part == "transcription" else part for part in error.get("loc", ()))]
        detail.append({"loc": loc, "msg": error.get("msg"), "type": error.get("type")})
    return HTTPException(status_code=422, detail=detail)


async def _read_attachments(uploads: list[UploadFile] | None, settings: Settings) -> list[ExtractedAttachment]:
    uploads = [u for u in (uploads or []) if u.filename or u.size]  # algunos clientes mandan un campo vacío
    if len(uploads) > settings.attachment_max_files:
        raise HTTPException(
            status_code=422,
            detail=f"Demasiados adjuntos: {len(uploads)} (máximo {settings.attachment_max_files})",
        )
    extracted = []
    for upload in uploads:
        data = await upload.read()
        try:
            extracted.append(
                extract_text(
                    upload.filename or "adjunto",
                    data,
                    content_type=upload.content_type,
                    max_bytes=settings.attachment_max_bytes,
                    max_chars=settings.attachment_max_chars,
                )
            )
        except UnsupportedAttachment as exc:
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc
        except AttachmentTooLarge as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except AttachmentUnreadable as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return extracted


# --- Endpoints ---------------------------------------------------------------------


@router.post(
    "",
    response_model=SessionCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Crea una sesión de estimación vacía",
    description=(
        "Devuelve un `session_id` (UUID v4) que el cliente envía en cada `POST /api/v1/sessions/{session_id}/estimate`. "
        "La sesión guarda el historial de la conversación (ventana deslizante de `SESSION_MAX_TURNS` turnos) y la "
        "memoria del proyecto (`project_metadata`). Vive en memoria del proceso: se pierde al reiniciar el servicio."
    ),
)
async def create_session(
    settings: Settings = Depends(get_settings),
    store: SessionStore = Depends(get_session_store),
) -> SessionCreatedResponse:
    store.max_sessions = settings.session_max_sessions
    session = store.create(max_turns=settings.session_max_turns)
    return SessionCreatedResponse(
        session_id=session.session_id, created_at=session.created_at, max_turns=session.history.max_turns
    )


@router.get(
    "/{session_id}",
    response_model=SessionStateResponse,
    summary="Estado de una sesión: memoria del proyecto e historial efectivo",
    description="Para depurar y para el panel del cliente: qué sabe el sistema del proyecto y qué turnos siguen en la ventana.",
    responses={404: {"description": "La sesión no existe (o el servicio se reinició)."}},
)
async def get_session_state(session_id: str, store: SessionStore = Depends(get_session_store)) -> SessionStateResponse:
    return _state(_get_session(session_id, store))


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Elimina una sesión",
    responses={404: {"description": "La sesión no existe."}},
)
async def delete_session(session_id: str, store: SessionStore = Depends(get_session_store)) -> None:
    if not store.delete(session_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No existe la sesión '{session_id}'")


@router.post(
    "/{session_id}/estimate",
    response_model=SessionEstimationResponse,
    summary="Turno de estimación dentro de una sesión (transcripción + adjuntos, multipart/form-data)",
    description=(
        "Acepta `multipart/form-data` con la transcripción (`transcript`), los mismos parámetros tipados que "
        "`POST /api/v1/estimate` como campos de formulario y una lista opcional de adjuntos (`attachments`: PDF, "
        "Word .docx o texto). El texto de los adjuntos se extrae localmente y se concatena a la transcripción con "
        "separadores `--- attachment: <fichero> ---`. El system prompt se regenera con la memoria del proyecto "
        "(`project_metadata`), el historial de la ventana viaja como turnos anteriores y, tras la respuesta, "
        "historial y memoria se actualizan."
    ),
    responses={
        404: {"description": "La sesión no existe."},
        413: {"description": "Un adjunto supera el tamaño máximo."},
        415: {"description": "Formato de adjunto no soportado."},
        422: {"description": "Parámetros inválidos, versión de plantilla inexistente o adjunto ilegible."},
        500: {"description": "El servicio no está configurado (falta la API key del proveedor)."},
        502: {"description": "El proveedor del LLM ha devuelto un error."},
    },
)
async def estimate_in_session(
    session_id: str,
    transcript: Annotated[str, Form(description="Transcripción de la reunión (mínimo 20 caracteres).")],
    attachments: Annotated[
        list[UploadFile] | None,
        File(description="Documentación complementaria: PDF, Word (.docx) o texto (.txt, .md). Opcional."),
    ] = None,
    project_name: Annotated[str | None, Form()] = None,
    project_type: Annotated[ProjectType | None, Form()] = None,
    hourly_rate_eur: Annotated[float | None, Form()] = None,
    deadline_weeks: Annotated[int | None, Form()] = None,
    estimation_mode: Annotated[EstimationMode | None, Form()] = None,
    detail_level: Annotated[DetailLevel | None, Form()] = None,
    prompt_version: Annotated[str | None, Form()] = None,
    settings: Settings = Depends(get_settings),
    store: SessionStore = Depends(get_session_store),
) -> SessionEstimationResponse:
    session = _get_session(session_id, store)

    # Los campos del formulario se validan con el mismo contrato que el endpoint JSON (EstimationRequest).
    fields = {
        "transcription": transcript,
        "project_name": project_name,
        "project_type": project_type,
        "hourly_rate_eur": hourly_rate_eur,
        "deadline_weeks": deadline_weeks,
        "estimation_mode": estimation_mode,
        "detail_level": detail_level,
        "prompt_version": prompt_version,
    }
    try:
        request = EstimationRequest(**{k: v for k, v in fields.items() if v is not None and v != ""})
    except ValidationError as exc:
        raise _validation_error(exc.errors(include_url=False)) from exc

    extracted = await _read_attachments(attachments, settings)

    try:
        turn = await run_turn(session, request, extracted, settings)
    except PromptTemplateNotFound as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (LLMConfigurationError, PromptTemplateError) as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except LLMProviderError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    result = turn.result
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
    return SessionEstimationResponse(
        estimation=result.estimation,
        model=result.model,
        provider=result.provider,
        usage=usage,
        prompt=prompt,
        generated_at=datetime.now(timezone.utc),
        session_id=session.session_id,
        turn=turn.turn,
        window=MemoryWindowResponse(
            max_turns=session.history.max_turns,
            turns_in_window=turn.turns_in_window,
            dropped_turns=turn.dropped_turns,
            messages_sent=turn.messages_sent,
        ),
        project_metadata=turn.project_metadata,
        attachments=[
            AttachmentResponse(
                filename=a.filename, kind=a.kind, size_bytes=a.size_bytes, chars=a.chars,
                pages=a.pages, truncated=a.truncated, warning=a.warning,
            )
            for a in turn.attachments
        ],
    )
