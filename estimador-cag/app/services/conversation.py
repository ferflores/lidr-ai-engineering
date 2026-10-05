"""Un turno de la conversación: adjuntos → prompt con memoria → LLM → historial → project_metadata.

Es la pieza que une las demás sin que ninguna sepa de las otras:

- `attachments.py` convierte los ficheros en texto y los concatena a la transcripción.
- `llm_service.build_prompt` renderiza system (con `<project_metadata>`) y user; `call_llm` habla con el proveedor.
- `sessions.ConversationHistory` decide qué turnos anteriores viajan (ventana deslizante).
- `metadata_extractor` actualiza la memoria con lo que aportó el turno.

Orden deliberado dentro del turno:

1. Validar todo lo que puede fallar sin tocar la sesión (plantilla, API key, adjuntos ya extraídos).
2. `start_turn(user)`: el mensaje del usuario entra en la ventana (y, si hace falta, sale el turno más antiguo).
3. Llamada al LLM con `system + ventana`. Si falla, `discard_incomplete_turn()`: un error no deja un
   mensaje de usuario sin respuesta en el historial.
4. `complete_turn(assistant)` y actualización de la memoria con `ProjectMetadata.merged`.

El `lock` de la sesión serializa los turnos: dos peticiones concurrentes al mismo `session_id` no
intercalan sus mensajes.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

from app.config import Settings, get_settings
from app.schemas.estimation import EstimationRequest
from app.services.attachments import ExtractedAttachment, enrich_transcript
from app.services.llm_service import EstimationResult, build_prompt, build_result, call_llm, require_api_key
from app.services.metadata_extractor import extract_project_metadata
from app.services.sessions import ProjectMetadata, Session

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnResult:
    """Resultado de un turno: la estimación más lo que pasó con la memoria y la ventana."""

    result: EstimationResult
    turn: int                      # número de turno en la sesión (1, 2, 3...)
    attachments: tuple[ExtractedAttachment, ...]
    messages_sent: int             # mensajes enviados al LLM, system incluido
    turns_in_window: int           # turnos que quedan en la ventana tras este
    dropped_turns: int             # turnos descartados por la ventana en toda la sesión
    project_metadata: ProjectMetadata


async def run_turn(
    session: Session,
    request: EstimationRequest,
    attachments: Sequence[ExtractedAttachment] = (),
    settings: Settings | None = None,
) -> TurnResult:
    """Ejecuta un turno de estimación dentro de una sesión y actualiza su historial y su memoria.

    `request.transcription` es la transcripción tal cual la envió el cliente; aquí se le concatena el
    texto de los adjuntos antes de renderizar el prompt. Lanza los mismos errores que
    `generate_estimation` (`PromptTemplateNotFound`, `LLMConfigurationError`, `LLMProviderError`).
    """
    settings = settings or get_settings()
    attachments = tuple(attachments)

    async with session.lock:
        # 1) Todo lo que puede fallar sin tocar la sesión
        enriched = request.model_copy(update={"transcription": enrich_transcript(request.transcription, attachments)})
        rendered = build_prompt(enriched, settings, project_metadata=session.metadata)
        api_key = require_api_key(settings)

        # 2) El mensaje de usuario entra en la ventana
        session.history.start_turn(rendered.user)
        messages = session.history.to_messages_list(rendered.system)
        prompt = replace(rendered, history=tuple(messages[1:-1]))  # system y mensaje actual los aporta RenderedPrompt
        turn_number = session.turn_count + 1
        logger.info(
            "Sesión %s, turno %d: %d mensajes al LLM (%d turnos en ventana de %d, %d descartados), "
            "%d adjuntos con %d caracteres, memoria %s, plantilla %s",
            session.session_id[:8], turn_number, len(messages), len(session.history), session.history.max_turns,
            session.history.dropped_turns, len(attachments), sum(a.chars for a in attachments),
            "vacía" if session.metadata.is_empty() else "con datos", rendered.template.id,
        )

        # 3) Llamada al LLM; si falla, el turno se retira
        started = time.perf_counter()
        try:
            text, usage = await call_llm(prompt, settings, api_key)
        except Exception:
            session.history.discard_incomplete_turn()
            raise

        # 4) Cierre del turno y actualización de la memoria
        session.history.complete_turn(text)
        session.turn_count = turn_number
        session.metadata = session.metadata.merged(extract_project_metadata(request, text, attachments, current=session.metadata))
        session.touch()
        result = build_result(text, usage, settings, prompt, request, time.perf_counter() - started)
        return TurnResult(
            result=result,
            turn=turn_number,
            attachments=attachments,
            messages_sent=len(messages),
            turns_in_window=len(session.history),
            dropped_turns=session.history.dropped_turns,
            project_metadata=session.metadata,
        )
