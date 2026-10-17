"""Un turno de la conversación: adjuntos → compresión del historial → prompt → caché/LLM → historial → memoria → observación.

Es la pieza que une las demás sin que ninguna sepa de las otras:

- `attachments.py` convierte los ficheros en texto y los concatena a la transcripción.
- `sessions.ConversationHistory` decide qué turnos anteriores viajan (ventana deslizante) y cuáles salen.
- `summarizer.py` pliega los turnos que salen en el resumen acumulativo (puede llamar al LLM).
- `anchors.py` extrae del mensaje del usuario los hechos que deben sobrevivir a la ventana.
- `llm_service.build_prompt` renderiza system (con `<project_metadata>`, `<anchors>` y `<summary>`) y user.
- `tiering.py` decide con qué modelo y cuántos tokens de salida se atiende el turno.
- `response_cache.py` evita la llamada si el prompt ya se respondió (exacto o casi).
- `metadata_extractor` actualiza la memoria con lo que aportó el turno.
- `observability.py` emite el evento `turn_observed` con todo lo anterior agregado.

Orden deliberado dentro del turno:

1. Validar todo lo que puede fallar sin tocar la sesión (plantilla, API key, adjuntos ya extraídos).
2. `start_turn(user)`: el mensaje del usuario entra en la ventana y, si hace falta, sale el turno más
   antiguo, que se pliega en el resumen ANTES de llamar al LLM (así la llamada ya lo ve resumido).
3. Anclas del turno, system prompt con la memoria, las anclas y el resumen; tier según el contexto.
4. Caché; si no hay acierto, llamada al LLM. Si falla, `discard_incomplete_turn()`: un error no deja
   un mensaje de usuario sin respuesta en el historial (las anclas tampoco se guardan).
5. `complete_turn(assistant)`, memoria, anclas, tier y el evento `turn_observed`.

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
from app.services.anchors import extract_anchors, merge_anchors
from app.services.attachments import ExtractedAttachment, enrich_transcript
from app.services.llm_service import EstimationResult, TokenUsage, build_prompt, build_result, call_llm, require_api_key
from app.services.metadata_extractor import extract_project_metadata
from app.services.observability import TurnObservation, emit_turn_observed
from app.services.prompts import get_template
from app.services.response_cache import NONE, ResponseCache, get_response_cache
from app.services.semantic_cache_backend import make_cache_embedder
from app.services.sessions import Anchor, ProjectMetadata, Session
from app.services.summarizer import SummaryUpdate, fold_into_summary
from app.services.tiering import TierContext, TierDecision, resolve_tier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnResult:
    """Resultado de un turno: la estimación más lo que pasó con la memoria, la ventana y la observación."""

    result: EstimationResult
    turn: int                      # número de turno en la sesión (1, 2, 3...)
    attachments: tuple[ExtractedAttachment, ...]
    messages_sent: int             # mensajes enviados al LLM, system incluido
    turns_in_window: int           # turnos que quedan en la ventana tras este
    dropped_turns: int             # turnos descartados por la ventana en toda la sesión
    project_metadata: ProjectMetadata
    anchors: tuple[Anchor, ...] = ()
    summary: str = ""
    cache_hit_kind: str = NONE
    tier: TierDecision | None = None
    observation: dict | None = None  # el evento turn_observed tal cual se emitió


async def run_turn(
    session: Session,
    request: EstimationRequest,
    attachments: Sequence[ExtractedAttachment] = (),
    settings: Settings | None = None,
    cache: ResponseCache | None = None,
) -> TurnResult:
    """Ejecuta un turno de estimación dentro de una sesión y actualiza su historial, su memoria, sus anclas y su resumen.

    `request.transcription` es la transcripción tal cual la envió el cliente; aquí se le concatena el
    texto de los adjuntos antes de renderizar el prompt. Lanza los mismos errores que
    `generate_estimation` (`PromptTemplateNotFound`, `LLMConfigurationError`, `LLMProviderError`).
    """
    settings = settings or get_settings()
    if cache is None and settings.response_cache_enabled:
        cache = get_response_cache()
        cache.semantic_threshold = settings.effective_semantic_threshold
        cache.max_entries = settings.response_cache_max_entries
        cache.ttl_seconds = settings.response_cache_ttl_seconds
        cache.embed = make_cache_embedder(settings)
    attachments = tuple(attachments)
    attachments_chars = sum(a.chars for a in attachments)
    turn_started = time.perf_counter()

    async with session.lock:
        # 1) Todo lo que puede fallar sin tocar la sesión
        enriched = request.model_copy(update={"transcription": enrich_transcript(request.transcription, attachments)})
        get_template(settings, request.prompt_version)  # una versión inexistente es un error de la petición
        api_key = require_api_key(settings)
        turn_number = session.turn_count + 1
        user_message = build_prompt(enriched, settings, project_metadata=session.metadata).user

        # 2) El mensaje de usuario entra en la ventana; lo que sale se pliega en el resumen antes de llamar al LLM
        turn = session.history.start_turn(user_message)
        turn.index = turn_number
        summary_update: SummaryUpdate | None = None
        if session.history.last_dropped:
            summary_update = await fold_into_summary(session.summary, session.history.last_dropped, settings, api_key)
            session.summary = summary_update.summary

        # 3) Anclas del turno, prompt con memoria + anclas + resumen, y tier según lo que va a viajar
        anchors = merge_anchors(
            session.anchors,
            extract_anchors(request.transcription, request, turn_number, settings.anchors_per_turn),
            settings.anchors_max,
        )
        rendered = build_prompt(enriched, settings, project_metadata=session.metadata, anchors=anchors, summary=session.summary)
        messages = session.history.to_messages_list(rendered.system)
        prompt = replace(rendered, history=tuple(messages[1:-1]))  # system y mensaje actual los aporta RenderedPrompt
        context_chars = sum(len(m["content"]) for m in messages)
        tier = resolve_tier(
            TierContext(context_chars=context_chars, attachments_chars=attachments_chars, messages_in_window=len(messages) - 1),
            settings,
        )
        logger.info(
            "Sesión %s, turno %d: %d mensajes al LLM (%d turnos en ventana de %d, %d descartados), "
            "%d adjuntos con %d caracteres, %d anclas, resumen de %d caracteres, tier %s (%s), plantilla %s",
            session.session_id[:8], turn_number, len(messages), len(session.history), session.history.max_turns,
            session.history.dropped_turns, len(attachments), attachments_chars, len(anchors), len(session.summary),
            tier.tier, tier.rule, rendered.template.id,
        )

        # 4) Caché o llamada al LLM; si la llamada falla, el turno se retira
        cache_hit_kind = NONE
        hit = cache.lookup(rendered.system, messages[1:]) if cache is not None else None
        llm_started = time.perf_counter()
        if hit is not None:
            cache_hit_kind = hit.kind
            text, usage = hit.entry.text, TokenUsage(input_tokens=0, output_tokens=0)  # nada viajó al proveedor
        else:
            try:
                text, usage = await call_llm(prompt, settings, api_key, model=tier.model, max_tokens=tier.max_tokens)
            except Exception:
                session.history.discard_incomplete_turn()
                raise
            if cache is not None:
                cache.store(rendered.system, messages[1:], text, usage, tier.model)
        llm_latency_ms = int((time.perf_counter() - llm_started) * 1000)

        # 5) Cierre del turno: historial, memoria, anclas, tier
        session.history.complete_turn(text)
        session.turn_count = turn_number
        session.anchors = anchors
        session.metadata = session.metadata.merged(extract_project_metadata(request, text, attachments, current=session.metadata))
        session.last_resolved_tier, session.last_tier_rule = tier.tier, tier.rule
        session.touch()
        result = build_result(text, usage, settings, prompt, request, time.perf_counter() - turn_started, model=tier.model)

        # 6) Un único evento con todo lo observado en el turno
        main_cost = result.cost.total_usd if result.cost is not None else 0.0
        summarizer_usage = summary_update.usage if summary_update else None
        summarizer_cost = summary_update.cost.total_usd if summary_update and summary_update.cost else 0.0
        observation = TurnObservation(
            turn_index=turn_number,
            session_id=session.session_id,
            enriched_transcript_chars=len(enriched.transcription),
            attachments_total_chars=attachments_chars,
            messages_in_window=session.history.message_count,
            anchors_count=len(anchors),
            summary_chars=len(session.summary),
            tokens_in=(usage.input_tokens if usage else 0) + (summarizer_usage.input_tokens if summarizer_usage else 0),
            tokens_out=(usage.output_tokens if usage else 0) + (summarizer_usage.output_tokens if summarizer_usage else 0),
            cost_usd=round(main_cost + summarizer_cost, 6),
            latency_ms=int((time.perf_counter() - turn_started) * 1000),
            cache_hit_kind=cache_hit_kind,
            last_resolved_tier=tier.tier,
            last_tier_rule=tier.rule,
            model=tier.model,
            provider=settings.llm_provider,
            prompt_version=rendered.template.version,
            system_prompt_chars=len(rendered.system),
            context_chars=context_chars,
            dropped_turns=session.history.dropped_turns,
            summarizer_called=bool(summary_update and summary_update.called_llm),
            summarizer_mode=summary_update.mode if summary_update else "off",
            summarizer_tokens_in=summarizer_usage.input_tokens if summarizer_usage else 0,
            summarizer_tokens_out=summarizer_usage.output_tokens if summarizer_usage else 0,
            summarizer_cost_usd=round(summarizer_cost, 6),
            summarizer_latency_ms=summary_update.latency_ms if summary_update else 0,
            llm_latency_ms=llm_latency_ms,
        )
        session.last_turn_observed = emit_turn_observed(observation)

        return TurnResult(
            result=result,
            turn=turn_number,
            attachments=attachments,
            messages_sent=len(messages),
            turns_in_window=len(session.history),
            dropped_turns=session.history.dropped_turns,
            project_metadata=session.metadata,
            anchors=tuple(anchors),
            summary=session.summary,
            cache_hit_kind=cache_hit_kind,
            tier=tier,
            observation=session.last_turn_observed,
        )
