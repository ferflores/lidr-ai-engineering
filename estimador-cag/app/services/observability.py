"""Observación por turno: un único evento `turn_observed` con todo lo relevante de la llamada.

Hasta ahora cada turno dejaba varios logs sueltos (mensajes al LLM, tokens y coste, resumen...). Para medir
el sistema hace falta correlacionarlos, y correlacionar timestamps de logs distintos es frágil. Este módulo
agrega los 13 campos del ejercicio de la sesión 06 (más unos extras) en un solo evento structlog emitido
justo antes de devolver el turno, y lo guarda en la sesión (`last_turn_observed`) para que
`GET /api/v1/sessions/{id}` lo exponga: el runner del stress lo lee de ahí sin parsear stdout.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass, field

import structlog

from app.config import Settings

TURN_OBSERVED = "turn_observed"

# Los 13 campos mínimos del ejercicio, en este orden (el CSV del stress los vuelca así).
TURN_OBSERVED_FIELDS: tuple[str, ...] = (
    "turn_index",
    "session_id",
    "enriched_transcript_chars",
    "attachments_total_chars",
    "messages_in_window",
    "anchors_count",
    "summary_chars",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "latency_ms",
    "cache_hit_kind",
    "last_resolved_tier",
)


@dataclass(frozen=True)
class TurnObservation:
    """Lo que se observa de un turno. Los 13 campos del enunciado primero; después, extras útiles para leer las curvas."""

    turn_index: int                   # 1-based, dentro de la sesión
    session_id: str
    enriched_transcript_chars: int    # transcripción + adjuntos concatenados
    attachments_total_chars: int      # 0 si no hay adjuntos
    messages_in_window: int           # mensajes user/assistant en la ventana tras la compresión
    anchors_count: int
    summary_chars: int
    tokens_in: int                    # llamada principal + summarizer (0 si acierto de caché)
    tokens_out: int
    cost_usd: float                   # idem
    latency_ms: int                   # tiempo total del turno (summarizer incluido)
    cache_hit_kind: str               # "none" | "exact" | "semantic"
    last_resolved_tier: str
    # --- extras ---
    last_tier_rule: str = ""
    model: str = ""
    provider: str = ""
    prompt_version: str = ""
    system_prompt_chars: int = 0
    context_chars: int = 0            # system + mensajes de la ventana (lo que viajó)
    dropped_turns: int = 0
    summarizer_called: bool = False
    summarizer_mode: str = "off"
    summarizer_tokens_in: int = 0
    summarizer_tokens_out: int = 0
    summarizer_cost_usd: float = 0.0
    summarizer_latency_ms: int = 0
    llm_latency_ms: int = 0           # solo la llamada principal
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        data = asdict(self)
        extra = data.pop("extra", {}) or {}
        return {**data, **extra}


def configure_logging(settings: Settings | None = None) -> None:
    """structlog hacia stdout, en JSON (una línea por evento) o legible en consola."""
    fmt = settings.log_format if settings is not None else "json"
    renderer = structlog.processors.JSONRenderer(ensure_ascii=False) if fmt == "json" else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=False,
    )


def emit_turn_observed(observation: TurnObservation) -> dict:
    """Emite el evento y devuelve el diccionario emitido (el mismo que se guarda en la sesión)."""
    data = observation.as_dict()
    structlog.get_logger("estimador.turns").info(TURN_OBSERVED, **data)
    return data
