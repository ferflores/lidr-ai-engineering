"""Resumen acumulativo: los turnos que salen de la ventana deslizante se pliegan en un resumen corto.

Antes (pre-work de la sesión 05) un turno que salía de la ventana desaparecía entero. Ahora, justo antes
de llamar al LLM, los turnos descartados por `ConversationHistory.start_turn` se integran en
`Session.summary`, que viaja en el bloque `<summary>` del system prompt (plantilla `v4`).

Tres modos (`SUMMARY_MODE`):

- `llm` (por defecto): una llamada extra al mismo modelo con la plantilla `prompts/resumen/v1`. Es la
  opción fiel al enunciado del directo ("summarizer acumulativo"), y se paga: sus tokens, coste y
  latencia se suman a los del turno en `turn_observed` (`summarizer_*`). Si la llamada falla, se cae al
  modo heurístico y el turno no se pierde.
- `heuristic`: sin llamadas. Por cada turno descartado guarda el principio de lo que dijo el cliente y
  las líneas clave de la estimación (título, alcance, total). Determinista y gratis, pero no "entiende".
- `off`: el comportamiento anterior (el turno se descarta sin dejar rastro). Sirve de control.

El resumen está acotado (`SUMMARY_MAX_CHARS`): cuando no cabe, se pierde lo más antiguo. Ese tope es
una decisión de diseño que el stress mide, no un límite del modelo.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

from app.config import Settings
from app.services.llm_service import RenderedPrompt, TokenUsage, call_llm, cost_for
from app.services.pricing import CostEstimate
from app.services.prompts import load_template
from app.services.sessions import Turn

logger = logging.getLogger(__name__)

SUMMARIZER_TEMPLATE = "resumen"
SUMMARIZER_MAX_OUTPUT_TOKENS = 1024
ATTACHMENT_EXCERPT_CHARS = 1500   # lo que ve el summarizer de cada adjunto de un turno descartado
CLIENT_EXCERPT_CHARS = 400        # modo heurístico: cuánto se conserva de lo que dijo el cliente
ESTIMATION_EXCERPT_CHARS = 500    # cuánto de la estimación (título, alcance, total...)

_TRANSCRIPT_BLOCK = re.compile(r"<transcripcion>\n?(.*?)\n?</transcripcion>", re.DOTALL)
_ATTACHMENT_BLOCK = re.compile(r"--- attachment: (.+?) ---\n(.*?)\n--- fin del adjunto: \1 ---", re.DOTALL)
_KEY_LINES = re.compile(r"^(## Estimaci[oó]n:.*|\*\*(?:Total estimado|Coste estimado|Equipo recomendado|Duraci[oó]n estimada|Plazo objetivo):.*)$", re.MULTILINE)
_SCOPE = re.compile(r"^###\s*Resumen del alcance\s*\n(.+?)(?=\n#{2,3}\s|\n\*\*|\Z)", re.MULTILINE | re.DOTALL)


@dataclass(frozen=True)
class SummaryUpdate:
    summary: str
    mode: str                  # "off" | "heuristic" | "llm" | "llm-fallback"
    called_llm: bool
    usage: TokenUsage | None
    cost: CostEstimate | None
    latency_ms: int


def client_text(user_message: str) -> str:
    """Lo que dijo el cliente en un mensaje de usuario ya renderizado: el bloque <transcripcion>, con los
    adjuntos reducidos a un extracto (el summarizer no debe pagar 60.000 caracteres de PDF por turno)."""
    match = _TRANSCRIPT_BLOCK.search(user_message)
    text = match.group(1) if match else user_message

    def shrink(block: re.Match[str]) -> str:
        name, body = block.group(1), block.group(2)
        excerpt = body if len(body) <= ATTACHMENT_EXCERPT_CHARS else body[:ATTACHMENT_EXCERPT_CHARS].rstrip() + " […]"
        size = f"{len(body):,}".replace(",", ".")
        return f"[adjunto {name}, {size} caracteres] {excerpt}"

    return _ATTACHMENT_BLOCK.sub(shrink, text).strip()


def estimation_excerpt(assistant: str | None) -> str:
    """Título, alcance y líneas de totales de una estimación: lo que merece la pena recordar de ella."""
    if not assistant:
        return "(sin estimación)"
    parts = [m.group(1).strip() for m in _KEY_LINES.finditer(assistant)]
    scope = _SCOPE.search(assistant)
    if scope:
        parts.insert(1, " ".join(scope.group(1).split()))
    text = " | ".join(parts) if parts else " ".join(assistant.split())
    return text if len(text) <= ESTIMATION_EXCERPT_CHARS else text[:ESTIMATION_EXCERPT_CHARS].rstrip() + " […]"


def heuristic_summary(current: str, dropped: Sequence[Turn], max_chars: int) -> str:
    """Resumen sin LLM: una entrada por turno descartado, y se pierde lo más antiguo cuando no cabe."""
    entries = [e for e in current.split("\n") if e.strip()] if current else []
    for turn in dropped:
        said = client_text(turn.user)
        said = said if len(said) <= CLIENT_EXCERPT_CHARS else said[:CLIENT_EXCERPT_CHARS].rstrip() + " […]"
        entries.append(f"Turno {turn.index or '?'} — cliente: {' '.join(said.split())} → estimación: {estimation_excerpt(turn.assistant)}")
    while entries and len("\n".join(entries)) > max_chars:
        if len(entries) == 1:
            entries[0] = entries[0][: max_chars - 1].rstrip() + "…"
            break
        entries.pop(0)
    return "\n".join(entries)


async def fold_into_summary(
    current: str,
    dropped: Sequence[Turn],
    settings: Settings,
    api_key: str | None,
    model: str | None = None,
) -> SummaryUpdate:
    """Resumen de la sesión tras plegar los turnos descartados, según `SUMMARY_MODE`."""
    if not dropped or settings.summary_mode == "off":
        return SummaryUpdate(summary=current, mode="off", called_llm=False, usage=None, cost=None, latency_ms=0)
    if settings.summary_mode == "heuristic" or not api_key:
        return SummaryUpdate(
            summary=heuristic_summary(current, dropped, settings.summary_max_chars),
            mode="heuristic", called_llm=False, usage=None, cost=None, latency_ms=0,
        )

    template = load_template(SUMMARIZER_TEMPLATE, settings.summary_prompt_version)
    variables = {
        "max_chars": settings.summary_max_chars,
        "current_summary": current,
        "dropped": [
            {"turn": turn.index or "?", "user": client_text(turn.user), "assistant": estimation_excerpt(turn.assistant)}
            for turn in dropped
        ],
    }
    prompt = RenderedPrompt(template=template, system=template.render_system(**variables), user=template.render_user(**variables))
    started = time.perf_counter()
    try:
        text, usage = await call_llm(prompt, settings, api_key, model=model, max_tokens=SUMMARIZER_MAX_OUTPUT_TOKENS)
    except Exception as exc:  # el resumen nunca tumba el turno: se degrada al modo heurístico
        logger.warning("El summarizer ha fallado (%s); se usa el resumen heurístico", exc)
        return SummaryUpdate(
            summary=heuristic_summary(current, dropped, settings.summary_max_chars),
            mode="llm-fallback", called_llm=True, usage=None, cost=None,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    latency_ms = int((time.perf_counter() - started) * 1000)
    summary = " ".join(text.split())
    if len(summary) > settings.summary_max_chars:
        summary = summary[: settings.summary_max_chars - 1].rstrip() + "…"
    return SummaryUpdate(summary=summary, mode="llm", called_llm=True, usage=usage, cost=cost_for(usage, settings, model), latency_ms=latency_ms)
