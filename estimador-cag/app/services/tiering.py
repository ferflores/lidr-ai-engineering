"""Tier dinámico: con qué modelo y cuántos tokens de salida se atiende cada turno, decidido en runtime.

El tier NO lo elige el cliente: lo resuelven reglas deterministas sobre el contexto que va a viajar en
este turno (tamaño del system prompt más los mensajes de la ventana, tamaño de los adjuntos). Cada turno
registra qué tier se resolvió y qué regla lo decidió (`last_resolved_tier`, `last_tier_rule`), y ambos
van en el evento `turn_observed`.

Dos tiers:

- `standard`: el modelo del proveedor (`OPENAI_MODEL` / `ANTHROPIC_MODEL`) y `LLM_MAX_TOKENS`.
- `extended`: para turnos con mucho contexto. Por defecto usa el **mismo** modelo y los mismos tokens:
  así el tier se observa sin alterar las medidas (el ejercicio de la sesión 06 exige un único modelo).
  `TIER_EXTENDED_MODEL` y `TIER_EXTENDED_MAX_TOKENS` lo cambian cuando se quiera que el tier tenga efecto.

Las reglas se evalúan en orden y gana la primera que se cumple.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.config import Settings

STANDARD = "standard"
EXTENDED = "extended"


@dataclass(frozen=True)
class TierContext:
    context_chars: int        # system prompt + mensajes de la ventana (lo que va a viajar)
    attachments_chars: int    # texto de adjuntos del turno
    messages_in_window: int   # mensajes user/assistant de la ventana


@dataclass(frozen=True)
class TierDecision:
    tier: str
    rule: str
    model: str
    max_tokens: int


Rule = tuple[str, str, Callable[[TierContext, Settings], bool]]

RULES: tuple[Rule, ...] = (
    ("attachments_large", EXTENDED, lambda ctx, s: ctx.attachments_chars >= s.tier_extended_min_attachment_chars),
    ("context_large", EXTENDED, lambda ctx, s: ctx.context_chars >= s.tier_extended_min_context_chars),
    ("default", STANDARD, lambda ctx, s: True),
)


def resolve_tier(context: TierContext, settings: Settings) -> TierDecision:
    """Primera regla que se cumple → tier, y el tier → modelo y tokens de salida."""
    for name, tier, predicate in RULES:
        if predicate(context, settings):
            return TierDecision(tier=tier, rule=name, model=model_for(tier, settings), max_tokens=max_tokens_for(tier, settings))
    raise AssertionError("la regla 'default' siempre se cumple")  # pragma: no cover


def model_for(tier: str, settings: Settings) -> str:
    if tier == EXTENDED and settings.tier_extended_model:
        return settings.tier_extended_model
    return settings.model


def max_tokens_for(tier: str, settings: Settings) -> int:
    if tier == EXTENDED and settings.tier_extended_max_tokens:
        return settings.tier_extended_max_tokens
    return settings.llm_max_tokens
