"""Caché de respuestas del LLM en memoria del proceso: acierto exacto y acierto semántico.

- **Exacta**: misma clave = hash del system prompt y de todos los mensajes. Si dos turnos envían
  exactamente el mismo prompt (el mismo escenario repetido, un reintento del cliente), la respuesta se
  devuelve sin llamar al proveedor: coste 0, latencia de milisegundos.
- **Semántica** (determinista, sin embeddings): mismo system prompt y mismo historial, y el mensaje del
  usuario actual es "casi" el mismo: similitud de Jaccard sobre el conjunto de palabras normalizadas
  ≥ `SEMANTIC_CACHE_THRESHOLD`. Cubre reformulaciones mínimas ("añade un detalle" con un punto o una
  mayúscula de diferencia). Se eligió Jaccard sobre embeddings por la misma razón que las métricas del
  stress: determinismo y cero dependencias; el umbral alto (0.9) evita devolver una estimación vieja a
  una petición que sí cambia (p. ej. otro presupuesto).

Cada turno registra `cache_hit_kind` ∈ {"none", "exact", "semantic"} en `turn_observed`.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass

from app.services.llm_service import TokenUsage

NONE = "none"
EXACT = "exact"
SEMANTIC = "semantic"

_WORDS = re.compile(r"[^\w]+", re.UNICODE)


@dataclass(frozen=True)
class CacheEntry:
    exact_key: str
    context_key: str              # system + historial sin el mensaje actual
    user_words: frozenset[str]    # palabras normalizadas del mensaje actual
    text: str
    usage: TokenUsage | None
    model: str
    created_at: float


@dataclass(frozen=True)
class CacheHit:
    kind: str                     # "exact" | "semantic"
    entry: CacheEntry
    similarity: float


def _hash(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def normalized_words(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORDS.split(text.casefold()) if w)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class ResponseCache:
    """LRU con TTL. `lookup` antes de llamar al LLM, `store` después."""

    def __init__(self, max_entries: int = 256, ttl_seconds: float = 3600.0, semantic_threshold: float = 0.9):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.semantic_threshold = semantic_threshold
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = {EXACT: 0, SEMANTIC: 0}
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def keys(system: str, messages: list[dict[str, str]]) -> tuple[str, str, frozenset[str]]:
        exact_key = _hash(system, messages)
        context_key = _hash(system, messages[:-1])
        return exact_key, context_key, normalized_words(messages[-1]["content"] if messages else "")

    def lookup(self, system: str, messages: list[dict[str, str]]) -> CacheHit | None:
        self._expire()
        exact_key, context_key, words = self.keys(system, messages)
        entry = self._entries.get(exact_key)
        if entry is not None:
            self._entries.move_to_end(exact_key)
            self.hits[EXACT] += 1
            return CacheHit(kind=EXACT, entry=entry, similarity=1.0)
        if self.semantic_threshold < 1.0:
            best: tuple[float, CacheEntry] | None = None
            for candidate in self._entries.values():
                if candidate.context_key != context_key:
                    continue
                similarity = jaccard(words, candidate.user_words)
                if similarity >= self.semantic_threshold and (best is None or similarity > best[0]):
                    best = (similarity, candidate)
            if best is not None:
                self.hits[SEMANTIC] += 1
                return CacheHit(kind=SEMANTIC, entry=best[1], similarity=best[0])
        self.misses += 1
        return None

    def store(self, system: str, messages: list[dict[str, str]], text: str, usage: TokenUsage | None, model: str) -> CacheEntry:
        exact_key, context_key, words = self.keys(system, messages)
        entry = CacheEntry(
            exact_key=exact_key, context_key=context_key, user_words=words, text=text, usage=usage, model=model,
            created_at=time.monotonic(),
        )
        self._entries[exact_key] = entry
        self._entries.move_to_end(exact_key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return entry

    def clear(self) -> None:
        self._entries.clear()
        self.hits = {EXACT: 0, SEMANTIC: 0}
        self.misses = 0

    def stats(self) -> dict[str, int]:
        return {"entries": len(self._entries), "exact_hits": self.hits[EXACT], "semantic_hits": self.hits[SEMANTIC], "misses": self.misses}

    def _expire(self) -> None:
        now = time.monotonic()
        expired = [key for key, entry in self._entries.items() if now - entry.created_at > self.ttl_seconds]
        for key in expired:
            del self._entries[key]


_cache = ResponseCache()


def get_response_cache() -> ResponseCache:
    """Caché única del proceso (los tests la sustituyen o la vacían)."""
    return _cache
