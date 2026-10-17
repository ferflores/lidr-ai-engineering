"""Caché de respuestas del LLM en memoria del proceso: acierto exacto y acierto semántico.

- **Exacta**: misma clave = hash del system prompt y de todos los mensajes. Si dos turnos envían
  exactamente el mismo prompt (el mismo escenario repetido, un reintento del cliente), la respuesta se
  devuelve sin llamar al proveedor: coste 0, latencia de milisegundos.
- **Semántica**: mismo system prompt y mismo historial, y el mensaje del usuario actual es "casi" el mismo.
  Hay dos formas de medir ese "casi", y se elige con `SEMANTIC_CACHE_BACKEND`:

  - `jaccard` (por defecto): similitud de Jaccard sobre el conjunto de palabras normalizadas. Determinista,
    sin red y sin coste, pero solo reconoce reformulaciones que comparten palabras ("añade un detalle" con
    un punto o una mayúscula de diferencia). Fue la elección original del proyecto por la misma razón que
    las métricas del stress: determinismo y cero dependencias.
  - `embeddings`: similitud coseno entre los embeddings del mensaje. Reconoce *significados* parecidos
    aunque cambien las palabras, que es lo que pide el material de la sesión 04. Cuesta una llamada a la
    API de embeddings por consulta (unos 50-100 ms y 0,02 USD por millón de tokens) y deja de ser
    determinista, así que es una decisión de producto, no una mejora gratuita.

  El umbral (`SEMANTIC_CACHE_THRESHOLD`) no significa lo mismo en los dos: con Jaccard 0.9 es muy estricto
  (casi las mismas palabras); con embeddings el material recomienda 0.90-0.93 para un estimador, porque
  por debajo empieza a servir respuestas de peticiones que sí eran distintas.

Cada turno registra `cache_hit_kind` ∈ {"none", "exact", "semantic"} en `turn_observed`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from app.services.llm_service import TokenUsage

NONE = "none"
EXACT = "exact"
SEMANTIC = "semantic"

logger = logging.getLogger(__name__)

_WORDS = re.compile(r"[^\w]+", re.UNICODE)


@dataclass(frozen=True)
class CacheEntry:
    exact_key: str
    context_key: str              # system + historial sin el mensaje actual
    user_words: frozenset[str]    # palabras normalizadas del mensaje actual (backend jaccard)
    text: str
    usage: TokenUsage | None
    model: str
    created_at: float
    user_vector: tuple[float, ...] | None = None  # embedding del mensaje actual (backend embeddings)


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


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Coseno entre dos vectores. Se reimplementa aquí (en vez de importar `embedding_pipeline`) para que
    la caché no dependa del módulo de embeddings cuando el backend es Jaccard."""
    if len(a) != len(b):
        return 0.0
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class ResponseCache:
    """LRU con TTL. `lookup` antes de llamar al LLM, `store` después."""

    def __init__(
        self,
        max_entries: int = 256,
        ttl_seconds: float = 3600.0,
        semantic_threshold: float = 0.9,
        embed: Callable[[str], Sequence[float]] | None = None,
    ):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.semantic_threshold = semantic_threshold
        # Backend semántico: si hay función de embedding se compara por coseno; si no, por Jaccard.
        # Se inyecta en vez de importarse para que los tests no dependan de la red ni de una API key.
        self.embed = embed
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = {EXACT: 0, SEMANTIC: 0}
        self.misses = 0
        self.embedding_calls = 0
        self.embedding_errors = 0

    @property
    def semantic_backend(self) -> str:
        return "embeddings" if self.embed is not None else "jaccard"

    def _vector_for(self, text: str) -> tuple[float, ...] | None:
        """Embedding del mensaje, o None si no hay backend o la llamada falla.

        Un fallo del servicio de embeddings degrada la caché a fallo, nunca tumba la estimación: se
        pagará la llamada al LLM, que es exactamente lo que habría pasado sin caché.
        """
        if self.embed is None:
            return None
        try:
            self.embedding_calls += 1
            return tuple(self.embed(text))
        except Exception:  # noqa: BLE001 - cualquier fallo del embedder degrada a fallo de caché
            self.embedding_errors += 1
            logger.warning("La caché semántica no ha podido embeber el mensaje; se sigue sin acierto semántico")
            return None

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
            # Los candidatos son solo los del mismo contexto (mismo system y mismo historial): lo que se
            # compara es si cambió la *petición*, no la conversación entera.
            candidatos = [c for c in self._entries.values() if c.context_key == context_key]
            # El embedding solo se paga si hay con qué comparar.
            vector = self._vector_for(messages[-1]["content"]) if candidatos and messages else None

            best: tuple[float, CacheEntry] | None = None
            for candidate in candidatos:
                if vector is not None and candidate.user_vector is not None:
                    similarity = cosine(vector, candidate.user_vector)
                else:
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
            user_vector=self._vector_for(messages[-1]["content"]) if messages else None,
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
        self.embedding_calls = 0
        self.embedding_errors = 0

    def stats(self) -> dict[str, object]:
        return {
            "entries": len(self._entries),
            "exact_hits": self.hits[EXACT],
            "semantic_hits": self.hits[SEMANTIC],
            "misses": self.misses,
            "semantic_backend": self.semantic_backend,
            "embedding_calls": self.embedding_calls,
            "embedding_errors": self.embedding_errors,
        }

    def _expire(self) -> None:
        now = time.monotonic()
        expired = [key for key, entry in self._entries.items() if now - entry.created_at > self.ttl_seconds]
        for key in expired:
            del self._entries[key]


_cache = ResponseCache()


def get_response_cache() -> ResponseCache:
    """Caché única del proceso (los tests la sustituyen o la vacían)."""
    return _cache
