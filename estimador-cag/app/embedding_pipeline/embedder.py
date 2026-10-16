"""Embedder sobre la API de OpenAI (`text-embedding-3-small`).

- Lotes: `embed_many` envía hasta `batch_size` textos por llamada (`embeddings.create` acepta listas) en vez
  de una llamada por chunk.
- Reintentos: `RateLimitError` se reintenta 3 veces con esperas 1 s, 2 s, 4 s; cualquier otro error se propaga.
  El cliente se crea con `max_retries=0` para que los reintentos del SDK no se sumen a los nuestros.
- Observación: un evento structlog `embedding_batch_completed` por lote (chunks, tokens, latencia), siguiendo
  la convención de la sesión 03.
- Coste: `usage.prompt_tokens` de cada respuesta × precio de lista. El precio es una constante de módulo
  claramente etiquetada porque cambia con el tiempo.

El embedder no guarda estado entre llamadas (se puede compartir entre peticiones concurrentes); el uso de cada
operación se devuelve junto con el resultado en los métodos `*_with_usage`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import structlog
from openai import OpenAI, RateLimitError

from app.embedding_pipeline.schemas import Chunk, EmbeddedChunk

log = structlog.get_logger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536  # dimensión por defecto del modelo; no se fuerza una menor (Matryoshka queda para el directo)

# --- PRECIO DE LISTA (cambia con el tiempo) ------------------------------------------------------------
# text-embedding-3-small: 0,02 USD por millón de tokens de entrada. Comprobado el 2026-10-16 en
# https://openai.com/api/pricing/ . Si cambia, se actualiza aquí y el coste estimado de las ingestas lo sigue.
EMBEDDING_PRICE_USD_PER_MTOK = 0.02
# ------------------------------------------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 100  # recomendación del curso; la API admite hasta 2048 entradas por llamada
RETRY_DELAYS_SECONDS: tuple[float, ...] = (1.0, 2.0, 4.0)  # 3 reintentos con espera exponencial


def estimate_cost_usd(prompt_tokens: int, price_per_mtok: float = EMBEDDING_PRICE_USD_PER_MTOK) -> float:
    return round(prompt_tokens * price_per_mtok / 1_000_000, 8)


@dataclass
class EmbeddingUsage:
    """Lo que costó una operación de embedding (acumulado sobre sus lotes)."""

    model: str
    batches: int = 0
    prompt_tokens: int = 0  # tokens de entrada facturados por la API
    latency_ms: int = 0
    dimensions: int = 0

    @property
    def estimated_cost_usd(self) -> float:
        return estimate_cost_usd(self.prompt_tokens)


class OpenAIEmbedder:
    """Vectoriza textos con la API de embeddings de OpenAI, por lotes y con reintentos ante rate limit."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: OpenAI | None = None,
        model: str = EMBEDDING_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        retry_delays: tuple[float, ...] = RETRY_DELAYS_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        timeout_seconds: float = 60.0,
    ) -> None:
        if client is None:
            if not api_key:
                raise ValueError("OpenAIEmbedder needs an OpenAI API key (OPENAI_API_KEY) or an explicit client")
            # max_retries=0: los reintentos los gobierna este módulo (1 s, 2 s, 4 s), no el SDK.
            client = OpenAI(api_key=api_key, max_retries=0, timeout=timeout_seconds)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._client = client
        self.model = model
        self.batch_size = batch_size
        self._retry_delays = tuple(retry_delays)
        self._sleep = sleep

    # --- API pública ---------------------------------------------------------------------------------------

    def embed_one(self, text: str) -> list[float]:
        """Vector de un único texto (una llamada a la API)."""
        vectors, _usage = self.embed_texts_with_usage([text])
        return vectors[0]

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Vectoriza los chunks por lotes y devuelve cada uno con su `embedding`, en el mismo orden."""
        embedded, _usage = self.embed_many_with_usage(chunks)
        return embedded

    def embed_many_with_usage(self, chunks: list[Chunk]) -> tuple[list[EmbeddedChunk], EmbeddingUsage]:
        """Como `embed_many`, devolviendo además tokens facturados, lotes y latencia acumulada."""
        vectors, usage = self.embed_texts_with_usage([chunk.text for chunk in chunks], token_counts=[c.token_count for c in chunks])
        embedded = [EmbeddedChunk(**chunk.model_dump(), embedding=vector) for chunk, vector in zip(chunks, vectors)]
        return embedded, usage

    def embed_texts_with_usage(self, texts: list[str], token_counts: list[int] | None = None) -> tuple[list[list[float]], EmbeddingUsage]:
        """Vectoriza una lista de textos en lotes de `batch_size`. Es la primitiva que usan los demás métodos."""
        usage = EmbeddingUsage(model=self.model)
        if not texts:
            return [], usage
        vectors: list[list[float]] = []
        for batch_index, start in enumerate(range(0, len(texts), self.batch_size)):
            batch = texts[start : start + self.batch_size]
            batch_vectors, prompt_tokens, latency_ms = self._create_with_retry(batch)
            vectors.extend(batch_vectors)
            usage.batches += 1
            usage.prompt_tokens += prompt_tokens
            usage.latency_ms += latency_ms
            usage.dimensions = len(batch_vectors[0])
            log.info(
                "embedding_batch_completed",
                model=self.model,
                batch_index=batch_index,
                chunks=len(batch),
                prompt_tokens=prompt_tokens,
                counted_tokens=sum(token_counts[start : start + self.batch_size]) if token_counts else None,
                latency_ms=latency_ms,
                dimensions=usage.dimensions,
                cost_usd=estimate_cost_usd(prompt_tokens),
            )
        return vectors, usage

    # --- Internos --------------------------------------------------------------------------------------------

    def _create_with_retry(self, texts: list[str]) -> tuple[list[list[float]], int, int]:
        """Una llamada a `embeddings.create` con reintento exponencial solo ante `RateLimitError`."""
        attempt = 0
        while True:
            started = time.perf_counter()
            try:
                response = self._client.embeddings.create(model=self.model, input=texts)
            except RateLimitError:
                if attempt >= len(self._retry_delays):
                    log.error("embedding_rate_limit_exhausted", model=self.model, attempts=attempt + 1, chunks=len(texts))
                    raise
                delay = self._retry_delays[attempt]
                attempt += 1
                log.warning("embedding_rate_limited", model=self.model, attempt=attempt, retry_in_seconds=delay, chunks=len(texts))
                self._sleep(delay)
                continue
            latency_ms = int((time.perf_counter() - started) * 1000)
            data = sorted(response.data, key=lambda item: item.index)
            vectors = [list(item.embedding) for item in data]
            if len(vectors) != len(texts):
                raise RuntimeError(f"embedding API returned {len(vectors)} vectors for {len(texts)} inputs")
            prompt_tokens = int(getattr(response.usage, "prompt_tokens", 0) or 0)
            return vectors, prompt_tokens, latency_ms
