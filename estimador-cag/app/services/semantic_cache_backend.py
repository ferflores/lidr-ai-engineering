"""El backend de embeddings de la caché semántica, separado para que la caché no dependa de él.

`response_cache.py` solo conoce una función `str -> vector`. Quién la construye y con qué modelo vive
aquí, así que la caché se puede probar sin red y el módulo de embeddings no se carga cuando el backend
configurado es `jaccard`.

Coste y latencia, que es lo que convierte esto en una decisión de producto y no en una mejora gratuita:
cada consulta a la caché embebe el mensaje del usuario, unos 50-100 ms y 0,02 USD por millón de tokens.
Se paga en **todas** las peticiones, acierten o no; lo que se ahorra es la llamada al LLM de las que
aciertan. Compensa cuando la tasa de aciertos es alta, que es justo lo que el material pide medir antes
de activarlo.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from functools import lru_cache

from app.config import Settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _embedder(api_key: str, model: str):
    from app.embedding_pipeline.embedder import OpenAIEmbedder  # import perezoso: solo si hace falta

    return OpenAIEmbedder(api_key=api_key, model=model)


def make_cache_embedder(settings: Settings) -> Callable[[str], Sequence[float]] | None:
    """La función de embedding para la caché, o None si el backend es Jaccard o falta la API key."""
    if settings.semantic_cache_backend != "embeddings":
        return None
    api_key = (settings.openai_api_key or "").strip()
    if not api_key:
        logger.warning(
            "SEMANTIC_CACHE_BACKEND=embeddings pero no hay OPENAI_API_KEY; la caché semántica sigue con Jaccard"
        )
        return None
    embedder = _embedder(api_key, settings.embedding_model)
    return embedder.embed_one
