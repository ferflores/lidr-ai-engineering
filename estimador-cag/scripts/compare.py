#!/usr/bin/env python
"""Similitud coseno entre dos textos usando los embeddings de OpenAI (pre-work sesión 07).

    uv run python scripts/compare.py --text-a "OAuth 2.0 authentication backend for fintech" \
                                     --text-b "JWT-based authorization service for banking app"

    docker compose exec api python scripts/compare.py --text-a "..." --text-b "..."

Reutiliza `OpenAIEmbedder` (los dos textos viajan en una sola llamada por lotes) y `cosine_similarity`
(producto escalar / producto de normas, sin numpy). Necesita OPENAI_API_KEY en `.env` o en el entorno.
Con `--json` imprime el resultado como JSON (útil para scripts); los logs del embedder van a stderr.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Ejecutable como `python scripts/compare.py` desde la raíz del proyecto (local o dentro del contenedor):
# el paquete `app` vive en el directorio padre de `scripts/`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog  # noqa: E402

from app.config import get_settings  # noqa: E402  (carga .env)
from app.embedding_pipeline.embedder import OpenAIEmbedder, estimate_cost_usd  # noqa: E402
from app.embedding_pipeline.similarity import cosine_similarity, l2_norm  # noqa: E402


@dataclass
class Comparison:
    text_a: str
    text_b: str
    cosine_similarity: float
    model: str
    dimensions: int
    prompt_tokens: int
    estimated_cost_usd: float
    latency_ms: int
    norm_a: float
    norm_b: float

    def render(self) -> str:
        return (
            f"Text A: {self.text_a}\n"
            f"Text B: {self.text_b}\n"
            f"Cosine similarity: {self.cosine_similarity:.4f}\n"
            f"({self.model}, {self.dimensions} dims, {self.prompt_tokens} tokens, "
            f"{self.estimated_cost_usd:.8f} USD, {self.latency_ms} ms; norms {self.norm_a:.4f} / {self.norm_b:.4f})"
        )


def compare(text_a: str, text_b: str, embedder: OpenAIEmbedder) -> Comparison:
    """Embebe los dos textos (una llamada) y calcula su similitud coseno."""
    (vec_a, vec_b), usage = embedder.embed_texts_with_usage([text_a, text_b])
    return Comparison(
        text_a=text_a,
        text_b=text_b,
        cosine_similarity=cosine_similarity(vec_a, vec_b),
        model=usage.model,
        dimensions=len(vec_a),
        prompt_tokens=usage.prompt_tokens,
        estimated_cost_usd=estimate_cost_usd(usage.prompt_tokens),
        latency_ms=usage.latency_ms,
        norm_a=l2_norm(vec_a),
        norm_b=l2_norm(vec_b),
    )


def _configure_logs(verbose: bool) -> None:
    """Los eventos structlog del embedder van a stderr para no mezclarse con el resultado."""
    level = logging.INFO if verbose else logging.WARNING
    structlog.configure(
        processors=[structlog.processors.add_log_level, structlog.processors.KeyValueRenderer(key_order=["event"])],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cosine similarity between the embeddings of two texts.")
    parser.add_argument("--text-a", required=True, help="First text.")
    parser.add_argument("--text-b", required=True, help="Second text.")
    parser.add_argument("--model", default=None, help="Embedding model (default: EMBEDDING_MODEL from .env, text-embedding-3-small).")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    parser.add_argument("--verbose", action="store_true", help="Show the embedder's log events (stderr).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logs(args.verbose)
    settings = get_settings()
    api_key = (settings.openai_api_key or "").strip()
    if not api_key:
        print("error: OPENAI_API_KEY is not set (put it in .env or export it)", file=sys.stderr)
        return 2
    embedder = OpenAIEmbedder(api_key=api_key, model=args.model or settings.embedding_model)
    result = compare(args.text_a, args.text_b, embedder)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2) if args.json else result.render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
