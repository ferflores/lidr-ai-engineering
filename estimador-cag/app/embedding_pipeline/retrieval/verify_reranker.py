"""Smoke test of the reranker: download/load the model and score three pairs.

    uv run python -m app.embedding_pipeline.retrieval.verify_reranker
    docker compose exec api python -m app.embedding_pipeline.retrieval.verify_reranker
    ... --model cross-encoder/ms-marco-MiniLM-L-6-v2       # try another model without touching settings

The three documents are the exercise's own failure case: an e-commerce query against (1) a component of a
fashion e-commerce budget, (2) a component of a *payment gateway* budget (the vector neighbour that is useless
for estimating a shop) and (3) a component about industrial refrigeration telemetry. The model passes if the
e-commerce chunk scores highest and the telemetry chunk lowest. Exit code 0 on success, 1 otherwise, so the
check can gate a CI job.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

from app.config import get_settings
from app.embedding_pipeline.retrieval.reranker import CrossEncoderReranker, RerankerUnavailable

QUERY = "Online store for a fashion brand with product catalogue, shopping cart and card payments"

DOCUMENTS: tuple[tuple[str, str], ...] = (
    (
        "relevant (e-commerce storefront)",
        "[Project: Headless fashion e-commerce storefront with product catalog search and Stripe checkout]\n"
        "[Client sector: ecommerce | Year: 2024 | Main tech: nextjs]\n\n"
        "Component: Cart and Stripe checkout\n"
        "Description: Persistent cart for guests and logged-in users, promo codes, shipping rate calculation and "
        "Stripe Checkout with Apple Pay and Bizum, including webhook-driven order confirmation.\n"
        "Tech stack: nextjs, stripe, nestjs\nComplexity: high\nEstimated hours: 100",
    ),
    (
        "near miss (payment gateway)",
        "[Project: Payment gateway with card tokenisation, 3-D Secure and merchant dashboard]\n"
        "[Client sector: finance | Year: 2025 | Main tech: spring_boot]\n\n"
        "Component: Card tokenisation vault\n"
        "Description: PCI DSS scoped service that stores card data encrypted with HSM-backed keys and exposes "
        "tokens to the rest of the platform, with key rotation and detokenisation audit.\n"
        "Tech stack: spring_boot, java, postgresql, aws_kms\nComplexity: high\nEstimated hours: 150",
    ),
    (
        "irrelevant (industrial telemetry)",
        "[Project: IoT predictive maintenance platform for industrial refrigeration units]\n"
        "[Client sector: industrial | Year: 2023 | Main tech: go]\n\n"
        "Component: Telemetry ingestion\n"
        "Description: MQTT broker and Go ingestion service receiving temperature, pressure and compressor cycles "
        "from 2.000 units, with schema validation, deduplication and storage in a time-series database.\n"
        "Tech stack: go, mqtt, timescaledb, kubernetes\nComplexity: high\nEstimated hours: 130",
    ),
)


@dataclass
class VerificationReport:
    model: str
    device: str
    load_time_ms: int
    score_time_ms: int
    scores: list[tuple[str, float]]  # (label, score) in document order

    @property
    def passed(self) -> bool:
        ordered = sorted(self.scores, key=lambda item: item[1], reverse=True)
        return ordered[0][0].startswith("relevant") and ordered[-1][0].startswith("irrelevant")

    def render(self) -> str:
        lines = [
            f"model:      {self.model}",
            f"device:     {self.device}",
            f"load time:  {self.load_time_ms} ms",
            f"score time: {self.score_time_ms} ms for {len(self.scores)} pairs",
            f'query:      "{QUERY}"',
            "",
        ]
        for label, score in sorted(self.scores, key=lambda item: item[1], reverse=True):
            lines.append(f"  {score:>9.4f}  {label}")
        lines.append("")
        lines.append("PASS: the e-commerce chunk ranks first and the telemetry chunk last" if self.passed else "FAIL: unexpected order")
        return "\n".join(lines)


def run(reranker: CrossEncoderReranker) -> VerificationReport:
    reranker.load()
    started = time.perf_counter()
    scores = reranker.score(QUERY, [text for _label, text in DOCUMENTS])
    score_time_ms = int((time.perf_counter() - started) * 1000)
    return VerificationReport(
        model=reranker.model_name,
        device=reranker.device,
        load_time_ms=reranker.load_time_ms or 0,
        score_time_ms=score_time_ms,
        scores=[(label, score) for (label, _text), score in zip(DOCUMENTS, scores)],
    )


def main(argv: list[str] | None = None, reranker: CrossEncoderReranker | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load the cross-encoder reranker and score three test pairs.")
    parser.add_argument("--model", default=None, help="Model name (default: RERANK_MODEL from settings).")
    parser.add_argument("--device", default=None, help="Torch device (default: RERANK_DEVICE from settings, cpu).")
    args = parser.parse_args(argv)

    if reranker is None:
        settings = get_settings()
        reranker = CrossEncoderReranker(
            args.model or settings.rerank_model,
            batch_size=settings.rerank_batch_size,
            max_length=settings.rerank_max_length,
            device=args.device or settings.rerank_device,
        )
    try:
        report = run(reranker)
    except RerankerUnavailable as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(report.render())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
