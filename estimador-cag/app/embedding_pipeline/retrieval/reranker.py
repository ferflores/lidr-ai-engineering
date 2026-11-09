"""Cross-encoder reranker wrapper: load the model once, score (query, document) pairs, reorder.

A bi-encoder (the embedding model) encodes the query and each chunk *separately* and compares the vectors;
that is what makes it cheap enough to search a whole corpus, and also why "payment gateway" and "online
shop" end up neighbours: the vectors summarise topics, not the fit between this query and this chunk. A
cross-encoder reads the query and the chunk *together* in one forward pass and outputs a relevance score for
the pair. It is far more precise and far too slow to run against the corpus, hence recall-then-rerank: a
wide, cheap recall (top-50 from the vector/hybrid search) and a precise, expensive reorder of just those.

Design of the wrapper:

- **Lazy loading.** `sentence_transformers` (and torch underneath) is imported inside `load()`, not at module
  import: the service must start and serve vector search in milliseconds whether or not reranking is on.
  The router pre-loads it at startup when `RERANK_ENABLED=true` so the first request pays no cold start.
- **Injectable scorer.** Tests pass a `scorer` callable and never touch the model; production leaves it
  `None` and the wrapper builds a `CrossEncoder`.
- **Explicit device**, `cpu` by default: the production container has no GPU, and measuring latency on the
  laptop's MPS would report numbers the server will never see.
- **Stable ordering.** Equal scores keep the incoming order, so the recall stage breaks the ties.

Default model: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, a 118 M parameter multilingual MiniLM trained
on mMARCO (Spanish included). The corpus of this project is English, where `ms-marco-MiniLM-L-6-v2` (22 M)
would be faster; the multilingual one is the default because the project's real inputs are Spanish meeting
transcripts (session 09), and the model is a setting (`RERANK_MODEL`), not a constant.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import structlog

from app.config import DEFAULT_RERANKER_MODEL

log = structlog.get_logger(__name__)

# A scorer takes (query, document) pairs and returns one relevance score per pair, higher = more relevant.
Scorer = Callable[[Sequence[tuple[str, str]]], Sequence[float]]


class RerankerUnavailable(RuntimeError):
    """The model could not be loaded (missing weights, no network, broken install). The router maps it to 500."""


@dataclass(frozen=True)
class RerankedItem:
    """Position of a candidate in the incoming list and the score the cross-encoder gave the pair."""

    index: int
    score: float


class CrossEncoderReranker:
    """Scores (query, document) pairs with a sentence-transformers `CrossEncoder` and reorders candidates."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        batch_size: int = 32,
        max_length: int = 512,
        device: str = "cpu",
        scorer: Scorer | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._scorer = scorer
        self._model = None
        self.load_time_ms: int | None = None

    # --- Loading -------------------------------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None or self._scorer is not None

    def load(self) -> None:
        """Load the model once. Idempotent; raises `RerankerUnavailable` with the cause if it cannot."""
        if self.is_loaded:
            return
        started = time.perf_counter()
        try:
            # Heavy import (torch): deferred so the service starts fast when reranking is off.
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, max_length=self.max_length, device=self.device)
        except Exception as exc:  # noqa: BLE001 - whatever the hub/torch raise, the caller needs one type
            log.error("reranker_load_failed", model=self.model_name, device=self.device, error=type(exc).__name__, detail=str(exc))
            raise RerankerUnavailable(f"could not load reranker model {self.model_name!r}: {exc}") from exc
        self.load_time_ms = int((time.perf_counter() - started) * 1000)
        log.info("reranker_loaded", model=self.model_name, device=self.device, max_length=self.max_length, load_time_ms=self.load_time_ms)

    # --- Scoring ------------------------------------------------------------------------------------------

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """One relevance score per document, in the documents' order. Loads the model on first use."""
        if not documents:
            return []
        self.load()
        pairs = [(query, document) for document in documents]
        started = time.perf_counter()
        if self._scorer is not None:
            raw = self._scorer(pairs)
        else:
            raw = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)
        scores = [float(value) for value in raw]
        if len(scores) != len(documents):
            raise RuntimeError(f"reranker returned {len(scores)} scores for {len(documents)} documents")
        log.info(
            "rerank_completed",
            model=self.model_name,
            pairs=len(pairs),
            batch_size=self.batch_size,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        return scores

    def rerank(self, query: str, documents: Sequence[str], top_k: int | None = None) -> list[RerankedItem]:
        """Score and reorder: best first, ties keep the incoming order, optionally truncated to `top_k`."""
        scores = self.score(query, documents)
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        if top_k is not None:
            order = order[:top_k]
        return [RerankedItem(index=index, score=scores[index]) for index in order]
