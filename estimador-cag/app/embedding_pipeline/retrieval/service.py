"""The retrieval pipeline behind `POST /search`: vector or hybrid recall, optional cross-encoder rerank.

    query ─► embed ─► vector top-N ──┐
                                     ├─► RRF fusion ─► top-N ─► (rerank) ─► top-k
    query ─► full-text top-N ────────┘        (hybrid only)        (optional)

The two switches (`mode`, `rerank`) give the four configurations the exercise measures:

    A  vector  + no rerank    the session 08 baseline: top-k straight from `ORDER BY embedding <=> q`
    B  hybrid  + no rerank    each branch recalls N, RRF fuses, top-k of the fused list
    C  vector  + rerank       vector recalls N, the cross-encoder reorders, top-k
    D  hybrid  + rerank       hybrid fused top-N, the cross-encoder reorders, top-k

`N` is `candidates` (settings `SEARCH_CANDIDATES`, 50 by default): the recall width of the
recall-then-rerank pattern. Configuration A recalls exactly k rows because nothing reorders them later.

Every stage is timed separately and returned with the results: the exercise asks what the reranking *costs*,
and a single total would hide that the embedding call to OpenAI is the same in all four configurations.

The two SQL branches run sequentially on the request's `AsyncSession` (one session cannot run two
statements concurrently). With 73 chunks each branch is single-digit milliseconds, so parallelising them
with two sessions would buy nothing measurable here; it is the obvious next step on a large corpus.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.config import SearchMode
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.repository import search_chunks
from app.embedding_pipeline.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from app.embedding_pipeline.retrieval.lexical import lexical_search
from app.embedding_pipeline.retrieval.reranker import CrossEncoderReranker
from app.embedding_pipeline.schemas import RankSignals, SearchResult

log = structlog.get_logger(__name__)

DEFAULT_CANDIDATES = 50

VECTOR_BRANCH = "vector"
LEXICAL_BRANCH = "lexical"


@dataclass
class RetrievalTimings:
    """Milliseconds per stage. A stage that did not run stays at 0."""

    embedding_ms: int = 0
    vector_ms: int = 0
    lexical_ms: int = 0
    fusion_ms: int = 0
    rerank_ms: int = 0
    total_ms: int = 0


@dataclass
class RetrievalOutcome:
    results: list[SearchResult]
    mode: SearchMode
    reranked: bool
    candidates: int  # how many chunks reached the final stage (the pool the rerank or the cut to k saw)
    timings: RetrievalTimings


def _elapsed_ms(since: float) -> int:
    return int((time.perf_counter() - since) * 1000)


def fuse_branches(vector_hits: list[SearchResult], lexical_hits: list[SearchResult], *, rrf_k: int) -> list[SearchResult]:
    """RRF over the two branches; each fused result keeps both ranks so the response explains itself.

    The row data (content, metadata, distance) is taken from the vector hit when the chunk came through
    that branch, otherwise from the lexical hit: a chunk only the keyword branch found has no distance.
    """
    by_id_vector = {hit.chunk_id: hit for hit in vector_hits}
    by_id_lexical = {hit.chunk_id: hit for hit in lexical_hits}
    fused = reciprocal_rank_fusion(
        {VECTOR_BRANCH: [hit.chunk_id for hit in vector_hits], LEXICAL_BRANCH: [hit.chunk_id for hit in lexical_hits]},
        k=rrf_k,
    )
    results: list[SearchResult] = []
    for candidate in fused:
        source = by_id_vector.get(candidate.key) or by_id_lexical[candidate.key]
        results.append(
            source.model_copy(
                update={
                    "score": candidate.score,
                    "ranking": RankSignals(
                        vector_rank=candidate.ranks.get(VECTOR_BRANCH),
                        lexical_rank=candidate.ranks.get(LEXICAL_BRANCH),
                        rrf_score=candidate.score,
                    ),
                }
            )
        )
    return results


def apply_rerank_scores(pool: list[SearchResult], scores: list[float], k: int) -> list[SearchResult]:
    """Reorder the pool by cross-encoder score (ties keep the recall order) and keep the top-k."""
    order = sorted(range(len(pool)), key=lambda index: (-scores[index], index))
    return [
        pool[index].model_copy(
            update={"score": scores[index], "ranking": pool[index].ranking.model_copy(update={"rerank_score": scores[index]})}
        )
        for index in order[:k]
    ]


async def retrieve(
    session: AsyncSession,
    *,
    query: str,
    embedder: OpenAIEmbedder,
    mode: SearchMode,
    k: int,
    rerank: bool,
    reranker: CrossEncoderReranker | None = None,
    candidates: int = DEFAULT_CANDIDATES,
    rrf_k: int = DEFAULT_RRF_K,
) -> RetrievalOutcome:
    """Run the pipeline for one query. Errors propagate untouched (OpenAI, SQLAlchemy, RerankerUnavailable)."""
    if rerank and reranker is None:
        raise ValueError("rerank=True requires a reranker")
    timings = RetrievalTimings()
    started = time.perf_counter()

    stage = time.perf_counter()
    # The OpenAI SDK is synchronous: off the event loop, same as the ingest path.
    query_vector = await run_in_threadpool(embedder.embed_one, query)
    timings.embedding_ms = _elapsed_ms(stage)

    # Recall width: wide when something reorders the list afterwards, exactly k for the plain baseline.
    recall_n = candidates if (mode == "hybrid" or rerank) else k

    stage = time.perf_counter()
    vector_hits = await search_chunks(session, query_vector, recall_n)
    timings.vector_ms = _elapsed_ms(stage)

    if mode == "hybrid":
        stage = time.perf_counter()
        lexical_hits = await lexical_search(session, query, recall_n)
        timings.lexical_ms = _elapsed_ms(stage)
        stage = time.perf_counter()
        # The fused list can hold up to 2N distinct chunks; the rerank stage sees at most N of them.
        pool = fuse_branches(vector_hits, lexical_hits, rrf_k=rrf_k)[:candidates]
        timings.fusion_ms = _elapsed_ms(stage)
    else:
        pool = vector_hits

    if rerank:
        stage = time.perf_counter()
        # Torch releases the GIL while it computes; still, keep the model off the event loop.
        scores = await run_in_threadpool(reranker.score, query, [hit.content for hit in pool])
        results = apply_rerank_scores(pool, scores, k)
        timings.rerank_ms = _elapsed_ms(stage)
    else:
        results = pool[:k]

    timings.total_ms = _elapsed_ms(started)
    return RetrievalOutcome(results=results, mode=mode, reranked=rerank, candidates=len(pool), timings=timings)
