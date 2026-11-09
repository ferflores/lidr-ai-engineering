"""Retrieval over the vectorised budgets: hybrid search and reranking (session 10 pre-work).

    lexical.py    keyword branch: PostgreSQL full-text search over the generated `content_tsv` column
    fusion.py     Reciprocal Rank Fusion of the vector and lexical rankings (pure function)
    reranker.py   cross-encoder wrapper: lazy model loading, pair scoring, stable reordering
    service.py    the pipeline: recall (vector | hybrid) → optional rerank → top-k, with per-stage timings
    verify_reranker.py   `python -m ...verify_reranker`: the model downloads, loads and orders three pairs sensibly

The vector branch itself stays where session 08 left it (`app/embedding_pipeline/repository.py`).
"""

from app.embedding_pipeline.retrieval.fusion import DEFAULT_RRF_K, FusedCandidate, reciprocal_rank_fusion
from app.embedding_pipeline.retrieval.lexical import build_lexical_statement, lexical_search
from app.embedding_pipeline.retrieval.reranker import CrossEncoderReranker, RerankedItem, RerankerUnavailable
from app.embedding_pipeline.retrieval.service import (
    DEFAULT_CANDIDATES,
    RetrievalOutcome,
    RetrievalTimings,
    fuse_branches,
    retrieve,
)

__all__ = [
    "DEFAULT_CANDIDATES",
    "DEFAULT_RRF_K",
    "CrossEncoderReranker",
    "FusedCandidate",
    "RerankedItem",
    "RerankerUnavailable",
    "RetrievalOutcome",
    "RetrievalTimings",
    "build_lexical_statement",
    "fuse_branches",
    "lexical_search",
    "reciprocal_rank_fusion",
    "retrieve",
]
