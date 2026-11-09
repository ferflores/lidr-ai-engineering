"""Keyword branch of hybrid search: PostgreSQL full-text search over `chunks.content_tsv`.

What the vector branch misses, this one catches: exact names. "WooCommerce", "HL7", "OEE", "Stripe" or
"PSD2" are single tokens whose meaning an embedding model dilutes among the rest of the sentence, while a
tsvector match on them is binary and cheap. The two signals are fused by rank in `fusion.py`.

Two decisions worth stating:

1. **Same configuration as the column.** The query is normalised with `TEXT_SEARCH_CONFIG` (the regconfig
   the generated column uses, see migration 0002): stemming and stop words have to be applied identically on
   both sides or "payments" in the query never meets "payment" in a chunk.
2. **OR semantics, not AND.** `plainto_tsquery` / `websearch_to_tsquery` join every term with `&`. A query
   here is a project description of 15-30 words, and requiring all of them in one 60-word chunk returns
   nothing (measured: zero rows for all five golden queries on the 73-chunk corpus). The query is therefore
   built by `plainto_tsquery` (which quotes and normalises the lexemes safely) and then every `&` is
   rewritten to `|`, so any overlapping keyword is a match and `ts_rank_cd` orders by how many terms match
   and how close together they appear. Only the order matters downstream: RRF ignores the score itself.
"""

from __future__ import annotations

from sqlalchemy import Text, cast, func, literal, select
from sqlalchemy.dialects.postgresql import REGCONFIG, TSQUERY
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import TEXT_SEARCH_CONFIG, Chunk
from app.embedding_pipeline.schemas import RankSignals, SearchResult

# `ts_rank_cd` normalisation flag 1: divide the rank by 1 + log(document length). Without it the longest
# chunk of the corpus (a 130-word component description) outranks shorter chunks that match the same terms.
TS_RANK_LENGTH_NORMALISATION = 1


def build_lexical_tsquery(query: str):
    """`plainto_tsquery(config, query)` with every AND turned into an OR (see module docstring)."""
    regconfig = cast(literal(TEXT_SEARCH_CONFIG), REGCONFIG)
    and_query = cast(func.plainto_tsquery(regconfig, query), Text)
    return cast(func.replace(and_query, "&", "|"), TSQUERY)


def build_lexical_statement(query: str, n: int):
    """Top-n chunks whose tsvector matches any query term, best `ts_rank_cd` first.

    The `@@` in the WHERE is what lets Postgres use the GIN index; the rank is projected so the result
    carries its own evidence (same reasoning as `distance` in the vector statement). Separated from the
    `execute` so the generated SQL can be checked in tests without a database.
    """
    tsquery = build_lexical_tsquery(query)
    rank = func.ts_rank_cd(Chunk.content_tsv, tsquery, TS_RANK_LENGTH_NORMALISATION)
    return (
        select(
            Chunk.id,
            Chunk.document_id,
            Chunk.chunk_type,
            Chunk.content,
            Chunk.chunk_metadata,
            rank.label("rank"),
        )
        .where(Chunk.content_tsv.op("@@")(tsquery))
        .order_by(rank.desc(), Chunk.id)
        .limit(n)
    )


async def lexical_search(session: AsyncSession, query: str, n: int) -> list[SearchResult]:
    """Execute the keyword search. `distance` is None (no vector involved); `score` is the ts_rank_cd."""
    rows = (await session.execute(build_lexical_statement(query, n))).all()
    return [
        SearchResult(
            chunk_id=row.id,
            document_id=row.document_id,
            chunk_type=row.chunk_type,
            content=row.content,
            distance=None,
            score=float(row.rank),
            ranking=RankSignals(lexical_rank=position),
            metadata=row.chunk_metadata or {},
        )
        for position, row in enumerate(rows, start=1)
    ]
