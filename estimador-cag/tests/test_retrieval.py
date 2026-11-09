"""Hybrid search and reranking (session 10) without PostgreSQL, without the model and without credits.

Four blocks:

1. **Pure pieces**: Reciprocal Rank Fusion (the arithmetic and the tie-breaking), the SQL of the lexical
   branch compiled to the Postgres dialect, and the agreement between migration 0002 and the model.
2. **Reranker wrapper** with an injected scorer: lazy loading, stable ordering, truncation, the smoke test
   module's verdict.
3. **The pipeline through `POST /search`** with a session double that answers the vector statement and the
   lexical statement with different rows: the four configurations A-D, the request switches versus the
   service defaults, the per-stage timings, the 500 when the model cannot load.
4. **Eval metrics and the shipped golden set**: precision@k and budget recall over budget ids, percentiles,
   and that every budget the golden set names exists in the corpus.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import TSVECTOR

from app.config import DEFAULT_RERANKER_MODEL, get_settings
from app.db.models import TEXT_SEARCH_CONFIG, TSVECTOR_EXPRESSION, Chunk
from app.db.session import get_db_session
from app.embedding_pipeline.retrieval import (
    CrossEncoderReranker,
    RerankerUnavailable,
    build_lexical_statement,
    fuse_branches,
    reciprocal_rank_fusion,
)
from app.embedding_pipeline.retrieval import verify_reranker
from app.embedding_pipeline.retrieval.service import apply_rerank_scores
from app.embedding_pipeline.router import RERANKER_UNAVAILABLE_DETAIL, get_embedder, get_reranker
from app.embedding_pipeline.schemas import RankSignals, SearchResponse, SearchResult
from app.main import app
from evals.retrieval.metrics import LatencySummary, percentile, precision_at_k, relevant_budgets_recall_at_k
from evals.retrieval.run import CONFIGURATIONS, GOLDEN_SET, RunRecord, load_golden_set, summarize
from tests.conftest import make_settings
from tests.test_embedding_pipeline import make_embedder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIGRATION_FILE = PROJECT_ROOT / "alembic" / "versions" / "0002_chunks_fulltext.py"
DATA_FILE = PROJECT_ROOT / "data" / "budgets_sample.json"


# --- 1. Pure pieces ----------------------------------------------------------------------------------------


def test_rrf_rewards_agreement_between_branches():
    """A chunk both branches return outscores one that a single branch ranks first.

    Between the two shared chunks, ranks (1, 3) beat ranks (2, 2) by a hair (1/61 + 1/63 > 2/62): the
    reciprocal is convex, so one very good position is worth slightly more than two average ones.
    """
    fused = reciprocal_rank_fusion({"vector": [1, 2, 3], "lexical": [3, 2, 9]}, k=60)
    assert [candidate.key for candidate in fused] == [3, 2, 1, 9]
    third, second = fused[0], fused[1]
    assert third.ranks == {"vector": 3, "lexical": 1} and third.score == pytest.approx(1 / 63 + 1 / 61)
    assert second.ranks == {"vector": 2, "lexical": 2} and second.score == pytest.approx(2 / 62)
    assert third.score > second.score
    assert fused[2].ranks == {"vector": 1} and fused[2].score == pytest.approx(1 / 61)
    assert fused[3].ranks == {"lexical": 3} and fused[3].score == pytest.approx(1 / 63)


def test_rrf_k_damps_the_top_positions():
    """With k = 0 the first position weighs ten times the tenth; with k = 60, barely more. Same lists, the order flips."""
    rankings = {"vector": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "lexical": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1]}
    with_k_0 = reciprocal_rank_fusion(rankings, k=0)
    with_k_60 = reciprocal_rank_fusion(rankings, k=60)
    # k = 0: 1/1 + 1/11 (chunk 1) vs 1/11 + 1/1 (chunk 11): tie, broken by the best rank then the key.
    assert [c.key for c in with_k_0[:2]] == [1, 11]
    # k = 60: still the two chunks both branches saw, far ahead of any single-branch chunk.
    assert {c.key for c in with_k_60[:2]} == {1, 11}
    assert with_k_60[2].score < with_k_60[1].score
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(rankings, k=-1)


def test_rrf_a_branch_votes_once_per_key_and_breaks_ties_deterministically():
    fused = reciprocal_rank_fusion({"vector": [5, 5, 5], "lexical": [7]})
    assert [c.key for c in fused] == [5, 7]  # same score (both rank 1): lower key first
    assert fused[0].ranks == {"vector": 1}
    assert fused[0].score == fused[1].score == pytest.approx(1 / 61)
    assert reciprocal_rank_fusion({}) == []


def test_lexical_statement_normalises_the_query_with_the_column_config_and_uses_or_semantics():
    sql = str(
        build_lexical_statement("fashion store with card payments", 50).compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )
    assert f"CAST('{TEXT_SEARCH_CONFIG}' AS REGCONFIG)" in sql  # same regconfig as the generated column
    assert "plainto_tsquery(" in sql and "replace(" in sql and "'&', '|'" in sql  # AND -> OR
    assert "AS TSQUERY" in sql
    assert "chunks.content_tsv @@" in sql  # the operator the GIN index serves
    assert "ts_rank_cd(chunks.content_tsv" in sql and ", 1)" in sql  # length-normalised rank
    assert "ORDER BY" in sql and "DESC" in sql and "LIMIT 50" in sql
    assert "<=>" not in sql  # no vector arithmetic in the lexical branch


def test_migration_0002_agrees_with_the_model():
    spec = importlib.util.spec_from_file_location("migration_0002", MIGRATION_FILE)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    assert migration.revision == "0002_chunks_fulltext" and migration.down_revision == "0001_initial_schema"
    assert migration.TEXT_SEARCH_CONFIG == TEXT_SEARCH_CONFIG == "english"
    assert migration.TSVECTOR_EXPRESSION == TSVECTOR_EXPRESSION
    column = Chunk.__table__.c.content_tsv
    assert isinstance(column.type, TSVECTOR)
    assert column.computed is not None and column.computed.persisted is True  # STORED: the only kind Postgres can index
    assert str(column.computed.sqltext) == TSVECTOR_EXPRESSION
    gin = {index.name: index for index in Chunk.__table__.indexes}[migration.INDEX_NAME]
    assert [c.name for c in gin.columns] == ["content_tsv"]
    assert gin.dialect_options["postgresql"]["using"] == "gin"
    source = MIGRATION_FILE.read_text(encoding="utf-8")
    assert 'postgresql_using="gin"' in source and "persisted=True" in source


# --- 2. Reranker wrapper -----------------------------------------------------------------------------------


def scorer_by_keyword(**scores: float):
    """Scorer double: the score of a (query, document) pair is the first keyword found in the document."""

    def score(pairs):
        out = []
        for _query, document in pairs:
            out.append(next((value for keyword, value in scores.items() if keyword in document), 0.0))
        return out

    return score


def test_reranker_is_lazy_and_reorders_with_stable_ties():
    lazy = CrossEncoderReranker("some/model")
    assert lazy.is_loaded is False and lazy.load_time_ms is None  # nothing imported, nothing downloaded

    reranker = CrossEncoderReranker("fake", scorer=scorer_by_keyword(shop=0.9, payments=0.4))
    assert reranker.is_loaded is True
    documents = ["payments vault", "online shop A", "telemetry", "online shop B"]
    items = reranker.rerank("fashion store", documents)
    assert [(item.index, item.score) for item in items] == [(1, 0.9), (3, 0.9), (0, 0.4), (2, 0.0)]
    assert [item.index for item in reranker.rerank("fashion store", documents, top_k=2)] == [1, 3]
    assert reranker.score("q", []) == []
    with pytest.raises(ValueError):
        CrossEncoderReranker("fake", batch_size=0)


def test_reranker_rejects_a_scorer_that_returns_the_wrong_count():
    reranker = CrossEncoderReranker("fake", scorer=lambda pairs: [1.0])
    with pytest.raises(RuntimeError):
        reranker.score("q", ["a", "b"])


def test_verify_reranker_passes_with_a_sensible_model_and_fails_otherwise(capsys):
    sensible = CrossEncoderReranker("fake", scorer=scorer_by_keyword(Stripe=0.9, tokenisation=0.3, MQTT=-0.5))
    assert verify_reranker.main([], reranker=sensible) == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "model:      fake" in out

    upside_down = CrossEncoderReranker("fake", scorer=scorer_by_keyword(Stripe=-1.0, tokenisation=0.3, MQTT=0.9))
    assert verify_reranker.main([], reranker=upside_down) == 1
    assert "FAIL" in capsys.readouterr().out


def test_verify_reranker_reports_a_model_that_cannot_load(capsys):
    class Broken(CrossEncoderReranker):
        def load(self) -> None:
            raise RerankerUnavailable("no weights here")

    assert verify_reranker.main([], reranker=Broken("missing/model")) == 1
    assert "no weights here" in capsys.readouterr().err


# --- 3. The pipeline through POST /search ------------------------------------------------------------------


def vector_row(chunk_id: int, distance: float, budget: str, name: str):
    return SimpleNamespace(
        id=chunk_id,
        document_id=int(budget[-3:]),
        chunk_type="budget_component",
        content=f"[Project: {budget}] Component: {name}",
        chunk_metadata={"budget_id": budget, "component_name": name},
        distance=distance,
    )


def lexical_row(chunk_id: int, rank: float, budget: str, name: str):
    return SimpleNamespace(
        id=chunk_id,
        document_id=int(budget[-3:]),
        chunk_type="budget_component",
        content=f"[Project: {budget}] Component: {name}",
        chunk_metadata={"budget_id": budget, "component_name": name},
        rank=rank,
    )


class DispatchingSession:
    """Session double that tells the two statements apart by their SQL and records the LIMIT each asked for."""

    def __init__(self, vector_rows, lexical_rows):
        self.vector_rows = vector_rows
        self.lexical_rows = lexical_rows
        self.calls: list[tuple[str, int]] = []  # (branch, limit)

    async def execute(self, statement):
        sql = str(statement.compile(dialect=postgresql.dialect()))
        limit = statement._limit_clause.value
        if "<=>" in sql:
            self.calls.append(("vector", limit))
            rows = self.vector_rows[:limit]
        elif "ts_rank_cd" in sql:
            self.calls.append(("lexical", limit))
            rows = self.lexical_rows[:limit]
        else:  # pragma: no cover - a third statement would be a bug
            raise AssertionError(f"unexpected statement: {sql}")
        return SimpleNamespace(all=lambda: list(rows))


# The exercise's failure case in miniature: the vector branch likes the payment gateway (shares the words
# "card" and "checkout"), the lexical branch finds the shops by their exact component names, and only the
# fashion storefront appears in both.
VECTOR_ROWS = [
    vector_row(1, 0.40, "BUD-2025-003", "Card tokenisation vault"),
    vector_row(2, 0.42, "BUD-2024-021", "Cart and Stripe checkout"),
    vector_row(3, 0.44, "BUD-2025-003", "3-D Secure 2 authentication flow"),
    vector_row(4, 0.46, "BUD-2024-021", "Product catalog service"),
    vector_row(5, 0.48, "BUD-2023-025", "Telemetry ingestion"),
    vector_row(6, 0.50, "BUD-2024-014", "OAuth 2.0 authentication backend"),
]
LEXICAL_ROWS = [
    lexical_row(2, 0.9, "BUD-2024-021", "Cart and Stripe checkout"),
    lexical_row(7, 0.8, "BUD-2025-008", "WooCommerce migration and catalog"),
    lexical_row(8, 0.7, "BUD-2022-004", "In-app payments"),
    lexical_row(4, 0.6, "BUD-2024-021", "Product catalog service"),
]
FAKE_SCORES = {"Cart and Stripe": 0.95, "Product catalog": 0.90, "WooCommerce": 0.80, "In-app payments": 0.70, "Telemetry": -0.9}


@pytest.fixture
def retrieval_api(request):
    """TestClient with the embedder, the session and the reranker replaced by doubles.

    Parametrise indirectly with a dict of `make_settings` overrides to change the service defaults.
    """
    overrides = getattr(request, "param", None) or {}
    session = DispatchingSession(VECTOR_ROWS, LEXICAL_ROWS)
    embedder, fake_api, _ = make_embedder()
    reranker = CrossEncoderReranker("fake-cross-encoder", scorer=scorer_by_keyword(**FAKE_SCORES))
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_db_session] = lambda: session
    app.dependency_overrides[get_reranker] = lambda: reranker
    app.dependency_overrides[get_settings] = lambda: make_settings(openai_api_key="sk-test", **overrides)
    try:
        yield SimpleNamespace(client=TestClient(app), session=session, embeddings_api=fake_api, reranker=reranker)
    finally:
        app.dependency_overrides.clear()


def ids(body: SearchResponse) -> list[int]:
    return [result.chunk_id for result in body.results]


def test_configuration_a_vector_without_rerank_is_the_session_08_baseline(retrieval_api):
    response = retrieval_api.client.post("/search", json={"query": "fashion store", "k": 3, "mode": "vector", "rerank": False})
    assert response.status_code == 200, response.text
    body = SearchResponse.model_validate(response.json())
    assert body.mode == "vector" and body.reranked is False and body.candidates == 3
    assert ids(body) == [1, 2, 3]  # the vector order, untouched: the gateway wins
    assert retrieval_api.session.calls == [("vector", 3)]  # exactly k rows, nothing reorders them later
    first = body.results[0]
    assert first.distance == 0.40 and first.score == pytest.approx(0.60)
    assert first.ranking == RankSignals(vector_rank=1)
    assert body.timings.lexical_ms == 0 and body.timings.rerank_ms == 0 and body.timings.total_ms >= 0
    assert retrieval_api.embeddings_api.calls == [["fashion store"]]  # one embedding call, as before


def test_configuration_b_hybrid_fuses_both_branches_with_rrf(retrieval_api):
    response = retrieval_api.client.post("/search", json={"query": "fashion store", "k": 4, "mode": "hybrid", "rerank": False})
    body = SearchResponse.model_validate(response.json())
    assert body.mode == "hybrid" and body.reranked is False
    # Both branches recall the full candidate width (50), not k.
    assert retrieval_api.session.calls == [("vector", 50), ("lexical", 50)]
    # Chunks found by both branches rise above the vector-only top hit.
    assert ids(body) == [2, 4, 1, 7]
    cart = body.results[0]
    assert cart.ranking.vector_rank == 2 and cart.ranking.lexical_rank == 1
    assert cart.ranking.rrf_score == pytest.approx(1 / 62 + 1 / 61) and cart.score == cart.ranking.rrf_score
    assert cart.distance == 0.42  # came through the vector branch too: keeps its distance
    woocommerce = body.results[3]
    assert woocommerce.distance is None and woocommerce.ranking.vector_rank is None and woocommerce.ranking.lexical_rank == 2
    assert body.candidates == 8  # 6 vector + 4 lexical, 2 shared
    assert len(body.results) == 4


def test_configuration_c_vector_with_rerank_recalls_wide_then_keeps_the_top_k(retrieval_api):
    response = retrieval_api.client.post("/search", json={"query": "fashion store", "k": 2, "mode": "vector", "rerank": True})
    body = SearchResponse.model_validate(response.json())
    assert body.mode == "vector" and body.reranked is True
    assert retrieval_api.session.calls == [("vector", 50)]  # recall-then-rerank: wide recall, not k
    assert body.candidates == 6  # everything the corpus double had
    assert ids(body) == [2, 4]  # the cross-encoder demotes the gateway
    assert [r.ranking.rerank_score for r in body.results] == [0.95, 0.90]
    assert body.results[0].score == 0.95 and body.results[0].ranking.vector_rank == 2
    assert body.timings.rerank_ms >= 0


def test_configuration_d_hybrid_with_rerank(retrieval_api):
    response = retrieval_api.client.post("/search", json={"query": "fashion store", "k": 4, "mode": "hybrid", "rerank": True})
    body = SearchResponse.model_validate(response.json())
    assert body.mode == "hybrid" and body.reranked is True
    assert retrieval_api.session.calls == [("vector", 50), ("lexical", 50)]
    assert ids(body) == [2, 4, 7, 8]  # the three shops first; the gateway (score 0.0) and telemetry (-0.9) are out
    assert body.results[2].ranking.lexical_rank == 2 and body.results[2].ranking.rerank_score == 0.80
    assert body.results[2].ranking.rrf_score is not None  # fusion evidence survives the rerank


@pytest.mark.parametrize("retrieval_api", [{"search_mode": "hybrid", "rerank_enabled": True}], indirect=True)
def test_request_switches_default_to_the_service_settings_and_can_override_them(retrieval_api):
    body = SearchResponse.model_validate(retrieval_api.client.post("/search", json={"query": "q", "k": 2}).json())
    assert body.mode == "hybrid" and body.reranked is True

    body = SearchResponse.model_validate(retrieval_api.client.post("/search", json={"query": "q", "k": 2, "mode": "vector", "rerank": False}).json())
    assert body.mode == "vector" and body.reranked is False


@pytest.mark.parametrize("body", [{"query": "q", "mode": "keyword"}, {"query": "q", "rerank": "yes please"}])
def test_invalid_switch_values_are_422(retrieval_api, body):
    assert retrieval_api.client.post("/search", json=body).status_code == 422


def test_rerank_requested_but_model_unavailable_is_a_500_with_a_useful_detail(retrieval_api):
    class Broken(CrossEncoderReranker):
        def load(self) -> None:
            raise RerankerUnavailable("no weights")

    app.dependency_overrides[get_reranker] = lambda: Broken("missing/model")
    response = retrieval_api.client.post("/search", json={"query": "q", "rerank": True})
    assert response.status_code == 500
    assert response.json()["detail"] == RERANKER_UNAVAILABLE_DETAIL
    # Vector search keeps working on the same service.
    assert retrieval_api.client.post("/search", json={"query": "q", "rerank": False, "mode": "vector"}).status_code == 200


def test_health_exposes_the_retrieval_switches(client):
    body = client.get("/health").json()
    assert body["search_mode"] == "vector" and body["rerank_enabled"] is False  # the test settings
    assert body["rerank_model"] == DEFAULT_RERANKER_MODEL
    assert body["reranker_loaded"] is False  # no model was loaded by running the tests


def test_fuse_branches_and_apply_rerank_scores_directly():
    vector = [SearchResult(chunk_id=1, document_id=1, chunk_type="c", content="a", distance=0.3, score=0.7, ranking=RankSignals(vector_rank=1))]
    lexical = [
        SearchResult(chunk_id=1, document_id=1, chunk_type="c", content="a", score=0.9, ranking=RankSignals(lexical_rank=1)),
        SearchResult(chunk_id=2, document_id=2, chunk_type="c", content="b", score=0.5, ranking=RankSignals(lexical_rank=2)),
    ]
    fused = fuse_branches(vector, lexical, rrf_k=60)
    assert [r.chunk_id for r in fused] == [1, 2]
    assert fused[0].distance == 0.3
    assert (fused[0].ranking.vector_rank, fused[0].ranking.lexical_rank) == (1, 1)
    assert fused[0].ranking.rrf_score == pytest.approx(2 / 61) and fused[0].score == fused[0].ranking.rrf_score
    assert fused[1].distance is None and fused[1].ranking.vector_rank is None and fused[1].ranking.lexical_rank == 2

    reranked = apply_rerank_scores(fused, [0.1, 0.8], k=5)
    assert [r.chunk_id for r in reranked] == [2, 1]
    assert reranked[0].score == 0.8 and reranked[0].ranking.rerank_score == 0.8 and reranked[0].ranking.rrf_score is not None
    assert apply_rerank_scores(fused, [0.5, 0.5], k=1)[0].chunk_id == 1  # ties keep the recall order


# --- 4. Eval metrics and the golden set --------------------------------------------------------------------


def test_precision_and_budget_recall_at_k():
    retrieved = ["BUD-A", "BUD-B", "BUD-A", "BUD-C", "BUD-D"]
    assert precision_at_k(retrieved, {"BUD-A"}, 5) == pytest.approx(0.4)
    assert precision_at_k(retrieved, {"BUD-A", "BUD-C"}, 5) == pytest.approx(0.6)
    assert precision_at_k(["BUD-A"], {"BUD-A"}, 5) == pytest.approx(0.2)  # fewer than k results count as misses
    assert relevant_budgets_recall_at_k(retrieved, {"BUD-A", "BUD-C", "BUD-Z"}, 5) == pytest.approx(2 / 3)
    assert relevant_budgets_recall_at_k(retrieved, {"BUD-A"}, 2) == 1.0
    with pytest.raises(ValueError):
        precision_at_k(retrieved, {"BUD-A"}, 0)
    with pytest.raises(ValueError):
        relevant_budgets_recall_at_k(retrieved, set(), 5)


def test_percentiles_and_latency_summary():
    samples = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert percentile(samples, 50) == 50 and percentile(samples, 95) == 100 and percentile(samples, 0) == 10
    summary = LatencySummary.from_samples([300, 120, 150])
    assert summary.samples == 3 and summary.median_ms == 150 and summary.p95_ms == 300 and summary.min_ms == 120
    with pytest.raises(ValueError):
        percentile([], 50)


@pytest.mark.parametrize("golden_file", [GOLDEN_SET, GOLDEN_SET.with_name("golden_set_paraphrases.json")])
def test_shipped_golden_sets_are_consistent_with_the_corpus(golden_file):
    k, queries = load_golden_set(golden_file)
    assert k == 5 and len(queries) == 5
    budgets = {budget["budget_id"] for budget in json.loads(DATA_FILE.read_text(encoding="utf-8"))}
    for query in queries:
        assert set(query.relevant_budget_ids) <= budgets, query.id
        assert set(query.distractors) <= budgets, query.id
        assert query.rationale and len(query.query.split()) >= 15  # project descriptions, not keywords


def test_summarize_aggregates_precision_per_query_and_latency_per_configuration():
    def record(config, query_id, repetition, precision, ms, rerank_ms=0):
        return RunRecord(
            config=config, query_id=query_id, repetition=repetition, precision_at_k=precision, budget_recall_at_k=1.0,
            server_ms=ms, client_ms=ms + 2, embedding_ms=100, vector_ms=5, lexical_ms=0, fusion_ms=0, rerank_ms=rerank_ms,
            candidates=5, top_budget_ids=["BUD-X"] * 5, top_labels=["x"] * 5,
        )

    records = [
        record("A", "q1", 0, 0.4, 110), record("A", "q1", 1, 0.4, 130),
        record("A", "q2", 0, 1.0, 120), record("A", "q2", 1, 1.0, 900),
        record("C", "q1", 0, 0.8, 400, 280), record("C", "q1", 1, 1.0, 420, 300),
        record("C", "q2", 0, 1.0, 410, 290), record("C", "q2", 1, 1.0, 430, 310),
    ]
    summaries = summarize(records, CONFIGURATIONS)
    assert [s.config.label for s in summaries] == ["A", "C"]  # B and D had no records
    a, c = summaries
    assert a.precision_at_k == pytest.approx(0.7) and a.precision_stable is True
    # Nearest-rank percentiles: the median of four samples is the second one, and one slow request only moves the p95.
    assert a.latency.median_ms == 120 and a.latency.p95_ms == 900 and a.client_latency.median_ms == 122
    assert c.precision_at_k == pytest.approx((0.9 + 1.0) / 2) and c.precision_stable is False
    assert c.stage_medians_ms["rerank_ms"] == 290 and c.stage_medians_ms["embedding_ms"] == 100
