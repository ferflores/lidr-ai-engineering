"""Run the four retrieval configurations against the golden set and measure precision@k and latency.

    uv run python -m evals.retrieval.run --http http://localhost:8000
    uv run python -m evals.retrieval.run --http http://localhost:8000 --repetitions 5 \
        --output evals/retrieval/results.csv --report evals/retrieval/REPORT.md
    docker compose run --rm api python -m evals.retrieval.run          # API_BASE_URL=http://api:8000 inside compose

    Configuration   Search    Reranking
    A               vector    no          the session 08 baseline
    B               hybrid    no          vector + full-text fused with RRF
    C               vector    yes         recall 50 by vector, cross-encoder picks the top-k
    D               hybrid    yes         recall 50 by hybrid, cross-encoder picks the top-k

Each configuration is invoked through the public endpoint with the request switches (`mode`, `rerank`), so
the four runs hit the same service, the same corpus and the same embedding model, and only the switches
change. Every query is sent `--repetitions` times: precision comes from the result lists (and the report says
whether it was stable across repetitions), latency from the server-side `timings` of every repetition, after
one uncounted warm-up request per configuration.

Output: a CSV with one row per (configuration, query, repetition) and a Markdown report with the comparative
table, the per-query precision and the top-k of every configuration annotated against the golden set. The
conclusions are written by hand in the README; this script produces the numbers, not the opinion.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.clients.estimator_api import DEFAULT_BASE_URL  # noqa: E402
from evals.retrieval.metrics import (  # noqa: E402
    LatencySummary,
    percentile,
    precision_at_k,
    relevant_budgets_recall_at_k,
)

GOLDEN_SET = Path(__file__).resolve().parent / "golden_set.json"
STAGES = ("embedding_ms", "vector_ms", "lexical_ms", "fusion_ms", "rerank_ms")


@dataclass(frozen=True)
class Configuration:
    label: str
    mode: str
    rerank: bool
    description: str

    @property
    def search_name(self) -> str:
        return "Híbrida" if self.mode == "hybrid" else "Vectorial"

    @property
    def rerank_name(self) -> str:
        return "Sí" if self.rerank else "No"


CONFIGURATIONS: tuple[Configuration, ...] = (
    Configuration("A", "vector", False, "vector top-k (session 08 baseline)"),
    Configuration("B", "hybrid", False, "vector + full-text, Reciprocal Rank Fusion"),
    Configuration("C", "vector", True, "vector top-50, cross-encoder rerank, top-k"),
    Configuration("D", "hybrid", True, "hybrid top-50, cross-encoder rerank, top-k"),
)


@dataclass(frozen=True)
class GoldenQuery:
    id: str
    query: str
    relevant_budget_ids: tuple[str, ...]
    rationale: str = ""
    distractors: dict[str, str] = field(default_factory=dict)


def load_golden_set(path: Path = GOLDEN_SET) -> tuple[int, list[GoldenQuery]]:
    """The golden set file: `k` and the annotated queries. Validates what a typo would silently break."""
    data = json.loads(path.read_text(encoding="utf-8"))
    k = int(data["k"])
    queries = [
        GoldenQuery(
            id=item["id"],
            query=item["query"],
            relevant_budget_ids=tuple(item["relevant_budget_ids"]),
            rationale=item.get("rationale", ""),
            distractors=dict(item.get("distractors", {})),
        )
        for item in data["queries"]
    ]
    ids = [query.id for query in queries]
    if len(set(ids)) != len(ids):
        raise ValueError("golden set query ids must be unique")
    for query in queries:
        if not query.relevant_budget_ids:
            raise ValueError(f"golden query {query.id} has no relevant budgets")
        overlap = set(query.relevant_budget_ids) & set(query.distractors)
        if overlap:
            raise ValueError(f"golden query {query.id} lists {sorted(overlap)} as both relevant and distractor")
    return k, queries


@dataclass
class RunRecord:
    """One request: which configuration, which query, which repetition, what came back and how long it took."""

    config: str
    query_id: str
    repetition: int
    precision_at_k: float
    budget_recall_at_k: float
    server_ms: int
    client_ms: int
    embedding_ms: int
    vector_ms: int
    lexical_ms: int
    fusion_ms: int
    rerank_ms: int
    candidates: int
    top_budget_ids: list[str]
    top_labels: list[str]  # "component (budget)" per result, for the report


def search(client: httpx.Client, query: str, k: int, config: Configuration) -> tuple[dict, int]:
    """One `POST /search` with the configuration's switches. Returns the body and the client wall-clock ms."""
    started = time.perf_counter()
    response = client.post("/search", json={"query": query, "k": k, "mode": config.mode, "rerank": config.rerank})
    client_ms = int((time.perf_counter() - started) * 1000)
    response.raise_for_status()
    return response.json(), client_ms


def describe(result: dict) -> str:
    meta = result.get("metadata") or {}
    return f"{meta.get('component_name', '?')} ({meta.get('budget_id', '?')})"


def evaluate_body(body: dict, golden: GoldenQuery, k: int, config: Configuration, repetition: int, client_ms: int) -> RunRecord:
    results = body["results"]
    budget_ids = [(result.get("metadata") or {}).get("budget_id", "?") for result in results]
    timings = body.get("timings") or {}
    return RunRecord(
        config=config.label,
        query_id=golden.id,
        repetition=repetition,
        precision_at_k=precision_at_k(budget_ids, golden.relevant_budget_ids, k),
        budget_recall_at_k=relevant_budgets_recall_at_k(budget_ids, golden.relevant_budget_ids, k),
        server_ms=int(body["search_time_ms"]),
        client_ms=client_ms,
        embedding_ms=int(timings.get("embedding_ms", 0)),
        vector_ms=int(timings.get("vector_ms", 0)),
        lexical_ms=int(timings.get("lexical_ms", 0)),
        fusion_ms=int(timings.get("fusion_ms", 0)),
        rerank_ms=int(timings.get("rerank_ms", 0)),
        candidates=int(body.get("candidates", 0)),
        top_budget_ids=budget_ids,
        top_labels=[describe(result) for result in results],
    )


def run_eval(
    base_url: str,
    queries: Sequence[GoldenQuery],
    *,
    k: int,
    repetitions: int,
    configs: Sequence[Configuration],
    warmup: int = 1,
    timeout: float = 120.0,
    bodies: dict[tuple[str, str, int], dict] | None = None,
) -> tuple[list[RunRecord], dict]:
    """All configurations × queries × repetitions against the service. `bodies` keeps the raw responses of
    repetition 0 (for the report's top-k listings) when the caller passes a dict."""
    records: list[RunRecord] = []
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        health = client.get("/health").json()
        for config in configs:
            for _ in range(warmup):
                search(client, queries[0].query, k, config)  # uncounted: model load, connection pool, caches
            for golden in queries:
                for repetition in range(repetitions):
                    body, client_ms = search(client, golden.query, k, config)
                    records.append(evaluate_body(body, golden, k, config, repetition, client_ms))
                    if bodies is not None and repetition == 0:
                        bodies[(config.label, golden.id, repetition)] = body
                print(f"  {config.label} {golden.id:<28} P@{k}={records[-1].precision_at_k:.2f}  {records[-1].server_ms:>5} ms", file=sys.stderr)
    return records, health


@dataclass
class ConfigSummary:
    config: Configuration
    precision_at_k: float
    precision_stable: bool  # same P@k in every repetition of every query
    budget_recall_at_k: float
    latency: LatencySummary
    client_latency: LatencySummary
    stage_medians_ms: dict[str, float]
    candidates: int


def summarize(records: Iterable[RunRecord], configs: Sequence[Configuration]) -> list[ConfigSummary]:
    records = list(records)
    summaries: list[ConfigSummary] = []
    for config in configs:
        own = [record for record in records if record.config == config.label]
        if not own:
            continue
        query_ids = sorted({record.query_id for record in own})
        per_query_precision = []
        stable = True
        for query_id in query_ids:
            values = [record.precision_at_k for record in own if record.query_id == query_id]
            per_query_precision.append(sum(values) / len(values))
            stable = stable and len(set(values)) == 1
        per_query_recall = [
            sum(record.budget_recall_at_k for record in own if record.query_id == query_id) / sum(1 for record in own if record.query_id == query_id)
            for query_id in query_ids
        ]
        summaries.append(
            ConfigSummary(
                config=config,
                precision_at_k=sum(per_query_precision) / len(per_query_precision),
                precision_stable=stable,
                budget_recall_at_k=sum(per_query_recall) / len(per_query_recall),
                latency=LatencySummary.from_samples([record.server_ms for record in own]),
                client_latency=LatencySummary.from_samples([record.client_ms for record in own]),
                stage_medians_ms={stage: percentile([getattr(record, stage) for record in own], 50) for stage in STAGES},
                candidates=max(record.candidates for record in own),
            )
        )
    return summaries


def write_csv(records: Sequence[RunRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        fields = [name for name in RunRecord.__dataclass_fields__ if name not in ("top_budget_ids", "top_labels")]
        writer.writerow([*fields, "top_budget_ids"])
        for record in records:
            row = asdict(record)
            writer.writerow([*(row[name] for name in fields), "|".join(record.top_budget_ids)])


def render_report(
    summaries: Sequence[ConfigSummary],
    records: Sequence[RunRecord],
    queries: Sequence[GoldenQuery],
    *,
    k: int,
    repetitions: int,
    base_url: str,
    health: dict,
    bodies: dict[tuple[str, str, int], dict],
) -> str:
    """Markdown: environment, the comparative table, precision per query and the annotated top-k lists."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Retrieval eval: hybrid search and reranking (session 10)",
        "",
        f"Generated by `python -m evals.retrieval.run` on {now}. Numbers, not opinion: the reading is in the README.",
        "",
        "## Environment",
        "",
        f"- API: `{base_url}` · corpus: 15 budgets / 73 chunks (`data/budgets_sample.json`) · k = {k} · repetitions per query = {repetitions} (+1 warm-up per configuration, not counted)",
        f"- Embedding model: `{health.get('embedding_model', '?')}` (OpenAI, one call per query in every configuration)",
        f"- Reranker: `{health.get('rerank_model', '?')}` on CPU · recall width: {max((s.candidates for s in summaries), default=0)} candidates",
        f"- Runner host: {platform.platform()} · service defaults: mode={health.get('search_mode', '?')}, rerank={health.get('rerank_enabled', '?')}",
        "",
        "## Comparative table",
        "",
        f"Precision@{k} = share of the {k} returned chunks whose budget is in the golden set (mean over the {len(queries)} queries). "
        f"Budgets@{k} = share of the relevant budgets that appear at least once in the top-{k}. Latency = server-side time of the whole request "
        "(query embedding + SQL + fusion + rerank), median and p95 over all repetitions of all queries; the last columns split the median by stage.",
        "",
        f"| Config | Búsqueda | Reranking | Precisión@{k} | Budgets@{k} | Latencia mediana | p95 | embedding | SQL vector | SQL full-text | RRF | rerank |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for summary in summaries:
        stages = summary.stage_medians_ms
        stable = "" if summary.precision_stable else " *"
        lines.append(
            f"| {summary.config.label} | {summary.config.search_name} | {summary.config.rerank_name} | "
            f"{summary.precision_at_k:.2f}{stable} | {summary.budget_recall_at_k:.2f} | "
            f"{summary.latency.median_ms:.0f} ms | {summary.latency.p95_ms:.0f} ms | "
            f"{stages['embedding_ms']:.0f} ms | {stages['vector_ms']:.0f} ms | {stages['lexical_ms']:.0f} ms | {stages['fusion_ms']:.0f} ms | {stages['rerank_ms']:.0f} ms |"
        )
    if any(not summary.precision_stable for summary in summaries):
        lines.append("")
        lines.append(f"\\* precision changed between repetitions of the same query (the embedding API is not bit-exact between calls).")
    lines += [
        "",
        f"Client-side wall clock (HTTP included), median: "
        + ", ".join(f"{s.config.label} {s.client_latency.median_ms:.0f} ms" for s in summaries)
        + ".",
        "",
        f"## Precision@{k} per query",
        "",
        "| Query | relevant budgets | " + " | ".join(summary.config.label for summary in summaries) + " |",
        "|---|---|" + "---|" * len(summaries),
    ]
    for golden in queries:
        cells = []
        for summary in summaries:
            values = [r.precision_at_k for r in records if r.config == summary.config.label and r.query_id == golden.id]
            cells.append(f"{sum(values) / len(values):.2f}" if values else "—")
        lines.append(f"| {golden.id} | {', '.join(golden.relevant_budget_ids)} | " + " | ".join(cells) + " |")
    lines += ["", f"## Top-{k} per query and configuration (repetition 0)", ""]
    lines.append("✓ = chunk of a relevant budget, ✗ = not relevant. In brackets: rank in the vector branch / rank in the lexical branch / cross-encoder score.")
    for golden in queries:
        lines += ["", f"### {golden.id}", "", f"> {golden.query}", "", f"Relevant: {', '.join(golden.relevant_budget_ids)}. {golden.rationale}", ""]
        if golden.distractors:
            lines.append("Distractors: " + "; ".join(f"**{budget}** — {why}" for budget, why in golden.distractors.items()))
            lines.append("")
        for summary in summaries:
            body = bodies.get((summary.config.label, golden.id, 0))
            if body is None:
                continue
            lines.append(f"**{summary.config.label} · {summary.config.search_name}, reranking {summary.config.rerank_name.lower()}** — {body['search_time_ms']} ms")
            lines.append("")
            for position, result in enumerate(body["results"], start=1):
                meta = result.get("metadata") or {}
                mark = "✓" if meta.get("budget_id") in golden.relevant_budget_ids else "✗"
                ranking = result.get("ranking") or {}
                signals = f"v{ranking.get('vector_rank') or '–'} / l{ranking.get('lexical_rank') or '–'}"
                if ranking.get("rerank_score") is not None:
                    signals += f" / ce {ranking['rerank_score']:.3f}"
                distance = f" d={result['distance']:.4f}" if result.get("distance") is not None else ""
                lines.append(f"{position}. {mark} {describe(result)}{distance} [{signals}]")
            lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the four retrieval configurations against the golden set.")
    parser.add_argument("--http", default=None, help="API base URL (default: API_BASE_URL or http://localhost:8000).")
    parser.add_argument("--golden", type=Path, default=GOLDEN_SET, help="Golden set JSON.")
    parser.add_argument("--k", type=int, default=None, help="Results per query (default: the golden set's k).")
    parser.add_argument("--repetitions", type=int, default=5, help="Requests per (configuration, query) for latency (default: 5).")
    parser.add_argument("--configs", default="A,B,C,D", help="Comma-separated subset of A,B,C,D.")
    parser.add_argument("--output", type=Path, default=None, help="CSV with one row per request.")
    parser.add_argument("--report", type=Path, default=None, help="Markdown report.")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    base_url = (args.http or os.environ.get("API_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    golden_k, queries = load_golden_set(args.golden)
    k = args.k or golden_k
    wanted = {label.strip().upper() for label in args.configs.split(",") if label.strip()}
    configs = [config for config in CONFIGURATIONS if config.label in wanted]
    if not configs:
        print(f"error: no known configuration in {args.configs!r}", file=sys.stderr)
        return 2

    print(f"Golden set: {len(queries)} queries · k={k} · {args.repetitions} repetitions · configs {''.join(c.label for c in configs)} · {base_url}", file=sys.stderr)
    bodies: dict[tuple[str, str, int], dict] = {}
    records, health = run_eval(base_url, queries, k=k, repetitions=args.repetitions, configs=configs, timeout=args.timeout, bodies=bodies)
    summaries = summarize(records, configs)

    print()
    print(f"{'config':<8}{'search':<10}{'rerank':<8}{'P@' + str(k):>6}{'budgets@' + str(k):>11}{'median ms':>11}{'p95 ms':>9}{'rerank ms':>11}")
    for summary in summaries:
        print(
            f"{summary.config.label:<8}{summary.config.mode:<10}{str(summary.config.rerank).lower():<8}"
            f"{summary.precision_at_k:>6.2f}{summary.budget_recall_at_k:>11.2f}"
            f"{summary.latency.median_ms:>11.0f}{summary.latency.p95_ms:>9.0f}{summary.stage_medians_ms['rerank_ms']:>11.0f}"
        )

    if args.output:
        write_csv(records, args.output)
        print(f"\nCSV: {args.output}", file=sys.stderr)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            render_report(summaries, records, queries, k=k, repetitions=args.repetitions, base_url=base_url, health=health, bodies=bodies),
            encoding="utf-8",
        )
        print(f"Report: {args.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
