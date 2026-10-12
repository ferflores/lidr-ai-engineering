"""Runner del stress del CAG (sesión 06, bloque 5): escenarios × tamaños de adjunto × repeticiones → CSV → REPORT.md.

    uv run python -m evals.stress.run --http http://localhost:8000 \\
        --scenarios growing,pivot,contradiction \\
        --attachment-sizes 0,5,20,50,100 \\
        --repeats 3 \\
        --output evals/stress/results.csv

    uv run python -m evals.stress.run                 # in-process (ASGI), lee .env; mismos flags
    uv run python -m evals.stress.run --fake --turns 8  # humo sin API key: LLM simulado (NO vale como baseline)

Reusa el patrón de `evals/run.py`: httpx contra un servicio (`--http`) o contra la app ASGI en el mismo
proceso. Por cada (escenario, tamaño, repetición) crea una sesión, manda los turnos del escenario (el
adjunto, si lo hay, va en el turno 1 con la estimación inicial), y por cada turno:

1. Lee el evento `turn_observed` **de la respuesta** de `POST /sessions/{id}/estimate` (campo `turn_observed`)
   y lo contrasta con `GET /sessions/{id}` (`last_turn_observed`). Se eligió exponer el evento por la API en
   vez de parsear stdout porque el runner funciona igual contra Docker, contra uvicorn o in-process, no
   depende del formato de log ni de reconciliar timestamps, y el evento es exactamente el mismo diccionario
   que se emite al log (misma función, `emit_turn_observed`).
2. Evalúa `LatencyBudgetMetric` y `CostBudgetMetric` sobre el evento, y `MemoryDriftMetric` sobre el
   snapshot de la sesión para cada hecho declarado en turnos anteriores (fact-tracker del escenario).
3. En el turno 1 con adjunto mide el recall de los marcadores del PDF en la estimación.
4. Escribe una fila en el CSV: todas las columnas del `turn_observed` más una por métrica.

Al terminar genera `REPORT.md` con `evals/stress/report.py` (se puede regenerar a mano desde el CSV).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from evals.metrics import contains
from evals.stress import report as report_module
from evals.stress.fixtures.build_pdfs import MARKER_TERMS, ensure_pdfs, pdf_path
from evals.stress.metrics import CostBudgetMetric, LatencyBudgetMetric, SessionSnapshot, recall_of_facts
from evals.stress.scenarios import Scenario, get_scenarios

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results.csv"
DEFAULT_REPORT = Path(__file__).resolve().parent / "REPORT.md"
API = "/api/v1"

TURN_OBSERVED_COLUMNS: tuple[str, ...] = (
    "turn_index", "session_id", "enriched_transcript_chars", "attachments_total_chars", "messages_in_window",
    "anchors_count", "summary_chars", "tokens_in", "tokens_out", "cost_usd", "latency_ms", "cache_hit_kind",
    "last_resolved_tier",
)
EXTRA_COLUMNS: tuple[str, ...] = (
    "last_tier_rule", "model", "provider", "prompt_version", "system_prompt_chars", "context_chars", "dropped_turns",
    "summarizer_called", "summarizer_mode", "summarizer_tokens_in", "summarizer_tokens_out", "summarizer_cost_usd",
    "summarizer_latency_ms", "llm_latency_ms",
)
DERIVED_COLUMNS: tuple[str, ...] = (
    "client_latency_ms", "cumulative_cost_usd", "estimation_chars", "title_matches_project",
    "attachment_markers_expected", "attachment_markers_recalled", "attachment_recall",
)
METRIC_COLUMNS: tuple[str, ...] = (
    "latency_budget_pass", "cost_budget_pass", "memory_drift_score", "facts_expected", "facts_recalled",
    "facts_missing", "facts_found_in", "stale_retained",
)
COLUMNS: tuple[str, ...] = ("scenario", "attachment_kb", "repeat", *TURN_OBSERVED_COLUMNS, *EXTRA_COLUMNS, *DERIVED_COLUMNS, *METRIC_COLUMNS)


class StressRun:
    def __init__(self, client: httpx.AsyncClient, latency_budget_ms: int, cost_budget_usd: float, max_turns: int | None):
        self.client = client
        self.latency_metric = LatencyBudgetMetric(latency_budget_ms)
        self.cost_metric = CostBudgetMetric(cost_budget_usd)
        self.max_turns = max_turns

    async def _post(self, path: str, **kwargs) -> dict:
        response = await self.client.post(path, **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(f"POST {path} -> HTTP {response.status_code}: {response.text[:400]}")
        return response.json()

    async def conversation(self, scenario: Scenario, size_kb: int, repeat: int, pdf: bytes | None) -> list[dict]:
        session_id = (await self._post(f"{API}/sessions"))["session_id"]
        rows: list[dict] = []
        cumulative_cost = 0.0
        turns = scenario.turns[: self.max_turns] if self.max_turns else scenario.turns
        try:
            for turn in turns:
                data = {"transcript": turn.transcript, **{k: str(v) for k, v in turn.params.items()}}
                files = None
                if pdf is not None and turn.turn_index == 1:
                    files = [("attachments", (pdf_path(size_kb).name, pdf, "application/pdf"))]
                started = time.perf_counter()
                body = await self._post(f"{API}/sessions/{session_id}/estimate", data=data, files=files)
                client_latency_ms = int((time.perf_counter() - started) * 1000)

                observed = body.get("turn_observed") or {}
                state = (await self.client.get(f"{API}/sessions/{session_id}")).json()
                if state.get("last_turn_observed", {}).get("turn_index") != observed.get("turn_index"):
                    raise RuntimeError("GET /sessions/{id}.last_turn_observed no coincide con el turn_observed de la respuesta")
                snapshot = SessionSnapshot.from_api(state)

                # Métricas de presupuesto sobre el evento; deriva de memoria sobre el snapshot
                latency = self.latency_metric.evaluate(observed)
                cost = self.cost_metric.evaluate(observed)
                facts = scenario.facts_expected_at(turn.turn_index)
                recall, recalled, missing = recall_of_facts(snapshot, facts)
                found_in = []
                for _, fact in facts:
                    places = [w for w in ("summary", "anchors", "metadata") if contains(snapshot.text_of(w), fact)]
                    if places:
                        found_in.append(f"{fact}@{'+'.join(places)}")
                stale = [fact for fact in scenario.retired_before(turn.turn_index) if any(contains(snapshot.text_of(w), fact) for w in ("summary", "anchors", "metadata"))]

                # Recall del contenido del adjunto en la estimación (solo el turno que lo lleva)
                markers_recalled = [m for m in MARKER_TERMS if contains(body["estimation"], m)] if files else []
                attachment_recall = round(len(markers_recalled) / len(MARKER_TERMS), 4) if files else ""

                cumulative_cost = round(cumulative_cost + float(observed.get("cost_usd") or 0.0), 6)
                title = body["estimation"].splitlines()[0] if body["estimation"] else ""
                row = {
                    "scenario": scenario.key,
                    "attachment_kb": size_kb,
                    "repeat": repeat,
                    **{column: observed.get(column, "") for column in (*TURN_OBSERVED_COLUMNS, *EXTRA_COLUMNS)},
                    "client_latency_ms": client_latency_ms,
                    "cumulative_cost_usd": cumulative_cost,
                    "estimation_chars": len(body["estimation"]),
                    "title_matches_project": contains(title, scenario.project_name),
                    "attachment_markers_expected": len(MARKER_TERMS) if files else 0,
                    "attachment_markers_recalled": len(markers_recalled),
                    "attachment_recall": attachment_recall,
                    "latency_budget_pass": latency.passed,
                    "cost_budget_pass": cost.passed,
                    "memory_drift_score": "" if recall is None else recall,
                    "facts_expected": len(facts),
                    "facts_recalled": len(recalled),
                    "facts_missing": "|".join(missing),
                    "facts_found_in": "|".join(found_in),
                    "stale_retained": "|".join(stale),
                }
                rows.append(row)
                print(
                    f"  {scenario.key:<13} kb={size_kb:<3} r={repeat} t={turn.turn_index:>2}  {observed.get('latency_ms', 0):>6} ms  "
                    f"{observed.get('tokens_in', 0):>6} tok  {float(observed.get('cost_usd') or 0):.6f} USD  "
                    f"cache={observed.get('cache_hit_kind', '?'):<8} tier={observed.get('last_resolved_tier', '?'):<8} "
                    f"anchors={observed.get('anchors_count', 0):<2} summary={observed.get('summary_chars', 0):<5} "
                    f"recall={'-' if recall is None else f'{recall:.2f}'}"
                    + (f"  attach_recall={attachment_recall}" if files else ""),
                    flush=True,
                )
        finally:
            await self.client.delete(f"{API}/sessions/{session_id}")
        return rows


def make_client(http: str | None, fake: bool) -> httpx.AsyncClient:
    if http:
        return httpx.AsyncClient(base_url=http.rstrip("/"), timeout=300.0)
    from app.main import app

    if fake:
        install_fake_llm()
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://in-process", timeout=300.0)


def install_fake_llm() -> None:
    """LLM simulado (el doble de los tests) con la plantilla v4, summarizer por LLM y caché activa: prueba el runner sin gastar."""
    sys.path.insert(0, str(ROOT))
    from app.config import get_settings
    from app.main import app
    from app.services import llm_service
    from tests.conftest import make_settings
    from tests.fakes import FakeEstimatorLLM

    llm_service.openai.AsyncOpenAI = FakeEstimatorLLM()  # type: ignore[assignment]
    app.dependency_overrides[get_settings] = lambda: make_settings(
        openai_api_key="sk-fake", prompt_version="v4", summary_mode="llm", response_cache_enabled=True
    )


async def service_meta(client: httpx.AsyncClient) -> dict:
    try:
        health = (await client.get("/health")).json()
    except Exception:  # noqa: BLE001
        health = {}
    return {k: health.get(k) for k in ("provider", "model", "prompt_template", "session_max_turns", "llm_configured")}


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stress del CAG: multi-turno, adjuntos grandes y ráfagas (repeticiones)")
    parser.add_argument("--http", help="URL del servicio (p. ej. http://localhost:8000). Sin ella, in-process (lee .env).")
    parser.add_argument("--scenarios", default="growing,pivot,contradiction")
    parser.add_argument("--attachment-sizes", default="0,5,20,50,100", help="KB de texto del adjunto del turno 1; 0 = sin adjunto")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--turns", type=int, default=None, help="Tope de turnos por escenario (por defecto, todos: 20)")
    parser.add_argument("--latency-budget-ms", type=int, default=4000)
    parser.add_argument("--cost-budget-usd", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--fake", action="store_true", help="LLM simulado in-process: valida el runner sin API key (no es un baseline)")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # la consola de Windows usa cp1252; las transcripciones llevan acentos
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # una línea por petición taparía el progreso

    scenarios = get_scenarios([s.strip() for s in args.scenarios.split(",") if s.strip()])
    sizes = [int(s) for s in args.attachment_sizes.split(",") if s.strip()]
    ensure_pdfs(tuple(s for s in sizes if s > 0))
    pdfs = {size: (pdf_path(size).read_bytes() if size > 0 else None) for size in sizes}

    started_at = datetime.now(timezone.utc)
    rows: list[dict] = []
    async with make_client(args.http, args.fake) as client:
        meta = await service_meta(client)
        print(f"Servicio: {meta}  ·  escenarios={[s.key for s in scenarios]}  tamaños={sizes} KB  repeats={args.repeats}"
              f"  budgets: {args.latency_budget_ms} ms / {args.cost_budget_usd} USD" + ("  [LLM SIMULADO]" if args.fake else ""))
        run = StressRun(client, args.latency_budget_ms, args.cost_budget_usd, args.turns)
        for scenario in scenarios:
            for size in sizes:
                for repeat in range(1, args.repeats + 1):
                    print(f"\n== {scenario.name} ({scenario.key}) · adjunto {size} KB · repetición {repeat}/{args.repeats} ==")
                    rows.extend(await run.conversation(scenario, size, repeat, pdfs[size]))
                    write_csv(args.output, rows)  # se vuelca tras cada conversación: un fallo a mitad no pierde lo medido

    write_csv(args.output, rows)
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "service": meta,
        "fake_llm": args.fake,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "rows": len(rows),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(rows)} filas en {args.output} (meta en {meta_path.name})")
    if not args.no_report:
        report_module.write_report(rows, args.report, meta=json.loads(meta_path.read_text(encoding="utf-8")))
        print(f"Reporte en {args.report}")
    return 0


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in COLUMNS})


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
