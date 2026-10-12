"""Runner de los casos golden: ejecuta cada caso contra el servicio y evalúa las tres métricas.

    uv run python -m evals.run                                  # in-process (TestClient), necesita .env con API key
    uv run python -m evals.run --http http://localhost:8000     # contra un servicio arrancado
    uv run python -m evals.run --cases landing-hubspot,api-pagos-stripe --output evals/results.json
    uv run python -m evals.run --fake                           # sin LLM: el doble de tests/fakes.py (humo del runner)

Modos:
- `actor` (por defecto): una llamada por caso a `POST /api/v1/estimate`.
- `acb`: Actor-Critic-Boss. No está implementado en este repositorio (era opcional en la sesión 05); el
  runner lo rechaza con un mensaje claro en vez de fingir que lo ejecuta.

Salida: una tabla por caso (métricas pasadas/falladas, coste, latencia), el resumen por métrica y, si se
pide, un JSON con todo. Devuelve código 1 si algún caso falla alguna métrica: así sirve en CI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from evals.metrics import EvalObservation, MetricResult, default_metrics, run_all_metrics, summarize

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path(__file__).resolve().parent / "golden_dataset.json"


@dataclass(frozen=True)
class GoldenCase:
    id: str
    transcript: str
    params: dict
    expected_terms: tuple[str, ...]
    max_cost_usd: float | None

    @property
    def detail_level(self) -> str:
        return str(self.params.get("detail_level", "completa"))


def load_cases(path: Path = GOLDEN, only: set[str] | None = None) -> list[GoldenCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = [
        GoldenCase(
            id=case["id"],
            transcript=case["transcript"],
            params=dict(case.get("params", {})),
            expected_terms=tuple(case.get("expected", {}).get("terms", [])),
            max_cost_usd=case.get("expected", {}).get("max_cost_usd", data.get("default_max_cost_usd")),
        )
        for case in data["cases"]
    ]
    if only:
        unknown = only - {c.id for c in cases}
        if unknown:
            raise SystemExit(f"Casos desconocidos: {', '.join(sorted(unknown))}")
        cases = [c for c in cases if c.id in only]
    return cases


def make_client(http: str | None, fake: bool) -> httpx.AsyncClient:
    """httpx contra un servicio real (`--http`) o contra la app ASGI en el mismo proceso."""
    if http:
        return httpx.AsyncClient(base_url=http.rstrip("/"), timeout=180.0)
    from app.main import app  # import tardío: solo hace falta en modo in-process

    if fake:
        install_fake_llm()
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://in-process", timeout=180.0)


def install_fake_llm() -> None:
    """Sustituye el cliente de OpenAI por el doble de los tests y fuerza una API key ficticia."""
    sys.path.insert(0, str(ROOT))
    from app.config import get_settings
    from app.main import app
    from app.services import llm_service
    from tests.conftest import make_settings
    from tests.fakes import FakeEstimatorLLM

    llm_service.openai.AsyncOpenAI = FakeEstimatorLLM()  # type: ignore[assignment]
    app.dependency_overrides[get_settings] = lambda: make_settings(openai_api_key="sk-fake", prompt_version="v4")


async def run_case(client: httpx.AsyncClient, case: GoldenCase) -> tuple[EvalObservation, dict]:
    payload = {"transcription": case.transcript, **case.params}
    started = time.perf_counter()
    response = await client.post("/api/v1/estimate", json=payload)
    latency_ms = int((time.perf_counter() - started) * 1000)
    if response.status_code != 200:
        raise RuntimeError(f"[{case.id}] HTTP {response.status_code}: {response.text[:300]}")
    body = response.json()
    usage = body.get("usage") or {}
    cost = (usage.get("cost") or {}).get("total_usd")
    observation = EvalObservation(
        case_id=case.id,
        estimation=body["estimation"],
        detail_level=case.detail_level,
        cost_usd=cost,
        tokens_in=usage.get("input_tokens"),
        tokens_out=usage.get("output_tokens"),
        latency_ms=latency_ms,
        expected_terms=case.expected_terms,
        max_cost_usd=case.max_cost_usd,
    )
    return observation, body


def _utf8_stdout() -> None:
    """La consola de Windows usa cp1252 por defecto; los casos golden llevan acentos."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.getLogger("httpx").setLevel(logging.WARNING)


def format_row(observation: EvalObservation, results: list[MetricResult]) -> str:
    marks = " ".join(f"{r.name}={'OK' if r.passed else 'FAIL'}({r.score:.2f})" for r in results)
    cost = f"{observation.cost_usd:.6f}" if observation.cost_usd is not None else "?"
    return f"{observation.case_id:<28} {cost:>9} USD {observation.latency_ms or 0:>6} ms  {marks}"


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evals golden del estimador")
    parser.add_argument("--http", help="URL del servicio (p. ej. http://localhost:8000). Sin ella, in-process.")
    parser.add_argument("--mode", choices=["actor", "acb"], default="actor")
    parser.add_argument("--cases", help="ids separados por comas (por defecto, todos)")
    parser.add_argument("--max-cost-usd", type=float, default=None, help="Tope global de coste si el caso no declara uno")
    parser.add_argument("--recall-threshold", type=float, default=1.0)
    parser.add_argument("--output", type=Path, help="JSON con observaciones y resultados")
    parser.add_argument("--fake", action="store_true", help="LLM simulado (in-process): prueba el runner sin API key")
    args = parser.parse_args(argv)
    _utf8_stdout()

    if args.mode == "acb":
        parser.error("el modo acb (Actor-Critic-Boss) no está implementado en este repositorio; usa --mode actor")

    cases = load_cases(only=set(args.cases.split(",")) if args.cases else None)
    metrics = default_metrics(max_cost_usd=args.max_cost_usd if args.max_cost_usd is not None else 0.01, recall_threshold=args.recall_threshold)
    report: list[dict] = []
    all_results: list[list[MetricResult]] = []

    async with make_client(args.http, args.fake) as client:
        for case in cases:
            observation, body = await run_case(client, case)
            results = run_all_metrics(observation, metrics)
            all_results.append(results)
            print(format_row(observation, results))
            report.append({"case": case.id, "observation": asdict(observation), "results": [r.as_dict() for r in results], "model": body.get("model")})

    totals = summarize(all_results)
    print()
    for name, stats in totals.items():
        print(f"{name:<20} pass_rate={stats['pass_rate']:.2%}  mean_score={stats['mean_score']:.3f}  ({stats['cases']} casos)")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"summary": totals, "cases": report}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nResultados en {args.output}")
    failed = sum(1 for results in all_results if not all(r.passed for r in results))
    print(f"\n{len(cases) - failed}/{len(cases)} casos pasan todas las métricas")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
