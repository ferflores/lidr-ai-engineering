#!/usr/bin/env python
"""Cinco consultas semánticas contra `POST /search`, formateadas para leerlas en terminal (sesión 08).

    docker compose run --rm api python query_examples.py
    uv run python query_examples.py                      # con la API arrancada en API_BASE_URL
    uv run python query_examples.py --k 5 --output output_examples.txt

Sustituye al `scripts/compare.py` de la sesión 07, que medía la similitud entre dos textos sueltos en memoria.
Ahora la medida se hace donde importa: contra el corpus ya persistido en PostgreSQL, por el mismo camino que
usaría una aplicación real (HTTP → embedding de la consulta → `ORDER BY embedding <=> query` → top-k).

Las cinco consultas no son cinco ejemplos cualesquiera: cada una comprueba una cosa distinta del
recuperador, y el bloque "Qué observar" dice qué habría que ver si el pipeline está bien. Eso es lo que
convierte el output en una medida y no en una demo."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.clients.estimator_api import DEFAULT_BASE_URL  # noqa: E402
from app.config import get_settings  # noqa: E402  (carga .env)

CONTENT_PREVIEW_CHARS = 120


@dataclass(frozen=True)
class ExampleQuery:
    """Una consulta con el ángulo que ejercita y lo que debería verse en el resultado."""

    label: str
    angle: str
    text: str
    expectation: str


# Las cinco del enunciado, con el texto literal que propone.
EXAMPLE_QUERIES: tuple[ExampleQuery, ...] = (
    ExampleQuery(
        label="1/5",
        angle="Componente directo conocido",
        text="REST API development with JWT authentication for financial sector",
        expectation=(
            "Sanity check: el corpus tiene un 'OAuth 2.0 authentication backend' de un proyecto de banca. "
            "Debería salir arriba y con la menor distancia de las cinco consultas."
        ),
    ),
    ExampleQuery(
        label="2/5",
        angle="Reformulación semántica",
        text="secure backend service with token-based access control for banking applications",
        expectation=(
            "Misma idea, vocabulario distinto (ni 'JWT' ni 'OAuth' ni 'REST'). Si recupera los mismos chunks "
            "que la consulta 1, los embeddings capturan significado y no coincidencia de palabras."
        ),
    ),
    ExampleQuery(
        label="3/5",
        angle="Dominio distinto",
        text="mobile application for restaurant reservations",
        expectation=(
            "No hay restaurantes en el corpus. Lo esperable son distancias claramente mayores: el top-k "
            "siempre devuelve k filas, así que la señal de 'no hay nada' es la distancia, no una lista vacía."
        ),
    ),
    ExampleQuery(
        label="4/5",
        angle="Consulta ambigua",
        text="integration with external system",
        expectation=(
            "Genérica: muchos chunks encajan a medias (pasarelas de pago, ERPs, conectores bancarios). "
            "Lo interesante es que las cinco distancias estén muy juntas: no hay un ganador claro."
        ),
    ),
    ExampleQuery(
        label="5/5",
        angle="Consulta muy específica",
        text="migration from monolith to microservices architecture using Kubernetes",
        expectation=(
            "Vocabulario técnico preciso. Mide si distingue entre tecnologías vecinas o si se conforma con "
            "recuperar cualquier cosa que suene a infraestructura."
        ),
    ),
)


def preview(content: str, chars: int = CONTENT_PREVIEW_CHARS) -> str:
    """Una línea: el contenido del chunk sin saltos, recortado."""
    flat = " ".join(content.split())
    return flat if len(flat) <= chars else flat[: chars - 1] + "…"


def identify(result: dict) -> str:
    """De qué componente y de qué presupuesto es el chunk, según su metadata.

    Hace falta porque el texto embebido empieza por la cabecera de contexto del presupuesto padre (decisión
    del chunker de la sesión 07), así que los primeros 120 caracteres de dos chunks del mismo presupuesto son
    idénticos. La metadata es lo que distingue uno de otro de un vistazo."""
    meta = result.get("metadata") or {}
    name = meta.get("component_name") or meta.get("chunk_ref") or "?"
    sector, year = meta.get("client_sector", "?"), meta.get("year", "?")
    hours, complexity = meta.get("estimated_hours", "?"), meta.get("complexity", "?")
    return f"{name}  ({sector}, {year}, {hours} h, {complexity})"


def render_query(example: ExampleQuery, body: dict) -> str:
    """Bloque de texto de una consulta: cabecera, qué observar, y el top-k (dos líneas por resultado)."""
    results = body["results"]
    lines = [
        "─" * 110,
        f"[{example.label}] {example.angle}",
        f'  Query:     "{example.text}"',
        f"  k={body['k']}  ·  {len(results)} resultado(s)  ·  {body['search_time_ms']} ms (embedding de la consulta + SQL)",
        f"  Qué observar: {example.expectation}",
        "",
        f"  Cada resultado: chunk_id · distancia coseno · chunk_type · componente (sector, año, horas, complejidad),",
        f"  y debajo los primeros {CONTENT_PREVIEW_CHARS} caracteres del content tal cual se embebió.",
        "",
    ]
    for position, result in enumerate(results, start=1):
        lines.append(
            f"  {position:>2}. #{result['chunk_id']:<5} d={result['distance']:.4f}  "
            f"{result['chunk_type']}  ·  {identify(result)}"
        )
        lines.append(f"      {preview(result['content'])}")
    if not results:
        lines.append("  (sin resultados: ¿está el corpus ingestado? `uv run python scripts/ingest_corpus.py`)")
    else:
        distances = [r["distance"] for r in results]
        lines.append("")
        lines.append(f"  rango de distancias: {min(distances):.4f} … {max(distances):.4f}  (margen {max(distances) - min(distances):.4f})")
    return "\n".join(lines)


def render_summary(measured: list[tuple[ExampleQuery, dict]]) -> str:
    """Tabla comparativa de las cinco consultas. Se calcula del resultado real, no se escribe a mano.

    La comparación entre consultas es lo que hace legible el experimento: la distancia coseno no tiene una
    escala absoluta con la que decir "0,41 es bueno". Lo que informa es que la consulta directa y su
    reformulación caigan en 0,41 y la de otro dominio en 0,53."""
    lines = [
        "",
        "LECTURA COMPARADA",
        "",
        f"  {'consulta':<28}  {'mejor':>7}  {'peor':>7}  {'margen':>7}  {'ms':>5}  primer resultado",
        f"  {'─' * 28}  {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 5}  {'─' * 46}",
    ]
    for example, body in measured:
        results = body["results"]
        if not results:
            lines.append(f"  {example.angle:<28}  {'—':>7}  {'—':>7}  {'—':>7}  {body['search_time_ms']:>5}  (sin resultados)")
            continue
        distances = [r["distance"] for r in results]
        meta = results[0].get("metadata") or {}
        lines.append(
            f"  {example.angle:<28}  {min(distances):>7.4f}  {max(distances):>7.4f}  "
            f"{max(distances) - min(distances):>7.4f}  {body['search_time_ms']:>5}  "
            f"{(meta.get('component_name') or '?')[:46]}"
        )
    return "\n".join(lines)


def run(base_url: str, k: int, timeout: float) -> tuple[str, int]:
    """Lanza las cinco consultas y devuelve el informe completo y el número de fallos."""
    blocks = [
        "Búsqueda semántica sobre los presupuestos históricos (PostgreSQL + pgvector, distancia coseno)",
        f"API: {base_url}  ·  k={k}  ·  sin índice vectorial (sequential scan)",
        "La distancia coseno va de 0 (idéntico) a 2 (opuesto); para embeddings de texto, menos es más cercano.",
        "",
    ]
    failures = 0
    measured: list[tuple[ExampleQuery, dict]] = []
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        for example in EXAMPLE_QUERIES:
            response = client.post("/search", json={"query": example.text, "k": k})
            if response.status_code != 200:
                failures += 1
                blocks.append("─" * 110)
                blocks.append(f"[{example.label}] {example.angle}\n  ❌ HTTP {response.status_code}: {response.text[:300]}")
                continue
            body = response.json()
            measured.append((example, body))
            blocks.append(render_query(example, body))
    blocks.append("─" * 110)
    if measured:
        blocks.append(render_summary(measured))
        blocks.append("")
    return "\n".join(blocks) + "\n", failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run five representative semantic queries against POST /search.")
    parser.add_argument("--api-base-url", default=None, help="API base URL (default: API_BASE_URL from the environment).")
    parser.add_argument("--k", type=int, default=5, help="Results per query (default: 5).")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds.")
    parser.add_argument("--output", type=Path, default=None, help="Also write the report to this file.")
    args = parser.parse_args(argv)

    get_settings()  # carga el .env
    base_url = (args.api_base_url or os.environ.get("API_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")

    report, failures = run(base_url, args.k, args.timeout)
    print(report, end="")
    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"\nEscrito en {args.output}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
