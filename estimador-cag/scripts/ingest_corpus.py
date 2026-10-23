#!/usr/bin/env python
"""Ingesta el corpus de presupuestos del curso llamando a `POST /embeddings/ingest`, un documento por presupuesto.

    uv run python scripts/ingest_corpus.py                      # 15 presupuestos de data/budgets_sample.json
    docker compose run --rm api python scripts/ingest_corpus.py
    uv run python scripts/ingest_corpus.py --reset               # borra documents/chunks antes de ingestar

El contrato del endpoint es **un documento por petición**, así que el corpus se recorre presupuesto a
presupuesto. El `source_path` identifica cada documento de forma única y honesta: el corpus del curso viene
en un único fichero, así que la ruta es `data/budgets_sample.json#<budget_id>` en vez de inventar 15 ficheros
con los mismos datos. Un segundo pase sobre el mismo corpus devuelve 409 en todos: el script lo cuenta como
"ya ingestado", no como error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clients.estimator_api import DEFAULT_BASE_URL  # noqa: E402
from app.config import get_settings  # noqa: E402  (carga .env)

DEFAULT_CORPUS = Path("data/budgets_sample.json")
DOCUMENT_TYPE = "historical_budget"


def source_path_for(corpus: Path, budget: dict) -> str:
    """Identificador estable del documento dentro del corpus."""
    return f"{corpus.as_posix()}#{budget['budget_id']}"


def reset_database(database_url: str) -> int:
    """Vacía `documents` y, en cascada, `chunks`. Devuelve los documentos que había.

    `TRUNCATE ... RESTART IDENTITY CASCADE` en vez de `DELETE`: reinicia las secuencias de `id`, así que una
    ingesta desde cero siempre produce los mismos `document_id` y `chunk_id`. Eso es lo que hace comparable
    `output_examples.txt` entre ejecuciones y entre máquinas. `CASCADE` aquí es el de TRUNCATE (alcanza a
    `chunks` por la foreign key), el mismo efecto que el `ON DELETE CASCADE` del esquema.
    """
    import asyncio

    from sqlalchemy import func, select, text

    from app.db.models import Document
    from app.db.session import get_sessionmaker

    async def _reset() -> int:
        async with get_sessionmaker(database_url)() as session:
            before = await session.scalar(select(func.count()).select_from(Document)) or 0
            await session.execute(text("TRUNCATE TABLE documents RESTART IDENTITY CASCADE"))
            await session.commit()
            return before

    return asyncio.run(_reset())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest the course budget corpus through POST /embeddings/ingest.")
    parser.add_argument("corpus", nargs="?", type=Path, default=DEFAULT_CORPUS, help=f"Budget JSON array (default: {DEFAULT_CORPUS}).")
    parser.add_argument("--api-base-url", default=None, help="API base URL (default: API_BASE_URL from .env).")
    parser.add_argument("--reset", action="store_true", help="Delete every document (chunks cascade) before ingesting.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds.")
    args = parser.parse_args(argv)

    settings = get_settings()
    # Misma convención que el resto del proyecto: API_BASE_URL del entorno (o del .env), con localhost
    # como valor por defecto. Dentro de compose el servicio `chat` ya lo recibe apuntando a http://api:8000.
    base_url = (args.api_base_url or os.environ.get("API_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")

    if args.reset:
        deleted = reset_database(settings.database_url)
        print(f"reset: {deleted} documento(s) borrado(s) (sus chunks caen en cascada)\n")

    budgets = json.loads(args.corpus.read_text(encoding="utf-8"))
    if not isinstance(budgets, list):
        print(f"error: {args.corpus} debe contener una lista de presupuestos", file=sys.stderr)
        return 2

    created = skipped = failed = 0
    total_chunks = 0
    print(f"Ingestando {len(budgets)} presupuesto(s) de {args.corpus} en {base_url}\n")
    with httpx.Client(base_url=base_url, timeout=args.timeout) as client:
        for budget in budgets:
            source_path = source_path_for(args.corpus, budget)
            payload = {"source_path": source_path, "document_type": DOCUMENT_TYPE, "content": budget}
            response = client.post("/embeddings/ingest", json=payload)
            if response.status_code == 200:
                body = response.json()
                created += 1
                total_chunks += body["chunks_created"]
                print(
                    f"  ✅ {budget['budget_id']:<14} document_id={body['document_id']:<4} "
                    f"chunks={body['chunks_created']:<3} dim={body['embedding_dimension']} "
                    f"{body['ingestion_time_ms']} ms"
                )
            elif response.status_code == 409:
                skipped += 1
                print(f"  ⏭  {budget['budget_id']:<14} ya ingestado (document_id={response.json()['document_id']})")
            else:
                failed += 1
                print(f"  ❌ {budget['budget_id']:<14} HTTP {response.status_code}: {response.text[:200]}", file=sys.stderr)

    print(f"\n{created} ingestado(s) · {skipped} ya estaba(n) · {failed} fallo(s) · {total_chunks} chunks nuevos")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
