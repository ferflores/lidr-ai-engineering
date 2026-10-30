#!/usr/bin/env python
"""Trace manual de una transcripción a través del sistema tal como quedó al cierre de la sesión 08.

Es un **script de cliente** para el diagnóstico de la sesión 09: no añade etapas al pipeline ni toca el
endpoint `POST /search`. Reutiliza tres piezas existentes tal cual:

- `JSONStructuralChunker.count_tokens` para contar tokens con el tokenizador del modelo de embeddings,
- `OpenAIEmbedder.embed_one` para vectorizar texto (la misma llamada que hace `POST /search`),
- `search_chunks` del repositorio para resolver el top-k por distancia coseno (la misma consulta SQL).

Lo que hace y por qué pasa por aquí en vez de por curl: el servicio **no expone** ningún endpoint que
vectorice texto libre (`/embeddings/ingest` solo acepta un presupuesto con el esquema `Budget`), y `POST
/search` limita la consulta a 2.000 caracteres, así que la transcripción completa (8.405) no cabe. Este
script hace esas dos cosas por debajo del HTTP para poder **observar** lo que el endpoint no deja observar.

Ocho pasos: vector de la transcripción completa, top-k con ese vector, dispersión contra todo el corpus,
lo que cabe por HTTP (2.000 caracteres), la intención escrita a mano como control, la ablación señal/ruido,
el efecto del idioma, los chunks más genéricos del corpus y el solape entre rankings.

Uso:
    uv run python examples/trace_02_ambiguous.py                       # trace completo
    uv run python examples/trace_02_ambiguous.py --transcript examples/transcripts/01_clear.txt

Requiere `OPENAI_API_KEY` en `.env` y el Postgres del proyecto levantado con el corpus ingestado
(`docker compose up -d postgres api && docker compose run --rm api alembic upgrade head &&
uv run python scripts/ingest_corpus.py`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
from pathlib import Path

from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402  (carga .env)
from app.db.models import Chunk as DbChunk  # noqa: E402
from app.db.session import get_sessionmaker  # noqa: E402
from app.embedding_pipeline.chunker import JSONStructuralChunker  # noqa: E402
from app.embedding_pipeline.embedder import OpenAIEmbedder  # noqa: E402
from app.embedding_pipeline.repository import search_chunks  # noqa: E402
from app.embedding_pipeline.similarity import cosine_similarity  # noqa: E402

DEFAULT_TRANSCRIPT = Path("examples/transcripts/02_ambiguous.txt")
HTTP_QUERY_MAX_LENGTH = 2000  # el `max_length` de `SearchRequest.query` (app/embedding_pipeline/schemas.py)
TOTAL_CHUNKS_IN_CORPUS = 200  # k alto para traer el corpus entero y medir la dispersión de distancias

# Lo que el cliente pide de verdad, extraído a mano de las dos frases útiles de 02_ambiguous.txt.
# Es el control del trace: si esta consulta recupera mejor que la transcripción completa, el problema está
# en la query, no en el corpus.
INTENT_QUERY = (
    "B2B web portal where workshop customers log in, see their own tier price based on purchase volume, "
    "place orders that land in the ERP, and request a quote for parts without a list price"
)

# La misma intención, en el idioma en el que habla el cliente. Aísla el efecto del idioma: el corpus está
# en inglés y las transcripciones en español.
INTENT_QUERY_ES = (
    "Portal B2B donde el taller entra con su usuario, ve su tarifa segun el volumen que compra, "
    "hace el pedido y este entra en el ERP, y pide presupuesto de las piezas que no tienen tarifa"
)

# Las tres frases de 02_ambiguous.txt que de verdad dicen qué quiere construir el cliente. Se localizan por
# subcadena para que la ablación siga funcionando si la transcripción se reformatea.
SIGNAL_MARKERS = (
    "Lo que de verdad necesitamos",
    "Y que lo que pida entre en el Sage",
    "Hay mucho recambio que no",
)


def rule(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def vector_report(label: str, vector: list[float], text: str, chunker: JSONStructuralChunker) -> None:
    norm = math.sqrt(sum(x * x for x in vector))
    print(
        json.dumps(
            {
                "label": label,
                "input_characters": len(text),
                "input_tokens": chunker.count_tokens(text),
                "dimensions": len(vector),
                "first_component": vector[0],
                "last_component": vector[-1],
                "first_3": vector[:3],
                "last_3": vector[-3:],
                "l2_norm": round(norm, 6),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def results_report(results, show_content_chars: int = 0) -> None:
    payload = []
    for rank, item in enumerate(results, start=1):
        meta = item.metadata
        row = {
            "rank": rank,
            "chunk_id": item.chunk_id,
            "document_id": item.document_id,
            "distance": round(item.distance, 6),
            "cosine_similarity": round(1 - item.distance, 6),
            "chunk_type": item.chunk_type,
            "budget_id": meta.get("budget_id"),
            "component_id": meta.get("component_id"),
            "component_name": meta.get("component_name"),
            "client_sector": meta.get("client_sector"),
            "main_technology": meta.get("main_technology"),
            "year": meta.get("year"),
            "estimated_hours": meta.get("estimated_hours"),
            "complexity": meta.get("complexity"),
        }
        if show_content_chars:
            row["content_excerpt"] = item.content[:show_content_chars].replace("\n", " | ")
        payload.append(row)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def spread_report(results) -> None:
    distances = [r.distance for r in results]
    print(
        json.dumps(
            {
                "chunks_scored": len(distances),
                "min_distance": round(min(distances), 6),
                "max_distance": round(max(distances), 6),
                "mean_distance": round(statistics.fmean(distances), 6),
                "stdev_distance": round(statistics.stdev(distances), 6),
                "range_max_minus_min": round(max(distances) - min(distances), 6),
                "top5_minus_top1": round(distances[4] - distances[0], 6),
                "top1_vs_worst_gap": round(max(distances) - min(distances), 6),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manual trace of a meeting transcript through the S08 system.")
    parser.add_argument("--transcript", type=Path, default=DEFAULT_TRANSCRIPT)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)

    settings = get_settings()
    api_key = (settings.openai_api_key or "").strip()
    if not api_key:
        print("error: falta OPENAI_API_KEY en .env", file=sys.stderr)
        return 1

    transcript = args.transcript.read_text(encoding="utf-8")
    truncated = transcript[:HTTP_QUERY_MAX_LENGTH]

    chunker = JSONStructuralChunker()
    embedder = OpenAIEmbedder(api_key=api_key, model=settings.embedding_model)
    sessionmaker = get_sessionmaker(settings.database_url)

    rule(f"PASO 1 — Vector de la transcripción completa ({args.transcript})")
    full_vector = embedder.embed_one(transcript)
    vector_report("full_transcript", full_vector, transcript, chunker)

    rule(f"PASO 2 — Top-{args.k} con el vector de la transcripción completa (cosine_distance, SQL)")
    async with sessionmaker() as session:
        top_full = await search_chunks(session, full_vector, args.k)
        results_report(top_full, show_content_chars=160)

        rule("PASO 2b — Dispersión de distancias de ese vector contra TODO el corpus")
        all_full = await search_chunks(session, full_vector, TOTAL_CHUNKS_IN_CORPUS)
        spread_report(all_full)

        rule(f"PASO 3 — Lo que SÍ cabe por HTTP: primeros {HTTP_QUERY_MAX_LENGTH} caracteres de la transcripción")
        truncated_vector = embedder.embed_one(truncated)
        vector_report("truncated_2000_chars", truncated_vector, truncated, chunker)
        top_truncated = await search_chunks(session, truncated_vector, args.k)
        results_report(top_truncated)

        rule("PASO 4 — Control: la intención del cliente escrita a mano (lo que el sistema debería haber buscado)")
        print(json.dumps({"intent_query": INTENT_QUERY}, indent=2, ensure_ascii=False))
        intent_vector = embedder.embed_one(INTENT_QUERY)
        vector_report("intent_query", intent_vector, INTENT_QUERY, chunker)
        top_intent = await search_chunks(session, intent_vector, args.k)
        results_report(top_intent)
        all_intent = await search_chunks(session, intent_vector, TOTAL_CHUNKS_IN_CORPUS)
        print("\n-- dispersión de la consulta de intención contra todo el corpus --")
        spread_report(all_intent)

        rule("PASO 5 — Ablación: la señal y el ruido de la transcripción, por separado")
        signal_lines = [line for line in transcript.splitlines() if any(m in line for m in SIGNAL_MARKERS)]
        noise_lines = [line for line in transcript.splitlines() if line not in signal_lines]
        print(json.dumps({"signal_lines": signal_lines}, indent=2, ensure_ascii=False))
        top_signal = top_noise = []
        # Los marcadores son de 02_ambiguous.txt: con otra transcripción no hay nada que ablacionar.
        pairs = (
            (("SOLO SEÑAL", "\n".join(signal_lines)), ("SOLO RUIDO", "\n".join(noise_lines)))
            if signal_lines
            else ()
        )
        if not pairs:
            print("(sin frases marcadas como señal en esta transcripción: ablación omitida)")
        for label, text in pairs:
            vector = embedder.embed_one(text)
            print(f"\n-- {label} --")
            vector_report(label.lower().replace(" ", "_"), vector, text, chunker)
            results = await search_chunks(session, vector, args.k)
            results_report(results)
            if label == "SOLO SEÑAL":
                top_signal = results
            else:
                top_noise = results

        rule("PASO 6 — Efecto del idioma: la misma intención en español y en inglés")
        for label, query in (("intent_es", INTENT_QUERY_ES), ("intent_en", INTENT_QUERY)):
            vector = embedder.embed_one(query)
            print(f"\n-- {label}: {query}")
            results_report(await search_chunks(session, vector, args.k))

        rule("PASO 7 — Chunks genéricos: los más cercanos al centroide del corpus")
        # El centroide es el "chunk medio" del corpus. Lo que queda cerca de él se parece un poco a todo, así
        # que tiende a colarse en cualquier ranking: es el candidato natural a falso positivo recurrente.
        rows = (await session.execute(select(DbChunk.id, DbChunk.chunk_metadata, DbChunk.embedding))).all()
        vectors = [[float(x) for x in row.embedding] for row in rows]
        centroid = [sum(v[i] for v in vectors) / len(vectors) for i in range(len(vectors[0]))]
        ranked = sorted(
            ((1 - cosine_similarity(v, centroid), row) for v, row in zip(vectors, rows)), key=lambda t: t[0]
        )
        print(
            json.dumps(
                {
                    "chunks": len(vectors),
                    "most_generic": [
                        {
                            "distance_to_centroid": round(d, 6),
                            "chunk_id": row.id,
                            "budget_id": row.chunk_metadata.get("budget_id"),
                            "component_id": row.chunk_metadata.get("component_id"),
                            "component_name": row.chunk_metadata.get("component_name"),
                            "client_sector": row.chunk_metadata.get("client_sector"),
                        }
                        for d, row in ranked[:5]
                    ],
                    "most_specific": [
                        {
                            "distance_to_centroid": round(d, 6),
                            "chunk_id": row.id,
                            "budget_id": row.chunk_metadata.get("budget_id"),
                            "component_id": row.chunk_metadata.get("component_id"),
                            "component_name": row.chunk_metadata.get("component_name"),
                        }
                        for d, row in ranked[-3:]
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        rule("PASO 8 — Solape entre los rankings")
        full_ids = [r.chunk_id for r in top_full]
        intent_ids = [r.chunk_id for r in top_intent]
        truncated_ids = [r.chunk_id for r in top_truncated]
        signal_ids = [r.chunk_id for r in top_signal]
        noise_ids = [r.chunk_id for r in top_noise]
        print(
            json.dumps(
                {
                    "top_k": args.k,
                    "full_transcript_chunk_ids": full_ids,
                    "truncated_chunk_ids": truncated_ids,
                    "intent_query_chunk_ids": intent_ids,
                    "signal_only_chunk_ids": signal_ids,
                    "noise_only_chunk_ids": noise_ids,
                    "full_vs_intent_overlap": sorted(set(full_ids) & set(intent_ids)),
                    "full_vs_truncated_overlap": sorted(set(full_ids) & set(truncated_ids)),
                    "full_vs_noise_overlap": sorted(set(full_ids) & set(noise_ids)),
                    "full_vs_signal_overlap": sorted(set(full_ids) & set(signal_ids)),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
