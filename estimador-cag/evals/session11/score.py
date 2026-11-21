# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["ragas==0.4.4", "langchain-community==0.3.31", "openai==3.3.0", "instructor==1.17.0", "socksio==1.0.0"]
# ///
"""Score saved service outputs with the four Ragas metrics, independently of the app environment."""

import argparse
import asyncio
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import mean

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

HERE = Path(__file__).resolve().parent
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
JUDGE = "gpt-4.1-mini-2025-04-14"
EMBEDDINGS = "text-embedding-3-small"


def write_table(report, path):
    labels = ("Faithfulness", "Answer relevancy", "Context precision", "Context recall")
    lines = ["# Evaluación Ragas · sesión 11", "",
             "| Consulta | " + " | ".join(labels) + " |",
             "|---|---:|---:|---:|---:|"]
    for row in [*report["rows"], {"id": "**Media**", **report["mean"]}]:
        values = " | ".join(f"{row[metric]:.3f}" for metric in METRICS)
        lines.append(f"| {row['id']} | {values} |")
    lines.extend(["", f"Juez `{report['judge']}`. Embeddings `{report['embedding_model']}`. "
                  f"Ragas {report['ragas']}.",
                  "", "[Puntuaciones sin redondear](scores.json) · [Respuestas y fragmentos](samples.json)",
                  "", "Las medias dan el mismo peso a cada consulta."])
    path.write_text("\n".join(lines) + "\n")


async def score(args):
    data = args.samples.read_bytes()
    samples = json.loads(data)
    if not samples or len({s["id"] for s in samples}) != len(samples):
        raise ValueError("Expected a nonempty sample file with unique query IDs.")
    rows = []
    async with AsyncOpenAI() as client:
        judge = llm_factory(args.judge, client=client, temperature=0, max_tokens=8192)
        embeddings = OpenAIEmbeddings(model=EMBEDDINGS, client=client)
        fidelity, relevance = Faithfulness(llm=judge), AnswerRelevancy(llm=judge, embeddings=embeddings)
        precision, recall = ContextPrecision(llm=judge), ContextRecall(llm=judge)
        for sample in samples:
            question, answer = sample["question"], sample["answer"]
            contexts, reference = sample["contexts"], sample["ground_truth"]
            results = await asyncio.gather(
                fidelity.ascore(user_input=question, response=answer, retrieved_contexts=contexts),
                relevance.ascore(user_input=question, response=answer),
                precision.ascore(user_input=question, reference=reference, retrieved_contexts=contexts),
                recall.ascore(user_input=question, reference=reference, retrieved_contexts=contexts),
            )
            row = {name: float(result.value) for name, result in zip(METRICS, results, strict=True)}
            if not all(math.isfinite(value) for value in row.values()):
                raise ValueError(f"Non-finite score on {sample['id']}: {row}")
            rows.append({"id": sample["id"], **row})
            print(sample["id"], row, flush=True)
    report = {
        "completed_at": datetime.now(timezone.utc).isoformat(), "ragas": version("ragas"),
        "judge": args.judge, "embedding_model": EMBEDDINGS, "generator_models": sorted({s["result"]["model"] for s in samples}),
        "sample_sha256": hashlib.sha256(data).hexdigest(), "rows": rows,
        "mean": {m: mean(row[m] for row in rows) for m in METRICS},
    }
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_table(report, args.table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=HERE / "samples.json")
    parser.add_argument("--out", type=Path, default=HERE / "scores.json")
    parser.add_argument("--table", type=Path, default=HERE / "baseline.md")
    parser.add_argument("--judge", default=JUDGE)
    asyncio.run(score(parser.parse_args()))
