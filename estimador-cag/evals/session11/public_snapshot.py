"""Replay the published S10 hybrid + reranking top five without accessing a database."""

import argparse
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path

from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.schemas import Budget
from app.services.cited_estimate import Passage, generate_from_passages
from evals.session11.collect import as_sample

HERE = Path(__file__).resolve().parent
PUBLIC_INPUTS = {
    "corpus": (HERE.parents[1] / "data" / "budgets_sample.json", "c86464b67743557521535f68493f43252fb7bd778c186a93a127334d2e807a89"),
    "ranking": (HERE.parent / "retrieval" / "REPORT.md", "3ca01e0aaf3b27053e7af8f47dae3e5fbcfea544efe459df9d230ac36fb384bc"),
    "golden": (HERE.parent / "retrieval" / "golden_set.json", "43e1d4c9e2c77ace0aaab5648b3a2fbb1172d381aa70563df7be20cff1c50b77"),
}


def read_public_file(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Unexpected contents for the published course fixture {path.name}")
    return data.decode()


def load_cases(corpus, ranking, golden):
    by_name = {}
    for raw in json.loads(corpus):
        budget = Budget.model_validate(raw)
        for component in budget.components:
            key = (budget.budget_id, component.name)
            if key in by_name:
                raise ValueError(f"Ambiguous component: {key}")
            by_name[key] = Passage(chunk_id=f"{budget.budget_id}::{component.component_id}",
                                   document_id=budget.budget_id, historical_hours=component.estimated_hours,
                                   text=JSONStructuralChunker.render_text(budget, component))
    cases = []
    for case in json.loads(golden)["queries"]:
        section = ranking.split(f"### {case['id']}\n", 1)[1].split("\n##", 1)[0]
        top_five = section.split("**D · Híbrida, reranking sí**", 1)[1].strip().split("\n\n")[1]
        names = re.findall(r"^\d+\. [✓✗] (.*?) \((BUD-[\d-]+)\) d=", top_five, re.M)
        if len(names) != 5:
            raise ValueError(f"Expected five ranked components for {case['id']}")
        cases.append((case, [by_name[document, name] for name, document in names]))
    return cases


async def replay(args):
    inputs = {name: read_public_file(getattr(args, name), digest)
              for name, (_, digest) in PUBLIC_INPUTS.items()}
    cases = load_cases(**inputs)
    samples = []
    for case, passages in cases:
        result = await generate_from_passages(case["query"], passages, args.model, os.environ["OPENAI_API_KEY"])
        samples.append(as_sample(case, result, passages, "published_session10_config_D"))
        print(f"{case['id']}: {len(result.structured.items)} lines", flush=True)
    args.out.write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name, (path, _) in PUBLIC_INPUTS.items():
        parser.add_argument(f"--{name}", type=Path, default=path)
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--out", type=Path, default=HERE / "samples.json")
    asyncio.run(replay(parser.parse_args()))
