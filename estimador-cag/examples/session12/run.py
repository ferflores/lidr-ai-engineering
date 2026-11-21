"""Run the project agent on an official course transcript and save its returned trace."""

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from app.agent.loop import run_agent
from app.agent.retrieval import budget_search
from app.config import get_settings
from app.db.models import Chunk
from app.db.session import get_sessionmaker
from app.embedding_pipeline.router import get_embedder, get_reranker
from reference_retrieval import search_budgets_stub


async def reference_search(args):
    filters = {"sectors": [args.filters.sector]} if args.filters and args.filters.sector else None
    return [{"id": item["budget_id"], "scope": item["content_preview"], "sector": item["sector"],
             "estimated_hours": int(item["estimated_hours"])} for item in search_budgets_stub(args.query, filters)]


async def run(args):
    settings = get_settings()
    if not settings.openai_api_key:
        raise SystemExit("Set OPENAI_API_KEY in the environment or in the project's .env file.")
    transcript = args.transcript.read_bytes()
    async with get_sessionmaker(settings.database_url)() as session, AsyncOpenAI(
            api_key=settings.openai_api_key) as client:
        backend = args.retrieval
        if backend == "auto":
            try:
                exists = await asyncio.wait_for(session.scalar(select(Chunk.id).limit(1)), timeout=4)
            except (OSError, TimeoutError, SQLAlchemyError):
                exists = None
            backend = "hybrid" if exists is not None else "reference"
        print(f"Model: {args.model}; retrieval: {backend}", file=sys.stderr)
        search = reference_search if backend == "reference" else budget_search(
            session, settings, get_embedder(settings), get_reranker(settings))
        result = await run_agent(client, transcript.decode("utf-8"), search, model=args.model)
    report = {"execution": "live_openai_responses", "retrieval": backend,
              "input": args.transcript.name, "input_sha256": hashlib.sha256(transcript).hexdigest(),
              "timestamp_utc": datetime.now(timezone.utc).isoformat(), **result}
    output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
        print(f"Trace saved to {args.output}", file=sys.stderr)
    else:
        print(output, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path, nargs="?", default=HERE / "sample_transcript_complex.txt")
    parser.add_argument("--model", choices=["gpt-5", "gpt-5-mini"], default="gpt-5")
    parser.add_argument("--retrieval", choices=["hybrid", "reference", "auto"], default="hybrid",
                        help="Auto uses the course reference dataset when the project database is empty or offline.")
    parser.add_argument("--output", type=Path)
    asyncio.run(run(parser.parse_args()))
