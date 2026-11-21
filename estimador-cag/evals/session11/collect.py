"""Collect the production generator's answer, displayed evidence and reference for each S10 case."""

import argparse
import asyncio
import json
from pathlib import Path

from app.config import get_settings
from app.schemas.estimation import EstimationRequest
from app.services.cited_estimate import generate_cited_estimate

HERE = Path(__file__).resolve().parent
GOLDEN = HERE.parent / "retrieval" / "golden_set.json"


async def collect(output):
    settings = get_settings()
    golden = json.loads(GOLDEN.read_text())
    samples = []
    for case in golden["queries"]:
        payload = EstimationRequest(transcription=case["query"], response_format="cited")
        result, passages = await generate_cited_estimate(payload, settings)
        answer = "\n".join(
            f"{line.component}: {str(line.hours) + ' hours' if line.grounded else 'insufficient evidence'}. "
            f"{line.rationale}" for line in result.structured.items)
        samples.append({
            "id": case["id"], "question": case["query"], "answer": answer,
            "contexts": [json.dumps(p.model_dump(exclude={"historical_hours"}), ensure_ascii=False) for p in passages],
            "ground_truth": case["ground_truth"], "result": result.model_dump(),
        })
        print(f"{case['id']}: {len(passages)} passages, {len(result.structured.items)} lines.", flush=True)
    output.write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "samples.json")
    asyncio.run(collect(parser.parse_args().out))
