"""Demonstrate the ID check on a recorded live estimate and one deliberately corrupted copy."""

import json
from pathlib import Path

from app.services.cited_estimate import Estimate, Passage, verify_citations

HERE = Path(__file__).resolve().parent


def build_example(samples):
    sample = next(s for s in samples if any(i["grounded"] for i in s["result"]["structured"]["items"]))
    result = sample["result"]
    original = Estimate.model_validate(result["structured"])
    passages = []
    for encoded in sample["contexts"]:
        raw = json.loads(encoded)
        amount = raw["text"].split("Estimated hours: ")[-1]
        passages.append(Passage(**raw, historical_hours=float(amount)))
    changed = original.model_copy(deep=True)
    line = next(line for line in changed.items if line.grounded)
    previous_id = line.sources[0].chunk_id
    line.sources[0].chunk_id = "INTENTIONAL-DANGLING-CHUNK"
    return {
        "query": sample["id"], "request_id": result["request_id"],
        "live_citation_report": result["citation_report"],
        "controlled_test": {
            "component": line.component, "original_id": previous_id, "injected_id": line.sources[0].chunk_id,
            "report_after_injection": [c.model_dump() for c in verify_citations(changed, passages)],
        },
    }


if __name__ == "__main__":
    samples = json.loads((HERE / "samples.json").read_text())
    print(json.dumps(build_example(samples), ensure_ascii=False, indent=2))
