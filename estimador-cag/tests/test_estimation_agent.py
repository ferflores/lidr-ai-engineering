"""Exercise the real Responses SDK and agent with a scripted HTTP transport and course reference data."""

import asyncio
import hashlib
import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from app.agent.loop import AgentRunError, run_agent
from app.agent.tools import CalculateEstimate, calculate_estimate
from app.main import app

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "session12"
sys.path.insert(0, str(EXAMPLE))
from run import reference_search

COMPONENTS = [
    ("Logistics backend", "business backend API orders route assignment tracking",
     "Find comparable order-management and shipment-tracking services."),
    ("SAP integration", "SAP ERP integration IDoc middleware retries",
     "Find SAP synchronization and middleware work."),
    ("Driver mobile app", "mobile app Android iOS offline proof of delivery",
     "Find mobile delivery projects with offline workflows."),
    ("Analytics dashboard", "analytics dashboard KPI reporting trends",
     "Find dashboard and reporting projects comparable to the requested KPIs."),
]


def call(name, call_id, arguments):
    return {"type": "function_call", "id": f"fc_{call_id}", "call_id": call_id,
            "name": name, "arguments": json.dumps(arguments), "status": "completed"}


def fixture_transport(request):
    body = json.loads(request.content)
    assert request.url.path == "/v1/responses"
    assert body["model"] == "gpt-5" and body["reasoning"]["effort"] == "medium"
    assert all(t["strict"] and t["type"] == "function" for t in body["tools"])
    previous = body.get("previous_response_id")
    if previous is None:
        assert body["input"][0]["content"] == (EXAMPLE / "sample_transcript_complex.txt").read_text()
        output = [call("search_budgets", f"search_{n}", {"component": name, "query": query,
                       "filters": None, "purpose": purpose}) for n, (name, query, purpose) in enumerate(COMPONENTS)]
        rid = "fixture_search"
    elif previous == "fixture_search":
        components = []
        for n, (name, _, _) in enumerate(COMPONENTS):
            item = body["input"][n]
            assert item["type"] == "function_call_output" and item["call_id"] == f"search_{n}"
            matches = json.loads(item["output"])["matches"][:2]
            components.append({"name": name, "reference_ids": [r["id"] for r in matches],
                               "reference_amounts": [r["estimated_hours"] for r in matches]})
        output = [call("calculate_estimate", "calculate_all", {"components": components,
                       "purpose": "Price the four scoped components using their matched references."})]
        rid = "fixture_calculation"
    else:
        assert previous == "fixture_calculation"
        assert body["input"][0]["call_id"] == "calculate_all"
        assert json.loads(body["input"][0]["output"])["complete"]
        output = [{"type": "message", "role": "assistant", "id": "fixture_message", "status": "completed",
                   "content": [{"type": "output_text", "text": "The four components have been priced.",
                                "annotations": []}]}]
        rid = "fixture_finish"
    return httpx.Response(200, json={"id": rid, "model": "scripted-test-double", "object": "response",
                          "status": "completed", "created_at": 0, "output": output,
                          "parallel_tool_calls": True, "tool_choice": "auto", "tools": []})


async def offline_run(max_rounds=12):
    transcript = (EXAMPLE / "sample_transcript_complex.txt").read_bytes()
    http = httpx.AsyncClient(transport=httpx.MockTransport(fixture_transport))
    async with AsyncOpenAI(api_key="offline-test", base_url="https://fixture.invalid/v1",
                           http_client=http) as client:
        result = await run_agent(client, transcript.decode(), reference_search, max_rounds=max_rounds)
    return {"execution": "offline_integration_test", "retrieval": "course_reference_stub",
            "decision_source": "Scripted HTTP responses in tests/test_estimation_agent.py. No live LLM call.",
            "input": "sample_transcript_complex.txt", "input_sha256": hashlib.sha256(transcript).hexdigest(),
            **result}


def test_complex_run():
    result = asyncio.run(offline_run())
    assert [s["action"] for s in result["trace"]] == ["search_budgets"] * 4 + ["calculate_estimate", "finish"]
    assert result["estimate"]["estimated_hours"] == 3496.1
    assert result["estimate"]["cost_eur"] == 174805
    assert all(r["reference_ids"] for r in result["estimate"]["components"])


def test_round_limit():
    with pytest.raises(AgentRunError, match="limit of 1"):
        asyncio.run(offline_run(max_rounds=1))


def test_missing_or_misattributed_evidence():
    from decimal import Decimal

    args = CalculateEstimate.model_validate({"components": [{"name": "ERP", "reference_ids": [],
                                             "reference_amounts": []}], "purpose": "Price ERP."})
    result = calculate_estimate(args, {"ERP": {}}, Decimal("50"))
    assert result["components"][0]["unbudgeted"] and not result["complete"]
    args.components[0].reference_ids = ["wrong-component"]
    args.components[0].reference_amounts = [100]
    with pytest.raises(ValueError, match="do not match"):
        calculate_estimate(args, {"ERP": {}, "other": {"wrong-component": 100}}, Decimal("50"))


def test_api_entrypoint_rejects_blank_transcript():
    assert TestClient(app).post("/api/v1/agent/estimate", json={"transcript": " " * 30}).status_code == 422


if __name__ == "__main__":
    print(json.dumps(asyncio.run(offline_run()), ensure_ascii=False, indent=2))
