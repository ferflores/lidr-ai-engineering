from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers import estimations
from app.services.cited_estimate import (
    Estimate, EstimateLineItem, Passage, SourceReference, checked_response, verify_citations,
)

PASSAGE = Passage(chunk_id="BUD-1::SAP", document_id="BUD-1",
                  text="Component: SAP confirmations\nEstimated hours: 60", historical_hours=60)


def line(**changes):
    fields = {"component": "Post production to SAP", "grounded": True, "hours": 60,
              "rationale": "The historical component covers the same integration.",
              "sources": [SourceReference(chunk_id=PASSAGE.chunk_id, document_id=PASSAGE.document_id,
                                          evidence="Estimated hours: 60")]}
    return EstimateLineItem(**{**fields, **changes})


def test_cited_and_unbudgeted_lines():
    estimate = Estimate(items=[line(), line(component="Sensor installation", hours=None,
                                            grounded=False, sources=[], rationale="No comparable work.")])
    result = checked_response(estimate, [PASSAGE], "test", "unit-request")
    assert [c.status for c in result.citation_report] == ["grounded", "insufficient_context"]
    assert result.subtotal_hours == 60 and not result.complete


@pytest.mark.parametrize(("change", "status"), [
    ({"chunk_id": "BUD-404::SAP"}, "dangling_citation"),
    ({"document_id": "BUD-2"}, "invalid_evidence"),
    ({"evidence": "Estimated hours: 120"}, "invalid_evidence"),
])
def test_invalid_citation_is_rejected(change, status):
    item = line()
    item.sources[0] = item.sources[0].model_copy(update=change)
    result = checked_response(Estimate(items=[item]), [PASSAGE], "test", "unit-request")
    assert result.citation_report[0].status == status
    assert result.structured.items[0].hours is None and not result.structured.items[0].sources
    assert not result.complete and result.subtotal_hours == 0


@pytest.mark.parametrize("item", [line(sources=[]), line(hours=90), line(grounded=False)])
def test_grounding_rules(item):
    assert verify_citations(Estimate(items=[item]), [PASSAGE])[0].status == "invalid_evidence"


def test_existing_estimation_endpoint_returns_citations(monkeypatch):
    result = checked_response(Estimate(items=[line()]), [PASSAGE], "test", "unit-request")
    monkeypatch.setattr(estimations, "generate_cited_estimate", AsyncMock(return_value=(result, [PASSAGE])))
    response = TestClient(app).post("/api/v1/estimate", json={
        "transcription": "We need to post daily production figures to SAP.",
        "response_format": "cited",
    })
    assert response.status_code == 200
    assert response.json()["request_id"] == "unit-request"
    assert response.json()["structured"]["items"][0]["sources"][0]["chunk_id"] == PASSAGE.chunk_id
