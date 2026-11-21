"""Expose the agent as a single transcript-to-estimate operation."""

from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.loop import AgentRunError, run_agent
from app.agent.retrieval import budget_search
from app.config import Settings, get_settings
from app.db.session import get_db_session
from app.embedding_pipeline.retrieval import RerankerUnavailable
from app.embedding_pipeline.router import get_embedder, get_reranker

router = APIRouter(tags=["Agent estimates"])


class EstimateRequest(BaseModel):
    transcript: str = Field(min_length=10, max_length=100_000, pattern=r"\S")
    hourly_rate_eur: Decimal = Field(default=Decimal("50"), gt=0, le=1000, decimal_places=2)


@router.post("/agent/estimate")
async def estimate(request: EstimateRequest, settings: Settings = Depends(get_settings),
                   session: AsyncSession = Depends(get_db_session)):
    if not settings.openai_api_key:
        raise HTTPException(503, "The OpenAI API key is missing.")
    search = budget_search(session, settings, get_embedder(settings), get_reranker(settings))
    try:
        async with AsyncOpenAI(api_key=settings.openai_api_key) as client:
            return await run_agent(client, request.transcript, search, hourly_rate=request.hourly_rate_eur)
    except (SQLAlchemyError, RerankerUnavailable) as exc:
        raise HTTPException(503, "Historical budget retrieval is unavailable.") from exc
    except OpenAIError as exc:
        raise HTTPException(502, "The OpenAI API could not complete the estimate.") from exc
    except AgentRunError as exc:
        raise HTTPException(502, str(exc)) from exc
