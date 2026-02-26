"""
routes.py
=========
Instrumented FastAPI route handlers demonstrating observability patterns.
"""

import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.collectors.anthropic_collector import InstrumentedAnthropic
from src.exporters.cost_estimator import CostEstimator

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Request / Response Models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024
    system_prompt: Optional[str] = "You are a helpful assistant."


class ChatResponse(BaseModel):
    response: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    finish_reason: str


class RAGRequest(BaseModel):
    query: str
    top_k: int = 3
    model: str = "claude-sonnet-4-6"


class CostEstimateRequest(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int


# ─── Route Handlers ───────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Single-turn LLM chat endpoint.

    Demonstrates:
    - Instrumented Anthropic client
    - Token usage extraction
    - Cost estimation per request
    - Structured logging with trace correlation
    """
    client = InstrumentedAnthropic()

    try:
        response = client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            system=request.system_prompt,
            messages=[{"role": "user", "content": request.message}],
        )

        content = response.content[0].text if response.content else ""
        cost = CostEstimator.estimate(
            response.model,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

        logger.info(
            "Chat request completed",
            extra={
                "model": response.model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cost_usd": cost,
            }
        )

        return ChatResponse(
            response=content,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            estimated_cost_usd=cost,
            finish_reason=response.stop_reason or "unknown",
        )

    except Exception as exc:
        logger.error("Chat request failed", extra={"error": str(exc)}, exc_info=True)
        raise HTTPException(status_code=502, detail=f"LLM API error: {str(exc)}")


@router.post("/rag/query")
async def rag_query(request: RAGRequest):
    """
    RAG pipeline demo endpoint.

    Demonstrates multi-stage trace spans:
    1. Retrieval stage
    2. Reranking stage
    3. Generation stage

    The full pipeline is visible as a single distributed trace in Jaeger/Tempo.
    """
    from opentelemetry import trace
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("rag_pipeline") as pipeline_span:
        pipeline_span.set_attribute("rag.query", request.query[:200])
        pipeline_span.set_attribute("rag.top_k", request.top_k)

        # Stage 1: Retrieval (mocked for demo)
        with tracer.start_as_current_span("retrieval") as retrieval_span:
            retrieval_span.set_attribute("llm.pipeline.stage", "retrieval")
            # In production: vector DB query here
            mock_docs = [
                {"id": f"doc_{i}", "content": f"Relevant content {i} for: {request.query}", "score": 0.9 - i * 0.1}
                for i in range(request.top_k)
            ]
            retrieval_span.set_attribute("retrieval.documents_returned", len(mock_docs))
            retrieval_span.set_attribute("retrieval.top_score", mock_docs[0]["score"])

        # Stage 2: Context assembly
        context = "\n\n".join([doc["content"] for doc in mock_docs])

        # Stage 3: Generation
        with tracer.start_as_current_span("generation") as gen_span:
            gen_span.set_attribute("llm.pipeline.stage", "generation")
            client = InstrumentedAnthropic()
            augmented_prompt = f"Context:\n{context}\n\nQuestion: {request.query}"

            response = client.messages.create(
                model=request.model,
                max_tokens=512,
                system="Answer the question based only on the provided context. Be concise.",
                messages=[{"role": "user", "content": augmented_prompt}],
            )

            gen_span.set_attribute(
                "gen_ai.usage.input_tokens", response.usage.input_tokens
            )
            gen_span.set_attribute(
                "gen_ai.usage.output_tokens", response.usage.output_tokens
            )

        answer = response.content[0].text if response.content else ""

        return {
            "query": request.query,
            "answer": answer,
            "sources": [{"id": doc["id"], "score": doc["score"]} for doc in mock_docs],
            "model": response.model,
            "token_usage": {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            },
        }


@router.post("/cost/estimate")
async def estimate_cost(request: CostEstimateRequest):
    """Estimate cost for a given model and token counts."""
    cost = CostEstimator.estimate(request.model, request.input_tokens, request.output_tokens)
    return {
        "model": request.model,
        "input_tokens": request.input_tokens,
        "output_tokens": request.output_tokens,
        "estimated_cost_usd": cost,
    }


@router.get("/cost/monthly-projection")
async def monthly_projection(
    model: str = "claude-sonnet-4-6",
    daily_requests: int = 1000,
    avg_input_tokens: int = 500,
    avg_output_tokens: int = 200,
):
    """Project monthly cost for capacity planning."""
    monthly = CostEstimator.monthly_budget_usd(
        daily_requests, avg_input_tokens, avg_output_tokens, model
    )
    return {
        "model": model,
        "daily_requests": daily_requests,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "projected_monthly_cost_usd": monthly,
    }
