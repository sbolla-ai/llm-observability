"""
main.py
=======
FastAPI demo application showing end-to-end LLM observability in action.

Principal SRE Note:
    This app demonstrates the "instrumented from day one" pattern.
    Every endpoint is fully traced with:
    - HTTP request spans (via opentelemetry-instrumentation-fastapi)
    - LLM call spans (via InstrumentedAnthropic)
    - Business metrics (tokens, cost, latency)
    - Structured logs correlated to traces

Run:
    uvicorn src.api.main:app --reload --port 8000

Endpoints:
    GET  /health          → Health check
    GET  /metrics         → Prometheus metrics
    POST /chat            → Single-turn LLM chat
    POST /rag/query       → RAG pipeline demo
    GET  /cost/estimate   → Cost estimation utility
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from prometheus_client import make_asgi_app

from src.api.routes import router
from src.collectors.llm_instrumentor import LLMObservabilitySDK

logger = logging.getLogger(__name__)

# Global SDK instance
sdk = LLMObservabilitySDK(
    service_name=os.getenv("OTEL_SERVICE_NAME", "llm-observability-demo"),
    service_version=os.getenv("SERVICE_VERSION", "1.0.0"),
    environment=os.getenv("ENVIRONMENT", "development"),
    otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
    enable_cost_tracking=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize SDK and instrumentation on startup."""
    sdk.initialize()

    # Auto-instrument HTTP outbound calls
    RequestsInstrumentor().instrument()

    logger.info("LLM Observability Demo App started")
    yield
    logger.info("LLM Observability Demo App shutting down")


app = FastAPI(
    title="LLM Observability Demo",
    description="End-to-end observability for AI workloads using OpenTelemetry",
    version="1.0.0",
    lifespan=lifespan,
)

# Auto-instrument FastAPI (adds spans for every HTTP request)
FastAPIInstrumentor.instrument_app(app)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Include API routes
app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "llm-observability-demo"}


@app.get("/")
async def root():
    return {
        "service": "LLM Observability Demo",
        "docs": "/docs",
        "metrics": "/metrics",
        "health": "/health",
    }
