# 🔭 LLM Observability Platform

> **End-to-End Observability for AI Workloads** — Built by a Principal SRE  
> Integrates OpenTelemetry, Python APIs, and Grafana to deliver deep insights into LLM performance, latency, token usage, and reliability.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-1.x-orange.svg)](https://opentelemetry.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/your-username/llm-observability/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/llm-observability/actions)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Use Cases](#use-cases)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Dashboards](#dashboards)
- [Prompting Claude for SRE Tasks](#prompting-claude-for-sre-tasks)
- [Contributing](#contributing)

---

## Overview

This platform was developed over **6 months** to solve a real production problem: AI workloads (LLM APIs, embedding pipelines, RAG systems) are black boxes. Standard APM tools don't understand tokens, model latency distributions, or hallucination-adjacent quality metrics.

**This repository solves that.** It instruments any Python-based LLM application with:

- **Distributed traces** across every LLM call, embedding, retrieval, and response
- **Custom metrics** for token throughput, model latency percentiles, cost estimation
- **Structured logs** with trace correlation for root cause analysis
- **Alerting rules** for SLO violations (p99 latency, error rate, token budget)

### Key Achievements
- Reduced MTTD (Mean Time to Detect) for LLM degradation from **hours → minutes**
- Surfaced 3 silent token budget overruns causing downstream failures
- Enabled per-model, per-team cost attribution in multi-tenant environments

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LLM Application Layer                        │
│   ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐   │
│   │  FastAPI App │  │  RAG Pipeline│  │  Batch Inference Jobs  │   │
│   └──────┬───────┘  └──────┬───────┘  └──────────┬─────────────┘   │
│          │                 │                      │                 │
│          └────────────────►│◄─────────────────────┘                 │
│                     ┌──────▼──────────────┐                         │
│                     │  OTel SDK (Python)  │  ← Auto + Manual Instr  │
│                     │  - Tracer           │                         │
│                     │  - MeterProvider    │                         │
│                     │  - LoggerProvider   │                         │
│                     └──────┬──────────────┘                         │
└────────────────────────────┼────────────────────────────────────────┘
                             │ OTLP gRPC/HTTP
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    OpenTelemetry Collector                           │
│   ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐   │
│   │  Receivers  │  │  Processors  │  │      Exporters         │   │
│   │  - OTLP     │  │  - Batch     │  │  - Prometheus          │   │
│   │  - Prometheus│  │  - Filter   │  │  - Jaeger/Tempo        │   │
│   │  - Filelog  │  │  - Enrich   │  │  - Loki (logs)         │   │
│   └─────────────┘  └──────────────┘  └────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
            ┌────────────────┼──────────────────┐
            ▼                ▼                  ▼
     ┌─────────────┐  ┌────────────┐   ┌──────────────┐
     │  Prometheus │  │   Tempo    │   │     Loki     │
     │  (Metrics)  │  │  (Traces)  │   │    (Logs)    │
     └──────┬──────┘  └─────┬──────┘   └──────┬───────┘
            └───────────────┼──────────────────┘
                            ▼
                    ┌───────────────┐
                    │    Grafana    │
                    │  Dashboards   │
                    │  + Alerting   │
                    └───────────────┘
```

### Component Responsibilities

| Component | Role |
|-----------|------|
| **OTel SDK (Python)** | Instruments LLM calls with traces, metrics, and logs |
| **LLMInstrumentor** | Custom auto-instrumentor for OpenAI, Anthropic, LangChain |
| **OTel Collector** | Receives, processes, and routes telemetry data |
| **Prometheus** | Stores and queries metrics (token rates, latency, errors) |
| **Grafana Tempo** | Distributed tracing backend with trace-to-metrics correlation |
| **Loki** | Log aggregation with trace ID correlation |
| **Grafana** | Unified dashboards, alerting, and on-call routing |

---

## Use Cases

### 1. LLM Latency SLO Monitoring
Track p50/p95/p99 response latency per model and alert when SLOs are breached.

### 2. Token Budget Management
Monitor token consumption per team/service and alert before budget overruns.

### 3. Error Rate & Retry Analysis
Detect silent retries, rate limit errors, and model fallback events.

### 4. Cost Attribution
Estimate per-request cost using token counts and model pricing tables.

### 5. RAG Pipeline Observability
Trace full retrieval-augmented generation pipelines: query → retrieve → rerank → generate.

### 6. Batch Job Monitoring
Track throughput, stall detection, and completion rates for offline inference jobs.

---

## Project Structure

```
llm-observability/
├── src/
│   ├── collectors/
│   │   ├── __init__.py
│   │   ├── llm_instrumentor.py      # Core OTel instrumentation
│   │   ├── anthropic_collector.py   # Anthropic API wrapper
│   │   ├── openai_collector.py      # OpenAI API wrapper
│   │   └── langchain_collector.py   # LangChain callback handler
│   ├── exporters/
│   │   ├── __init__.py
│   │   ├── prometheus_exporter.py   # Custom metrics definitions
│   │   └── cost_estimator.py        # Token cost calculator
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI demo app
│   │   └── routes.py                # Instrumented route handlers
│   └── dashboards/
│       └── grafana_llm_overview.json  # Grafana dashboard JSON
├── tests/
│   ├── test_instrumentor.py
│   ├── test_cost_estimator.py
│   └── test_api.py
├── configs/
│   ├── otel-collector.yaml          # OTel Collector config
│   ├── prometheus.yaml              # Prometheus scrape config
│   └── alerting_rules.yaml         # SLO alerting rules
├── k8s/
│   ├── deployment.yaml
│   ├── otel-collector.yaml
│   └── grafana-configmap.yaml
├── scripts/
│   ├── setup.sh
│   └── load_test.py                 # Generates sample telemetry
├── .github/
│   └── workflows/
│       └── ci.yml
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Git

### 1. Clone & Install

```bash
git clone https://github.com/your-username/llm-observability.git
cd llm-observability

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
cp .env.example .env
# Edit .env with your API keys
export ANTHROPIC_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=llm-observability-demo
```

### 3. Start the Observability Stack

```bash
docker-compose up -d
```

This starts:
- OTel Collector on `:4317` (gRPC) and `:4318` (HTTP)
- Prometheus on `:9090`
- Grafana on `:3000` (admin/admin)
- Jaeger on `:16686`
- Loki on `:3100`

### 4. Run the Demo App

```bash
python -m uvicorn src.api.main:app --reload --port 8000
```

### 5. Generate Test Traffic

```bash
python scripts/load_test.py
```

### 6. View Dashboards

Open Grafana at http://localhost:3000 → Dashboards → **LLM Observability Overview**

---

## Configuration

### OTel SDK Configuration (Python)

```python
from src.collectors.llm_instrumentor import LLMObservabilitySDK

sdk = LLMObservabilitySDK(
    service_name="my-llm-app",
    service_version="1.0.0",
    environment="production",
    otlp_endpoint="http://otel-collector:4317",
    enable_cost_tracking=True,
    token_budget_alert_threshold=0.8,  # Alert at 80% of budget
)
sdk.initialize()
```

### Instrument Your LLM Calls

```python
from src.collectors.anthropic_collector import InstrumentedAnthropic

client = InstrumentedAnthropic()  # Drop-in replacement

# All calls are automatically traced
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain SRE principles"}]
)
```

---

## Prompting Claude for SRE Tasks

A key part of this project was learning to craft effective prompts for Claude when working on complex SRE tasks. Here is the proven framework:

### The Principal SRE Prompt Formula

```
ROLE:     "Act as a Principal SRE with expertise in [specific domain]."
CONTEXT:  "We are running [system description] at [scale]."
TASK:     "I need [specific deliverable] that includes [components]."
FORMAT:   "Provide: 1) Architecture diagram (ASCII), 2) Step-by-step implementation,
           3) Production-ready code with error handling, 4) Runbook."
CONSTRAINTS: "Must use [tech stack]. Must handle [edge cases]. 
              Include unit tests. Follow [standards]."
```

### Example: Requesting Observability Code

```
Act as a Principal SRE specializing in observability and LLM systems.

Context: I'm building observability for a Python FastAPI service that calls 
Anthropic and OpenAI APIs. We process ~10k requests/day with SLOs of p99 < 2s 
and error rate < 0.1%.

Task: Create a complete OpenTelemetry instrumentation module that:
1. Auto-instruments all LLM API calls (traces + metrics + logs)
2. Captures: model name, token counts, latency, error type
3. Calculates estimated cost per request
4. Exports to OTel Collector via OTLP/gRPC
5. Includes Prometheus metrics for Grafana dashboards

Constraints:
- Python 3.11, opentelemetry-sdk 1.x, anthropic 0.x, openai 1.x
- Thread-safe, async-compatible
- Must not increase p99 latency by more than 5ms
- Include pytest tests with mocked LLM responses

Format: Complete, runnable Python modules with type hints and docstrings.
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs welcome!

---

## License

MIT — see [LICENSE](LICENSE)
