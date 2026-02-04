# Agentic Observability Platform
## Production-Grade Multi-Agent System for Anomaly Detection & Root Cause Analysis

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Enabled-green.svg)](https://fastapi.tiangolo.com/)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-Enabled-orange.svg)](https://opentelemetry.io/)

---

## Canonical Documentation

Read this first:

- **[MASTER_GUIDE.md](MASTER_GUIDE.md)** — single source of truth (entrypoints, pipeline I/O, intermediate outputs, storage, observability)

Supporting docs:

- [QUICKSTART.md](QUICKSTART.md)
- [API_TESTING_GUIDE.md](API_TESTING_GUIDE.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Overview

Enterprise-grade agentic observability platform leveraging multi-agent orchestration for real-time anomaly detection and automated root cause analysis. Built to demonstrate advanced ML engineering capabilities.

### Key Features

- **Multi-Agent Architecture**: Coordinated agents for detection, memory, reasoning, and action
- **Production Patterns**: Circuit breakers, exponential backoff, comprehensive observability
- **Async-First Design**: Full async/await for high-throughput processing
- **Enterprise Integrations**: OpenTelemetry; TimescaleDB/Qdrant via Docker Compose; optional LLM integration
- **RAG-Enhanced Analysis**: Context-aware anomaly investigation with vector memory

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    API Gateway (FastAPI)                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  Agent Coordinator                           │
│         (Circuit Breaker, Retry Logic, Metrics)             │
└─────┬────────┬────────┬────────┬──────────────────────────┘
      │        │        │        │
┌─────▼──┐ ┌──▼─────┐ ┌▼──────┐ ┌▼────────┐
│Anomaly │ │  RAG   │ │Reason │ │ Action  │
│Detect  │ │ Memory │ │ Agent │ │ Agent   │
└────┬───┘ └───┬────┘ └───┬───┘ └───┬─────┘
     │         │          │         │
     └─────────┴──────────┴─────────┘
               │
     ┌─────────▼──────────┐
     │  TimescaleDB +     │
     │  Vector Storage    │
     └────────────────────┘
```

### Agent Responsibilities

- **Anomaly Detection Agent**: Statistical & ML-based anomaly identification
- **RAG Memory Agent**: Contextual retrieval of historical incidents and patterns
- **Reasoning Agent**: Rule-based mock by default (pluggable LLM integration)
- **Action Agent**: Automated remediation and alerting orchestration

---

## Project Structure

```
src/agentic_observability/
├── agents/                 # Multi-agent implementations
│   ├── anomaly_detection/  # Anomaly detection algorithms
│   ├── rag_memory/         # RAG & memory management
│   ├── reasoning/          # LLM-based reasoning
│   └── action/             # Action execution
├── coordinator/            # Agent orchestration & workflow
├── ingestion/              # Telemetry data ingestion pipeline
├── rag/                    # RAG infrastructure
│   ├── embeddings/         # Embedding generation
│   └── retrieval/          # Vector search & retrieval
├── api/                    # REST API endpoints
│   └── routes/             # API route handlers
├── core/                   # Production patterns
│   ├── circuit_breaker.py  # Circuit breaker implementation
│   ├── retry.py            # Retry logic with backoff
│   └── metrics.py          # OpenTelemetry metrics
├── models/                 # Data models & schemas
├── storage/                # Database abstractions
└── config/                 # Configuration management
```

---

## Quick Start

### Prerequisites

- Python (project targets 3.11+)
- Optional: Docker Desktop (for the full stack: TimescaleDB, Redis, Qdrant, Prometheus, Grafana, Jaeger)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd agentic-observability

# Install dependencies
make install-dev
```

### Running the API (local)

```bash
# Run the API server
make run

# Open API docs
open http://localhost:8000/docs
```

---

## Configuration

Key configuration in `.env`:

- `ANTHROPIC_API_KEY`: Optional (only needed if you swap in an LLM-backed reasoning agent)
- `TIMESCALE_*`: TimescaleDB connection parameters
- `OTEL_*`: OpenTelemetry configuration
- `ANOMALY_THRESHOLD`: Detection sensitivity (0.0-1.0)
- `CIRCUIT_BREAKER_THRESHOLD`: Failure threshold before circuit opens

---

## Testing

```bash
# Run all tests
pytest -q
```

---

## Observability

The platform is instrumented with:

- **OpenTelemetry**: Distributed tracing for agent coordination
- **Prometheus**: Metrics for agent performance, latency, error rates
- **Structured Logging**: JSON logs with correlation IDs
- **Health Checks**: Liveness and readiness endpoints

Common endpoints:

- Health: http://localhost:8000/api/v1/health
- Readiness: http://localhost:8000/api/v1/health/ready
- Metrics (JSON): http://localhost:8000/api/v1/metrics

---

## Production Patterns

### Circuit Breaker
Prevents cascading failures when external services (LLM API, databases, vector stores) are degraded.

### Retry Logic
Exponential backoff with jitter for transient failures.

### Graceful Degradation
System continues operating with reduced functionality when agents fail.

### Rate Limiting
Per-agent rate limits to prevent API quota exhaustion.

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.11+ |
| **Framework** | FastAPI |
| **Reasoning** | MockReasoningAgent by default (LLM integration optional) |
| **Time-Series DB** | TimescaleDB |
| **Vector DB** | Qdrant / ChromaDB |
| **Observability** | OpenTelemetry, Prometheus |
| **Caching** | Redis |
| **Async Runtime** | asyncio, aiohttp |

---

## Roadmap

See [MASTER_GUIDE.md](MASTER_GUIDE.md#roadmap-safe-next-steps) for safe next steps that preserve production-grade patterns.

---

## Author

**Vishal Khadake**  
AI/ML Architect & Engineer 

Demonstrating:
- Multi-agent system design
- Production ML engineering
- Enterprise observability expertise
- Async Python mastery

---

## License

MIT License - See LICENSE file for details
