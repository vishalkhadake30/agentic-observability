# Quick Start Guide

## Initial Setup

Follow these steps to get the agentic observability demo running locally.

### 1. Environment Setup

```bash
# Navigate to project directory
cd ~/agentic-observability

# Create Python virtual environment (project targets Python 3.11+)
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or: .venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip
```

### 2. Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Or install development dependencies (includes testing tools)
pip install -r requirements-dev.txt
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys and configuration
# Optional: ANTHROPIC_API_KEY (only required if you swap in an LLM-backed reasoning agent)
nano .env  # or use your preferred editor
```

### 4. Start Infrastructure

This step is optional. You can run the API and pipeline without Docker; the RAG stage will run best-effort if Qdrant is unavailable.

```bash
# Start TimescaleDB, Redis, Qdrant, Jaeger, Prometheus, Grafana
make docker-up

# Verify services are running
docker-compose ps

# Check logs if needed
docker-compose logs -f
```

### 5. Database Setup

```bash
# Initialize database (when Alembic migrations are ready)
# alembic upgrade head
```

### 6. Run the Application

```bash
# Development mode with auto-reload
make run
# Equivalent:
# uvicorn agentic_observability.api.main:app --app-dir src --reload --port 8000

# Production mode
make run-prod
# Equivalent:
# uvicorn agentic_observability.api.main:app --app-dir src --host 0.0.0.0 --port 8000
```

### 7. Verify Installation

```bash
# Health check
curl http://localhost:8000/api/v1/health

# API documentation
open http://localhost:8000/docs

# Prometheus metrics
open http://localhost:9090

# Jaeger tracing
open http://localhost:16686

# Grafana dashboards
open http://localhost:3000
# Default credentials: admin/admin
```

---

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/unit/test_circuit_breaker.py -v

# Run with coverage report
pytest --cov=src/agentic_observability --cov-report=html
open htmlcov/index.html
```

### Code Quality

```bash
# Format code
make format

# Run linters
make lint

# Type checking
mypy src/
```

### Clean Build Artifacts

```bash
make clean
```

---

## Project Structure Overview

```
agentic-observability/
├── src/agentic_observability/       # Main application package
│   ├── agents/                      # Multi-agent implementations
│   │   ├── anomaly_detection/       # Statistical & ML anomaly detection
│   │   ├── rag_memory/              # RAG & context retrieval
│   │   ├── reasoning/               # Mock reasoning (LLM integration optional)
│   │   └── action/                  # Automated remediation
│   ├── coordinator/                 # Agent orchestration
│   ├── ingestion/                   # Data ingestion pipeline
│   ├── rag/                         # RAG infrastructure
│   │   ├── embeddings/              # Embedding generation
│   │   └── retrieval/               # Vector search
│   ├── api/                         # FastAPI REST API
│   │   └── routes/                  # API endpoints
│   ├── core/                        # Production patterns
│   │   ├── circuit_breaker.py       # Circuit breaker
│   │   ├── retry.py                 # Retry with backoff
│   │   └── metrics.py               # OpenTelemetry metrics
│   ├── models/                      # Pydantic models
│   ├── storage/                     # Database layer
│   └── config/                      # Configuration management
├── tests/                           # Test suite
│   ├── unit/                        # Unit tests
│   └── integration/                 # Integration tests
├── docker/                          # Docker configurations
├── docker-compose.yml               # Infrastructure orchestration
├── requirements.txt                 # Production dependencies
├── pyproject.toml                   # Project configuration
├── Makefile                         # Common commands
└── README.md                        # Project documentation
```

---

## Key Features Demonstrated

### 1. Multi-Agent Architecture
- **Anomaly Detection Agent**: Identifies statistical anomalies
- **RAG Memory Agent**: Context-aware retrieval from historical data
- **Reasoning Agent**: Rule-based mock by default (pluggable LLM integration)
- **Action Agent**: Automated remediation execution

### 2. Production Patterns
- **Circuit Breaker**: Prevents cascading failures
- **Retry Logic**: Exponential backoff with jitter
- **Observability**: OpenTelemetry, Prometheus, Jaeger
- **Structured Logging**: JSON logs with correlation IDs
- **Health Checks**: Kubernetes-ready liveness/readiness probes

### 3. Technology Stack
- **Framework**: FastAPI (async/await)
- **Reasoning**: Mock by default (LLM integration optional)
- **Time-Series DB**: TimescaleDB
- **Vector DB**: Qdrant
- **Caching**: Redis
- **Observability**: OpenTelemetry + Prometheus + Jaeger + Grafana

---

## Next Steps

1. **Enhance Agents**
   - Expand anomaly detection algorithms
   - Improve RAG retrieval quality and ranking
   - Optional LLM reasoning integration
   - Add real alerting/remediation integrations

2. **Database Schema**
   - TimescaleDB hypertables for metrics
   - Vector embeddings storage
   - Anomaly history tracking

3. **Agent Coordinator**
   - Multi-agent workflow orchestration
   - Circuit breaker integration
   - Async task queue

4. **Advanced Features**
   - Real-time streaming ingestion
   - Predictive anomaly detection
   - Automated root cause correlation
   - Self-healing capabilities

---

## Troubleshooting

### Docker Services Won't Start
```bash
# Check if ports are already in use
lsof -i :5432  # TimescaleDB
lsof -i :6379  # Redis
lsof -i :6333  # Qdrant

# Stop existing services or change ports in docker-compose.yml
```

### Import Errors
```bash
# Ensure you're in virtual environment
which python
# Should show: /path/to/venv/bin/python

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### API Won't Start
```bash
# Check .env file exists and has required keys
cat .env | grep ANTHROPIC_API_KEY

# Check Python version
python --version  # Should be 3.11+
```

---

## Support

For questions or issues specific to this demonstration project, please refer to the comprehensive documentation in [README.md](README.md) or examine the inline code documentation.

This project demonstrates ML engineering patterns for multi-agent systems.
