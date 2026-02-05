# System Architecture

## Production-Grade Agentic Observability Demo

---

## Architecture Overview

This document describes the architecture of a production-grade multi-agent system for anomaly detection and root cause analysis, designed to demonstrate Engineering capabilities.

Canonical runtime walkthrough:


This file includes both **implemented** components and **planned/illustrative** components. The master guide is the source of truth for what runs today.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Client Applications                         │
│                   (Web UI, CLI, External Systems)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           │ HTTPS/REST
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                     API Gateway (FastAPI)                        │
│  • Request Validation (Pydantic)                                 │
│  • Structured Logging                                            │
│  • OpenTelemetry Instrumentation (basic)                         │
│  • (Planned) Authentication / Rate limiting                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    Agent Coordinator                             │
│  ┌──────────────────────────────────────────────────┐           │
│  │           Orchestration Layer                     │           │
│  │  • Workflow Management                            │           │
│  │  • Circuit Breaker Protection                     │           │
│  │  • Retry Logic with Backoff                       │           │
│  │  • Metrics Collection                             │           │
│  └──────────────────────────────────────────────────┘           │
└──────┬────────┬────────┬────────┬─────────────────────────────┘
       │        │        │        │
       │        │        │        │
   ┌───▼───┐┌──▼─────┐┌─▼──────┐┌▼────────┐
   │Anomaly││  RAG   ││Reason- ││ Action  │
   │Detect ││ Memory ││ing     ││ Agent   │
   │Agent  ││ Agent  ││Agent   ││         │
   └───┬───┘└───┬────┘└───┬────┘└───┬─────┘
       │        │         │         │
       └────────┴─────────┴─────────┘
                │
     ┌──────────▼───────────┐
     │   Data & Storage     │
     │  ┌────────────────┐  │
    │  │ TimescaleDB    │  │  (Present in docker-compose; not wired yet)
    │  │ (PostgreSQL)   │  │
     │  └────────────────┘  │
     │  ┌────────────────┐  │
    │  │ Qdrant         │  │  Vector embeddings (RAG memory)
    │  │ (Vector Store) │  │  Semantic search
     │  └────────────────┘  │
     │  ┌────────────────┐  │
    │  │     Redis      │  │  (Present in docker-compose; not wired yet)
    │  │   (Cache)      │  │
     │  └────────────────┘  │
     └────────────────────┘
```

---

## Agent Architecture

### 1. Anomaly Detection Agent

**Responsibility**: Identify statistical and ML-based anomalies in time-series data

**Implemented components**:
- Statistical detection (Z-score, robust Z-score via MAD, IQR, moving averages)

**Planned/illustrative**:
- ML models (Isolation Forest, Autoencoders)
- Seasonal decomposition
- Real-time streaming analysis

**Input**: Time-series metrics, logs, traces
**Output**: Anomaly events with confidence scores

### 2. RAG Memory Agent

**Responsibility**: Contextual retrieval from historical incident data

**Implemented components**:
- Embedding generation (SentenceTransformers)
- Vector similarity search (Qdrant)

**Planned/illustrative**:
- Alternative embedding backends
- More advanced filtering/ranking

**Input**: Anomaly description, metadata
**Output**: Similar historical incidents, patterns, resolutions

### 3. Reasoning Agent

**Responsibility**: Root cause analysis (rule-based mock by default; LLM-based reasoning is a planned swap)

**Implemented components**:
- Rule-based pattern matching with structured outputs

**Planned/illustrative**:
- LLM-backed analysis (Claude/GPT)
- Prompting strategies and richer reasoning chains

**Input**: Anomaly + RAG context
**Output**: Root cause hypotheses, investigation steps

### 4. Action Agent

**Responsibility**: Alerts + simulated remediation (mock implementation)

**Components**:
- Runbook execution
- External system integration (PagerDuty, Slack, etc.)
- Remediation validation
- Rollback mechanisms

**Input**: Reasoning output, remediation plan
**Output**: Action results, notifications

---

## Production Patterns

### Circuit Breaker Pattern

**Purpose**: Prevent cascading failures when external services degrade

**States**:
- `CLOSED`: Normal operation, requests pass through
- `OPEN`: Service failing, reject requests immediately
- `HALF_OPEN`: Testing if service recovered

**Configuration**:
```python
CircuitBreaker(
    failure_threshold=5,      # Open after 5 failures
    timeout=60,               # Wait 60s before half-open
  name="llm_api"
)
```

**Applied To**:
- LLM API calls (optional)
- TimescaleDB queries
- Vector database searches
- External API integrations

### Retry Logic with Exponential Backoff

**Purpose**: Handle transient failures gracefully

**Features**:
- Exponential backoff: `delay = base_delay * (2 ^ attempt)`
- Jitter: Random variance to prevent thundering herd
- Selective retry: Only retry specific exception types
- Max attempts: Configurable retry limit

**Configuration**:
```python
retry_with_backoff(
    func=api_call,
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    jitter=True
)
```

### Observability Stack

**Metrics** (Prometheus + OpenTelemetry):
- Agent execution latency
- Success/failure rates
- Circuit breaker state
- API request counts

**Tracing** (Jaeger + OpenTelemetry):
- Distributed traces across agents
- Request correlation IDs
- Span attributes for debugging

**Logging** (Structlog):
- JSON-formatted logs
- Correlation IDs
- Context binding
- Log levels: DEBUG, INFO, WARNING, ERROR

---

## Data Flow

### Anomaly Detection Workflow

```
1. Metric Ingestion
   ↓
2. Anomaly Detection Agent
   • Statistical analysis
   • ML model inference
   • Threshold checking
   ↓
3. RAG Memory Agent (if anomaly detected)
   • Generate embedding
   • Vector similarity search
   • Retrieve historical context
   ↓
4. Reasoning Agent
   • Combine anomaly + RAG context
  • Optional LLM API call with retry + circuit breaker
   • Generate root cause hypotheses
   ↓
5. Action Agent
   • Execute remediation
   • Send alerts
   • Update incident database
   ↓
6. Response
   • Return to client
   • Store in TimescaleDB
```

---

## Scalability Considerations

### Horizontal Scaling

- **API Layer**: Stateless FastAPI instances behind load balancer
- **Agent Pool**: Worker pool with task queue (Celery + Redis)
- **Database**: TimescaleDB read replicas for query distribution
- **Vector Store**: Qdrant distributed cluster

### Performance Optimization

- **Async I/O**: All I/O operations use async/await
- **Connection Pooling**: Database connection pools
- **Caching**: Redis for frequently accessed data
- **Batch Processing**: Group similar operations

### Reliability

- **Health Checks**: Kubernetes liveness/readiness probes
- **Graceful Shutdown**: Proper cleanup on SIGTERM
- **Database Failover**: TimescaleDB HA configuration
- **Circuit Breakers**: Prevent cascading failures

---

## Security

### API Security

- **Authentication**: API key or JWT tokens
- **Authorization**: Role-based access control (RBAC)
- **Rate Limiting**: Per-client request limits
- **Input Validation**: Pydantic models

### Data Security

- **Encryption at Rest**: TimescaleDB encryption
- **Encryption in Transit**: TLS for all connections
- **Secrets Management**: Environment variables, not hardcoded
- **Audit Logging**: All actions logged with user context

---

## Deployment

### Container Orchestration

```yaml
# Kubernetes Deployment Example
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentic-observability
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: agentic-observability:latest
        ports:
        - containerPort: 8000
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: anthropic-key
        livenessProbe:
          httpGet:
            path: /api/v1/health/live
            port: 8000
        readinessProbe:
          httpGet:
            path: /api/v1/health/ready
            port: 8000
```

### Infrastructure as Code

- **Docker Compose**: Local development environment
- **Kubernetes**: Production deployment
- **Terraform**: Cloud infrastructure provisioning
- **Helm Charts**: Kubernetes package management

---

## Monitoring & Alerting

### Metrics Dashboard (Grafana)

**Agent Performance**:
- Execution latency percentiles (p50, p95, p99)
- Success/failure rates
- Throughput (requests/second)

**System Health**:
- Circuit breaker state
- Database connection pool usage
- Memory and CPU utilization

**Business Metrics**:
- Anomalies detected per hour
- Root cause analysis accuracy
- Mean time to detection (MTTD)
- Mean time to resolution (MTTR)

### Alerts

- Agent failure rate > 5%
- API latency p95 > 500ms
- Circuit breaker OPEN state
- Database connection pool exhausted

---

## Testing Strategy

### Unit Tests

- Individual agent logic
- Circuit breaker behavior
- Retry logic with different failure scenarios
- Data model validation

### Integration Tests

- Multi-agent workflow
- Database operations
- External API integration
- End-to-end anomaly detection

### Load Tests

- API throughput under concurrent requests
- Agent performance with high volume
- Database query performance

---

## Technology Stack Summary

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **API Framework** | FastAPI | Async REST API |
| **Language** | Python 3.11+ | Application code |
| **Reasoning** | Mock by default; optional LLM (e.g., Claude) | Root cause hypotheses |
| **Time-Series DB** | TimescaleDB | Metrics storage |
| **Vector DB** | Qdrant | Embeddings & search |
| **Cache** | Redis | Caching & queue |
| **Observability** | OpenTelemetry | Metrics & traces |
| **Metrics** | Prometheus | Metrics collection |
| **Tracing** | Jaeger | Distributed tracing |
| **Dashboards** | Grafana | Visualization |
| **Orchestration** | Docker Compose / K8s | Deployment |

---

## Design Decisions

### Why FastAPI?

- **Async Native**: Built on Starlette/ASGI for high concurrency
- **Type Safety**: Pydantic integration for validation
- **OpenAPI**: Automatic API documentation
- **Performance**: Comparable to Node.js and Go

### Why an LLM (optional)?

- **Reasoning Capability**: Useful for causal hypothesis generation
- **Context Window**: Helps when combining anomaly + incident context
- **Reliability**: Needs careful prompting and validation
- **API Stability**: Choose a provider with good SLAs

### Why TimescaleDB?

- **PostgreSQL Compatibility**: Rich SQL features
- **Time-Series Optimized**: Hypertables for efficient queries
- **Compression**: Automatic data compression
- **pgvector Extension**: Vector embeddings in same DB (alternative)

### Why Multi-Agent Architecture?

- **Separation of Concerns**: Each agent has single responsibility
- **Scalability**: Agents can scale independently
- **Resilience**: Agent failure doesn't crash entire system
- **Flexibility**: Easy to add/modify agents

---

This architecture demonstrates production-grade thinking.
