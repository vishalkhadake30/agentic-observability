"""
FastAPI Application Entry Point

Production-ready API with OpenTelemetry instrumentation,
health checks, and comprehensive error handling.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from ..config.settings import settings
from ..core.logging_config import configure_logging, add_context, clear_context
from ..core.metrics import get_metrics_collector, start_metrics_server
from ..coordinator.coordinator import AgentCoordinator
from ..agents.anomaly_detection.detector import AnomalyDetectionAgent
from ..agents.rag_memory.memory import RAGMemoryAgent
from ..agents.reasoning.reasoning import MockReasoningAgent, HuggingFaceReasoningAgent
from ..agents.action.action import MockActionAgent
from .routes import health, anomalies

# Configure logging before anything else
configure_logging(
    log_level=settings.log_level,
    log_dir=settings.log_dir,
    service_name=settings.otel_service_name,
    enable_file_logging=settings.enable_file_logging
)

logger = structlog.get_logger()

# Global coordinator instance
coordinator = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan management"""
    global coordinator
    
    # Startup
    logger.info("starting_application", env=settings.app_env)
    
    # Initialize metrics server (skip in dev if port is busy)
    try:
        start_metrics_server(port=9090)
        logger.info("prometheus_metrics_server_started", port=9090)
    except OSError as e:
        logger.warning("prometheus_metrics_server_skipped", reason=str(e), 
                      info="Continuing without Prometheus metrics endpoint")
    
    # Initialize agents
    logger.info("initializing_agents")
    anomaly_agent = AnomalyDetectionAgent(
        name="anomaly-detection",
        circuit_breaker_threshold=settings.circuit_breaker_threshold,
        max_retries=settings.max_agent_retries
    )
    
    rag_agent = RAGMemoryAgent(
        name="rag-memory",
        vector_db_url=settings.vector_db_url,
        collection_name=settings.vector_collection_name,
        circuit_breaker_threshold=settings.circuit_breaker_threshold,
        max_retries=settings.max_agent_retries
    )
    
    # Initialize reasoning agent - HuggingFace LLM with mock fallback
    reasoning_agent = HuggingFaceReasoningAgent(
        name="reasoning",
        hf_token=settings.huggingface_token,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        use_mock_fallback=True,  # Always fallback to mock if HF fails
        circuit_breaker_threshold=settings.circuit_breaker_threshold,
        max_retries=settings.max_agent_retries
    )
    
    action_agent = MockActionAgent(
        name="action",
        dry_run=settings.app_env == "development",
        circuit_breaker_threshold=settings.circuit_breaker_threshold,
        max_retries=settings.max_agent_retries
    )
    
    # Initialize coordinator
    logger.info("initializing_coordinator")
    coordinator = AgentCoordinator(
        anomaly_agent=anomaly_agent,
        rag_agent=rag_agent,
        reasoning_agent=reasoning_agent,
        action_agent=action_agent,
        name="main-coordinator"
    )
    
    await coordinator.initialize()
    
    # Set coordinator in route modules
    health.set_coordinator(coordinator)
    anomalies.set_coordinator(coordinator)
    
    logger.info("application_ready", agents=4)
    
    yield
    
    # Shutdown
    logger.info("shutting_down_application")
    if coordinator:
        await coordinator.cleanup()
    logger.info("application_shutdown_complete")


# Create FastAPI app
app = FastAPI(
    title="Agentic Observability Platform",
    description="Production-grade multi-agent system for anomaly detection and root cause analysis",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with correlation ID"""
    import uuid
    
    # Generate correlation ID for request tracing
    correlation_id = str(uuid.uuid4())
    
    # Add correlation ID to logging context
    add_context(
        correlation_id=correlation_id,
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else None
    )
    
    logger.info(
        "request_received",
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else None
    )
    
    response = await call_next(request)
    
    # Record metrics
    metrics = get_metrics_collector()
    metrics.increment_api_requests(
        endpoint=request.url.path,
        method=request.method,
        status_code=response.status_code
    )
    
    # Clear context after request completes
    clear_context()
    
    return response


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        error=str(exc),
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.app_env == "development" else None
        }
    )


# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(anomalies.router, prefix="/api/v1/pipeline", tags=["pipeline"])


# Instrument with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "agentic_observability.api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower()
    )
