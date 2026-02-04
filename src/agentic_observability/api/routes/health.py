"""
Health Check Endpoints

Provides liveness and readiness checks with coordinator integration.
"""

from fastapi import APIRouter, status
from pydantic import BaseModel
from typing import Dict, Any
import structlog

from ...config.settings import settings

logger = structlog.get_logger()

router = APIRouter()

# Global coordinator instance (set by main.py)
_coordinator = None


def set_coordinator(coordinator):
    """Set global coordinator instance"""
    global _coordinator
    _coordinator = coordinator


def get_coordinator():
    """Get coordinator instance"""
    return _coordinator


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    version: str
    environment: str
    checks: Dict[str, Any]


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """
    Liveness probe - checks if application is running.
    
    Returns HTTP 200 if application is alive.
    """
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        environment=settings.app_env,
        checks={
            "api": "ok"
        }
    )


@router.get("/health/ready", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def readiness_check():
    """
    Readiness probe - checks if application can accept traffic.
    
    Checks:
    - Coordinator initialization
    - Agent health status
    - Circuit breaker states
    """
    coordinator = get_coordinator()
    
    if not coordinator:
        return HealthResponse(
            status="not_ready",
            version="0.1.0",
            environment=settings.app_env,
            checks={
                "api": "ok",
                "coordinator": "not_initialized"
            }
        )
    
    is_healthy = coordinator.is_healthy()
    
    checks = {
        "api": "ok",
        "coordinator": "ok" if is_healthy else "degraded",
        "anomaly_agent": "ok" if coordinator.anomaly_agent.is_healthy() else "degraded",
        "rag_agent": "ok" if coordinator.rag_agent.is_healthy() else "degraded",
        "reasoning_agent": "ok" if coordinator.reasoning_agent.is_healthy() else "degraded",
        "action_agent": "ok" if coordinator.action_agent.is_healthy() else "degraded"
    }
    
    all_healthy = all(v == "ok" for v in checks.values())
    
    return HealthResponse(
        status="ready" if all_healthy else "not_ready",
        version="0.1.0",
        environment=settings.app_env,
        checks=checks
    )


@router.get("/health/live", status_code=status.HTTP_200_OK)
async def liveness_check():
    """Simple liveness check for container orchestration"""
    return {"status": "alive"}


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Get comprehensive system metrics for Prometheus/monitoring
    
    Returns coordinator and agent metrics
    """
    coordinator = get_coordinator()
    
    if not coordinator:
        return {"error": "Coordinator not initialized"}
    
    return coordinator.get_metrics()
