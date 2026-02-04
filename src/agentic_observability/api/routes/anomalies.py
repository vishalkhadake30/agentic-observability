"""
Pipeline Execution Endpoints

API routes for processing metrics through the complete 4-agent pipeline.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import structlog

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


class MetricsInput(BaseModel):
    """Input model for metrics processing"""
    service_name: str = Field(..., description="Name of the service")
    metric_name: str = Field(default="response_time", description="Name of the metric")
    metrics: List[float] = Field(..., description="List of metric values")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "service_name": "api-server",
                "metric_name": "response_time_ms",
                "metrics": [95.2, 96.1, 98.7, 150.3, 175.2, 200.1, 225.4, 250.8],
                "metadata": {"environment": "production", "region": "us-west-2"}
            }
        }


class PipelineResponse(BaseModel):
    """Response model for pipeline execution"""
    correlation_id: str
    success: bool
    stage: str
    anomaly_detected: bool
    root_cause: Optional[str] = None
    confidence: float = 0.0
    recommendations: List[str] = []
    alerts_sent: List[Dict[str, Any]] = []
    actions_executed: List[Dict[str, Any]] = []
    total_duration_ms: float
    stage_durations: Dict[str, float]
    error: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "correlation_id": "pipeline-1234567890.123",
                "success": True,
                "stage": "completed",
                "anomaly_detected": True,
                "root_cause": "Memory leak detected in api-server causing response time degradation",
                "confidence": 0.85,
                "recommendations": [
                    "Restart api-server service",
                    "Monitor memory usage for next 30 minutes",
                    "Review recent code deployments"
                ],
                "alerts_sent": [
                    {"channel": "slack", "severity": "error", "status": "sent"}
                ],
                "actions_executed": [
                    {"type": "restart_service", "target": "api-server", "status": "completed"}
                ],
                "total_duration_ms": 1234.56,
                "stage_durations": {
                    "anomaly_detection": 100.0,
                    "rag_retrieval": 200.0,
                    "reasoning": 500.0,
                    "action": 434.56
                }
            }
        }


@router.post("/process", response_model=PipelineResponse, status_code=status.HTTP_200_OK)
async def process_metrics(request: MetricsInput) -> PipelineResponse:
    """
    Process metrics through complete 4-stage pipeline
    
    **Pipeline Stages:**
    1. **Anomaly Detection** - Statistical analysis to detect anomalies
    2. **RAG Retrieval** - Find similar historical incidents from vector DB
    3. **Reasoning** - Claude AI analyzes root cause with context
    4. **Action** - Execute remediation and send alerts
    
    **Returns:**
    - Anomaly detection results
    - Root cause analysis with confidence score
    - Recommended actions
    - Alerts sent (Slack, PagerDuty, etc.)
    - Actions executed (service restarts, scaling, etc.)
    - Performance metrics (duration per stage)
    """
    coordinator = get_coordinator()
    
    if not coordinator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Coordinator not initialized. System starting up."
        )
    
    if not coordinator.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System is unhealthy. One or more agents are degraded."
        )
    
    logger.info(
        "pipeline_request_received",
        service=request.service_name,
        metric=request.metric_name,
        data_points=len(request.metrics)
    )
    
    try:
        # Prepare input for coordinator
        input_data = {
            "service_name": request.service_name,
            "metric_name": request.metric_name,
            "metrics": request.metrics
        }
        
        if request.metadata:
            input_data.update(request.metadata)
        
        # Execute pipeline through coordinator
        result = await coordinator.process_metrics(input_data)
        
        logger.info(
            "pipeline_request_completed",
            service=request.service_name,
            correlation_id=result.correlation_id,
            success=result.success,
            anomaly_detected=result.anomaly_detected,
            duration_ms=result.total_duration_ms
        )
        
        # Convert to response model
        return PipelineResponse(
            correlation_id=result.correlation_id,
            success=result.success,
            stage=result.stage.value,
            anomaly_detected=result.anomaly_detected,
            root_cause=result.root_cause,
            confidence=result.confidence,
            recommendations=result.recommendations,
            alerts_sent=result.alerts_sent,
            actions_executed=result.actions_executed,
            total_duration_ms=result.total_duration_ms,
            stage_durations=result.stage_durations,
            error=result.error
        )
        
    except Exception as e:
        logger.error(
            "pipeline_request_failed",
            service=request.service_name,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failed: {str(e)}"
        )


@router.get("/{correlation_id}")
async def get_pipeline_status(correlation_id: str) -> Dict[str, Any]:
    """
    Get status of a pipeline execution by correlation ID
    
    **Note:** This requires implementing a pipeline result store (Redis/DB).
    Currently returns placeholder.
    """
    logger.info("pipeline_status_requested", correlation_id=correlation_id)
    
    # TODO: Implement pipeline result storage and retrieval
    return {
        "correlation_id": correlation_id,
        "message": "Pipeline result storage not yet implemented",
        "note": "Results are currently returned synchronously in POST /process response"
    }
