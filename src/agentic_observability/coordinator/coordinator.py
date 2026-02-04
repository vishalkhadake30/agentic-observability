"""
Multi-Agent Coordinator

Orchestrates the complete incident response pipeline:
1. Anomaly Detection → Detect issues in metrics/logs
2. RAG Memory → Retrieve similar historical incidents
3. Reasoning → Analyze root cause
4. Action → Execute remediation and alerts

WHY THIS ARCHITECTURE:
- Sequential pipeline ensures proper data flow between agents
- Each agent can fail independently without breaking the pipeline
- Coordinator provides unified interface for the entire system
- Metrics tracked across the entire pipeline
- Circuit breaker prevents cascading failures

PRODUCTION PATTERNS:
- Pipeline Pattern: Sequential processing with data transformation
- Saga Pattern: Coordinate distributed transactions across agents
- Bulkhead Pattern: Isolate agent failures
- Observability: Track end-to-end latency and success rates
"""

from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import structlog

from ..agents.anomaly_detection.detector import AnomalyDetectionAgent
from ..agents.rag_memory.memory import RAGMemoryAgent
from ..agents.reasoning.reasoning import ReasoningAgent, MockReasoningAgent
from ..agents.action.action import ActionAgent, MockActionAgent
from ..core.circuit_breaker import CircuitBreakerError
from ..core.metrics import get_metrics_collector

logger = structlog.get_logger()


class PipelineStage(Enum):
    """Pipeline execution stages"""
    ANOMALY_DETECTION = "anomaly_detection"
    RAG_RETRIEVAL = "rag_retrieval"
    REASONING = "reasoning"
    ACTION = "action"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineResult:
    """Result of complete pipeline execution"""
    correlation_id: str
    stage: PipelineStage
    success: bool
    
    # Stage outputs
    anomaly_detected: bool = False
    anomaly_data: dict[str, Any] = field(default_factory=dict)
    
    similar_incidents: list[dict[str, Any]] = field(default_factory=list)
    
    root_cause: Optional[str] = None
    confidence: float = 0.0
    recommendations: list[str] = field(default_factory=list)
    
    alerts_sent: list[dict[str, Any]] = field(default_factory=list)
    actions_executed: list[dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    total_duration_ms: float = 0.0
    stage_durations: dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    error: Optional[str] = None


class AgentCoordinator:
    """
    Coordinates multi-agent pipeline for incident response.
    
    WHY THIS DESIGN:
    - Separates orchestration from agent logic (single responsibility)
    - Agents are injected, enabling easy testing with mocks
    - Pipeline can be extended with new agents without changing core logic
    - Each stage is independent and can fail gracefully
    
    USAGE:
        coordinator = AgentCoordinator(
            anomaly_agent=AnomalyDetectionAgent(...),
            rag_agent=RAGMemoryAgent(...),
            reasoning_agent=MockReasoningAgent(...),
            action_agent=MockActionAgent(...)
        )
        
        await coordinator.initialize()
        
        result = await coordinator.process_metrics({
            "service_name": "api-server",
            "metrics": [95.2, 96.1, 98.7, 150.3, 175.2]
        })
    """
    
    def __init__(
        self,
        anomaly_agent: AnomalyDetectionAgent,
        rag_agent: RAGMemoryAgent,
        reasoning_agent: ReasoningAgent,
        action_agent: ActionAgent,
        name: str = "agent-coordinator"
    ):
        """
        Initialize coordinator with agent dependencies.
        
        Args:
            anomaly_agent: Agent for anomaly detection
            rag_agent: Agent for historical context retrieval
            reasoning_agent: Agent for root cause analysis
            action_agent: Agent for remediation and alerts
            name: Coordinator name for logging
        """
        self.name = name
        self.logger = logger.bind(coordinator=name)
        
        # Agent dependencies
        self.anomaly_agent = anomaly_agent
        self.rag_agent = rag_agent
        self.reasoning_agent = reasoning_agent
        self.action_agent = action_agent
        
        # Metrics
        self._metrics_collector = get_metrics_collector()
        self._total_pipelines = 0
        self._successful_pipelines = 0
        self._failed_pipelines = 0
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """
        Initialize all agents in the pipeline.
        
        WHY SEPARATE INIT:
        - Allows async initialization (DB connections, model loading)
        - Fails fast if any agent can't initialize
        - Provides clear error messages about which agent failed
        """
        self.logger.info("initializing_coordinator", agents=4)
        
        try:
            # Initialize all agents in parallel for speed
            await asyncio.gather(
                self.anomaly_agent.initialize(),
                self.rag_agent.initialize(),
                self.reasoning_agent.initialize(),
                self.action_agent.initialize()
            )
            
            self._initialized = True
            self.logger.info("coordinator_initialized", agents=4)
            
        except Exception as e:
            self.logger.error(
                "coordinator_initialization_failed",
                error=str(e),
                exc_info=True
            )
            raise
    
    async def cleanup(self) -> None:
        """Cleanup all agent resources"""
        self.logger.info("cleaning_up_coordinator")
        
        # Cleanup in reverse order
        await asyncio.gather(
            self.action_agent.cleanup(),
            self.reasoning_agent.cleanup(),
            self.rag_agent.cleanup(),
            self.anomaly_agent.cleanup(),
            return_exceptions=True  # Don't fail if one cleanup fails
        )
        
        self._initialized = False
        self.logger.info("coordinator_cleanup_complete")
    
    async def process_metrics(
        self,
        input_data: dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> PipelineResult:
        """
        Process metrics through the complete pipeline.
        
        Pipeline stages:
        1. Anomaly Detection: Identify if metrics are anomalous
        2. RAG Retrieval: Find similar historical incidents
        3. Reasoning: Analyze root cause with context
        4. Action: Execute remediation and send alerts
        
        Args:
            input_data: Must contain:
                - service_name: Name of the service
                - metrics: List of metric values OR
                - metric_name: Name of metric
                - metric_value: Current value
                - historical_data: Historical values (optional)
            correlation_id: Optional correlation ID for tracing
            
        Returns:
            PipelineResult with complete execution details
        """
        if not self._initialized:
            raise RuntimeError("Coordinator not initialized. Call initialize() first.")
        
        # Generate correlation ID if not provided
        if correlation_id is None:
            correlation_id = f"pipeline-{datetime.utcnow().timestamp()}"
        
        self.logger = logger.bind(
            coordinator=self.name,
            correlation_id=correlation_id
        )
        
        start_time = asyncio.get_event_loop().time()
        stage_durations: dict[str, float] = {}
        
        self.logger.info(
            "pipeline_started",
            service_name=input_data.get("service_name", "unknown")
        )
        
        # Initialize result
        result = PipelineResult(
            correlation_id=correlation_id,
            stage=PipelineStage.ANOMALY_DETECTION,
            success=False
        )
        
        try:
            # Stage 1: Anomaly Detection
            anomaly_result = await self._run_anomaly_detection(
                input_data,
                stage_durations
            )
            
            result.anomaly_detected = anomaly_result.get("is_anomaly", False)
            result.anomaly_data = anomaly_result
            
            # If no anomaly, pipeline completes successfully
            if not result.anomaly_detected:
                self.logger.info(
                    "no_anomaly_detected",
                    service_name=input_data.get("service_name")
                )
                result.stage = PipelineStage.COMPLETED
                result.success = True
                result.stage_durations = stage_durations
                result.total_duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                
                self._total_pipelines += 1
                self._successful_pipelines += 1
                
                return result
            
            # Stage 2: RAG Retrieval
            result.stage = PipelineStage.RAG_RETRIEVAL
            rag_result = await self._run_rag_retrieval(
                anomaly_result,
                input_data,
                stage_durations
            )
            
            result.similar_incidents = rag_result.get("similar_incidents", [])
            
            # Stage 3: Reasoning
            result.stage = PipelineStage.REASONING
            reasoning_result = await self._run_reasoning(
                anomaly_result,
                rag_result,
                stage_durations
            )
            
            result.root_cause = reasoning_result.get("root_cause")
            result.confidence = reasoning_result.get("confidence", 0.0)
            result.recommendations = reasoning_result.get("recommendations", [])
            
            # Stage 4: Action
            result.stage = PipelineStage.ACTION
            action_result = await self._run_action(
                reasoning_result,
                anomaly_result,
                stage_durations
            )
            
            result.alerts_sent = action_result.get("alerts_sent", [])
            result.actions_executed = action_result.get("actions_executed", [])
            
            # Pipeline completed successfully
            result.stage = PipelineStage.COMPLETED
            result.success = True
            
            self._total_pipelines += 1
            self._successful_pipelines += 1
            
            end_time = asyncio.get_event_loop().time()
            result.total_duration_ms = (end_time - start_time) * 1000
            result.stage_durations = stage_durations
            
            self.logger.info(
                "pipeline_completed",
                duration_ms=result.total_duration_ms,
                anomaly_detected=result.anomaly_detected,
                root_cause=result.root_cause,
                confidence=result.confidence,
                alerts_sent=len(result.alerts_sent),
                actions_executed=len(result.actions_executed)
            )
            
            return result
            
        except CircuitBreakerError as e:
            # Circuit breaker opened - system is protecting itself
            result.stage = PipelineStage.FAILED
            result.success = False
            result.error = f"Circuit breaker open: {str(e)}"
            
            self._total_pipelines += 1
            self._failed_pipelines += 1
            
            self.logger.error(
                "pipeline_failed_circuit_breaker",
                stage=result.stage.value,
                error=str(e)
            )
            
            return result
            
        except Exception as e:
            # Unexpected failure
            result.stage = PipelineStage.FAILED
            result.success = False
            result.error = str(e)
            
            self._total_pipelines += 1
            self._failed_pipelines += 1
            
            self.logger.error(
                "pipeline_failed",
                stage=result.stage.value,
                error=str(e),
                exc_info=True
            )
            
            return result
    
    async def _run_anomaly_detection(
        self,
        input_data: dict[str, Any],
        stage_durations: dict[str, float]
    ) -> dict[str, Any]:
        """Run anomaly detection stage"""
        stage_start = asyncio.get_event_loop().time()
        
        self.logger.info("stage_started", stage="anomaly_detection")
        
        # Normalize input schema for AnomalyDetectionAgent.
        values = input_data.get("values")
        if values is None:
            values = input_data.get("metrics")

        anomaly_input = {
            **input_data,
            "values": values or [],
        }

        result = await self.anomaly_agent.execute(anomaly_input)
        
        stage_end = asyncio.get_event_loop().time()
        duration = (stage_end - stage_start) * 1000
        stage_durations["anomaly_detection"] = duration
        
        self.logger.info(
            "stage_completed",
            stage="anomaly_detection",
            duration_ms=duration,
            is_anomaly=result.get("is_anomaly", False)
        )
        
        return result
    
    async def _run_rag_retrieval(
        self,
        anomaly_result: dict[str, Any],
        input_data: dict[str, Any],
        stage_durations: dict[str, float]
    ) -> dict[str, Any]:
        """Run RAG retrieval stage"""
        stage_start = asyncio.get_event_loop().time()
        
        self.logger.info("stage_started", stage="rag_retrieval")
        
        # Prepare query for RAG
        query_text = anomaly_result.get("anomaly_description") or anomaly_result.get("explanation") or ""
        query_data = {
            "query": str(query_text),
            "service_name": input_data.get("service_name", "unknown"),
            "metric_name": input_data.get("metric_name", "unknown"),
            "top_k": 5
        }

        try:
            result = await self.rag_agent.execute(query_data)
        except Exception as e:
            # Graceful degradation: RAG enriches context but should not prevent
            # root-cause analysis from running.
            self.logger.warning(
                "rag_retrieval_failed",
                error=str(e),
                error_type=type(e).__name__,
                msg="Proceeding without historical context",
            )
            result = {
                "similar_incidents": [],
                "query_embedding": [],
                "total_found": 0,
                "search_time_ms": 0.0,
                "error": str(e),
            }
        
        stage_end = asyncio.get_event_loop().time()
        duration = (stage_end - stage_start) * 1000
        stage_durations["rag_retrieval"] = duration
        
        self.logger.info(
            "stage_completed",
            stage="rag_retrieval",
            duration_ms=duration,
            similar_incidents=len(result.get("similar_incidents", []))
        )
        
        return result
    
    async def _run_reasoning(
        self,
        anomaly_result: dict[str, Any],
        rag_result: dict[str, Any],
        stage_durations: dict[str, float]
    ) -> dict[str, Any]:
        """Run reasoning stage"""
        stage_start = asyncio.get_event_loop().time()
        
        self.logger.info("stage_started", stage="reasoning")
        
        # Prepare reasoning input
        similar_incidents = rag_result.get("similar_incidents", [])
        reasoning_data = {
            # Provide both key styles for compatibility.
            "anomaly": anomaly_result,
            "anomaly_data": anomaly_result,
            "similar_incidents": similar_incidents,
            "historical_incidents": similar_incidents,
        }
        
        result = await self.reasoning_agent.execute(reasoning_data)
        
        stage_end = asyncio.get_event_loop().time()
        duration = (stage_end - stage_start) * 1000
        stage_durations["reasoning"] = duration
        
        self.logger.info(
            "stage_completed",
            stage="reasoning",
            duration_ms=duration,
            root_cause=result.get("root_cause"),
            confidence=result.get("confidence", 0.0)
        )
        
        return result
    
    async def _run_action(
        self,
        reasoning_result: dict[str, Any],
        anomaly_result: dict[str, Any],
        stage_durations: dict[str, float]
    ) -> dict[str, Any]:
        """Run action stage"""
        stage_start = asyncio.get_event_loop().time()
        
        self.logger.info("stage_started", stage="action")
        
        # Prepare action input
        action_data = {
            "root_cause": reasoning_result.get("root_cause", "Unknown"),
            "confidence": reasoning_result.get("confidence", 0.0),
            "recommendations": reasoning_result.get("recommendations", []),
            "anomaly_data": anomaly_result,
            "severity": self._determine_severity(
                reasoning_result.get("confidence", 0.0),
                anomaly_result
            )
        }
        
        result = await self.action_agent.execute(action_data)
        
        stage_end = asyncio.get_event_loop().time()
        duration = (stage_end - stage_start) * 1000
        stage_durations["action"] = duration
        
        self.logger.info(
            "stage_completed",
            stage="action",
            duration_ms=duration,
            alerts_sent=len(result.get("alerts_sent", [])),
            actions_executed=len(result.get("actions_executed", []))
        )
        
        return result
    
    def _determine_severity(
        self,
        confidence: float,
        anomaly_result: dict[str, Any]
    ) -> str:
        """
        Determine alert severity based on confidence and anomaly data.
        
        WHY: Severity affects notification channels and urgency.
        High confidence + severe anomaly = critical alert
        """
        anomaly_score = anomaly_result.get("anomaly_score", 0.0)
        
        if confidence >= 0.9 and anomaly_score >= 0.8:
            return "critical"
        elif confidence >= 0.75 and anomaly_score >= 0.6:
            return "error"
        elif confidence >= 0.6:
            return "warning"
        else:
            return "info"
    
    def get_metrics(self) -> dict[str, Any]:
        """Get coordinator metrics"""
        success_rate = 0.0
        if self._total_pipelines > 0:
            success_rate = (self._successful_pipelines / self._total_pipelines) * 100
        
        return {
            "coordinator": self.name,
            "initialized": self._initialized,
            "total_pipelines": self._total_pipelines,
            "successful_pipelines": self._successful_pipelines,
            "failed_pipelines": self._failed_pipelines,
            "success_rate": round(success_rate, 2),
            "agents": {
                "anomaly_detection": self.anomaly_agent.get_metrics(),
                "rag_memory": self.rag_agent.get_metrics(),
                "reasoning": self.reasoning_agent.get_metrics(),
                "action": self.action_agent.get_metrics()
            }
        }
    
    def is_healthy(self) -> bool:
        """
        Check if coordinator is healthy.
        
        Coordinator is healthy if:
        - Initialized
        - All agents are healthy
        """
        if not self._initialized:
            return False
        
        return all([
            self.anomaly_agent.is_healthy(),
            self.rag_agent.is_healthy(),
            self.reasoning_agent.is_healthy(),
            self.action_agent.is_healthy()
        ])
