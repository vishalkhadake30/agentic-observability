"""
Base Agent Abstract Class

Production-grade base agent with circuit breaker, retry logic, and comprehensive metrics.

WHY THIS ARCHITECTURE:
- Circuit breaker prevents cascading failures across dependent services (learned from eBay's 
  platform stability incidents where a single service degradation took down entire clusters)
- Exponential backoff prevents thundering herd when services recover
- Metrics provide observability into agent health (you can't fix what you can't measure)
- State management enables graceful degradation and recovery

PRODUCTION PATTERNS IMPLEMENTED:
- Circuit Breaker: Based on Netflix Hystrix pattern for fault tolerance
- Exponential Backoff: Inspired by AWS SDK retry strategies
- Metrics Collection: Following Prometheus/OpenTelemetry best practices
- State Machine: Finite state machine pattern for predictable behavior
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from enum import Enum
import asyncio
import time
import random
from dataclasses import dataclass, field
import structlog

from ..core.circuit_breaker import CircuitBreaker, CircuitBreakerError, CircuitState
from ..core.metrics import get_metrics_collector

logger = structlog.get_logger()


class AgentState(Enum):
    """
    Agent execution states for state machine.
    
    WHY: Explicit state tracking prevents race conditions and enables proper 
    graceful degradation. In production, you need to know if an agent failed 
    vs. is waiting for retry vs. circuit is open.
    """
    IDLE = "idle"
    PROCESSING = "processing"
    FAILED = "failed"
    CIRCUIT_OPEN = "circuit_open"
    RECOVERING = "recovering"


@dataclass
class AgentMetrics:
    """
    Comprehensive metrics for agent performance tracking.
    
    WHY: In production observability platforms, every component must be measurable.
    These metrics answer critical questions:
    - Is the agent healthy? (success_rate)
    - Is it performant? (avg_latency, p95_latency)
    - Is it failing silently? (total_failures, circuit_breaker_trips)
    
    PATTERN: Follows Prometheus metric naming conventions and Google SRE practices
    """
    total_executions: int = 0
    total_successes: int = 0
    total_failures: int = 0
    circuit_breaker_trips: int = 0
    total_retries: int = 0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    latencies: list[float] = field(default_factory=list)  # Python 3.11+ type hint
    
    @property
    def success_rate(self) -> float:
        """Success rate as percentage (0-100)"""
        if self.total_executions == 0:
            return 0.0
        return (self.total_successes / self.total_executions) * 100
    
    @property
    def p95_latency_ms(self) -> float:
        """
        95th percentile latency in milliseconds.
        
        WHY P95: P95 is more meaningful than average for SLAs. It tells you 
        what 95% of users experience, filtering out worst-case outliers.
        """
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx] if idx < len(sorted_latencies) else sorted_latencies[-1]


class BaseAgent(ABC):
    """
    Production-grade base agent with resilience patterns.
    
    DESIGN PHILOSOPHY:
    This class embodies production lessons learned from building large-scale distributed systems:
    
    1. CIRCUIT BREAKER: After seeing cascading failures take down entire platforms when a 
       downstream service degrades, circuit breakers are non-negotiable. They prevent retry 
       storms and give failing services breathing room to recover.
    
    2. EXPONENTIAL BACKOFF: Thundering herd problems are real. When 1000 agents retry 
       simultaneously, you just DDoS your own infrastructure. Exponential backoff with 
       jitter spreads the load.
    
    3. METRICS EVERYWHERE: "It worked on my machine" doesn't fly in production. Every 
       execution must be measured, logged, and traceable. This enables data-driven 
       troubleshooting and capacity planning.
    
    4. GRACEFUL DEGRADATION: Systems should degrade gracefully, not catastrophically. 
       Agent state management allows the system to continue operating with reduced 
       functionality rather than complete failure.
    
    USAGE:
        class MyAgent(BaseAgent):
            async def _execute_impl(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                # Your agent logic here
                return {"result": "processed"}
        
        agent = MyAgent(
            name="my-agent",
            circuit_breaker_threshold=5,
            circuit_breaker_timeout=60,
            max_retries=3
        )
        await agent.initialize()
        result = await agent.execute({"data": "test"})
    """
    
    def __init__(
        self,
        name: str,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: int = 60,
        max_retries: int = 3,
        base_retry_delay: float = 1.0,
        max_retry_delay: float = 60.0,
        config: Optional[dict[str, Any]] = None  # Python 3.11+ type hint
    ) -> None:
        """
        Initialize base agent with resilience patterns.
        
        CIRCUIT BREAKER PATTERN (Netflix Hystrix):
        When a downstream service fails repeatedly, stop sending requests to give it 
        time to recover. This prevents cascading failures and retry storms.
        
        EXPONENTIAL BACKOFF (AWS SDK Pattern):
        Retry with increasing delays: 1s, 2s, 4s, 8s... This prevents thundering herd 
        when services recover and gives systems breathing room.
        
        Args:
            name: Agent identifier (used in logs and metrics)
            circuit_breaker_threshold: Failures before circuit opens (default: 5)
            circuit_breaker_timeout: Seconds before attempting circuit recovery (default: 60)
            max_retries: Maximum retry attempts (default: 3)
            base_retry_delay: Initial retry delay in seconds (default: 1.0)
            max_retry_delay: Maximum retry delay in seconds (default: 60.0)
            config: Optional agent-specific configuration
        """
        self.name: str = name
        self.config: dict[str, Any] = config or {}
        self.logger = logger.bind(agent=name)
        
        # Circuit breaker for fault isolation (Netflix Hystrix pattern)
        self._circuit_breaker: CircuitBreaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            timeout=circuit_breaker_timeout,
            name=f"{name}_circuit"
        )
        
        # Retry configuration (AWS SDK exponential backoff pattern)
        self._max_retries: int = max_retries
        self._base_retry_delay: float = base_retry_delay
        self._max_retry_delay: float = max_retry_delay
        
        # State management (Finite State Machine pattern)
        self._state: AgentState = AgentState.IDLE
        self._initialized: bool = False
        
        # Metrics tracking (Prometheus/OpenTelemetry best practices)
        self._metrics: AgentMetrics = AgentMetrics()
        self._metrics_collector = get_metrics_collector()
        
        self.logger.info(
            "agent_initialized",
            circuit_threshold=circuit_breaker_threshold,
            circuit_timeout=circuit_breaker_timeout,
            max_retries=max_retries
        )
    
    async def initialize(self) -> None:
        """
        Initialize agent resources.
        
        Override this in subclasses to set up:
        - Database connections
        - ML model loading
        - External service clients
        - Caches
        
        WHY SEPARATE INIT: Allows for async initialization and proper resource 
        management. Constructor should be fast; initialization can be slow.
        """
        self.logger.info("initializing_agent")
        self._initialized = True
        self._state = AgentState.IDLE
    
    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Execute agent with full resilience patterns.
        
        This method wraps the actual implementation (_execute_impl) with:
        - Circuit breaker protection (Netflix Hystrix pattern)
        - Retry logic with exponential backoff (AWS SDK pattern)
        - Comprehensive metrics collection (Prometheus best practices)
        - State management (Finite State Machine pattern)
        
        WHY THIS APPROACH: Separating orchestration (this method) from implementation 
        (_execute_impl) allows subclasses to focus on business logic while getting 
        production patterns for free.
        
        Args:
            input_data: Input data for agent processing
            
        Returns:
            Dict containing agent output and metadata
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If all retries exhausted
        """
        if not self._initialized:
            self.logger.error("agent_not_initialized", agent=self.name)
            raise RuntimeError(f"Agent {self.name} not initialized. Call initialize() first.")
        
        # PATTERN: Circuit Breaker (Netflix Hystrix)
        # Check circuit breaker state before attempting execution
        if self._circuit_breaker.state == CircuitState.OPEN:
            # Check if circuit should transition to half-open
            if self._circuit_breaker._should_attempt_reset():
                self._circuit_breaker._state = CircuitState.HALF_OPEN
                self.logger.info(
                    "circuit_half_open",
                    agent=self.name,
                    state="half_open"
                )
            else:
                self._state = AgentState.CIRCUIT_OPEN
                self._metrics.circuit_breaker_trips += 1
                
                self.logger.warning(
                    "circuit_breaker_open",
                    agent=self.name,
                    total_trips=self._metrics.circuit_breaker_trips,
                    circuit_timeout=self._circuit_breaker.timeout
                )
                
                raise CircuitBreakerError(f"Circuit breaker open for agent {self.name}")
        
        start_time = time.time()
        last_exception: Optional[Exception] = None
        
        self.logger.info(
            "agent_execution_started",
            agent=self.name,
            state=self._state.value,
            input_size=len(str(input_data))
        )
        
        # PATTERN: Retry with Exponential Backoff (AWS SDK)
        # Retry loop with exponential backoff and jitter
        for attempt in range(self._max_retries + 1):
            try:
                self._state = AgentState.PROCESSING
                
                self.logger.debug(
                    "executing_agent",
                    attempt=attempt + 1,
                    max_attempts=self._max_retries + 1,
                    state=self._state.value
                )
                
                # Execute with circuit breaker protection
                result = await self._circuit_breaker.call(
                    self._execute_impl,
                    input_data
                )
                
                # Success path
                self._state = AgentState.IDLE
                self._record_success(start_time)
                
                self.logger.info(
                    "agent_execution_completed",
                    agent=self.name,
                    attempts=attempt + 1,
                    latency_ms=(time.time() - start_time) * 1000
                )
                
                return {
                    **result,
                    "metadata": {
                        "agent": self.name,
                        "attempts": attempt + 1,
                        "latency_ms": (time.time() - start_time) * 1000,
                        "state": self._state.value
                    }
                }
                
            except Exception as e:
                last_exception = e
                self.logger.warning(
                    "agent_execution_failed",
                    attempt=attempt + 1,
                    max_attempts=self._max_retries + 1,
                    error=str(e),
                    error_type=type(e).__name__,
                    will_retry=(attempt < self._max_retries)
                )
                
                # If we have more retries, calculate backoff
                if attempt < self._max_retries:
                    self._metrics.total_retries += 1
                    delay = self._calculate_backoff_delay(attempt)
                    
                    self.logger.info(
                        "retrying_agent_execution",
                        retry_delay_ms=delay * 1000,
                        attempt=attempt + 1,
                        next_attempt=attempt + 2,
                        backoff_formula=f"{self._base_retry_delay} * 2^{attempt}"
                    )
                    
                    self._state = AgentState.RECOVERING
                    await asyncio.sleep(delay)
                else:
                    # All retries exhausted
                    self._state = AgentState.FAILED
                    self._record_failure(start_time)
                    
                    self.logger.error(
                        "agent_execution_failed_permanently",
                        agent=self.name,
                        total_attempts=attempt + 1,
                        total_latency_ms=(time.time() - start_time) * 1000,
                        final_error=str(e)
                    )
                    
                    raise last_exception
        
        # Should never reach here, but for type safety
        self._state = AgentState.FAILED
        self._record_failure(start_time)
        raise last_exception or RuntimeError("Unknown execution failure")
    
    @abstractmethod
    async def _execute_impl(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Actual agent implementation - MUST be overridden by subclasses.
        
        This is where your agent's core logic goes. Focus on the business logic;
        resilience patterns are handled by the base class.
        
        PATTERN: Template Method Pattern
        The base class (execute) handles orchestration, subclasses implement specifics.
        
        Args:
            input_data: Input data for agent processing
            
        Returns:
            Dict containing agent output (without metadata)
            
        Example:
            async def _execute_impl(self, input_data: dict[str, Any]) -> dict[str, Any]:
                # Call external API
                result = await self.external_service.process(input_data)
                # Transform data
                return {"output": result}
        """
        pass
    
    async def cleanup(self) -> None:
        """
        Cleanup agent resources.
        
        Override this to properly close:
        - Database connections
        - File handles
        - External service clients
        
        WHY: Proper cleanup prevents resource leaks in long-running systems.
        """
        self.logger.info("cleaning_up_agent")
        self._initialized = False
        self._state = AgentState.IDLE
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff with jitter.
        
        PATTERN: Exponential Backoff with Jitter (AWS SDK)
        
        WHY JITTER: Without jitter, all retrying agents wake up simultaneously,
        creating a thundering herd. Jitter spreads the retry attempts over time.
        
        Formula: min(max_delay, base_delay * 2^attempt) * random(0.5, 1.0)
        
        Example progression (base=1.0, no jitter):
        - Attempt 0: 1s
        - Attempt 1: 2s
        - Attempt 2: 4s
        - Attempt 3: 8s
        
        With jitter, these become ranges: [0.5s-1s], [1s-2s], [2s-4s], [4s-8s]
        
        Args:
            attempt: Current retry attempt number (0-indexed)
            
        Returns:
            Delay in seconds with jitter applied
        """
        # Exponential component: base_delay * 2^attempt
        delay = min(
            self._max_retry_delay,
            self._base_retry_delay * (2 ** attempt)
        )
        
        # Add jitter: random between 50% and 100% of calculated delay
        # This prevents thundering herd when multiple agents retry simultaneously
        jitter = random.uniform(0.5, 1.0)
        final_delay = delay * jitter
        
        self.logger.debug(
            "backoff_delay_calculated",
            attempt=attempt,
            base_delay=delay,
            jitter_multiplier=jitter,
            final_delay=final_delay
        )
        
        return final_delay
    
    def _record_success(self, start_time: float) -> None:
        """
        Record successful execution metrics.
        
        PATTERN: Metrics Collection (Prometheus/OpenTelemetry)
        Every execution should be measured for observability.
        
        Args:
            start_time: Execution start timestamp
        """
        latency_ms = (time.time() - start_time) * 1000
        
        self._metrics.total_executions += 1
        self._metrics.total_successes += 1
        self._metrics.latencies.append(latency_ms)
        
        # Update latency stats
        self._metrics.min_latency_ms = min(self._metrics.min_latency_ms, latency_ms)
        self._metrics.max_latency_ms = max(self._metrics.max_latency_ms, latency_ms)
        self._metrics.avg_latency_ms = sum(self._metrics.latencies) / len(self._metrics.latencies)
        
        # Record to OpenTelemetry for external monitoring (Prometheus, Grafana, etc.)
        self._metrics_collector.record_agent_latency(self.name, latency_ms / 1000)
        self._metrics_collector.increment_agent_requests(self.name, status="success")
        
        self.logger.info(
            "agent_execution_success",
            latency_ms=round(latency_ms, 2),
            total_executions=self._metrics.total_executions,
            success_rate=round(self._metrics.success_rate, 2),
            avg_latency_ms=round(self._metrics.avg_latency_ms, 2),
            p95_latency_ms=round(self._metrics.p95_latency_ms, 2)
        )
    
    def _record_failure(self, start_time: float) -> None:
        """
        Record failed execution metrics.
        
        PATTERN: Metrics Collection (Prometheus/OpenTelemetry)
        Failures must be tracked separately for alerting and SLO monitoring.
        
        Args:
            start_time: Execution start timestamp
        """
        latency_ms = (time.time() - start_time) * 1000
        
        self._metrics.total_executions += 1
        self._metrics.total_failures += 1
        
        # Record to OpenTelemetry for external monitoring
        self._metrics_collector.increment_agent_errors(self.name, error_type="execution_failed")
        self._metrics_collector.increment_agent_requests(self.name, status="failure")
        
        self.logger.error(
            "agent_execution_failed",
            latency_ms=round(latency_ms, 2),
            total_executions=self._metrics.total_executions,
            total_failures=self._metrics.total_failures,
            success_rate=round(self._metrics.success_rate, 2),
            failure_rate=round(100 - self._metrics.success_rate, 2)
        )
    
    def is_healthy(self) -> bool:
        """
        Health check for agent.
        
        WHY: Kubernetes and other orchestrators need health checks to know when 
        to restart or remove unhealthy instances. This follows the Kubernetes 
        liveness probe pattern.
        
        Returns:
            True if agent is healthy and ready to accept requests
        """
        is_healthy = (
            self._initialized 
            and self._state != AgentState.FAILED
            and self._circuit_breaker.state != CircuitState.OPEN
        )
        
        if not is_healthy:
            self.logger.warning(
                "agent_unhealthy",
                initialized=self._initialized,
                state=self._state.value,
                circuit_state=self._circuit_breaker.state.value
            )
        
        return is_healthy
    
    def get_health_status(self) -> dict[str, Any]:
        """
        Get detailed health status including circuit breaker state.
        
        WHY: Detailed health status helps ops teams diagnose issues quickly.
        This is more informative than a simple boolean health check.
        
        PATTERN: Health Check Endpoint (Kubernetes Readiness Probe)
        
        Returns:
            Dict with detailed health information including:
            - Overall health status
            - Agent state
            - Circuit breaker state
            - Initialization status
            - Recent success rate
        """
        is_healthy = self.is_healthy()
        
        health_status = {
            "healthy": is_healthy,
            "agent": self.name,
            "state": self._state.value,
            "initialized": self._initialized,
            "circuit_breaker": {
                "state": self._circuit_breaker.state.value,
                "failure_count": self._circuit_breaker._failure_count,
                "threshold": self._circuit_breaker.failure_threshold,
                "timeout_seconds": self._circuit_breaker.timeout
            },
            "metrics": {
                "success_rate": round(self._metrics.success_rate, 2),
                "total_executions": self._metrics.total_executions,
                "circuit_breaker_trips": self._metrics.circuit_breaker_trips
            },
            "timestamp": time.time()
        }
        
        self.logger.debug("health_status_checked", **health_status)
        
        return health_status
    
    def get_metrics(self) -> dict[str, Any]:
        """
        Get current agent metrics.
        
        WHY: Enables runtime inspection and debugging. Critical for production 
        troubleshooting and capacity planning.
        
        PATTERN: Metrics Endpoint (Prometheus /metrics pattern)
        
        Returns:
            Dict with comprehensive metrics
        """
        metrics = {
            "agent": self.name,
            "state": self._state.value,
            "circuit_breaker_state": self._circuit_breaker.state.value,
            "total_executions": self._metrics.total_executions,
            "total_successes": self._metrics.total_successes,
            "total_failures": self._metrics.total_failures,
            "success_rate": round(self._metrics.success_rate, 2),
            "circuit_breaker_trips": self._metrics.circuit_breaker_trips,
            "total_retries": self._metrics.total_retries,
            "avg_latency_ms": round(self._metrics.avg_latency_ms, 2),
            "min_latency_ms": round(self._metrics.min_latency_ms, 2) if self._metrics.min_latency_ms != float('inf') else 0,
            "max_latency_ms": round(self._metrics.max_latency_ms, 2),
            "p95_latency_ms": round(self._metrics.p95_latency_ms, 2),
            "is_healthy": self.is_healthy()
        }
        
        self.logger.debug("metrics_retrieved", agent=self.name)
        
        return metrics
    
    def reset_circuit_breaker(self) -> None:
        """
        Manually reset circuit breaker.
        
        WHY: Sometimes ops needs to manually recover a circuit (e.g., after 
        fixing a downstream service). This should be used sparingly and only 
        after confirming the underlying issue is resolved.
        
        PATTERN: Manual Circuit Breaker Reset (Netflix Hystrix Dashboard)
        """
        old_state = self._circuit_breaker.state.value
        self._circuit_breaker.reset()
        
        # Reset agent state to IDLE when circuit is manually reset
        if self._state == AgentState.CIRCUIT_OPEN or self._state == AgentState.FAILED:
            self._state = AgentState.IDLE
        
        self.logger.warning(
            "circuit_breaker_manually_reset",
            agent=self.name,
            old_state=old_state,
            new_state=self._circuit_breaker.state.value,
            reason="manual_operator_intervention"
        )
        
        if self._state == AgentState.CIRCUIT_OPEN:
            self._state = AgentState.IDLE
            self.logger.info(
                "agent_state_transitioned",
                from_state="circuit_open",
                to_state=self._state.value
            )
    
    def __repr__(self) -> str:
        """
        String representation for debugging.
        
        WHY: Makes debugging easier when agents are printed in logs or debuggers.
        Shows critical state at a glance.
        
        Returns:
            Readable string representation of agent
        """
        return (
            f"<{self.__class__.__name__}("
            f"name='{self.name}', "
            f"state={self._state.value}, "
            f"circuit={self._circuit_breaker.state.value}, "
            f"initialized={self._initialized}, "
            f"success_rate={self._metrics.success_rate:.1f}%, "
            f"executions={self._metrics.total_executions}"
            f")>"
        )
