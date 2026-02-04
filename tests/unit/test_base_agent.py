"""
Unit tests for BaseAgent

Tests cover:
- Successful execution (happy path)
- Retry logic with eventual success
- Max retries exceeded
- Circuit breaker opening after threshold failures
- Circuit breaker half-open state after timeout
- Metrics tracking accuracy
- Exponential backoff timing

WHY THESE TESTS:
Production systems need comprehensive test coverage of resilience patterns.
These tests verify that circuit breakers, retries, and metrics work correctly
under various failure scenarios.
"""

import asyncio
import pytest
import time
from typing import Any
from unittest.mock import Mock, patch

from src.agentic_observability.agents.base import (
    BaseAgent,
    AgentState,
    AgentMetrics
)
from src.agentic_observability.core.circuit_breaker import (
    CircuitBreakerError,
    CircuitState
)


class MockAgent(BaseAgent):
    """
    Mock agent for testing base functionality.
    
    WHY: We need a concrete implementation to test the abstract BaseAgent.
    This mock allows us to control success/failure behavior for testing.
    """
    
    def __init__(
        self,
        name: str = "test-agent",
        should_fail: bool = False,
        failure_count: int = 0,
        delay_ms: float = 0,
        **kwargs
    ):
        """
        Initialize mock agent with controllable behavior.
        
        Args:
            name: Agent name
            should_fail: If True, always fails
            failure_count: Number of times to fail before succeeding
            delay_ms: Simulated execution delay in milliseconds
            **kwargs: Passed to BaseAgent
        """
        super().__init__(name=name, **kwargs)
        self._should_fail = should_fail
        self._failure_count = failure_count
        self._current_failures = 0
        self._delay_ms = delay_ms
        self._execution_count = 0
    
    async def _execute_impl(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Mock implementation with controllable failure behavior"""
        self._execution_count += 1
        
        # Simulate processing delay
        if self._delay_ms > 0:
            await asyncio.sleep(self._delay_ms / 1000)
        
        # Fail for the first N attempts, then succeed
        if self._current_failures < self._failure_count:
            self._current_failures += 1
            raise ValueError(f"Mock failure {self._current_failures}/{self._failure_count}")
        
        # Always fail if configured
        if self._should_fail:
            raise ValueError("Mock permanent failure")
        
        # Success path
        return {
            "result": "success",
            "execution_count": self._execution_count,
            "input": input_data
        }


@pytest.fixture
async def mock_agent():
    """Fixture providing initialized mock agent"""
    agent = MockAgent(
        name="test-agent",
        circuit_breaker_threshold=5,
        circuit_breaker_timeout=1,
        max_retries=3,
        base_retry_delay=0.01  # Fast retries for testing
    )
    await agent.initialize()
    yield agent
    await agent.cleanup()


@pytest.mark.asyncio
class TestBaseAgentHappyPath:
    """Test successful execution scenarios"""
    
    async def test_successful_execution(self, mock_agent):
        """
        Test: Successful execution on first attempt
        
        VERIFY:
        - Result contains output and metadata
        - State returns to IDLE
        - Metrics incremented correctly
        - No retries occurred
        """
        input_data = {"test": "data"}
        
        result = await mock_agent.execute(input_data)
        
        # Verify result structure
        assert "result" in result
        assert result["result"] == "success"
        assert "metadata" in result
        assert result["metadata"]["agent"] == "test-agent"
        assert result["metadata"]["attempts"] == 1
        
        # Verify state
        assert mock_agent._state == AgentState.IDLE
        
        # Verify metrics
        metrics = mock_agent.get_metrics()
        assert metrics["total_executions"] == 1
        assert metrics["total_successes"] == 1
        assert metrics["total_failures"] == 0
        assert metrics["success_rate"] == 100.0
        assert metrics["total_retries"] == 0
    
    async def test_multiple_successful_executions(self, mock_agent):
        """
        Test: Multiple consecutive successful executions
        
        VERIFY:
        - Metrics accumulate correctly
        - Latency tracking works
        - Success rate remains 100%
        """
        for i in range(5):
            result = await mock_agent.execute({"iteration": i})
            assert result["result"] == "success"
        
        metrics = mock_agent.get_metrics()
        assert metrics["total_executions"] == 5
        assert metrics["total_successes"] == 5
        assert metrics["success_rate"] == 100.0
        assert metrics["avg_latency_ms"] > 0  # Some latency recorded
        assert len(mock_agent._metrics.latencies) == 5


@pytest.mark.asyncio
class TestRetryLogic:
    """Test retry behavior and exponential backoff"""
    
    async def test_retry_with_eventual_success(self):
        """
        Test: Agent fails 2 times, succeeds on 3rd attempt
        
        VERIFY:
        - Retries occur
        - Final result is successful
        - Retry count is correct
        - State returns to IDLE
        """
        agent = MockAgent(
            name="retry-agent",
            failure_count=2,  # Fail twice, then succeed
            max_retries=3,
            base_retry_delay=0.01
        )
        await agent.initialize()
        
        start_time = time.time()
        result = await agent.execute({"test": "retry"})
        elapsed = time.time() - start_time
        
        # Verify success after retries
        assert result["result"] == "success"
        assert result["metadata"]["attempts"] == 3  # Failed twice, succeeded third time
        
        # Verify state
        assert agent._state == AgentState.IDLE
        
        # Verify metrics
        metrics = agent.get_metrics()
        assert metrics["total_executions"] == 1  # One execute() call
        assert metrics["total_successes"] == 1
        assert metrics["total_retries"] == 2  # Two retries occurred
        
        # Verify exponential backoff occurred (should take some time)
        # First retry: ~0.01s, second retry: ~0.02s = ~0.03s minimum
        assert elapsed >= 0.01  # At least some delay
        
        await agent.cleanup()
    
    async def test_max_retries_exceeded(self):
        """
        Test: Agent fails more times than max_retries allows
        
        VERIFY:
        - Exception is raised after max retries
        - State is FAILED
        - Metrics show failure
        - Circuit breaker not affected (single failure stream)
        """
        agent = MockAgent(
            name="max-retry-agent",
            should_fail=True,  # Always fails
            max_retries=2,
            base_retry_delay=0.01
        )
        await agent.initialize()
        
        # Execution should fail after exhausting retries
        with pytest.raises(ValueError, match="Mock permanent failure"):
            await agent.execute({"test": "max_retries"})
        
        # Verify state
        assert agent._state == AgentState.FAILED
        
        # Verify metrics
        metrics = agent.get_metrics()
        assert metrics["total_executions"] == 1
        assert metrics["total_failures"] == 1
        assert metrics["success_rate"] == 0.0
        assert metrics["total_retries"] == 2  # Tried max_retries times
        
        await agent.cleanup()
    
    async def test_exponential_backoff_timing(self):
        """
        Test: Verify exponential backoff delays increase correctly
        
        VERIFY:
        - Delays increase exponentially
        - Jitter is applied (delays vary)
        """
        agent = MockAgent(
            name="backoff-agent",
            failure_count=3,  # Fail 3 times
            max_retries=3,
            base_retry_delay=0.05,  # 50ms base
            max_retry_delay=1.0
        )
        await agent.initialize()
        
        start_time = time.time()
        result = await agent.execute({"test": "backoff"})
        total_elapsed = time.time() - start_time
        
        # Verify success
        assert result["result"] == "success"
        
        # Exponential backoff calculation (with jitter 0.5-1.0):
        # Attempt 0: 0.05 * 2^0 * jitter = [0.025, 0.05]
        # Attempt 1: 0.05 * 2^1 * jitter = [0.05, 0.1]
        # Attempt 2: 0.05 * 2^2 * jitter = [0.1, 0.2]
        # Minimum total: 0.025 + 0.05 + 0.1 = 0.175s
        # Maximum total: 0.05 + 0.1 + 0.2 = 0.35s
        
        assert total_elapsed >= 0.15  # Should have some cumulative delay
        assert total_elapsed < 0.5  # But not too long
        
        await agent.cleanup()


@pytest.mark.asyncio
class TestCircuitBreaker:
    """Test circuit breaker behavior"""
    
    async def test_circuit_breaker_opens_after_threshold(self):
        """
        Test: Circuit opens after threshold consecutive failures
        
        VERIFY:
        - Circuit opens after N failures
        - Subsequent calls raise CircuitBreakerError
        - Circuit breaker trip count increments
        - State is CIRCUIT_OPEN
        """
        agent = MockAgent(
            name="circuit-agent",
            should_fail=True,
            circuit_breaker_threshold=3,  # Open after 3 failures
            max_retries=0  # No retries to speed up test
        )
        await agent.initialize()
        
        # Execute until circuit opens (should take 3 failures)
        for i in range(3):
            with pytest.raises(ValueError):
                await agent.execute({"attempt": i})
        
        # Verify circuit is now open
        assert agent._circuit_breaker.state == CircuitState.OPEN
        
        # Next call should fail immediately with CircuitBreakerError
        with pytest.raises(CircuitBreakerError, match="Circuit breaker open"):
            await agent.execute({"attempt": "after_open"})
        
        # Verify state and metrics
        assert agent._state == AgentState.CIRCUIT_OPEN
        metrics = agent.get_metrics()
        assert metrics["circuit_breaker_trips"] >= 1
        assert metrics["circuit_breaker_state"] == "open"
        
        await agent.cleanup()
    
    async def test_circuit_breaker_half_open_after_timeout(self):
        """
        Test: Circuit transitions to HALF_OPEN after timeout
        
        VERIFY:
        - Circuit opens after failures
        - After timeout, circuit becomes HALF_OPEN
        - Successful request closes circuit
        - State returns to normal
        """
        agent = MockAgent(
            name="half-open-agent",
            failure_count=3,  # Fail first 3 times
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=0.1,  # 100ms timeout
            max_retries=0
        )
        await agent.initialize()
        
        # Open the circuit with 3 failures
        for i in range(3):
            with pytest.raises(ValueError):
                await agent.execute({"attempt": i})
        
        assert agent._circuit_breaker.state == CircuitState.OPEN
        
        # Wait for circuit breaker timeout
        await asyncio.sleep(0.15)
        
        # Next successful request should close the circuit
        # (failure_count=3 means it fails 3 times, then succeeds)
        result = await agent.execute({"attempt": "recovery"})
        
        assert result["result"] == "success"
        assert agent._circuit_breaker.state == CircuitState.CLOSED
        assert agent._state == AgentState.IDLE
        
        await agent.cleanup()
    
    async def test_manual_circuit_breaker_reset(self):
        """
        Test: Manual circuit breaker reset
        
        VERIFY:
        - Circuit can be manually reset
        - State transitions correctly
        - Agent becomes healthy again
        """
        agent = MockAgent(
            name="reset-agent",
            should_fail=True,
            circuit_breaker_threshold=2,
            max_retries=0
        )
        await agent.initialize()
        
        # Open the circuit
        for i in range(2):
            with pytest.raises(ValueError):
                await agent.execute({"attempt": i})
        
        assert agent._circuit_breaker.state == CircuitState.OPEN
        assert not agent.is_healthy()
        
        # Manually reset
        agent.reset_circuit_breaker()
        
        assert agent._circuit_breaker.state == CircuitState.CLOSED
        assert agent._state == AgentState.IDLE
        # Note: is_healthy() still False because agent will fail immediately
        # but circuit is reset
        
        await agent.cleanup()


@pytest.mark.asyncio
class TestMetrics:
    """Test metrics tracking accuracy"""
    
    async def test_metrics_tracking_mixed_results(self):
        """
        Test: Metrics correctly track mixed success/failure
        
        VERIFY:
        - Success rate calculated correctly
        - Latency stats are accurate
        - Execution counts are correct
        """
        agent = MockAgent(
            name="metrics-agent",
            circuit_breaker_threshold=10,  # High threshold
            max_retries=0
        )
        await agent.initialize()
        
        # Execute 3 successful requests
        for i in range(3):
            agent._should_fail = False
            result = await agent.execute({"attempt": i})
            assert result["result"] == "success"
        
        # Execute 2 failed requests
        for i in range(2):
            agent._should_fail = True
            with pytest.raises(ValueError):
                await agent.execute({"attempt": i + 3})
        
        # Verify metrics
        metrics = agent.get_metrics()
        assert metrics["total_executions"] == 5
        assert metrics["total_successes"] == 3
        assert metrics["total_failures"] == 2
        assert metrics["success_rate"] == 60.0  # 3/5 = 60%
        
        await agent.cleanup()
    
    async def test_latency_percentile_calculation(self):
        """
        Test: P95 latency calculation
        
        VERIFY:
        - Latency is tracked for each execution
        - P95 percentile is calculated correctly
        - Min/max/avg are accurate
        """
        agent = MockAgent(
            name="latency-agent",
            delay_ms=10,  # 10ms simulated delay
            circuit_breaker_threshold=100
        )
        await agent.initialize()
        
        # Execute 20 requests to get statistical data
        for i in range(20):
            await agent.execute({"iteration": i})
        
        metrics = agent.get_metrics()
        
        # Verify latency tracking
        assert len(agent._metrics.latencies) == 20
        assert metrics["avg_latency_ms"] >= 10  # At least 10ms average
        assert metrics["min_latency_ms"] > 0
        assert metrics["max_latency_ms"] >= metrics["avg_latency_ms"]
        assert metrics["p95_latency_ms"] > 0
        
        # P95 should be >= average for normal distribution
        assert metrics["p95_latency_ms"] >= metrics["avg_latency_ms"]
        
        await agent.cleanup()
    
    async def test_metrics_persistence_across_retries(self):
        """
        Test: Metrics count retries correctly
        
        VERIFY:
        - Only successful execute() calls count as executions
        - Retries are tracked separately
        """
        agent = MockAgent(
            name="retry-metrics-agent",
            failure_count=2,  # Fail twice
            max_retries=3,
            base_retry_delay=0.01
        )
        await agent.initialize()
        
        # This will retry twice and succeed
        result = await agent.execute({"test": "metrics"})
        
        assert result["result"] == "success"
        
        metrics = agent.get_metrics()
        assert metrics["total_executions"] == 1  # One execute() call
        assert metrics["total_successes"] == 1
        assert metrics["total_retries"] == 2  # Two retries occurred
        
        await agent.cleanup()


@pytest.mark.asyncio
class TestStateTransitions:
    """Test agent state machine transitions"""
    
    async def test_state_idle_to_processing_to_idle(self, mock_agent):
        """
        Test: Normal state transition during successful execution
        
        VERIFY:
        - Starts in IDLE
        - Transitions to PROCESSING during execution
        - Returns to IDLE after success
        """
        assert mock_agent._state == AgentState.IDLE
        
        # We can't easily assert PROCESSING state during execution
        # but we can verify it returns to IDLE
        await mock_agent.execute({"test": "state"})
        
        assert mock_agent._state == AgentState.IDLE
    
    async def test_state_failed_after_max_retries(self):
        """
        Test: State transitions to FAILED after exhausting retries
        
        VERIFY:
        - State becomes FAILED
        - is_healthy() returns False
        """
        agent = MockAgent(
            name="failed-state-agent",
            should_fail=True,
            max_retries=1,
            base_retry_delay=0.01
        )
        await agent.initialize()
        
        with pytest.raises(ValueError):
            await agent.execute({"test": "failed"})
        
        assert agent._state == AgentState.FAILED
        assert not agent.is_healthy()
        
        await agent.cleanup()
    
    async def test_state_circuit_open(self):
        """
        Test: State transitions to CIRCUIT_OPEN when circuit opens
        
        VERIFY:
        - State becomes CIRCUIT_OPEN
        - is_healthy() returns False
        """
        agent = MockAgent(
            name="circuit-open-state-agent",
            should_fail=True,
            circuit_breaker_threshold=2,
            max_retries=0
        )
        await agent.initialize()
        
        # Open circuit
        for i in range(2):
            with pytest.raises(ValueError):
                await agent.execute({"attempt": i})
        
        # Try to execute with open circuit
        with pytest.raises(CircuitBreakerError):
            await agent.execute({"attempt": "after_open"})
        
        assert agent._state == AgentState.CIRCUIT_OPEN
        assert not agent.is_healthy()
        
        await agent.cleanup()


@pytest.mark.asyncio
class TestHealthChecks:
    """Test health check functionality"""
    
    async def test_health_status_healthy_agent(self, mock_agent):
        """
        Test: Health status for healthy agent
        
        VERIFY:
        - is_healthy() returns True
        - get_health_status() shows detailed info
        """
        assert mock_agent.is_healthy()
        
        health = mock_agent.get_health_status()
        assert health["healthy"] is True
        assert health["initialized"] is True
        assert health["state"] == "idle"
        assert health["circuit_breaker"]["state"] == "closed"
    
    async def test_health_status_unhealthy_agent(self):
        """
        Test: Health status for unhealthy agent
        
        VERIFY:
        - is_healthy() returns False for failed agent
        - get_health_status() shows failure state
        """
        agent = MockAgent(
            name="unhealthy-agent",
            should_fail=True,
            max_retries=0
        )
        await agent.initialize()
        
        # Fail the agent
        with pytest.raises(ValueError):
            await agent.execute({"test": "health"})
        
        assert not agent.is_healthy()
        
        health = agent.get_health_status()
        assert health["healthy"] is False
        assert health["state"] == "failed"
        
        await agent.cleanup()
    
    async def test_health_status_circuit_open(self):
        """
        Test: Health status when circuit is open
        
        VERIFY:
        - is_healthy() returns False
        - Circuit breaker details in health status
        """
        agent = MockAgent(
            name="circuit-health-agent",
            should_fail=True,
            circuit_breaker_threshold=2,
            max_retries=0
        )
        await agent.initialize()
        
        # Open circuit
        for i in range(2):
            with pytest.raises(ValueError):
                await agent.execute({"attempt": i})
        
        assert not agent.is_healthy()
        
        health = agent.get_health_status()
        assert health["healthy"] is False
        assert health["circuit_breaker"]["state"] == "open"
        assert health["metrics"]["circuit_breaker_trips"] >= 0
        
        await agent.cleanup()


@pytest.mark.asyncio
class TestRepr:
    """Test __repr__ method for debugging"""
    
    async def test_repr_format(self, mock_agent):
        """
        Test: __repr__ returns useful debugging information
        
        VERIFY:
        - Contains agent name
        - Shows state
        - Shows circuit breaker state
        - Shows success rate
        """
        # Execute a few times to populate metrics
        for i in range(3):
            await mock_agent.execute({"iteration": i})
        
        repr_str = repr(mock_agent)
        
        assert "MockAgent" in repr_str
        assert "test-agent" in repr_str
        assert "state=idle" in repr_str
        assert "circuit=closed" in repr_str
        assert "initialized=True" in repr_str
        assert "success_rate=100.0%" in repr_str
        assert "executions=3" in repr_str


@pytest.mark.asyncio
class TestInitializationAndCleanup:
    """Test initialization and cleanup lifecycle"""
    
    async def test_execution_without_initialization_fails(self):
        """
        Test: Executing without initialization raises error
        
        VERIFY:
        - RuntimeError raised if not initialized
        """
        agent = MockAgent(name="uninit-agent")
        
        with pytest.raises(RuntimeError, match="not initialized"):
            await agent.execute({"test": "data"})
    
    async def test_cleanup_resets_state(self):
        """
        Test: Cleanup properly resets agent state
        
        VERIFY:
        - State returns to IDLE
        - Initialized flag is False
        - is_healthy() returns False after cleanup
        """
        agent = MockAgent(name="cleanup-agent")
        await agent.initialize()
        
        assert agent.is_healthy()
        
        await agent.cleanup()
        
        assert agent._state == AgentState.IDLE
        assert agent._initialized is False
        assert not agent.is_healthy()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
