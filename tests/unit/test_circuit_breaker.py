"""
Sample test for circuit breaker functionality
"""

import pytest
import asyncio
from src.agentic_observability.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState
)


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_failures():
    """Test that circuit opens after threshold failures"""
    breaker = CircuitBreaker(failure_threshold=3, timeout=1, name="test")
    
    async def failing_function():
        raise ValueError("Test failure")
    
    # First 3 failures should work but increment counter
    for _ in range(3):
        with pytest.raises(ValueError):
            await breaker.call(failing_function)
    
    # Circuit should now be open
    assert breaker.state == CircuitState.OPEN
    
    # Next call should raise CircuitBreakerError
    with pytest.raises(CircuitBreakerError):
        await breaker.call(failing_function)


@pytest.mark.asyncio
async def test_circuit_breaker_resets_on_success():
    """Test that circuit resets after successful call in half-open state"""
    breaker = CircuitBreaker(failure_threshold=2, timeout=0.1, name="test")
    
    async def sometimes_failing(should_fail: bool):
        if should_fail:
            raise ValueError("Failure")
        return "success"
    
    # Trigger failures to open circuit
    for _ in range(2):
        with pytest.raises(ValueError):
            await breaker.call(sometimes_failing, True)
    
    assert breaker.state == CircuitState.OPEN
    
    # Wait for timeout
    await asyncio.sleep(0.2)
    
    # Successful call should reset to closed
    result = await breaker.call(sometimes_failing, False)
    assert result == "success"
    assert breaker.state == CircuitState.CLOSED
