"""
Sample test for retry logic
"""

import pytest
import asyncio
from src.agentic_observability.core.retry import (
    retry_with_backoff,
    RetryExhaustedError
)


@pytest.mark.asyncio
async def test_retry_succeeds_after_failures():
    """Test that retry eventually succeeds after transient failures"""
    call_count = 0
    
    async def sometimes_failing():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Transient failure")
        return "success"
    
    result = await retry_with_backoff(
        sometimes_failing,
        max_retries=3,
        base_delay=0.01,
        name="test"
    )
    
    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_exhausted_after_max_attempts():
    """Test that retry raises error after max attempts"""
    async def always_failing():
        raise ValueError("Permanent failure")
    
    with pytest.raises(RetryExhaustedError):
        await retry_with_backoff(
            always_failing,
            max_retries=2,
            base_delay=0.01,
            name="test"
        )


@pytest.mark.asyncio
async def test_retry_with_specific_exceptions():
    """Test that retry only catches specified exceptions"""
    async def failing_with_runtime_error():
        raise RuntimeError("Should not be retried")
    
    with pytest.raises(RuntimeError):
        await retry_with_backoff(
            failing_with_runtime_error,
            max_retries=2,
            base_delay=0.01,
            exceptions=(ValueError,),  # Only retry ValueError
            name="test"
        )
