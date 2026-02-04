"""
Retry Logic with Exponential Backoff

Provides resilient retry mechanisms for transient failures
with exponential backoff and jitter.
"""

import asyncio
import random
from typing import Any, Callable, Optional, Type, Tuple
import structlog

logger = structlog.get_logger()


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted"""
    pass


async def retry_with_backoff(
    func: Callable,
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    name: str = "operation",
    **kwargs: Any
) -> Any:
    """
    Retry async function with exponential backoff.
    
    Args:
        func: Async function to retry
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff calculation
        jitter: Add random jitter to prevent thundering herd
        exceptions: Tuple of exceptions to catch and retry
        name: Operation name for logging
        **kwargs: Keyword arguments for func
        
    Returns:
        Function result
        
    Raises:
        RetryExhaustedError: If all retries are exhausted
    """
    retry_logger = logger.bind(operation=name)
    last_exception: Optional[Exception] = None
    
    for attempt in range(max_retries + 1):
        try:
            result = await func(*args, **kwargs)
            
            if attempt > 0:
                retry_logger.info(
                    "retry_succeeded",
                    attempt=attempt,
                    total_attempts=max_retries + 1
                )
            
            return result
            
        except exceptions as e:
            last_exception = e
            
            if attempt >= max_retries:
                retry_logger.error(
                    "retry_exhausted",
                    attempts=attempt + 1,
                    error=str(e)
                )
                raise RetryExhaustedError(
                    f"Failed after {attempt + 1} attempts: {str(e)}"
                ) from e
            
            # Calculate delay with exponential backoff
            delay = min(base_delay * (exponential_base ** attempt), max_delay)
            
            # Add jitter to prevent thundering herd
            if jitter:
                delay = delay * (0.5 + random.random())
            
            retry_logger.warning(
                "retry_attempt",
                attempt=attempt + 1,
                total_attempts=max_retries + 1,
                delay=f"{delay:.2f}s",
                error=str(e)
            )
            
            await asyncio.sleep(delay)
    
    # Should never reach here, but for type safety
    if last_exception:
        raise last_exception
    raise RetryExhaustedError("Retry logic failed unexpectedly")


class RetryConfig:
    """Configuration for retry behavior"""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
