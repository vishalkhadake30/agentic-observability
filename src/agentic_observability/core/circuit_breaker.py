"""
Circuit Breaker Implementation

Prevents cascading failures by stopping calls to failing services.
Based on the Circuit Breaker pattern with CLOSED, OPEN, HALF_OPEN states.
"""

import asyncio
import time
from enum import Enum
from typing import Any, Callable, Optional
import structlog

logger = structlog.get_logger()


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit is open"""
    pass


class CircuitBreaker:
    """
    Async circuit breaker for protecting external service calls.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        timeout: Seconds to wait before attempting half-open
        name: Circuit breaker identifier for logging
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        name: str = "default"
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.name = name
        self.logger = logger.bind(circuit_breaker=name)
        
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = CircuitState.CLOSED
    
    @property
    def state(self) -> CircuitState:
        """Current circuit state"""
        return self._state
    
    async def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
        """
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._state = CircuitState.HALF_OPEN
                self.logger.info("circuit_half_open", state="half_open")
            else:
                self.logger.warning("circuit_open", state="open")
                raise CircuitBreakerError(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self) -> None:
        """Handle successful call"""
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            self.logger.info("circuit_closed", state="closed")
        
        self._failure_count = 0
        self._last_failure_time = None
    
    def _on_failure(self) -> None:
        """Handle failed call"""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self.logger.warning(
                "circuit_opened",
                state="open",
                failure_count=self._failure_count
            )
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self._last_failure_time is None:
            return False
        return (time.time() - self._last_failure_time) >= self.timeout
    
    def reset(self) -> None:
        """Manually reset circuit breaker"""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self.logger.info("circuit_reset", state="closed")
