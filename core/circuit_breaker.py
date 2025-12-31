"""
Circuit Breaker Pattern
========================

Prevents cascading failures by temporarily blocking calls to failing services.
"""

import time
import threading
from enum import Enum
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Prevents cascading failures by:
    1. Counting consecutive failures
    2. Opening circuit after threshold
    3. Blocking calls while open
    4. Testing recovery after timeout
    
    Example:
        >>> cb = CircuitBreaker(threshold=3, timeout=60.0)
        >>> result = cb.call(risky_function, arg1, arg2)
    """
    
    def __init__(self, threshold: int = 5, timeout: float = 60.0, name: str = "default"):
        """
        Initialize circuit breaker.
        
        Args:
            threshold: Number of failures before opening circuit
            timeout: Seconds before attempting to close circuit
            name: Identifier for logging
        """
        self.threshold = threshold
        self.timeout = timeout
        self.name = name
        
        self.failures = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self._lock = threading.RLock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Original exception from func
        """
        with self._lock:
            # Check if circuit should transition
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.timeout:
                    logger.info(f"[CircuitBreaker:{self.name}] Transitioning to HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Retry after {self.timeout - (time.time() - self.last_failure_time):.1f}s"
                    )
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            
            # Success - reset or close circuit
            with self._lock:
                if self.state == CircuitState.HALF_OPEN:
                    logger.info(f"[CircuitBreaker:{self.name}] Transitioning to CLOSED")
                    self.state = CircuitState.CLOSED
                self.failures = 0
            
            return result
            
        except Exception as e:
            # Failure - increment counter and maybe open circuit
            with self._lock:
                self.failures += 1
                self.last_failure_time = time.time()
                
                if self.failures >= self.threshold:
                    if self.state != CircuitState.OPEN:
                        logger.warning(
                            f"[CircuitBreaker:{self.name}] Opening circuit after "
                            f"{self.failures} failures"
                        )
                        self.state = CircuitState.OPEN
            
            raise
    
    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        with self._lock:
            self.failures = 0
            self.last_failure_time = None
            self.state = CircuitState.CLOSED
            logger.info(f"[CircuitBreaker:{self.name}] Manually reset to CLOSED")
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self.state
    
    def get_failures(self) -> int:
        """Get current failure count."""
        with self._lock:
            return self.failures
