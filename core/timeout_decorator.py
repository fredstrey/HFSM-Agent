"""
Timeout Decorator
=================

Adds timeout support to functions using threading.
"""

import threading
import functools
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Raised when function execution exceeds timeout."""
    pass


def with_timeout(timeout_seconds: float):
    """
    Decorator to add timeout to a function.
    
    Uses threading to enforce timeout. If function doesn't complete
    within timeout_seconds, raises TimeoutError.
    
    Example:
        >>> @with_timeout(5.0)
        ... def slow_function():
        ...     time.sleep(10)
        ...     return "done"
        >>> 
        >>> slow_function()  # Raises TimeoutError after 5s
    
    Args:
        timeout_seconds: Maximum execution time in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            result = [None]
            exception = [None]
            
            def target():
                """Target function for thread."""
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            # Start function in thread
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            
            # Wait for completion or timeout
            thread.join(timeout_seconds)
            
            # Check if timed out
            if thread.is_alive():
                logger.error(
                    f"Function '{func.__name__}' exceeded timeout of {timeout_seconds}s"
                )
                raise TimeoutError(
                    f"Function '{func.__name__}' execution exceeded {timeout_seconds}s timeout"
                )
            
            # Check for exception
            if exception[0]:
                raise exception[0]
            
            return result[0]
        
        return wrapper
    return decorator


def with_configurable_timeout(get_timeout: Callable[[], float]):
    """
    Decorator with dynamic timeout from callable.
    
    Useful when timeout should come from configuration.
    
    Example:
        >>> config = AgentConfig()
        >>> @with_configurable_timeout(lambda: config.tool_timeout)
        ... def my_function():
        ...     pass
    
    Args:
        get_timeout: Callable that returns timeout in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            timeout = get_timeout()
            return with_timeout(timeout)(func)(*args, **kwargs)
        return wrapper
    return decorator
