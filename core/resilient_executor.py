"""
Resilient Tool Executor
========================

Wraps tool executor with circuit breaker for fault tolerance.
"""

from typing import Any, Dict
from core.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from core.executor import IToolExecutor
import logging

logger = logging.getLogger(__name__)


class ResilientToolExecutor(IToolExecutor):
    """
    Tool executor with circuit breaker protection.
    
    Prevents cascading failures by blocking calls to repeatedly failing tools.
    Each tool has its own circuit breaker.
    
    Example:
        >>> base_executor = ToolExecutor(registry)
        >>> cb = CircuitBreaker(threshold=5, timeout=60.0)
        >>> resilient_executor = ResilientToolExecutor(base_executor, cb)
        >>> result = resilient_executor.execute("my_tool", {"arg": "value"})
    """
    
    def __init__(self, executor: IToolExecutor, circuit_breaker: CircuitBreaker):
        """
        Initialize resilient executor.
        
        Args:
            executor: Base tool executor
            circuit_breaker: Circuit breaker instance
        """
        self.executor = executor
        self.circuit_breaker = circuit_breaker
    
    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute tool through circuit breaker.
        
        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments
            
        Returns:
            Tool execution result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open for this tool
            Exception: Original exception from tool
        """
        try:
            # Execute through circuit breaker
            result = self.circuit_breaker.call(
                self.executor.execute,
                tool_name,
                arguments
            )
            return result
            
        except CircuitBreakerOpenError as e:
            logger.error(f"Circuit breaker open for tool '{tool_name}': {e}")
            # Return error result instead of raising
            return {
                "success": False,
                "error": f"Tool '{tool_name}' temporarily unavailable (circuit breaker open)",
                "circuit_breaker_state": "open"
            }
        except Exception as e:
            logger.error(f"Tool '{tool_name}' execution failed: {e}")
            raise
