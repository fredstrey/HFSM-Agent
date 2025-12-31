"""
Async Execution Context
========================

Thread-safe execution context using asyncio.Lock instead of threading.RLock.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio


class AsyncExecutionContext(BaseModel):
    """
    Async execution context with asyncio.Lock for concurrency safety.
    
    All mutation methods are async and use lock for thread safety.
    """
    # Original user query
    user_query: str = Field(..., description="Original user query")
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        # Async lock (not serialized)
        object.__setattr__(self, '_lock', asyncio.Lock())
    
    # Memory for arbitrary data storage
    memory: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary memory")
    
    # Tool calls history
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Tool calls history")
    
    # Execution state
    current_iteration: int = Field(default=0, description="Current iteration")
    max_iterations: int = Field(default=3, description="Maximum iterations")
    
    # Timestamp & Metrics
    timestamp: datetime = Field(default_factory=datetime.now, description="Execution timestamp")
    start_time: datetime = Field(default_factory=datetime.now, description="Start time")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")

    async def add_tool_call(self, tool_name: str, arguments: Dict[str, Any], result: Any = None):
        """Add a tool call to the history (async-safe)"""
        async with self._lock:
            self.tool_calls.append({
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
                "iteration": self.current_iteration,
                "timestamp": datetime.now().isoformat()
            })

    async def set_memory(self, key: str, value: Any):
        """Store value in memory (async-safe)"""
        async with self._lock:
            self.memory[key] = value

    async def get_memory(self, key: str, default: Any = None) -> Any:
        """Retrieve value from memory (async-safe)"""
        async with self._lock:
            return self.memory.get(key, default)

    def snapshot(self) -> Dict[str, Any]:
        """
        Create a serializable snapshot (sync method for compatibility).
        """
        return {
            "user_query": self.user_query,
            "memory": {k: str(v) for k, v in self.memory.items()},
            "tool_calls": self.tool_calls,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "timestamp": self.timestamp.isoformat(),
            "start_time": self.start_time.isoformat(),
            "metrics": self.metrics
        }
