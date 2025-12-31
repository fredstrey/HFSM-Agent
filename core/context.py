from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import threading

class ExecutionContext(BaseModel):
    """
    Generic execution context that maintains state during processing.
    
    Thread-safe: All mutation methods use internal lock.
    """
    # Original user query
    user_query: str = Field(..., description="Original user query")
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        # Thread safety lock (not serialized)
        object.__setattr__(self, '_lock', threading.RLock())
    
    # Memory for arbitrary data storage (e.g., intermediate results, settings)
    memory: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary memory for execution")
    
    # Tool calls history
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Tool calls history")
    
    # Execution state
    current_iteration: int = Field(default=0, description="Current iteration")
    max_iterations: int = Field(default=3, description="Maximum iterations")
    
    # Timestamp & Metrics
    timestamp: datetime = Field(default_factory=datetime.now, description="Execution timestamp")
    start_time: datetime = Field(default_factory=datetime.now, description="Start execution time")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")

    def add_tool_call(self, tool_name: str, arguments: Dict[str, Any], result: Any = None):
        """Add a tool call to the history (thread-safe)"""
        with self._lock:
            self.tool_calls.append({
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
                "iteration": self.current_iteration,
                "timestamp": datetime.now().isoformat()
            })

    def set_memory(self, key: str, value: Any):
        """Store value in memory (thread-safe)"""
        with self._lock:
            self.memory[key] = value

    def get_memory(self, key: str, default: Any = None) -> Any:
        """Retrieve value from memory (thread-safe)"""
        with self._lock:
            return self.memory.get(key, default)

    def snapshot(self) -> Dict[str, Any]:
        """
        Create a serializable snapshot of the current context.
        Manually constructs dict to absolutely ensure no non-serializable objects (like LLMClient) are included.
        """
        # Manual construction is safer than model_dump when we have mixed types in memory
        data = {
            "user_query": self.user_query,
            "tool_calls": self.tool_calls,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "metrics": self.metrics,
            "memory": {} 
        }

        unsafe_keys = ["llm", "registry", "executor", "thread_pool", "agent"]
        
        if self.memory:
            for k, v in self.memory.items():
                if k not in unsafe_keys:
                    data["memory"][k] = v
                    
        return data

    @classmethod
    def load_from_snapshot(cls, snapshot: Dict[str, Any]) -> 'ExecutionContext':
        """
        Create an ExecutionContext instance from a snapshot dictionary.
        """
        return cls(**snapshot)
