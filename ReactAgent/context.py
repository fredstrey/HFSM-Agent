from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class ExecutionContext(BaseModel):
    """
    Generic execution context that maintains state during processing
    """
    # Original user query
    user_query: str = Field(..., description="Original user query")
    
    # Memory for arbitrary data storage (e.g., intermediate results, settings)
    memory: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary memory for execution")
    
    # Tool calls history
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Tool calls history")
    
    # Execution state
    current_iteration: int = Field(default=0, description="Current iteration")
    max_iterations: int = Field(default=3, description="Maximum iterations")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.now, description="Execution timestamp")

    def add_tool_call(self, tool_name: str, arguments: Dict[str, Any], result: Any = None):
        """Add a tool call to the history"""
        self.tool_calls.append({
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "iteration": self.current_iteration,
            "timestamp": datetime.now().isoformat()
        })

    def set_memory(self, key: str, value: Any):
        """Store value in memory"""
        self.memory[key] = value

    def get_memory(self, key: str, default: Any = None) -> Any:
        """Retrieve value from memory"""
        return self.memory.get(key, default)
