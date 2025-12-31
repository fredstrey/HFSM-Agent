from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import ValidationError
from .registry import ToolRegistry

class IToolExecutor(ABC):
    """Interface for tool execution strategies."""
    
    @abstractmethod
    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return structured result."""
        pass

class ToolExecutor(IToolExecutor):
    """Safely executes tools registered in the ToolRegistry"""
    
    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.registry = registry or ToolRegistry()
    
    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with automatic validation
        
        Returns:
            Dict containing success status, result or error, and tool name
        """
        tool_metadata = self.registry.get(tool_name)
        if not tool_metadata:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "tool_name": tool_name
            }
        
        function = tool_metadata["function"]
        args_model = tool_metadata["args_model"]
        
        try:
            # Validate arguments with Pydantic
            validated_args = args_model(**arguments)
            
            # Execute function
            result = function(**validated_args.model_dump())
            
            return {
                "success": True,
                "result": result,
                "tool_name": tool_name
            }
            
        except ValidationError as e:
            return {
                "success": False,
                "error": f"Validation error: {str(e)}",
                "tool_name": tool_name
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution error: {str(e)}",
                "tool_name": tool_name
            }
