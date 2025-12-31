from typing import Dict, Any, List, Callable, Optional, Type
from pydantic import BaseModel

class ToolRegistry:
    """Registry to store and manage tool metadata"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ToolRegistry, cls).__new__(cls)
            cls._instance.tools = {}
        return cls._instance
    
    def register(
        self,
        name: str,
        description: str,
        function: Callable,
        args_model: Type[BaseModel],
        return_type: Any = Any
    ):
        """Register a tool"""
        self.tools[name] = {
            "name": name,
            "description": description,
            "function": function,
            "args_model": args_model,
            "return_type": return_type
        }

    def register_tool(self, tool_func: Callable):
        """Register a tool decorated with @tool"""
        if not hasattr(tool_func, "_tool_name"):
            raise ValueError(f"Function {tool_func.__name__} must be decorated with @tool")
            
        self.register(
            name=tool_func._tool_name,
            description=tool_func._tool_description,
            function=tool_func,
            args_model=tool_func._args_model
        )
    
    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """Get tool metadata by name"""
        return self.tools.get(name)
    
    def get_tool(self, name: str):
        """Get tool object for execution (returns object with .function attribute)"""
        tool_data = self.tools.get(name)
        if not tool_data:
            return None
        
        # Return object with function attribute for executor
        class ToolWrapper:
            def __init__(self, data):
                self.function = data["function"]
                self.name = data["name"]
                self.description = data["description"]
        
        return ToolWrapper(tool_data)
    
    def list(self) -> List[str]:
        """List all tool names"""
        return list(self.tools.keys())
    
    def to_openai_format(self) -> List[Dict[str, Any]]:
        """Convert registry to OpenAI function calling format"""
        openai_tools = []
        for name, tool in self.tools.items():
            schema = tool["args_model"].model_json_schema()
            
            # Remove title from properties as it's not needed by OpenAI
            if "properties" in schema:
                for prop in schema["properties"].values():
                    prop.pop("title", None)
            
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool["description"],
                    "parameters": {
                        "type": "object",
                        "properties": schema.get("properties", {}),
                        "required": schema.get("required", [])
                    }
                }
            })
        return openai_tools
