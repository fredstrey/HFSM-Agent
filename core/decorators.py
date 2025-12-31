import inspect
from typing import Callable, Optional, Any, get_type_hints, Type
from functools import wraps
from pydantic import BaseModel, create_model, Field
from .registry import ToolRegistry

def tool(
    name: Optional[str] = None,
    description: Optional[str] = None
):
    """
    Decorator to register a function as a tool in the ReactAgent framework
    
    Args:
        name: Name of the tool (uses function name if not specified)
        description: Description of the tool (uses docstring if not specified)
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_description = description or (func.__doc__ or "").strip()
        
        # Extract type hints and signature
        type_hints = get_type_hints(func)
        sig = inspect.signature(func)
        
        # Create Pydantic schema automatically for arguments
        fields = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            param_type = type_hints.get(param_name, Any)
            param_default = param.default if param.default != inspect.Parameter.empty else ...
            
            # Use basic description
            param_description = f"Parameter {param_name}"
            
            fields[param_name] = (param_type, Field(default=param_default, description=param_description))
        
        # Create Pydantic model dynamically
        ArgsModel = create_model(
            f"{tool_name.capitalize()}Args",
            **fields
        )
        
        # Return type
        return_type = type_hints.get('return', Any)
        
        # Register the tool in the singleton registry
        registry = ToolRegistry()
        registry.register(
            name=tool_name,
            description=tool_description,
            function=func,
            args_model=ArgsModel,
            return_type=return_type
        )
        
        # Wrapper to maintain original function behavior and attach metadata
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
            wrapper._is_async = True
        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            wrapper._is_async = False
        
        wrapper._tool_name = tool_name
        wrapper._tool_description = tool_description
        wrapper._args_model = ArgsModel
        
        return wrapper
    
    return decorator
