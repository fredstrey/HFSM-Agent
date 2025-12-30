"""
Decorator para registrar funções como tools
"""
import inspect
from typing import Callable, Optional, Any, get_type_hints
from functools import wraps
from pydantic import BaseModel, create_model, Field
from .registry import ToolRegistry


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None
):
    """
    Decorator para registrar uma função como tool
    
    Usage:
        @tool(name="search", description="Busca documentos")
        def search_docs(query: str, limit: int = 3) -> SearchResult:
            ...
    
    Args:
        name: Nome da tool (usa nome da função se não especificado)
        description: Descrição da tool (usa docstring se não especificado)
    """
    def decorator(func: Callable) -> Callable:
        # Nome da tool
        tool_name = name or func.__name__
        
        # Descrição da tool
        tool_description = description or (func.__doc__ or "").strip()
        
        # Extrai type hints
        type_hints = get_type_hints(func)
        sig = inspect.signature(func)
        
        # Cria schema Pydantic automaticamente
        fields = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            param_type = type_hints.get(param_name, Any)
            param_default = param.default if param.default != inspect.Parameter.empty else ...
            
            # Extrai descrição do docstring se possível
            param_description = f"Parameter {param_name}"
            
            fields[param_name] = (param_type, Field(default=param_default, description=param_description))
        
        # Cria modelo Pydantic dinamicamente
        ArgsModel = create_model(
            f"{tool_name.capitalize()}Args",
            **fields
        )
        
        # Tipo de retorno
        return_type = type_hints.get('return', Any)
        
        # Registra a tool
        registry = ToolRegistry()
        registry.register(
            name=tool_name,
            description=tool_description,
            function=func,
            args_model=ArgsModel,
            return_type=return_type
        )
        
        # Wrapper que mantém a função original
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Adiciona metadados
        wrapper._tool_name = tool_name
        wrapper._tool_description = tool_description
        wrapper._args_model = ArgsModel
        
        return wrapper
    
    return decorator
