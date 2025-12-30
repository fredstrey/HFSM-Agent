"""
Framework modular para function calling com decorators
"""

__version__ = "1.0.0"

from .core.decorators import tool
from .core.registry import ToolRegistry
from .core.executor import ToolExecutor
from .agents.function_agent import FunctionAgent

__all__ = [
    "tool",
    "ToolRegistry",
    "ToolExecutor",
    "FunctionAgent",
]
