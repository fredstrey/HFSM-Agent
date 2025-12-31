"""
Async Tool Executor
===================

Executes tools concurrently using asyncio instead of threading.
"""

import asyncio
import logging
from typing import Any, Dict, List
from core.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AsyncToolExecutor:
    """
    Async tool executor using asyncio.gather for concurrency.
    
    Supports both async and sync tools:
    - Async tools: Called directly with await
    - Sync tools: Wrapped with asyncio.to_thread
    """
    
    def __init__(self, registry: ToolRegistry):
        """
        Initialize async executor.
        
        Args:
            registry: Tool registry
        """
        self.registry = registry
    
    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute single tool (async).
        
        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool not found
            Exception: Tool execution errors
        """
        tool = self.registry.get_tool(tool_name)
        
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in registry")
        
        logger.info(f"🛠️ [AsyncExecutor] Executing: {tool_name}")
        
        try:
            # Check if tool is async
            if getattr(tool.function, '_is_async', False):
                # Call async tool directly
                result = await tool.function(**arguments)
            else:
                # Wrap sync tool in thread pool
                result = await asyncio.to_thread(
                    tool.function,
                    **arguments
                )
            
            logger.info(f"✅ [AsyncExecutor] {tool_name} completed")
            return result
            
        except Exception as e:
            logger.error(f"❌ [AsyncExecutor] {tool_name} failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool_name
            }
    
    async def execute_parallel(self, tool_calls: List[Dict]) -> List[Any]:
        """
        Execute multiple tools concurrently.
        
        Args:
            tool_calls: List of dicts with 'tool_name' and 'arguments'
            
        Returns:
            List of results in same order as input
        """
        logger.info(f"🚀 [AsyncExecutor] Executing {len(tool_calls)} tools concurrently")
        
        # Create tasks for all tools
        tasks = [
            self.execute(call['tool_name'], call['arguments'])
            for call in tool_calls
        ]
        
        # Execute all concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error dicts
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "tool_name": tool_calls[i]['tool_name']
                })
            else:
                processed_results.append(result)
        
        logger.info(f"✅ [AsyncExecutor] All {len(tool_calls)} tools completed")
        return processed_results
