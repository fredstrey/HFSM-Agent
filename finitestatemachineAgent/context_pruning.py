"""
Async Context Pruning
======================

Context pruning utilities for managing token limits in long conversations.
Prevents context window overflow by truncating old tool results.
"""

import logging
from typing import Optional
from core.context_async import AsyncExecutionContext

logger = logging.getLogger("ContextPruning")


class AsyncContextPruner:
    """
    Async context pruner - truncates old tool call results to prevent token overflow.
    
    Strategy: Keep full results for recent tool calls, truncate older ones.
    """
    
    def __init__(
        self, 
        strategy: str = "cut_last_n",
        keep_recent: int = 4,
        max_length: int = 200
    ):
        """
        Initialize async context pruner.
        
        Args:
            strategy: Pruning strategy (currently only "cut_last_n" supported)
            keep_recent: Number of recent tool calls to keep full results (default: 4)
            max_length: Max length for truncated results (default: 200 chars)
        """
        self.strategy = strategy
        self.keep_recent = keep_recent
        self.max_length = max_length
    
    async def prune(self, context: AsyncExecutionContext):
        """
        Prune context to prevent token overflow.
        
        Creates a pruned view of tool calls in memory as "active_tool_calls".
        Original tool_calls remain intact for AnswerState.
        
        Args:
            context: Execution context to prune
        """
        if not context.tool_calls:
            return
        
        if self.strategy == "cut_last_n":
            total_calls = len(context.tool_calls)
            pruned_calls = []
            
            for i, call in enumerate(context.tool_calls):
                # Check if this is a "recent" call
                is_recent = (total_calls - i) <= self.keep_recent
                
                # Create a shallow copy to modify result for display/prompt only
                call_copy = call.copy()
                
                raw_result = str(call.get("result", ""))
                if not is_recent and len(raw_result) > self.max_length:
                    call_copy["result"] = raw_result[:self.max_length] + "... [TRUNCATED - OLD CONTEXT]"
                
                pruned_calls.append(call_copy)
            
            logger.info(f"✂️ [Pruner] Original calls: {len(context.tool_calls)} | Pruned view: {len(pruned_calls)} | Recent (full): {self.keep_recent}")
            await context.set_memory("active_tool_calls", pruned_calls)
