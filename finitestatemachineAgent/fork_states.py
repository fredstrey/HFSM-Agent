"""
Research Fork State
====================

Specialized state for fork execution that bypasses the full Router logic.
Forks execute with minimal context and a focused research goal.
"""

import logging
import traceback
import json
from abc import ABC, abstractmethod
from typing import Optional
from finitestatemachineAgent.transition import Transition

logger = logging.getLogger("AsyncAgentEngine")

# Import base classes to avoid circular dependency
from core.context_async import AsyncExecutionContext

class AsyncHierarchicalState(ABC):
    """Base class for async hierarchical states."""
    def __init__(self, parent: Optional['AsyncHierarchicalState'] = None):
        self.parent = parent
    
    @abstractmethod
    async def handle(self, context: AsyncExecutionContext):
        pass
    
    def find_state_by_type(self, type_name: str):
        """Traverse hierarchy to find state."""
        if self.parent:
            return self.parent.find_state_by_type(type_name)
        raise Exception(f"State provider for {type_name} not found")



class ResearchForkState(AsyncHierarchicalState):
    """
    Entry point for fork execution.
    
    Instead of going through RouterState (which would add unnecessary LLM calls),
    forks start here with a focused research goal.
    
    Flow: ResearchForkState -> ToolState (if tools needed) -> ForkSummaryState
    """
    def __init__(self, parent, llm, registry, tool_choice=None):
        super().__init__(parent)
        self.llm = llm
        self.registry = registry
        self.tool_choice = tool_choice
    
    async def handle(self, context: AsyncExecutionContext):
        # Get branch goal from memory (set by ForkDispatchState)
        branch_goal = await context.get_memory("branch_goal", "")
        branch_id = await context.get_memory("branch_id", "unknown")
        
        if not branch_goal:
            logger.error(f"❌ [ResearchFork:{branch_id}] No branch goal found")
            return Transition(to="ForkSummaryState", reason="Missing branch goal")
        
        # 🔥 Enhanced logging for observability
        logger.info("=" * 70)
        logger.info(f"🔬 [ResearchFork:{branch_id}] STARTING FORK EXECUTION")
        logger.info(f"   📋 Task: {branch_goal}")
        logger.info(f"   🎯 Branch ID: {branch_id}")
        logger.info("=" * 70)
        
        # Build focused research prompt
        system_instruction = """You are a research worker executing a specific research task.

Your job:
1. Analyze the research goal
2. Select appropriate tools to gather information
3. Execute efficiently

Do NOT ask questions. Do NOT engage in conversation. Focus on the task."""
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Research goal: {branch_goal}"}
        ]
        
        # Safety check
        await context.increment_llm_call()
        
        # Call LLM with tools
        response = await self.llm.chat_with_tools(
            messages=messages,
            tools=self.registry.to_openai_format(),
            tool_choice=self.tool_choice
        )
        
        # Track usage
        usage = response.get('usage', {})
        if usage:
            await context.accumulate_usage(usage)
        
        # Check for tool calls
        if response.get("tool_calls"):
            logger.info(f"🔧 [ResearchFork:{branch_id}] {len(response['tool_calls'])} tool(s) selected")
            
            # Store tool calls
            for call in response["tool_calls"]:
                await context.add_tool_call(
                    tool_name=call["function"]["name"],
                    arguments=json.loads(call["function"]["arguments"]),
                    result=None
                )
            
            # Execute tools
            return Transition(to="ToolState", reason="Tools selected for research")
        
        # No tools needed - store direct research notes
        elif response.get("content"):
            logger.info(f"📝 [ResearchFork:{branch_id}] Direct research notes provided")
            await context.set_memory("research_notes", response["content"])
            return Transition(to="ForkContractState", reason="Direct research completed")
        
        else:
            logger.warning(f"⚠️ [ResearchFork:{branch_id}] No tools or content generated")
            return Transition(to="ForkContractState", reason="Empty research result")


class ForkContractState(AsyncHierarchicalState):
    """
    Extracts structured contracts from fork execution results via Strategy.
    """
    def __init__(self, parent, llm, contract_strategy=None):
        super().__init__(parent)
        self.llm = llm
        
        # Default strategy: Epistemic (current behavior)
        if contract_strategy:
            self.contract_strategy = contract_strategy
        else:
            from finitestatemachineAgent.contract_strategies import EpistemicContractStrategy
            self.contract_strategy = EpistemicContractStrategy(llm)
    
    async def handle(self, context: AsyncExecutionContext):
        branch_id = await context.get_memory("branch_id", "unknown")
        branch_goal = await context.get_memory("branch_goal", "")
        
        logger.info(f"📊 [ForkContract] Extracting claims using {type(self.contract_strategy).__name__}...")
        
        try:
            # Delegate to strategy
            fork_result = await self.contract_strategy.extract(context, branch_id, branch_goal)
            
            # Store contract
            await context.set_memory("fork_contract", fork_result.model_dump())
            
            logger.info(f"✅ [ForkContract:{branch_id}] Contract created via strategy")
            if fork_result.claims:
                 logger.info(f"   Claims: {len(fork_result.claims)}")
            
        except Exception as e:
            logger.error(f"❌ [ForkContract] Strategy failed: {e!r}")
            logger.error(traceback.format_exc())
            # Fallback (Empty Result)
            from finitestatemachineAgent.fork_contracts import ForkResult
            empty_result = ForkResult(branch_id=branch_id, confidence=0.0)
            await context.set_memory("fork_contract", empty_result.model_dump())
        
        # Terminal state for fork
        return None


# Keep old name as alias for backward compatibility
ForkSummaryState = ForkContractState
