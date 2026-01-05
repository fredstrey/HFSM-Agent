"""
Async HFSM Agent
================

Async version of the Hierarchical Finite State Machine Agent.
All states and transitions are async for better concurrency.
"""

from __future__ import annotations

import json
import logging
import asyncio
from datetime import datetime
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, List, Dict, Any, Literal

from pydantic import BaseModel, Field
from core.context_async import AsyncExecutionContext, SafetyMonitor
from core.executor_async import AsyncToolExecutor
from finitestatemachineAgent.transition import Transition
from finitestatemachineAgent.fork_states import ResearchForkState, ForkSummaryState, ForkContractState
from finitestatemachineAgent.fork_contracts import ForkResult, MergedContract, UncertainTopic
from finitestatemachineAgent.context_pruning import AsyncContextPruner

# Setup logging
logger = logging.getLogger("AsyncAgentEngine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    # Stream Handler (Console)
    sh = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    # File Handler (logs/agent.log)
    import os
    if not os.path.exists("logs"):
        os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler("logs/agent.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

class DummyLogger:
    """No-op logger for disabled logging."""
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

# =================================================================================================
# Domain Models for Parallel Execution
# =================================================================================================

class BranchSpec(BaseModel):
    """Specification for a parallel execution branch."""
    id: str
    goal: str
    constraints: List[str] = Field(default_factory=list)

class ParallelPlan(BaseModel):
    """Plan for parallel execution strategy."""
    strategy: Literal["single", "parallel_research"]
    branches: List[BranchSpec] = Field(default_factory=list)
    merge_policy: str = "append"

# =================================================================================================
# Base Classes
# =================================================================================================

class AsyncHierarchicalState(ABC):
    """
    Base class for async states in the Hierarchical FSM.
    Enforces immutability via __slots__ to prevent accidental mutable state.
    """
    __slots__ = ("parent", "logger")
    
    def __init__(self, parent: Optional[AsyncHierarchicalState] = None, logger=None):
        self.parent = parent
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    async def handle(self, context: AsyncExecutionContext) -> Optional['Transition | AsyncHierarchicalState']:
        """
        Process context and return next state or transition.
        Can return:
        - Transition object (recommended for explicit semantics)
        - AsyncHierarchicalState instance (legacy support)
        - None (terminal state)
        """
        pass

    async def on_enter(self, context: AsyncExecutionContext):
        """Hook called when entering state."""
        pass

    async def on_exit(self, context: AsyncExecutionContext):
        """Hook called when exiting state."""
        pass


class AsyncContextPolicyState(AsyncHierarchicalState):
    """
    Middleware state that enforces context policies (pruning) before delegation.
    
    This state sits between root and reasoning layers, applying pruning
    whenever the state machine enters this layer.
    """
    __slots__ = ("pruner",)
    
    def __init__(
        self, 
        parent: Optional[AsyncHierarchicalState] = None,
        pruner: Optional[AsyncContextPruner] = None,
        logger=None
    ):
        """
        Initialize context policy state.
        
        Args:
            parent: Parent state in hierarchy
            pruner: Context pruner instance (creates default if None)
            logger: Custom logger
        """
        super().__init__(parent, logger=logger)
        self.pruner = pruner or AsyncContextPruner()
    
    async def on_enter(self, context: AsyncExecutionContext):
        """
        Apply pruning when entering this state layer.
        
        Args:
            context: Execution context
        """
        await self.pruner.prune(context)
        active_calls = await context.get_memory("active_tool_calls", [])
        if active_calls:
            self.logger.info(f"✂️ [Policy] Pruned context applied. Active tool calls: {len(active_calls)}")
    
    async def handle(self, context: AsyncExecutionContext):
        """
        Delegate to child states (no specific handling).
        
        Args:
            context: Execution context
            
        Returns:
            None (delegates to children)
        """
        return None  # Delegate to active child

    def find_state_by_type(self, type_name: str) -> AsyncHierarchicalState:
        """Traverse hierarchy to find state."""
        if self.parent:
            return self.parent.find_state_by_type(type_name)
        raise Exception(f"State provider for {type_name} not found")


# =================================================================================================
# Hierarchy States
# =================================================================================================

class AgentRootState(AsyncHierarchicalState):
    """Root of the state hierarchy."""
    def __init__(self, logger=None):
        super().__init__(parent=None, logger=logger)
        self.find_state_by_type = None  # Will be set by engine

    async def handle(self, context: AsyncExecutionContext):
        return None


class ContextPolicyState(AsyncHierarchicalState):
    """Middleware for context management."""
    async def handle(self, context: AsyncExecutionContext):
        return None


class ReasoningState(AsyncHierarchicalState):
    """Reasoning layer parent."""
    async def handle(self, context: AsyncExecutionContext):
        return None


class ExecutionState(AsyncHierarchicalState):
    """Execution layer parent."""
    async def handle(self, context: AsyncExecutionContext):
        return None


class RecoveryState(AsyncHierarchicalState):
    """Recovery layer parent."""
    async def handle(self, context: AsyncExecutionContext):
        return None


class TerminalState(AsyncHierarchicalState):
    """Terminal state parent."""
    async def handle(self, context: AsyncExecutionContext):
        return None


# =================================================================================================
# Operational States
# =================================================================================================


class RouterState(AsyncHierarchicalState):
    """
    Async router state - decides whether to call tools or answer.
    """
    def __init__(self, parent, llm, registry, tool_choice=None, enable_parallel=False, post_hook=None):
        super().__init__(parent)
        self.llm = llm
        self.registry = registry
        self.tool_choice = tool_choice  # Store tool_choice config
        self.enable_parallel = enable_parallel  # Store parallel execution flag
        self.post_hook = post_hook  # 🔥 Domain-specific hook

    async def handle(self, context: AsyncExecutionContext):
        self.logger.info("🧠 [Router] Thinking...")
        
        # 🔥 CRITICAL: Ensure IntentAnalysis runs BEFORE parallel planning check
        # This allows intent analysis to enhance the query before planning
        intent_analyzed = await context.get_memory("intent_analyzed", False)
        is_root = context.parent is None

        
        # Research context is injected ONLY in AnswerState for better control
        # - Check for parallel planning AFTER IntentAnalysis
        # Only if enabled, not checked yet, AND IS ROOT CONTEXT (no parents allowed to fork)
        parallel_checked = await context.get_memory("parallel_checked", False)
        
        if (self.enable_parallel and not parallel_checked and is_root):
            
            # Mark as checked so we don't loop back here endlessly
            await context.set_memory("parallel_checked", True)
            return Transition(to="ParallelPlanningState", reason="Parallel planning check required")

        # Build messages
        system_instruction = await context.get_memory("system_instruction", "")
        messages = [{"role": "system", "content": system_instruction}]

        # Add chat history
        # Skip history if intent was analyzed (query is already enhanced)
        # or if this is a fork (context pollution)
        intent_analyzed = await context.get_memory("intent_analyzed", False)
        is_root = context.parent is None
        
        should_include_history = is_root and not intent_analyzed
        
        history = await context.get_memory("chat_history", [])
        if history and should_include_history:
            self.logger.info(f"📜 [Router] Including {len(history)} history messages")
            messages.extend(history)
        elif history:
            self.logger.info("🧹 [Router] Skipping history (Intent analyzed or Fork)")

        messages.append({"role": "user", "content": context.user_query})
        
        # Add previous tool calls to context so LLM knows what was already executed
        if context.tool_calls:
            for call in context.tool_calls:
                # Add assistant message with tool call
                tool_call_id = f"call_{call['tool_name']}_{call.get('iteration', 0)}"
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": call["tool_name"],
                            "arguments": json.dumps(call["arguments"])
                        }
                    }]
                })
                
                # Add tool response
                if call.get("result") is not None:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": call["tool_name"],
                        "content": str(call["result"])
                    })

        # 🛡️ Safety Check Before LLM Call
        await context.increment_llm_call()

        # Call LLM with tools (async) - Use configured tool_choice
        response = await self.llm.chat_with_tools(
            messages=messages,
            tools=self.registry.to_openai_format(),
            tool_choice=self.tool_choice  # Use configured value
        )

        self.logger.info(f"📊 [Router] Token usage: {response.get('usage', {})}")
        
        # Track token usage in context
        usage = response.get('usage', {})
        if usage:
            await context.accumulate_usage(usage)
        
        # Debug: Log what the LLM returned
        if response.get("content"):
            self.logger.debug(f"🔍 [Router] LLM also returned content: {response['content'][:100]}...")

        # Check for tool calls
        if response.get("tool_calls"):
            self.logger.info(f"🔧 [Router] {len(response['tool_calls'])} tool(s) selected")

            # Store tool calls (async)
            for call in response["tool_calls"]:
                await context.add_tool_call(
                    tool_name=call["function"]["name"],
                    arguments=json.loads(call["function"]["arguments"]),
                    result=None
                )

            # Proceed to tool execution
            transition = Transition(to="ToolState", reason="Tools selected by LLM", metadata={"tool_count": len(response["tool_calls"])})
        else:
            self.logger.warning("[Router] No tool calls generated by LLM.")
            transition = Transition(to="AnswerState", reason="Direct answer generation")
        
        # 🔥 Apply post-hook if provided (domain-specific logic)
        if self.post_hook:
            hook_result = await self.post_hook(context, transition)
            # Only override if hook returns a new transition (None means "keep original")
            if hook_result is not None:
                transition = hook_result
        
        return transition


class ToolState(AsyncHierarchicalState):
    """
    Async tool execution state.
    """
    def __init__(self, parent, executor: AsyncToolExecutor, skip_validation: bool = True):
        super().__init__(parent)
        self.executor = executor
        self.skip_validation = skip_validation

    async def handle(self, context: AsyncExecutionContext):
        self.logger.info("🛠️ [Tool] Executing tools...")

        # Get pending tool calls
        pending = [c for c in context.tool_calls if c.get("result") is None]

        if not pending:
            self.logger.warning("⚠️ [Tool] No pending tool calls")
            return Transition(to="AnswerState", reason="No pending tool calls")

        # Execute all tools concurrently (async)
        results = await self.executor.execute_parallel(pending)

        # Update context atomically using public API
        await context.update_tool_results(pending, results)
        
        for call in pending:
            self.logger.info(f"✅ [Tool] Result for {call['tool_name']}: {str(call['result'])[:100]}...")

        # Decide next state based on validation config
        if self.skip_validation:
            self.logger.info("⏩ [Tool] Skipping validation (configured)")
            
            # 🔥 CRITICAL: Check if we're in a fork context
            branch_id = await context.get_memory("branch_id")
            if branch_id:
                # We're in a fork - go to ForkContractState
                self.logger.info(f"🌿 [Tool] Fork context detected (branch: {branch_id}), proceeding to contract extraction")
                return Transition(to="ForkContractState", reason="Fork execution complete")
            else:
                # Normal flow - go to AnswerState
                return Transition(to="AnswerState", reason="Tools executed, validation skipped")
        else:
            self.logger.info("🔍 [Tool] Proceeding to validation")
            return Transition(to="ValidationState", reason="Validation required")


class ValidationState(AsyncHierarchicalState):
    """
    Async validation state with customizable validation logic.
    """
    def __init__(self, parent, llm, validation_fn=None):
        super().__init__(parent)
        self.llm = llm
        self.validation_fn = validation_fn  # Custom validation function

    async def _validate(self, context: AsyncExecutionContext, tool_name: str, result: any) -> bool:
        """Internal validation logic."""
        # Use custom validation function if provided
        if self.validation_fn:
            is_valid = await self.validation_fn(context, tool_name, result)
            self.logger.info(f"✅ [Validation] Custom validation result: {is_valid}")
        else:
            # Default: simple LLM-based validation
            messages = [
                {"role": "system", "content": "Validate if the tool results answer the query."},
                {"role": "user", "content": f"Query: {context.user_query}"},
                {"role": "user", "content": f"Results: {json.dumps(result)}"}
            ]

            # 🛡️ Safety Check Before LLM Call
            await context.increment_llm_call()

            response = await self.llm.chat(messages)
            is_valid = "true" in response["content"].lower()
            self.logger.info(f"✅ [Validation] Default validation result: {is_valid}")
        
        return is_valid

    async def handle(self, context: AsyncExecutionContext):
        self.logger.info("🔍 [Validation] Checking data...")

        # Get last tool call atomically (thread-safe)
        async with context._lock:
            if not context.tool_calls:
                self.logger.warning("⚠️ [Validation] No tool calls to validate")
                return Transition(to="AnswerState", reason="No tools to validate")
            
            last_call = context.tool_calls[-1]
            tool_name = last_call.get("tool_name")
            result = last_call.get("result")
        
        # Validate
        is_valid = await self._validate(context, tool_name, result)
        
        if is_valid:
            self.logger.info("✅ [Validation] Data is valid")
            
            # 🔥 CRITICAL: Check if we're in a fork context
            branch_id = await context.get_memory("branch_id")
            if branch_id:
                # We're in a fork - go to ForkContractState
                self.logger.info(f"🌿 [Validation] Fork context detected (branch: {branch_id}), proceeding to contract extraction")
                return Transition(to="ForkContractState", reason="Validation passed in fork")
            else:
                # Normal flow - return to RouterState to allow more searches
                self.logger.info("🔄 [Validation] Returning to RouterState for potential additional searches")
                return Transition(to="RouterState", reason="Validation passed")
        else:
            self.logger.warning("⚠️ [Validation] Data is invalid")
            return Transition(to="RetryState", reason="Validation failed", metadata={"tool": tool_name})


class ParallelPlanningState(AsyncHierarchicalState):
    """
    Async parallel planning state with customizable planning logic.
    Decides whether to execute tools in parallel or sequentially.
    """
    def __init__(self, parent, llm, planner_fn=None, planning_system_prompt=None, max_branches=3):
        super().__init__(parent)
        self.llm = llm
        self.planner_fn = planner_fn  # Custom planner function
        self.planning_system_prompt = planning_system_prompt  # Custom/enhanced system prompt
        self.max_branches = max_branches  # - Max branches allowed
    

    
    async def _default_llm_plan(self, context: AsyncExecutionContext) -> ParallelPlan:
        """Default LLM-based planning logic with customizable system prompt."""
        
        # Default system prompt
        default_prompt = """You are a planning module for a tool execution system.

Analyze the user's request and decide if it should be split into independent parallel research branches.

Rules:
- Branches must be INDEPENDENT (no dependencies between them)
- Each branch should research a different aspect
- Only use parallel if it provides clear value
- Do NOT plan tool execution, only research goals
- MAX BRANCHES ALLOWED: {max_branches} (if you need more, prioritize top {max_branches}) 

If parallel is NOT useful, respond with:
{{"strategy": "single"}}

If parallel IS useful, respond with:
{{
  "strategy": "parallel_research",
  "branches": [
    {{"id": "branch_1", "goal": "Research X", "constraints": []}},
    {{"id": "branch_2", "goal": "Research Y", "constraints": []}}
  ],
  "merge_policy": "append"
}}

Respond with valid JSON only."""

        # Determine final system prompt
        if self.planning_system_prompt is None:
            # Use default system prompt formatted with limit
            system_prompt = default_prompt.format(max_branches=self.max_branches)
        elif callable(self.planning_system_prompt):
            # Callable: incremental enhancement
            # If the user uses a callable, we assume they WANT the base prompt instructions
            # so we format the default prompt first
            base_prompt = default_prompt.format(max_branches=self.max_branches)
            system_prompt = self.planning_system_prompt(base_prompt, context)
        else:
            # String: complete override
            # If user overrides, they are responsible for limits in instructions, 
            # but we will still enforce in code.
            system_prompt = self.planning_system_prompt

        # Construct messages properly (avoid str.format on JSON braces)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User query: {context.user_query}"}
        ]
        
        try:
            # 🛡️ Safety Check Before LLM Call
            await context.increment_llm_call()
            
            response = await self.llm.chat(messages)
            content = response["content"]
            
            # Extract JSON if wrapped in markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
                
            plan_data = json.loads(content.strip())
            
            # Parse into ParallelPlan
            if plan_data.get("strategy") == "parallel_research":
                branches = [
                    BranchSpec(
                        id=b["id"],
                        goal=b["goal"],
                        constraints=b.get("constraints", [])
                    )
                    for b in plan_data.get("branches", [])
                ]
                return ParallelPlan(
                    strategy="parallel_research",
                    branches=branches,
                    merge_policy=plan_data.get("merge_policy", "append")
                )
            else:
                return ParallelPlan(strategy="single")
        
        except Exception as e:
            self.logger.error(f"❌ [ParallelPlan] Planning failed: {e}")
            return ParallelPlan(strategy="single")  # Safe fallback
    
    async def handle(self, context: AsyncExecutionContext):
        self.logger.info("🧩 [ParallelPlan] Evaluating parallelization strategy...")
        
        # Use custom planner if provided
        if self.planner_fn:
            plan = await self.planner_fn(context)
            self.logger.info(f"✅ [ParallelPlan] Custom planner result: {plan.strategy}")
        else:
            # Default: LLM-based planning
            plan = await self._default_llm_plan(context)
            self.logger.info(f"✅ [ParallelPlan] LLM planner result: {plan.strategy}")
        
        # Validate plan
        if not isinstance(plan, ParallelPlan):
            self.logger.warning("⚠️ [ParallelPlan] Invalid plan type, falling back to single")
            return Transition(to="RouterState", reason="Invalid plan type", metadata={"fallback": True})
        
        # Store plan in context
        # Use model_dump() for Pydantic serialization safety
        await context.set_memory("parallel_plan", plan.model_dump())
        
        # Decide next state based on strategy
        if plan.strategy == "parallel_research" and plan.branches:
            self.logger.info(f"🔀 [ParallelPlan] Parallel execution with {len(plan.branches)} branches")
            return Transition(to="ForkDispatchState", reason="Parallel execution strategy selected", metadata={"branches": len(plan.branches)})
        else:
            self.logger.info("➡️ [ParallelPlan] Single path execution")
            # Mark as checked ensures Router continues
            return Transition(to="RouterState", reason="Single path strategy selected")


class ForkDispatchState(AsyncHierarchicalState):
    """
    Dispatches parallel execution branches using context forking.
    Each branch runs independently with its own context.
    """
    def __init__(self, parent, engine):
        super().__init__(parent)
        self.engine = engine  # Reference to engine for dispatch
    
    async def handle(self, context: AsyncExecutionContext):
        self.logger.info("🔀 [ForkDispatch] Starting parallel execution...")
        
        # Get plan from context
        plan_data = await context.get_memory("parallel_plan")
        if not plan_data:
            self.logger.warning("⚠️ [ForkDispatch] No valid plan, falling back")
            return Transition(to="RouterState", reason="Missing parallel plan")
        
        # Handle both dict (serialized) and object (in-memory) cases
        if isinstance(plan_data, dict):
            try:
                plan = ParallelPlan(**plan_data)
            except Exception as e:
                self.logger.error(f"❌ [ForkDispatch] Invalid plan data: {e}")
                return Transition(to="RouterState", reason="Invalid plan data")
        elif isinstance(plan_data, ParallelPlan):
            plan = plan_data
        else:
            self.logger.error(f"❌ [ForkDispatch] Unknown plan type: {type(plan_data)}")
            return Transition(to="RouterState", reason="Unknown plan type")
            
        if not plan.branches:
            self.logger.warning("⚠️ [ForkDispatch] No branches in plan, falling back")
            return Transition(to="RouterState", reason="Empty branches in plan")
        
        self.logger.info(f"🔀 [ForkDispatch] Spawning {len(plan.branches)} branches")
        
        # Create forked contexts
        fork_contexts = []
        for branch in plan.branches:
            fork_ctx = context.fork()
            await fork_ctx.set_memory("branch_id", branch.id)
            await fork_ctx.set_memory("branch_goal", branch.goal)
            await fork_ctx.set_memory("branch_constraints", branch.constraints)
            
            # Increment recursion depth
            current_depth = await context.get_memory("parallel_depth", 0)
            await fork_ctx.set_memory("parallel_depth", current_depth + 1)
            
            # Override user query with branch goal
            fork_ctx.user_query = f"{context.user_query}\n\nBranch goal: {branch.goal}"
            
            # - Detailed logging for observability
            self.logger.info(f"🌿 [ForkDispatch] Creating fork '{branch.id}':")
            self.logger.info(f"   📋 Task: {branch.goal}")
            self.logger.info(f"   📝 Query: {fork_ctx.user_query[:200]}..." if len(fork_ctx.user_query) > 200 else f"   📝 Query: {fork_ctx.user_query}")
            if branch.constraints:
                self.logger.info(f"   ⚠️  Constraints: {', '.join(branch.constraints)}")
            
            fork_contexts.append((branch.id, fork_ctx))
        
        # Execute all forks in parallel
        try:
            results = await asyncio.gather(
                *[self._execute_fork(branch_id, fork_ctx) for branch_id, fork_ctx in fork_contexts],
                return_exceptions=True
            )
            
            # Filter out failed forks
            successful_forks = []
            for (branch_id, fork_ctx), result in zip(fork_contexts, results):
                if isinstance(result, Exception):
                    self.logger.error(f"❌ [ForkDispatch] Branch {branch_id} failed: {result}")
                else:
                    successful_forks.append(fork_ctx)
                    self.logger.info(f"✅ [ForkDispatch] Branch {branch_id} completed")
            
            # - Accumulate token usage AND sources from all forks back to parent
            total_fork_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            all_fork_sources = []
            all_merged_tools = []
            
            for fork_ctx in successful_forks:
                # Tokens
                fork_usage = await fork_ctx.get_total_usage()
                total_fork_tokens["prompt_tokens"] += fork_usage.get("prompt_tokens", 0)
                total_fork_tokens["completion_tokens"] += fork_usage.get("completion_tokens", 0)
                total_fork_tokens["total_tokens"] += fork_usage.get("total_tokens", 0)
                
                # Sources
                fork_sources = await fork_ctx.get_memory("sources_used", [])
                if fork_sources:
                    for source in fork_sources:
                        if source not in all_fork_sources:
                            all_fork_sources.append(source)

                # Tool Calls (for merged_tool_calls)
                if fork_ctx.tool_calls:
                    all_merged_tools.extend(fork_ctx.tool_calls)
            
            # Save merged tool calls for LegalAI source extraction
            await context.set_memory("merged_tool_calls", all_merged_tools)
            
            # Update parent context with accumulated tokens
            current_tokens = await context.get_memory("total_usage", {})
            new_tokens = {
                "prompt_tokens": current_tokens.get("prompt_tokens", 0) + total_fork_tokens["prompt_tokens"],
                "completion_tokens": current_tokens.get("completion_tokens", 0) + total_fork_tokens["completion_tokens"],
                "total_tokens": current_tokens.get("total_tokens", 0) + total_fork_tokens["total_tokens"]
            }
            await context.set_memory("total_usage", new_tokens)
            
            # Update parent context with aggregated sources
            current_sources = await context.get_memory("sources_used", [])
            for source in all_fork_sources:
                if source not in current_sources:
                    current_sources.append(source)
            await context.set_memory("sources_used", current_sources)
            
            self.logger.info(f"📊 [ForkDispatch] Aggregated tokens: {total_fork_tokens['total_tokens']}")
            self.logger.info(f"📚 [ForkDispatch] Aggregated sources: {len(all_fork_sources)}")
            
            if total_fork_tokens["total_tokens"] > 0:
                await context.accumulate_usage(total_fork_tokens)
                self.logger.info(f"📊 [ForkDispatch] Accumulated {total_fork_tokens['total_tokens']} tokens from {len(successful_forks)} fork(s)")
            
            # Store results
            await context.set_memory("fork_results", successful_forks)
            
            if successful_forks:
                return Transition(to="MergeState", reason="Forks completed", metadata={"successful_forks": len(successful_forks)})
            else:
                self.logger.warning("⚠️ [ForkDispatch] All forks failed, falling back")
                return Transition(to="RouterState", reason="All forks failed")
        
        except Exception as e:
            self.logger.error(f"❌ [ForkDispatch] Parallel execution failed: {e}")
            return Transition(to="RouterState", reason="Parallel execution exception", metadata={"error": str(e)})
    
    async def _execute_fork(self, branch_id: str, fork_ctx: AsyncExecutionContext):
        """Execute a single fork through the engine."""
        self.logger.info(f"🌿 [Fork:{branch_id}] Starting execution")
        
        # - Start from ResearchForkState (not RouterState)
        # This bypasses the full Router logic for efficiency
        # Pass explicitly to dispatch
        await self.engine.dispatch(fork_ctx, initial_state_name="ResearchForkState")
        
        self.logger.info(f"🌿 [Fork:{branch_id}] Execution complete")
        return fork_ctx


class MergeState(AsyncHierarchicalState):
    """
    Merges results from parallel fork executions.
    Supports customizable merge strategies.
    """
    def __init__(self, parent, merge_fn=None):
        super().__init__(parent)
        self.merge_fn = merge_fn  # Custom merge function
    
    async def handle(self, context: AsyncExecutionContext):
        self.logger.info("🧬 [Merge] Consolidating fork results...")
        
        # Get fork results
        fork_results = await context.get_memory("fork_results", [])
        
        if not fork_results:
            self.logger.warning("⚠️ [Merge] No fork results to merge")
            return Transition(to="RouterState", reason="No results to merge")
        
        self.logger.info(f"🧬 [Merge] Merging {len(fork_results)} fork results")
        
        # Use custom merge function if provided
        if self.merge_fn:
            try:
                plan_data = await context.get_memory("parallel_plan")
                plan = None
                if isinstance(plan_data, dict):
                    plan = ParallelPlan(**plan_data)
                elif isinstance(plan_data, ParallelPlan):
                    plan = plan_data
                    
                merged = await self.merge_fn(context, fork_results, plan)
                self.logger.info("✅ [Merge] Custom merge completed")
            except Exception as e:
                self.logger.error(f"❌ [Merge] Custom merge failed: {e}, using default")
                merged = self._contract_merge(fork_results)
        else:
            # - Use deterministic contract merge by default
            merged = self._contract_merge(fork_results)
            self.logger.info("✅ [Merge] Contract merge completed")
        
        # Store merged results as research_context (used by SynthesisState or AnswerState)
        await context.set_memory("research_context", merged)
        
        # 🔥 CRITICAL: Route based on whether synthesis is needed
        # Synthesis is only needed when we have fork results (parallel execution)
        fork_results = await context.get_memory("fork_results", [])
        
        if fork_results and len(fork_results) > 0:
            # We have fork results - go to synthesis
            self.logger.info(f"🧬 [Merge] Routing to synthesis ({len(fork_results)} forks)")
            return Transition(to="SemanticSynthesisState", reason="Contracts merged, ready for synthesis", metadata={"branches": len(fork_results)})
        else:
            # No forks - skip synthesis, go directly to router
            self.logger.info("🔄 [Merge] No forks, skipping synthesis")
            return Transition(to="RouterState", reason="No synthesis needed", metadata={"branches": 0})
    
    def _contract_merge(self, fork_results: List[AsyncExecutionContext]) -> dict:
        """
        Deterministic contract merge (no LLM, domain-agnostic).
        
        Detects consensus and conflicts in claims from multiple forks.
        """
        
        all_claims = {}
        all_coverage = set()
        all_uncertain = []  # List of UncertainTopic dicts
        
        # Collect all claims by key
        for fork_ctx in fork_results:
            contract_data = fork_ctx.memory.get("fork_contract")
            if not contract_data:
                self.logger.warning(f"⚠️ [Merge] Fork {fork_ctx.memory.get('branch_id')} has no contract")
                continue
            
            # 🔥 DEBUG: Log each fork's contract
            self.logger.info(f"📦 [Merge] Processing fork {fork_ctx.memory.get('branch_id')}:")
            self.logger.info(f"   Contract keys: {list(contract_data.keys())}")
            self.logger.info(f"   Claims count: {len(contract_data.get('claims', []))}")
            
            try:
                contract = ForkResult(**contract_data)
                
                # 🔥 DEBUG: Log claims being added
                self.logger.info(f"   Adding {len(contract.claims)} claims to merge:")
                for claim in contract.claims:
                    self.logger.info(f"      - {claim.key}: {claim.value}")
                
                for claim in contract.claims:
                    if claim.key not in all_claims:
                        all_claims[claim.key] = []
                    all_claims[claim.key].append({
                        "value": claim.value,
                        "evidence": [ev.model_dump() for ev in claim.evidence],  # Serialize Evidence objects
                        "confidence": claim.confidence,
                        "branch_id": contract.branch_id
                    })
                
                all_coverage.update(contract.coverage)
                
                # 🔥 EPISTEMIC: Collect uncertain topics (don't invalidate claims)
                for uncertain in contract.uncertain_topics:
                    all_uncertain.append(uncertain.model_dump())
                
            except Exception as e:
                self.logger.error(f"❌ [Merge] Failed to parse fork contract: {e}")
                continue
        
        # Detect consensus and conflicts
        resolved = {}
        conflicts = {}
        
        for key, claim_list in all_claims.items():
            # Extract unique values (convert to string for comparison)
            unique_values = {}
            for claim in claim_list:
                value_str = str(claim["value"])
                if value_str not in unique_values:
                    unique_values[value_str] = []
                unique_values[value_str].append(claim)
            
            if len(unique_values) == 1:
                # 🔥 EPISTEMIC: Preserve all evidence variants (consensus)
                value_str = list(unique_values.keys())[0]
                variants = unique_values[value_str]
                
                resolved[key] = {
                    "value": variants[0]["value"],
                    "variants": [
                        {
                            "evidence": v["evidence"],
                            "confidence": v["confidence"],
                            "branch_id": v["branch_id"]
                        }
                        for v in variants
                    ]
                }
                self.logger.debug(f"✅ [Merge] Consensus on '{key}': {resolved[key]['value']} ({len(variants)} variants)")
            else:
                # Conflict - multiple different values
                
                # 🔥 SPECIAL CASE: Concatenate 'summary' claims instead of conflicting
                if key == "summary":
                    concatenated_summary = " | ".join([str(val) for val in unique_values.keys()])
                    # Create a synthetic merged claim
                    resolved[key] = {
                        "value": concatenated_summary,
                        "variants": [
                           {"evidence": c["evidence"], "confidence": c["confidence"], "source": c["branch_id"]}
                           for sublist in unique_values.values() for c in sublist
                        ],
                        # Average confidence
                        "confidence": sum(c["confidence"] for sublist in unique_values.values() for c in sublist) / sum(len(sublist) for sublist in unique_values.values()),
                        "is_concatenated": True
                    }
                    self.logger.info(f"✅ [Merge] Concatenated conflicting summaries ({len(unique_values)} variants)")
                
                else:
                    # Logic for normal keys
                    conflicts[key] = []
                    for value_str, variants in unique_values.items():
                        conflicts[key].extend(variants)
                    self.logger.warning(f"⚠️ [Merge] Conflict on '{key}': {len(conflicts[key])} different values")
        
        # 🔥 EPISTEMIC: Uncertainty reduces coverage, never invalidates claims
        uncertain_coverage = set(u["topic"] for u in all_uncertain)
        final_coverage = list(all_coverage - uncertain_coverage)
        
        # Build MergedContract
        merged_contract = MergedContract(
            resolved=resolved,
            conflicts=conflicts,
            coverage=final_coverage,
            uncertain_topics=[UncertainTopic(**u) for u in all_uncertain],
            total_forks=len(fork_results)
        )
        
        # 🔥 EPISTEMIC METRICS (internal, not exposed to user)
        total_claims = len(resolved) + len(conflicts)
        total_uncertain = len(all_uncertain)
        
        if total_claims + total_uncertain > 0:
            omission_rate = total_uncertain / (total_claims + total_uncertain)
            self.logger.info(f"📊 [Merge] Epistemic metrics:")
            self.logger.info(f"   Omission rate: {omission_rate:.2%}")
        
        # Count inferred claims
        inferred_count = sum(
            1 for claim_variants in resolved.values()
            for variant in claim_variants.get("variants", [])
            if any(ev.get("type") == "inferred" for ev in variant.get("evidence", []))
        )
        
        if inferred_count + total_uncertain > 0:
            inference_ratio = inferred_count / (inferred_count + total_uncertain)
            self.logger.info(f"   Inference ratio: {inference_ratio:.2%} (higher = better reasoning)")
        
        self.logger.info(f"📊 [Merge] Contract merge summary:")
        self.logger.info(f"   ✅ Resolved: {len(resolved)} claims")
        self.logger.info(f"   ⚠️  Conflicts: {len(conflicts)} claims")
        self.logger.info(f"   📋 Coverage: {len(final_coverage)} topics")
        self.logger.info(f"   ❓ Uncertain: {total_uncertain} topics")
        
        return merged_contract.model_dump()



class SemanticSynthesisState(AsyncHierarchicalState):
    """
    Optional synthesis state that consolidates fork contracts into natural language.
    
    Only active when parallel execution is enabled.
    Uses pluggable synthesis strategy (LLM, rules, hybrid).
    """
    
    def __init__(self, parent, synthesis_strategy):
        super().__init__(parent)
        self.synthesis_strategy = synthesis_strategy
    
    async def handle(self, context: AsyncExecutionContext):
        self.logger.info("🧬 [Synthesis] Consolidating fork outputs...")
        
        # Get merged contracts
        research_context = await context.get_memory("research_context")
        
        if not research_context or not isinstance(research_context, dict):
            self.logger.warning("⚠️ [Synthesis] No research context to synthesize")
            return Transition(to="RouterState", reason="No synthesis needed")
        
        # Build synthesis request
        from finitestatemachineAgent.synthesis_contracts import SynthesisRequest
        
        # Extract text from contracts
        fork_outputs = []
        
        # Add resolved claims
        if research_context.get("resolved"):
            resolved_text = "**Consensus Findings:**\n"
            for key, claim in research_context["resolved"].items():
                resolved_text += f"- {key}: {claim['value']}\n"
                if claim.get('evidence'):
                    resolved_text += f"  Evidence: {', '.join(claim['evidence'])}\n"
            fork_outputs.append(resolved_text)
        
        # Add conflicts
        if research_context.get("conflicts"):
            conflicts_text = "**Conflicting Findings:**\n"
            for key, claims in research_context["conflicts"].items():
                conflicts_text += f"- {key}:\n"
                for i, claim in enumerate(claims, 1):
                    conflicts_text += f"  {i}. {claim['value']} (from {claim.get('branch_id', 'unknown')})\n"
            fork_outputs.append(conflicts_text)
        
        # Add coverage and omissions
        if research_context.get("coverage"):
            fork_outputs.append(f"**Topics Covered:** {', '.join(research_context['coverage'])}")
        
        if research_context.get("omissions"):
            fork_outputs.append(f"**Information Gaps:** {', '.join(research_context['omissions'])}")
        
        if not fork_outputs:
            self.logger.warning("⚠️ [Synthesis] No outputs to synthesize")
            return Transition(to="RouterState", reason="Empty research context")
        
        # Get metadata for better synthesis
        user_language = await context.get_memory("user_language", "pt")
        todo_list = await context.get_memory("todo_list", [])
        
        # Build constraints
        constraints = []
        constraints.append(f"Output Language: {user_language.upper()}")
        
        if todo_list:
            constraints.append("Must address the following points if data allows:")
            for task in todo_list:
                constraints.append(f"- {task}")

        request = SynthesisRequest(
            task_description=context.user_query,
            fork_outputs=fork_outputs,
            constraints=constraints,
            output_format="markdown"
        )
        
        # Synthesize using strategy
        try:
            result = await self.synthesis_strategy.synthesize(request)
            
            # 🔥 Structural Guardrails (NOT semantic)
            if not result.answer or len(result.answer.strip()) == 0:
                raise ValueError("Empty synthesis result")
            
            if len(result.answer) > 50000:  # Token limit approximation
                self.logger.warning(f"⚠️ [Synthesis] Result too long ({len(result.answer)} chars), truncating")
                result.answer = result.answer[:50000] + "\n\n[Response truncated due to length]"
            
            # Store synthesis result
            await context.set_memory("synthesis_result", {
                "answer": result.answer,
                "confidence": result.confidence,
                "gaps": result.gaps,
                "inconsistencies": result.inconsistencies
            })
            
            self.logger.info(f"✅ [Synthesis] Consolidated {len(fork_outputs)} outputs into {len(result.answer)} chars")
            
            return Transition(to="AnswerState", reason="Synthesis complete")
            
        except Exception as e:
            self.logger.error(f"❌ [Synthesis] Failed: {e}")
            # Fallback: skip synthesis, use raw contracts
            self.logger.info("🔄 [Synthesis] Falling back to raw contracts")
            return Transition(to="AnswerState", reason="Synthesis failed, using raw contracts")


class AnswerState(TerminalState):
    """
    Async answer state with streaming.
    """
    def __init__(self, parent, llm):
        super().__init__(parent)
        self.llm = llm

    async def handle(self, context: AsyncExecutionContext):
        self.logger.info("✅ [Answer] Generating final response...")

        # Build messages
        system_instruction = await context.get_memory("system_instruction", "")
        
        # 🔥 Redirect Feature Override
        redirect_mode = await context.get_memory("redirect_mode", False)
        if redirect_mode:
            redirect_prompt = await context.get_memory("redirect_system_prompt")
            if redirect_prompt:
                system_instruction = redirect_prompt
                self.logger.info("↪️ [Answer] Using Redirect System Prompt")

        messages = [{"role": "system", "content": system_instruction}]

        # Get intent analysis metadata (Available for both flows)
        user_language = await context.get_memory("user_language")  # No default here to check if it exists
        todo_list = await context.get_memory("todo_list", [])

        # Determine language instruction
        if user_language:
            lang_instruction = f"You MUST answer in {user_language.upper()}."
        else:
            lang_instruction = "You MUST answer in the SAME LANGUAGE as the user's request."

        # Check if we have synthesis result
        synthesis_result = await context.get_memory("synthesis_result")
        
        if synthesis_result and synthesis_result.get("answer"):
            # SIMPLIFIED FLOW: Only synthesis + original query
            self.logger.info(f"📚 [Answer] Using synthesis-only mode ({len(synthesis_result['answer'])} chars)")
            
            # Just add the user query
            messages.append({"role": "user", "content": context.user_query})
            
            # Build todo list text if available
            todo_text = ""
            if todo_list:
                todo_text = "\n\n**Key points to address:**\n"
                for i, task in enumerate(todo_list, 1):
                    todo_text += f"{i}. {task}\n"
            
            # Inject synthesized context
            messages.append({
                "role": "system",
                "content": f"""Use the following synthesized research findings to answer the user's question.

USER'S QUESTION:
{context.user_query}

CRITICAL INSTRUCTIONS:
1. **LANGUAGE**: {lang_instruction}
   - DO NOT translate technical terms if they are standard in the field.

2. **Use the findings**: The provided text is a synthesis of multiple sources. Use it as the primary source of truth.
3. **Natural presentation**: Present the answer naturally to the user.{todo_text}

SYNTHESIZED FINDINGS:
{synthesis_result['answer']}"""
            })
            
        else:
            # FALLBACK: Original complex flow with history and tool calls
            self.logger.info("📚 [Answer] Using fallback mode (no synthesis)")
            
            # Add history
            history = await context.get_memory("chat_history", [])
            if history:
                messages.extend(history)

            messages.append({"role": "user", "content": context.user_query})

            # Add tool results if any
            if context.tool_calls:
                for call in context.tool_calls:
                    if call.get("result"):
                        tool_call_id = f"call_{call['tool_name']}_{call.get('iteration', 0)}"
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": call["tool_name"],
                                    "arguments": json.dumps(call["arguments"])
                                }
                            }]
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": call["tool_name"],
                            "content": str(call.get("result", ""))
                        })

            # Build fallback prompt
            fallback_instruction = f"Based on the tool results, provide a COMPREHENSIVE and FINAL answer to the user's request. {lang_instruction} Do NOT provide an introduction saying you will explain; JUST EXPLAIN directly and professionally. Act as if this is the final report. Do NOT call more tools."
            
            if todo_list:
                fallback_instruction += "\n\nKey points to address:\n"
                for i, task in enumerate(todo_list, 1):
                    fallback_instruction += f"{i}. {task}\n"

            messages.append({
                "role": "system",
                "content": fallback_instruction
            })


        # Fallback to research context (contracts) if no synthesis
            research_context = await context.get_memory("research_context")
            if research_context and isinstance(research_context, dict):
                # Check if it's a MergedContract (has 'resolved' and 'conflicts' keys)
                if "resolved" in research_context and "conflicts" in research_context:
                    self.logger.info(f"📚 [Answer] Injecting contract from {research_context.get('total_forks', 0)} fork(s)")
                    
                    # Format contract for LLM
                    contract_text = "# Research Contracts\n\n"
                    
                    # Resolved claims (consensus)
                    if research_context.get("resolved"):
                        contract_text += "## ✅ RESOLVED CLAIMS (Consensus)\n\n"
                        for key, claim in research_context["resolved"].items():
                            contract_text += f"- **{key}**: {claim['value']}\n"
                            contract_text += f"  - Evidence: {', '.join(claim.get('evidence', []))}\n"
                            contract_text += f"  - Confidence: {claim.get('confidence', 1.0):.2f}\n\n"
                    
                    # Conflicts (disagreements)
                    if research_context.get("conflicts"):
                        contract_text += "## ⚠️ CONFLICTS (Multiple Values Found)\n\n"
                        for key, claims in research_context["conflicts"].items():
                            contract_text += f"- **{key}**:\n"
                            for i, claim in enumerate(claims, 1):
                                contract_text += f"  {i}. {claim['value']} (from {claim.get('branch_id', 'unknown')})\n"
                                contract_text += f"     - Evidence: {', '.join(claim.get('evidence', []))}\n"
                            contract_text += "\n"
                    
                    # Coverage and omissions
                    if research_context.get("coverage"):
                        contract_text += f"## 📋 Coverage\n{', '.join(research_context['coverage'])}\n\n"
                    
                    if research_context.get("omissions"):
                        contract_text += f"## ❌ Omissions\n{', '.join(research_context['omissions'])}\n\n"
                    
                    # 🔥 DEBUG: Log contract being injected
                    self.logger.info(f"📝 [Answer] Contract text being injected ({len(contract_text)} chars):")
                    
                    messages.append({
                        "role": "system",
                        "content": """You have access to research contracts from parallel investigations. Use this information to answer the user's question.

CRITICAL INSTRUCTIONS:
1. **LANGUAGE CONSISTENCY**: You MUST answer in the SAME LANGUAGE as the user's query.
   - If user asks in Portuguese -> Answer in Portuguese
   - If user asks in English -> Answer in English
   - If user asks in Spanish -> Answer in Spanish
   - DO NOT translate technical terms if they are standard in the field, but explain them if needed.

2. **Use the information**: Leverage RESOLVED claims confidently (all sources agreed)
3. **Present conflicts naturally**: If there are CONFLICTS, present all viewpoints without choosing
4. **Be transparent about gaps**: Mention OMISSIONS clearly when information is missing
5. **Natural presentation**: DO NOT mention "contracts", "claims", "resolved", or "conflicts" in your response
6. **User-facing language**: Present findings as if you researched them directly

Remember: The user should NOT see the internal contract structure. Present information naturally in the CORRECT LANGUAGE."""
                    })
                    messages.append({
                        "role": "user",
                        "content": contract_text
                    })
                else:
                    # Legacy format (old semantic merge)
                    self.logger.info(f"📚 [Answer] Injecting legacy research from {research_context.get('total_branches', 0)} branches")
                    messages.append({
                        "role": "system",
                        "content": "Use the following research results to answer the user's question:"
                    })

        # 2. Add User Query
        messages.append({"role": "user", "content": context.user_query})
        
        # 3. Call LLM (Streaming or Non-Streaming)
        enable_streaming = await context.get_memory("enable_streaming", True)
        
        if enable_streaming:
            # Stream response
            self.logger.info("🌊 [Answer] Streaming initialized")
            # 🛡️ Safety Check Before LLM Call (Streaming)
            await context.increment_llm_call()
            
            # Create wrapper to capture usage from stream
            llm_stream = self.llm.chat_stream(messages)
            
            async def stream_with_usage():
                """Wrapper that captures usage from stream metadata."""
                async for token in llm_stream:
                    # Check if token is usage metadata (dict with __usage__ key)
                    if isinstance(token, dict) and "__usage__" in token:
                        # Store usage in context
                        await context.accumulate_usage(token["__usage__"])
                    else:
                        # Regular token, yield it
                        yield token
            
            # Store wrapped stream for consumption
            await context.set_memory("answer_stream", stream_with_usage())
            
            # Store total requests (same as non-streaming mode)
            if hasattr(context, 'safety_monitor'):
                await context.set_memory("total_requests", context.safety_monitor.count)
        else:
            # Non-streaming response
            self.logger.info("📝 [Answer] Generating complete response")
            # 🛡️ Safety Check Before LLM Call (Non-Streaming)
            await context.increment_llm_call()
            # 🔥 Fix: chat() only takes messages, not context
            response = await self.llm.chat(messages)
            
            # Manually track usage if available
            if "usage" in response:
                await context.accumulate_usage(response["usage"])
                
            final_answer = response.get("content", "")
            
            # Store for access
            await context.set_memory("final_answer", final_answer)
            
            # Store total requests (engine-level metric)
            if hasattr(context, 'safety_monitor'):
                await context.set_memory("total_requests", context.safety_monitor.count)
            
            # Create fake stream for compatibility
            async def stream_complete():
                for char in final_answer:
                    yield char
            
            await context.set_memory("answer_stream", stream_complete())
            
        return None  # Terminal state


class RetryState(AsyncHierarchicalState):
    """
    Async retry state.
    """
    def __init__(self, parent):
        super().__init__(parent)

    async def handle(self, context: AsyncExecutionContext):
        self.logger.warning("⚠️ [Retry] Attempting recovery...")

        await context.increment_iteration()

        if context.current_iteration >= context.max_iterations:
            self.logger.warning("⚠️ [Retry] Max retries reached. Proceeding to AnswerState (Best Effort).")
            return Transition(to="AnswerState", reason="Max retries reached")
        else:
            self.logger.warning(f"⚠️ [Retry] Attempt {context.current_iteration}/{context.max_iterations}")
            return Transition(to="RouterState", reason="Retrying execution", metadata={"attempt": context.current_iteration})


class FailState(TerminalState):
    """
    Async fail state.
    """
    def __init__(self, parent):
        super().__init__(parent)

    async def handle(self, context: AsyncExecutionContext):
        self.logger.error("❌ [Fail] Terminating agent.")
        return None


class IntentAnalysisState(AsyncHierarchicalState):
    """
    Standard state for analyzing user intent before routing.
    
    Analyzes:
    - Intent
    - Context from history
    - Todo list for execution
    - Complexity & Tool needs (for redirect)
    - Language
    """
    def __init__(self, parent, llm, system_instruction):
        super().__init__(parent)
        self.llm = llm
        self.system_instruction = system_instruction

    async def handle(self, context: AsyncExecutionContext):
        self.logger.info("🧠 [IntentAnalysis] Analyzing user intent...")
        self.logger.info(f"🔍 [IntentAnalysis] DEBUG: Query = '{context.user_query}'")
        
        # Check if already analyzed to prevent loops
        if await context.get_memory("intent_analyzed"):
            return Transition(to="RouterState", reason="Already analyzed")

        try:
            chat_history = await context.get_memory("chat_history", [])
            current_query = context.user_query
            
            # Build prompt - Comprehensive analysis
            messages = [{
                "role": "system",
                "content": """Você é um assistente de análise de intenção. Sua tarefa é analisar a pergunta do usuário e retornar um JSON estruturado.

TAREFA: Analise a pergunta e retorne APENAS um JSON válido com a seguinte estrutura:

{
    "intent": "Descrição clara e concisa da intenção do usuário",
    "context_from_history": ["Fato relevante 1 do histórico", "Fato relevante 2"],
    "enhanced_query": "Pergunta enriquecida com contexto (se necessário)",
    "todo_list": [
        "Passo 1: O que fazer primeiro",
        "Passo 2: O que analisar",
        "Passo 3: Como responder"
    ],
    "language": "pt",
    "complexity": "simple",
    "needs_tools": false
}

CAMPOS OBRIGATÓRIOS:

1. **intent**: Resumo da intenção principal (ex: "Buscar informações sobre X", "Comparar Y e Z")

2. **context_from_history**: Lista de fatos relevantes do histórico de conversa. Deixe vazio [] se não houver histórico relevante.

3. **enhanced_query**: A pergunta original enriquecida com contexto do histórico. Se não houver contexto relevante, repita a pergunta original.

4. **todo_list**: Lista de 2-4 passos técnicos para resolver a tarefa:
   - Para perguntas simples: ["Entender a pergunta", "Responder diretamente"]
   - Para perguntas complexas: ["Buscar informação X", "Analisar Y", "Sintetizar resposta"]

5. **language**: Código ISO do idioma (pt, en, es, fr, etc.)

6. **complexity**: 
   - "simple" = saudação, pergunta trivial, conversa casual
   - "complex" = requer pesquisa, raciocínio, análise

7. **needs_tools**: 
   - true = precisa buscar dados externos (documentos, APIs, cálculos)
   - false = pode responder com conhecimento geral

EXEMPLOS:

Pergunta: "Olá, tudo bem?"
{
    "intent": "Saudação casual",
    "context_from_history": [],
    "enhanced_query": "Olá, tudo bem?",
    "todo_list": ["Responder à saudação de forma amigável"],
    "language": "pt",
    "complexity": "simple",
    "needs_tools": false
}

Pergunta: "Quais são os direitos do consumidor na CF/88?"
{
    "intent": "Buscar informações sobre direitos do consumidor na Constituição Federal",
    "context_from_history": [],
    "enhanced_query": "Quais são os direitos do consumidor na CF/88?",
    "todo_list": [
        "Buscar artigos da CF/88 sobre direitos do consumidor",
        "Identificar dispositivos constitucionais relevantes",
        "Sintetizar os direitos encontrados"
    ],
    "language": "pt",
    "complexity": "complex",
    "needs_tools": true
}

IMPORTANTE:
- Retorne APENAS o JSON, sem texto adicional antes ou depois
- Não adicione comentários ou explicações
- Certifique-se de que o JSON está válido e completo"""
            }]
            
            if chat_history:
                messages.extend(chat_history[-5:])
                
            messages.append({"role": "user", "content": f"Pergunta atual: {current_query}\n\nRetorne APENAS o JSON da análise."})
            
            # Call LLM
            response = await self.llm.chat(messages)
            
            # Track usage
            if "usage" in response:
                await context.accumulate_usage(response["usage"])
            
            content = response.get("content", "").strip()
            
            # Robust JSON extraction
            analysis = None
            try:
                if not content:
                    raise ValueError("Empty response")
                
                # Clean markdown
                if content.startswith("```json"):
                    content = content.replace("```json", "", 1)
                if content.startswith("```"):
                    content = content.replace("```", "", 1)
                if content.endswith("```"):
                    content = content.rsplit("```", 1)[0]
                
                content = content.strip()
                
                # Extract JSON object using brace counting
                start_idx = content.find("{")
                if start_idx == -1:
                    raise ValueError("No JSON found")
                
                brace_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(content)):
                    if content[i] == "{":
                        brace_count += 1
                    elif content[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                json_str = content[start_idx:end_idx]
                analysis = json.loads(json_str)
                self.logger.info("✅ [IntentAnalysis] JSON parsed successfully")
            except Exception as e:
                self.logger.warning(f"⚠️ [IntentAnalysis] JSON parsing failed: {e}")
                analysis = None
            
            # Fallback
            if analysis is None:
                self.logger.info("🔄 [IntentAnalysis] Using fallback")
                analysis = {
                    "intent": "unknown",
                    "context_from_history": [],
                    "enhanced_query": current_query,
                    "todo_list": [],
                    "language": "pt",
                    "complexity": "simple",
                    "needs_tools": False
                }
            
            # Store in context
            await context.set_memory("intent_analysis", analysis)
            await context.set_memory("todo_list", analysis.get("todo_list", []))
            await context.set_memory("intent_analyzed", True)
            await context.set_memory("user_language", analysis.get("language", "pt"))
            
            # Enhanced query
            if analysis.get("enhanced_query"):
                context.user_query = analysis.get("enhanced_query")
                self.logger.info(f"✨ [IntentAnalysis] Query enhanced: {context.user_query}")
            
            # Log results with full JSON
            self.logger.info("🔍 [IntentAnalysis] DEBUG: About to log detailed analysis...")
            self.logger.info(f"✅ [IntentAnalysis] Intent: {analysis.get('intent', 'unknown')}")
            self.logger.info(f"📊 [IntentAnalysis] Complexity: {analysis.get('complexity', 'unknown')}")
            self.logger.info(f"🔧 [IntentAnalysis] Needs tools: {analysis.get('needs_tools', False)}")
            self.logger.info(f"🌍 [IntentAnalysis] Language: {analysis.get('language', 'unknown')}")
            self.logger.info(f"📝 [IntentAnalysis] Todo list ({len(analysis.get('todo_list', []))} items):")
            for i, task in enumerate(analysis.get('todo_list', []), 1):
                self.logger.info(f"   {i}. {task}")
            self.logger.info(f"📋 [IntentAnalysis] Full analysis: {json.dumps(analysis, ensure_ascii=False, indent=2)}")
                
            # Redirect Check
            complexity = analysis.get("complexity", "complex")
            needs_tools = analysis.get("needs_tools", True)
            
            if complexity == "simple" and not needs_tools:
                self.logger.info("🚀 [IntentAnalysis] Redirecting to AnswerState (Simple Query)")
                await context.set_memory("redirect_mode", True)
                return Transition(to="AnswerState", reason="Simple query redirect")
                
            return Transition(to="RouterState", reason="Analysis complete")
            
        except Exception as e:
            self.logger.error(f"❌ [IntentAnalysis] Failed: {e}")
            await context.set_memory("intent_analyzed", True)
            return Transition(to="RouterState", reason="Analysis failed")

# =================================================================================================
# Agent Engine
# =================================================================================================

class AsyncAgentEngine:
    """
    Async Hierarchical Finite State Machine Agent Engine.
    """

    def __init__(
        self,
        llm,
        registry,
        executor: AsyncToolExecutor,
        system_instruction: str = "",
        tool_choice: Optional[str] = None,
        skip_validation: bool = True,  # Default: No validation
        validation_fn=None,  # Custom validation function
        
        # - Parallel execution support
        enable_parallel_planning: bool = False,
        parallel_plan_fn=None,           # Custom planning function
        planning_system_prompt=None,     # Custom/enhanced system prompt for LLM planner
        merge_fn=None,                   # Custom merge strategy
        max_parallel_branches: int = 3,  # - Max branches per fork (width limit)
        # 🔥 Safety Config
        max_global_requests: int = 50,
        post_router_hook=None,  # - Optional hook to intercept router transitions
        initial_state: Optional[str] = None,  # - Custom initial state
        # 🔥 Intent Analysis Config
        enable_intent_analysis: bool = False,  # - Enable built-in intent analysis
        intent_analysis_llm: Optional['AsyncLLMClient'] = None,  # - LLM for intent analysis
        # 🔥 Redirect Feature
        redirect_system_prompt: str = "Você é um assistente útil. Responda a pergunta do usuário de forma direta.",
        
        # 🔥 Recovery Config
        max_retries: int = 3,  # Max retries for validation failure
        
        # 🔥 Persistence Config
        enable_snapshots: bool = True,  # Enable snapshot saving
        enable_logging: bool = True,  # Enable logging
        
        # 🔥 Context Pruning Config
        enable_context_pruning: bool = False,  # Enable context pruning (opt-in)
        context_pruner: Optional[AsyncContextPruner] = None,  # Custom pruner
        pruner_keep_recent: int = 4,  # Keep last N tool calls full
        pruner_max_length: int = 200,  # Max length for truncated results
        
        # 🔥 Strategy Config
        contract_strategy = None,
        synthesis_strategy = None
    ):
        """
        Initialize Async Agent Engine with HFSM architecture.
        
        Args:
            llm: Async LLM client
            registry: Tool registry
            executor: Async tool executor
            system_instruction: System prompt
            tool_choice: Tool selection mode
            skip_validation: Skip validation state
            validation_fn: Custom validation function
            enable_parallel_planning: Enable parallel execution
            planning_system_prompt: Custom planning prompt
            post_router_hook: Hook after router decisions
            merge_fn: Custom merge function for parallel results
            max_parallel_branches: Max parallel branches per fork
            max_global_requests: Global safety limit for LLM calls
            initial_state: Custom initial state name (default: RouterState)
            enable_intent_analysis: Enable built-in intent analysis before RouterState
            intent_analysis_llm: LLM client for intent analysis (defaults to main llm)
        """
        self.llm = llm
        self.registry = registry
        self.executor = executor
        self.system_instruction = system_instruction
        self.tool_choice = tool_choice
        self.skip_validation = skip_validation
        self.validation_fn = validation_fn
        
        # Store parallel execution config
        self.enable_parallel_planning = enable_parallel_planning
        self.parallel_plan_fn = parallel_plan_fn
        self.planning_system_prompt = planning_system_prompt
        self.merge_fn = merge_fn
        self.max_parallel_branches = max_parallel_branches
        self.max_global_requests = max_global_requests
        
        # 🔥 Custom initial state
        self.initial_state_name = initial_state
        
        # 🔥 Intent Analysis Config
        self.enable_intent_analysis = enable_intent_analysis
        self.intent_analysis_llm = intent_analysis_llm or llm
        
        # 🔥 Redirect Feature
        self.redirect_system_prompt = redirect_system_prompt
        
        # 🔥 Recovery Config
        self.max_retries = max_retries
        
        # 🔥 Persistence Config
        self.enable_snapshots = enable_snapshots
        self.enable_logging = enable_logging
        
        # 🔥 Strategy Injection
        self.contract_strategy = contract_strategy
        self.synthesis_strategy = synthesis_strategy
        
        # 🔥 Safety Config
        self.post_router_hook = post_router_hook  # 🔥 Store hook
        
        # 🔥 Custom state registration system
        self.custom_states = {}  # name -> state instance
        self.transition_overrides = []  # List of override rules

        self.states = {}

        # Setup Logger
        if self.enable_logging:
            self.logger = logging.getLogger("AsyncAgentEngine")
        else:
            self.logger = DummyLogger()

        # Initialize hierarchy
        self.root = AgentRootState(logger=self.logger)
        self.states["AgentRootState"] = self.root

        # 🔥 Context Pruning: Insert ContextPolicyState if enabled
        if enable_context_pruning:
            # Create or use custom pruner
            if context_pruner:
                self.pruner = context_pruner
            else:
                self.pruner = AsyncContextPruner(
                    keep_recent=pruner_keep_recent,
                    max_length=pruner_max_length
                )
            
            # Insert policy layer in hierarchy
            self.policy = AsyncContextPolicyState(self.root, self.pruner, logger=self.logger)
            self.states["AsyncContextPolicyState"] = self.policy
            self.logger.info(f"✂️ [Engine] Context pruning enabled (keep_recent={pruner_keep_recent}, max_length={pruner_max_length})")
        else:
            # No pruning - policy points directly to root
            self.policy = self.root
        self.states["ContextPolicyState"] = self.policy

        self.reasoning = ReasoningState(self.policy)
        self.states["ReasoningState"] = self.reasoning

        self.execution = ExecutionState(self.policy)
        self.states["ExecutionState"] = self.execution

        self.recovery = RecoveryState(self.policy)
        self.states["RecoveryState"] = self.recovery

        self.terminal = TerminalState(self.policy)
        self.states["TerminalState"] = self.terminal

        # Operational states
        self.router_state = RouterState(
            self.reasoning, 
            self.llm, 
            self.registry, 
            self.tool_choice,
            enable_parallel=self.enable_parallel_planning,
            post_hook=self.post_router_hook  # 🔥 Pass hook
        )
        self.states["RouterState"] = self.router_state

        self.tool_state = ToolState(self.execution, self.executor, self.skip_validation)
        self.states["ToolState"] = self.tool_state

        self.validation_state = ValidationState(self.execution, self.llm, self.validation_fn)
        self.states["ValidationState"] = self.validation_state
        
        # Inject logger into all states
        for state in self.states.values():
            if hasattr(state, 'logger'):
                state.logger = self.logger

        self.answer_state = AnswerState(self.reasoning, self.llm)
        self.states["AnswerState"] = self.answer_state

        self.retry_state = RetryState(self.execution)
        self.states["RetryState"] = self.retry_state

        self.fail_state = FailState(self.root)
        self.states["FailState"] = self.fail_state
        
        # Add parallel execution states (only if enabled)
        if self.enable_parallel_planning:
            self.parallel_planning_state = ParallelPlanningState(
                self.reasoning, 
                self.llm, 
                self.parallel_plan_fn,
                self.planning_system_prompt,
                max_branches=self.max_parallel_branches  # Pass width limit
            )
            self.states["ParallelPlanningState"] = self.parallel_planning_state
            
            self.fork_dispatch_state = ForkDispatchState(self.execution, self)
            self.states["ForkDispatchState"] = self.fork_dispatch_state
            
            self.merge_state = MergeState(self.execution, self.merge_fn)
            self.states["MergeState"] = self.merge_state
            
            # - Fork-specific states
            self.research_fork_state = ResearchForkState(
                self.reasoning,
                self.llm,
                self.registry,
                self.tool_choice
            )
            self.states["ResearchForkState"] = self.research_fork_state
            
            # Use ForkContractState with Strategy
            self.fork_contract_state = ForkContractState(
                self.terminal, 
                self.llm,
                contract_strategy=self.contract_strategy
            )
            self.states["ForkContractState"] = self.fork_contract_state
            # Backward compatibility alias
            self.states["ForkSummaryState"] = self.fork_contract_state
            
            # - Add semantic synthesis state with Strategy
            from finitestatemachineAgent.llm_synthesis_strategy import LLMSynthesisStrategy
            synthesis_strat = self.synthesis_strategy or LLMSynthesisStrategy(self.llm, temperature=0.3)
            
            self.synthesis_state = SemanticSynthesisState(self.reasoning, synthesis_strat)
            self.states["SemanticSynthesisState"] = self.synthesis_state
            
            self.logger.info("✅ Parallel execution states initialized")

            # 🔥 FORK OVERRIDES: Ensure forks define contracts instead of answering
            # Redirect Validation -> Answer to ForkContract
            self.override_transition(
                from_state="ValidationState",
                condition=lambda ctx, trans: ctx.memory.get("branch_id") is not None and trans.to == "AnswerState",
                new_target="ForkContractState"
            )
            # Redirect Tool -> Answer to ForkContract (if skipping validation)
            self.override_transition(
                from_state="ToolState",
                condition=lambda ctx, trans: ctx.memory.get("branch_id") is not None and trans.to == "AnswerState",
                new_target="ForkContractState"
            )

        # 🔥 Standard Optional States (Intent Analysis)
        if self.enable_intent_analysis:
            self.intent_analysis_state = IntentAnalysisState(
                self.policy, 
                self.intent_analysis_llm or self.llm,
                self.system_instruction  # Pass system instruction
            )
            self.states["IntentAnalysisState"] = self.intent_analysis_state
            # Inject logger
            self.intent_analysis_state.logger = self.logger
            
        # Response Validator (Standard)
        self.response_validator_state = ResponseValidatorState(
            self.terminal if hasattr(self, 'terminal') else self.root, 
            self.llm
        )
        self.states["ResponseValidatorState"] = self.response_validator_state

        # Service locator
        self.root.find_state_by_type = self._find_state_provider

    def _find_state_provider(self, state_type):
        """Find state by type."""
        for state in self.states.values():
            if type(state).__name__ == state_type:
                return state
        raise ValueError(f"State {state_type} not found")
    
    def _save_snapshot(self, context: AsyncExecutionContext, event_name: str):
        """Save context snapshot to disk/log."""
        if not self.enable_snapshots:
            return  # Snapshot saving disabled

        try:
            snapshot = context.snapshot()
            
            # Disk Persistence
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"snapshot_{timestamp}_{event_name}.json"
            directory = "logs/snapshots"
            
            import os
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                
            filepath = os.path.join(directory, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
                
            self.logger.debug(f"💾 Snapshot saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save snapshot: {e}")
    
    # 🔥 Custom State Registration API
    
    def register_state(self, name: str, state):
        """
        Register a custom state in the engine.
        
        Allows domain-specific agents to inject custom states without modifying engine.
        
        Args:
            name: State name (e.g., "TodoListState")
            state: State instance (must inherit from AsyncHierarchicalState)
        
        Example:
            todo_state = TodoListState(parent=engine.reasoning, llm=llm)
            engine.register_state("TodoListState", todo_state)
        """
        self.custom_states[name] = state
        self.states[name] = state
        self.logger.info(f"🔧 [Engine] Registered custom state: {name}")
    
    def override_transition(
        self, 
        from_state: str, 
        condition: callable,  # (context, transition) -> bool
        new_target: str
    ):
        """
        Override a transition based on condition.
        
        Allows domain-specific agents to redirect transitions without modifying engine.
        
        Args:
            from_state: Source state name (e.g., "RouterState")
            condition: Function (context, transition) -> bool to check if override applies
            new_target: Target state name to redirect to (e.g., "TodoListState")
        
        Example:
            # Inject TodoList before ParallelPlanning
            engine.override_transition(
                from_state="RouterState",
                condition=lambda ctx, trans: trans.to == "ParallelPlanningState",
                new_target="TodoListState"
            )
        """
        # Validate target state exists
        if new_target not in self.states and new_target not in self.custom_states:
            self.logger.warning(f"⚠️ [Engine] Override target '{new_target}' not registered yet")
        
        self.transition_overrides.append({
            "from": from_state,
            "condition": condition,
            "target": new_target
        })
        self.logger.info(f"🔧 [Engine] Override registered: {from_state} -> {new_target}")
    
    async def _analyze_intent(self, context: AsyncExecutionContext):
        """
        Built-in intent analysis (runs before RouterState in main flow).
        
        Analyzes user query + chat history to:
        - Extract user intent
        - Generate structured todo list
        - Detect language
        - Enhance query with context from history
        """
        self.logger.info("=" * 80)
        self.logger.info("🧠 [IntentAnalysis] Analyzing user intent...")
        self.logger.info(f"📝 [IntentAnalysis] Query: {context.user_query[:100]}...")
        self.logger.info("=" * 80)
        
        self.logger.info("🔥🔥🔥 [DEBUG] IntentAnalysisState.handle() v3.0 LOADED")
        
        try:
            # Get chat history
            chat_history = await context.get_memory("chat_history", [])
            current_query = context.user_query
            
            self.logger.info(f"[IntentAnalysis] Chat history length: {len(chat_history)}")
            
            # Build analysis prompt
            messages = [{
                "role": "system",
                "content": """Você é um assistente de análise de intenção.

TAREFA: Analise a pergunta do usuário e retorne APENAS um JSON válido.

Formato JSON OBRIGATÓRIO:
{
    "intent": "Resumo conciso do objetivo do usuário",
    "required_context": ["Fato relevante 1 extraído do histórico", "Fato relevante 2"],
    "enhanced_query": "Pergunta do usuário enriquecida com o contexto extraído (se necessário)",
    "todo_list": [
        "Ação 1: O que pesquisar/fazer",
        "Ação 2: O que analisar",
        "Ação 3: Como responder"
    ],
    "language": "pt",
    "complexity": "simple",
    "needs_tools": false
}

REGRAS:
- complexity: "simple" para saudações/trivial, "complex" para pesquisa
- needs_tools: true se precisa buscar dados externos
- required_context: vazio se histórico não for relevante
- enhanced_query: igual à pergunta se não houver ambiguidade
- Retorne APENAS o JSON, sem texto antes ou depois"""
            }]
            
            # Add chat history
            if chat_history:
                messages.extend(chat_history[-5:])  # Last 5 messages for context
            
            # Add current query
            messages.append({
                "role": "user",
                "content": f"Pergunta atual: {current_query}\n\nRetorne APENAS o JSON da análise, sem texto adicional."
            })
            
            self.logger.info(f"[IntentAnalysis] Calling LLM...")
            
            # Call LLM for intent analysis
            response = await self.intent_analysis_llm.chat(messages)

            # 🔥 usage tracking
            if "usage" in response:
                await context.accumulate_usage(response["usage"])
            
            # Extract JSON from response (handle markdown code blocks)
            content = response.get("content", "").strip()
            
            self.logger.info("🔥 [DEBUG] Using ROBUST JSON extraction v2.0")
            self.logger.info(f"[IntentAnalysis] Raw LLM response length: {len(content)} chars")
            
            # Robust JSON extraction with guaranteed fallback
            analysis = None
            try:
                import json
                
                if not content:
                    self.logger.warning("⚠️ [IntentAnalysis] Empty LLM response")
                    raise ValueError("Empty response content")

                # Remove markdown code blocks
                if content.startswith("```json"):
                    content = content.replace("```json", "", 1)
                if content.startswith("```"):
                    content = content.replace("```", "", 1)
                if content.endswith("```"):
                    content = content.rsplit("```", 1)[0]
                
                content = content.strip()
                
                # 🔥 ROBUST: Extract only the JSON object, ignore extra text
                # Find the first { and match its closing }
                start_idx = content.find("{")
                if start_idx == -1:
                    raise ValueError("No JSON object found in response")
                
                # Count braces to find the matching closing brace
                brace_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(content)):
                    if content[i] == "{":
                        brace_count += 1
                    elif content[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                # Extract only the JSON part
                json_str = content[start_idx:end_idx]
                
                analysis = json.loads(json_str)
                self.logger.info(f"✅ [IntentAnalysis] JSON parsed successfully")
            except Exception as e:
                # Catch ALL parsing errors - DO NOT RE-RAISE
                self.logger.warning(f"⚠️ [IntentAnalysis] JSON parsing failed: {e}")
                self.logger.warning(f"⚠️ [IntentAnalysis] Content preview: {content[:200] if content else 'EMPTY'}...")
                analysis = None  # Will trigger fallback below
            
            # Ensure analysis is set (fallback if parsing failed)
            if analysis is None:
                self.logger.info("🔄 [IntentAnalysis] Using fallback default analysis")
                analysis = {
                    "intent": "unknown",
                    "required_context": [],
                    "enhanced_query": current_query,
                    "todo_list": [],
                    "language": "pt",
                    "complexity": "simple",
                    "needs_tools": False
                }
            
            # Store analysis in context
            await context.set_memory("intent_analysis", analysis)
            await context.set_memory("todo_list", analysis.get("todo_list", []))
            await context.set_memory("user_language", analysis.get("language", "pt"))
            await context.set_memory("intent_analyzed", True)
            
            # Enhance the query with context
            enhanced_query = analysis.get("enhanced_query", current_query)
            context.user_query = enhanced_query
            
            # 🔥 LOG RESULTS
            self.logger.info(f"✅ [IntentAnalysis] Intent: {analysis.get('intent', 'unknown')}")
            self.logger.info(f"🌍 [IntentAnalysis] Language: {analysis.get('language', 'unknown')}")
            self.logger.info(f"📝 [IntentAnalysis] Todo list ({len(analysis.get('todo_list', []))} items):")
            for i, task in enumerate(analysis.get('todo_list', []), 1):
                self.logger.info(f"   {i}. {task}")
            
            # Log context from history if available
            # 🔥 Handle new key 'required_context' or old 'context_from_history'
            context_items = analysis.get('required_context') or analysis.get('context_from_history', [])
            
            # Store normalized context
            analysis['context_from_history'] = context_items
            await context.set_memory("intent_analysis", analysis)
            
            if context_items:
                self.logger.info(f"📚 [IntentAnalysis] Context from history:")
                for item in context_items:
                    self.logger.info(f"   - {item}")
            
            self.logger.info(f"🔄 [IntentAnalysis] Enhanced query: {enhanced_query[:100]}...")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"❌ [IntentAnalysis] Failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Set flag even on error to prevent infinite loop
            await context.set_memory("intent_analyzed", True)
            await context.set_memory("user_language", "pt")  # Default

    async def dispatch(self, context: AsyncExecutionContext, initial_state_name: str = None):
        """
        Async state machine dispatch loop with transition resolution.
        """
        # Check if in fork
        is_fork = await context.get_memory("branch_id") is not None
        
        current_state = None

        # 🔥 0. Forced initial state via arg
        if initial_state_name and initial_state_name in self.states:
            current_state = self.states[initial_state_name]
            self.logger.info(f"🎯 [Engine] Dispatch starting from forced state: {initial_state_name}")

        # 🔥 1. Start from valid global initial state (if not forced)
        elif self.initial_state_name and self.initial_state_name in self.states and not is_fork:
            current_state = self.states[self.initial_state_name]
            self.logger.info(f"🎯 [Engine] Starting from custom global initial state: {self.initial_state_name}")
            # Check if already analyzed to prevent loops
            intent_analyzed = await context.get_memory("intent_analyzed")
            if not intent_analyzed and self.enable_intent_analysis:
                current_state = self.states["IntentAnalysisState"]
                self.logger.info("🎯 [Engine] Starting with Intent Analysis (Initial State Override)")
            # else: current_state remains custom initial state

        # 🔥 2. Default flow: Check Intent Analysis before Router
        elif not is_fork and self.enable_intent_analysis:
            intent_analyzed = await context.get_memory("intent_analyzed")
            if not intent_analyzed:
                current_state = self.states["IntentAnalysisState"]
                self.logger.info("🎯 [Engine] Starting default flow with Intent Analysis")
            else:
                current_state = self.router_state

        else:
            current_state = self.router_state

        while current_state:
            state_name = type(current_state).__name__
            self.logger.info(f"📍 [Engine] Current state: {state_name}")

            # 🔥 MIDDLEWARE HOOK: Check lineage for ContextPolicy and trigger
            # This ensures pruning happens before the specific state logic
            temp_state = current_state
            while temp_state:
                if isinstance(temp_state, AsyncContextPolicyState):
                    await temp_state.on_enter(context)
                    break
                temp_state = temp_state.parent

            # Handle state (async)
            result = await current_state.handle(context)

            if result is None:
                self.logger.info(f"🏁 [Engine] Reached terminal state: {state_name}")
                break
            
            # 🔥 Apply transition overrides (custom state injection)
            if isinstance(result, Transition):
                original_target = result.to
                
                for override in self.transition_overrides:
                    if override["from"] == state_name:
                        try:
                            if override["condition"](context, result):
                                result = Transition(
                                    to=override["target"],
                                    reason=f"Override: {result.reason}",
                                    metadata=result.metadata
                                )
                                self.logger.info(f"🔀 [Override] {state_name}: {original_target} -> {result.to}")
                                break
                        except Exception as e:
                            self.logger.error(f"❌ [Override] Condition failed: {e}")
                            # Continue without override

            # Resolve transition
            if isinstance(result, Transition):
                next_state = self._find_state_provider(result.to)
                self.logger.info(f"🔄 Transition: {state_name} -> {result.to} (reason: {result.reason})")
                if result.metadata:
                    self.logger.debug(f"   Metadata: {result.metadata}")
            else:
                # Legacy: direct state return
                next_state = result
                next_name = type(next_state).__name__
                self.logger.info(f"🔄 Transition: {state_name} -> {next_name}")

            # Save snapshot
            self._save_snapshot(context, f"transition_{state_name}_to_{type(next_state).__name__}")

            current_state = next_state

    async def run(self, query: str, chat_history=None) -> AsyncExecutionContext:
        """
        Run agent (async).
        """
        # Create context with safety monitor
        monitor = SafetyMonitor(max_requests=self.max_global_requests)
        context = AsyncExecutionContext(
            user_query=query, 
            safety_monitor=monitor,
            max_iterations=self.max_retries
        )
        
        await context.set_memory("system_instruction", self.system_instruction)
        await context.set_memory("redirect_system_prompt", self.redirect_system_prompt)  # Store redirect prompt
        await context.set_memory("chat_history", chat_history or [])

        await self.dispatch(context)

        return context

    async def run_stream(self, query: str, chat_history=None) -> AsyncIterator[str]:
        """
        Run with streaming (async generator).
        """
        context = await self.run(query, chat_history)

        # Stream from context memory (not from state instance)
        stream = await context.get_memory("answer_stream")
        if stream:
            async for token in stream:
                yield token

# =================================================================================================
# To be implemented
# =================================================================================================

class ResponseValidatorState(AsyncHierarchicalState):
    """
    Standard state for validating final answers.
    """
    def __init__(self, parent, llm):
        super().__init__(parent)
        self.llm = llm
        
    async def handle(self, context: AsyncExecutionContext):
        return Transition(to="TerminalState", reason="Validator placeholder passed")


class HumanFeedbackState(AsyncHierarchicalState):
    """
    State for Human-in-the-Loop interaction.
    Pauses execution to request user feedback/approval before proceeding.
    """
    async def handle(self, context: AsyncExecutionContext):
        # TODO: Implement pause/resume mechanism via WebSocket or API callback
        # Logic:
        # 1. Send "Review Request" event to Client
        # 2. Save state snapshot
        # 3. Wait for external trigger/API call with approval
        return Transition(to="AnswerState", reason="Human feedback passed")