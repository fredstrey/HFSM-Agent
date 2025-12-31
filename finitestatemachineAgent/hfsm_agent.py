from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Generator, Optional, List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.context import ExecutionContext

# =================================================================================================
# 🔹 Phase 1: Hierarchy Fundamentals
# =================================================================================================

class HierarchicalState(ABC):
    """
    Base class for all states in the Hierarchical Finite State Machine.
    Supports parent-child relationships and delegation.
    """
    def __init__(self, parent: Optional[HierarchicalState] = None):
        self.parent = parent

    @abstractmethod
    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        """
        Process the current context and return the next state.
        If None is returned, the event is delegated to the parent.
        """
        pass

    def on_enter(self, context: ExecutionContext):
        """Optional hook called when entering this state."""
        pass

    def on_exit(self, context: ExecutionContext):
        """Optional hook called when exiting this state."""
        pass

    def find_state_by_type(self, type_name: str) -> HierarchicalState:
        """Traverse up the hierarchy to find a state provider."""
        if self.parent:
            return self.parent.find_state_by_type(type_name)
        raise Exception(f"State provider for {type_name} not found in hierarchy.")

# =================================================================================================
# 🔹 Phase 3: Define Superstates (Parents)
# =================================================================================================

class AgentRootState(HierarchicalState):
    """
    The root of the hierarchy. Handles global issues (fatal errors, generic fallbacks).
    Doesn't have a parent.
    """
    def __init__(self):
        super().__init__(parent=None)

    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        # If we reached the root without a transition, it's a deadlock or failure.
        print("❌ [Root] No state handled the context. Terminating.")
        return FailState(self)
        
class ReasoningState(HierarchicalState):
    """Parent for logic that involves thinking/routing."""
    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        return None # Delegate to parent

class ExecutionState(HierarchicalState):
    """Parent for tool execution and validation."""
    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        return None # Delegate to parent

class RecoveryState(HierarchicalState):
    """Parent for handling retries and errors."""
    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        return None # Delegate to parent

class TerminalState(HierarchicalState):
    """Marker for final states (Answer/Fail). Stops the engine."""
    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        return None # Delegate to parent

# =================================================================================================
# 🔹 Phase 4: Migrate Substates
# =================================================================================================

class RouterState(ReasoningState):
    def __init__(self, parent: HierarchicalState, llm, registry):
        super().__init__(parent)
        self.llm = llm
        self.registry = registry

    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        system_instruction = context.get_memory("system_instruction", "")
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": context.user_query},
        ]

        # Reconstruct chat history if available (not implemented generically in context yet)
        if hasattr(context, "chat_history") and context.chat_history:
             pass

        # Reconstruct history with tool calls (PRUNED FOR ROUTER)
        # Strategy: Keep full content only for the last 2 interactions. Truncate others.
        total_calls = len(context.tool_calls)
        for i, call in enumerate(context.tool_calls):
            # Check if this is a "recent" call (e.g., last 2)
            is_recent = (total_calls - i) <= 4
            
            tool_call_id = f"call_{call['tool_name']}_{call.get('iteration', 0)}_{i}"
            
            messages.append({
                "role": "assistant",
                "content": None, 
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": call["tool_name"],
                            "arguments": json.dumps(call["arguments"], ensure_ascii=False)
                        }
                    }
                ]
            })

            try:
                raw_result = str(call.get("result", ""))
                # PRUNING LOGIC
                if not is_recent and len(raw_result) > 200:
                    content_str = raw_result[:200] + "... [TRUNCATED - OLD CONTEXT]"
                else:
                    content_str = raw_result
            except:
                content_str = str(call.get("result", ""))
                
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": call["tool_name"],
                "content": content_str
            })

        print("🧠 [Router] Thinking...")
        response = self.llm.chat_with_tools(
            messages=messages,
            tools=self.registry.to_openai_format()
        )
        
        # Accumulate usage
        usage = response.get("usage", {})
        total_usage = context.get_memory("total_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        total_usage["total_tokens"] += usage.get("total_tokens", 0)
        context.set_memory("total_usage", total_usage)

        if response.get("tool_calls"):
            context.set_memory("pending_tool_calls", response["tool_calls"])
            # Transition to ToolState (which is under ExecutionState)
            return ToolState(self.parent.find_state_by_type("ExecutionState"), context.get_memory("executor"))

        context.set_memory("last_llm_content", response.get("content", ""))
        return AnswerState(self.parent.find_state_by_type("TerminalState"), self.llm)

class ToolState(ExecutionState):
    def __init__(self, parent: HierarchicalState, executor, max_workers: int = 4):
        super().__init__(parent)
        self.executor = executor
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        futures_map = {}
        pending_calls = context.get_memory("pending_tool_calls", [])

        if not pending_calls:
             return None 

        for call in pending_calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"])
            except:
                args = {}
            
            print(f"🛠️ [Tool] Executing: {name}")
            future = self.pool.submit(self.executor.execute, name, args)
            futures_map[future] = call

        for future in as_completed(futures_map):
            call = futures_map[future]
            try:
                result_map = future.result()
                result = result_map.get("result") if result_map.get("success") else result_map.get("error")
            except Exception as e:
                result = str(e)
                
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"])
            except:
                args = {}
                
            context.add_tool_call(name, args, result)

        context.set_memory("pending_tool_calls", [])
        
        return ValidationState(self.parent, context.get_memory("llm"))


class ValidationState(ExecutionState):
    def __init__(self, parent: HierarchicalState, llm):
        super().__init__(parent)
        self.llm = llm

    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        print("🔍 [Validation] Checking data...")
        prompt = f"""
        Você é um validador lógico.
        Verifique se as informações coletadas são suficientes para responder.
        
        PERGUNTA:
        {context.user_query}
        
        DADOS:
        {json.dumps(context.tool_calls, ensure_ascii=False, default=str)}
        
        Responda apenas:
        {{"valid": true}} ou {{"valid": false}}
        """

        try:
            response_dict = self.llm.chat([{"role": "user", "content": prompt}])
            
            # Accumulate usage
            usage = response_dict.get("usage", {})
            total_usage = context.get_memory("total_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            total_usage["total_tokens"] += usage.get("total_tokens", 0)
            context.set_memory("total_usage", total_usage)
            
            response_content = response_dict.get("content", "")
            
            clean_resp = response_content.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean_resp)
            is_valid = result.get("valid", False)
        except:
            is_valid = False

        if is_valid:
             # Go to Answer
             return AnswerState(self.parent.find_state_by_type("TerminalState"), self.llm)
        else:
             # Go to Retry
             return RetryState(self.parent.find_state_by_type("RecoveryState"))


class RetryState(RecoveryState):
    def __init__(self, parent: HierarchicalState):
        super().__init__(parent)

    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        print("⚠️ [Retry] Attempting recovery...")
        retry_count = context.get_memory("retry_count", 0)
        max_retries = context.get_memory("max_retries", 2)
        
        retry_count += 1
        context.set_memory("retry_count", retry_count)
        
        if retry_count > max_retries:
            print("❌ [Retry] Max retries reached.")
            return FailState(self.parent.find_state_by_type("TerminalState"))

        context.user_query = f"Refine melhor:\n{context.user_query}"
        
        # Back to Router (in Reasoning)
        return RouterState(self.parent.find_state_by_type("ReasoningState"), context.get_memory("llm"), context.get_memory("registry"))


class AnswerState(TerminalState):
    def __init__(self, parent: HierarchicalState, llm):
        super().__init__(parent)
        self.llm = llm
        self.generator = None

    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        """
        Constructs the final prompt with full context and initializes the streaming generator.
        
        This state does NOT return a next state immediately. It prepares the generator
        which the AgentEngine will yield from.
        """
        print("✅ [Answer] Generating...")
        
        system_instruction = context.get_memory("system_instruction", "")
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": context.user_query},
        ]
        
        # Append tool interactions to context
        for call in context.tool_calls:
            tool_call_id = f"call_{call['tool_name']}_{call.get('iteration', 0)}"
            messages.append({
                "role": "assistant",
                "content": None, 
                "tool_calls": [
                    {"id": tool_call_id, "type": "function", "function": {"name": call["tool_name"], "arguments": json.dumps(call["arguments"])}}
                ]
            })
            content_str = str(call.get("result", ""))
            messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": call["tool_name"], "content": content_str})

        messages.append({
            "role": "system",
            "content": "Based on the tool results above, provide a clear and direct answer to the user's question. Do NOT call any more tools. Just answer."
        })

        # Generator that streams and updates context usage side-effect
        self.generator = self.llm.chat_stream(messages, context=context)
        return None # Stay here / Finish

class FailState(TerminalState):
    def __init__(self, parent: HierarchicalState):
        super().__init__(parent)
        self.generator = None

    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        print("❌ [Fail] Terminating.")
        def fail_gen():
            yield "Erro: Não foi possível obter as informações necessárias após várias tentativas."
        self.generator = fail_gen()
        return None 

# =================================================================================================
# 🔹 Phase 2 & 8: Generic Engine
# =================================================================================================

class AgentEngine:
    def __init__(
        self,
        llm,
        registry,
        executor,
        system_instruction: str = ""
    ):
        self.llm = llm
        self.registry = registry
        self.executor = executor
        self.system_instruction = system_instruction
        
        # Initialize Hierarchy Root
        self.root = AgentRootState()
        self.reasoning = ReasoningState(self.root)
        self.execution = ExecutionState(self.root)
        self.recovery = RecoveryState(self.root)
        self.terminal = TerminalState(self.root)
        
        # Allow states to verify/find peers (basic service locator via root)
        # In a cleaner implementation, we would inject these fully.
        self.root.find_state_by_type = self._find_state_provider

    def _find_state_provider(self, type_name: str) -> HierarchicalState:
        if type_name == "ReasoningState": return self.reasoning
        if type_name == "ExecutionState": return self.execution
        if type_name == "RecoveryState": return self.recovery
        if type_name == "TerminalState": return self.terminal
        return self.root

    def dispatch(self, state: HierarchicalState, context: ExecutionContext) -> HierarchicalState:
        """
        The generic dispatch loop. Bubbles events up if handle() returns None.
        Returns the NEXT state.
        """
        start_state = state
        while state:
            next_state = state.handle(context)
            if next_state:
                return next_state # Transition found
            
            # Delegate to parent
            if state.parent:
                # print(f"⬆️ Delegating from {state.__class__.__name__} to {state.parent.__class__.__name__}")
                state = state.parent
            else:
                # Root reached and returned None -> Stop or Error?
                # If it's TerminalState, it's fine.
                if isinstance(start_state, TerminalState):
                    return start_state 
                return FailState(self.root) # Fallback

        return FailState(self.root)

    def run_stream(
        self, 
        query: str,
        chat_history: Optional[list] = None
    ) -> tuple[Generator[str, None, None], ExecutionContext]:
        
        context = ExecutionContext(user_query=query)
        context.set_memory("system_instruction", self.system_instruction)
        context.set_memory("chat_history", chat_history or [])
        context.set_memory("retry_count", 0)
        context.set_memory("max_retries", 2)
        context.set_memory("pending_tool_calls", [])
        
        # Inject dependencies into memory for states to access when recreating siblings
        context.set_memory("llm", self.llm)
        context.set_memory("registry", self.registry)
        context.set_memory("executor", self.executor)

        # Initial State
        current_state = RouterState(self.reasoning, self.llm, self.registry)

        while True:
            # print(f"🔄 [Engine] Current State: {current_state.__class__.__name__}")
            
            # Dispatch returns the NEW state (or the same if terminal)
            next_state = self.dispatch(current_state, context)
            
            # Check if Terminal
            if isinstance(next_state, TerminalState):
                # Ensure generator is ready
                gen = next_state.generator
                if not gen:
                     # Should have been set in handle()
                     next_state.handle(context) 
                     gen = next_state.generator
                
                # CLEANUP: Remove non-serializable objects from memory before returning context to API
                context.memory.pop("llm", None)
                context.memory.pop("registry", None)
                context.memory.pop("executor", None)
                
                return gen, context

            current_state = next_state
