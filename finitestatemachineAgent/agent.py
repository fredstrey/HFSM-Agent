"""
High-Level Agent API
====================

Developer-friendly interface for the HFSM Agent.
Inspired by LangChain, Pydantic AI, and other modern frameworks.

Usage:
    Basic:
        agent = Agent(llm_provider="openrouter", model="gemini-2.0")
        response = await agent.run("Hello!")
    
    Advanced:
        agent = Agent(
            llm_provider="openrouter",
            model="gemini-2.0",
            system_instruction="You are a Python expert",
            tools=[search_web, run_code],
            enable_parallel_planning=True
        )
        async for chunk in agent.stream("Explain decorators"):
            print(chunk, end="")
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, AsyncIterator, Union
from datetime import datetime

from finitestatemachineAgent.hfsm_agent_async import AsyncAgentEngine
from finitestatemachineAgent.contract_strategies import ForkContractStrategy, SimpleTextContractStrategy, EpistemicContractStrategy
from finitestatemachineAgent.llm_synthesis_strategy import LLMSynthesisStrategy, ConcatenationSynthesisStrategy
from finitestatemachineAgent.synthesis_strategy import SynthesisStrategy
from providers.llm_client_async import AsyncLLMClient
from providers.openrouter_async import AsyncOpenRouterProvider
from core.executor_async import AsyncToolExecutor
from core.registry import ToolRegistry
from core.context_async import AsyncExecutionContext, SafetyMonitor

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Response from Agent execution."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_usage: Dict[str, int] = field(default_factory=dict)
    tool_calls: List[Dict] = field(default_factory=list)
    confidence: Optional[float] = None
    
    def __str__(self) -> str:
        return self.content


class Agent:
    """
    High-level Agent interface for HFSM Agent.
    
    Simplifies agent creation and usage with sensible defaults.
    
    Example:
        >>> agent = Agent(llm_provider="openrouter", model="gemini-2.0")
        >>> response = await agent.run("What is Python?")
        >>> print(response.content)
    """
    
    def __init__(
        self,
        # LLM Configuration
        llm_provider: str = "openrouter",
        model: str = "xiaomi/mimo-v2-flash:free",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        
        # Persona & Prompts
        system_instruction: str = "You are a helpful assistant.",
        redirect_prompt: Optional[str] = None,
        
        # Tools
        tools: Optional[List[Callable]] = None,
        
        # Features
        enable_parallel_planning: bool = False,
        enable_intent_analysis: bool = True,
        max_parallel_branches: int = 3,
        
        # Safety & Limits
        max_global_requests: int = 50,
        skip_validation: bool = True,
        
        # Custom Functions & Hooks
        validation_fn: Optional[Callable] = None,
        planning_system_prompt: Optional[Callable] = None,
        post_router_hook: Optional[Callable] = None,
        
        # Workflow & Subagent Support
        initial_state: Optional[str] = None,  # e.g., "AnswerState" for simple agents
        
        # Strategy Configuration
        contract_strategy: Union[str, ForkContractStrategy] = "epistemic",
        synthesis_strategy: Union[str, SynthesisStrategy] = "llm",
        
        # Advanced (pass-through to AsyncAgentEngine)
        **kwargs
    ):
        """
        Initialize Agent with configuration.
        
        Args:
            llm_provider: LLM provider ("openrouter", "ollama", etc.)
            model: Model identifier
            api_key: API key (defaults to env var)
            base_url: Custom base URL for LLM provider
            system_instruction: Main system prompt/persona
            redirect_prompt: Simplified prompt for trivial queries
            tools: List of tool functions (decorated with @tool)
            enable_parallel_planning: Enable parallel execution
            enable_intent_analysis: Enable intent analysis state
            max_parallel_branches: Max parallel branches per fork
            contract_strategy: "epistemic" (default) or "simple" (text only)
            synthesis_strategy: "llm" (default) or "concat" (simple append)
            max_global_requests: Global safety limit for LLM calls
            skip_validation: Skip tool validation state
            **kwargs: Additional arguments passed to AsyncAgentEngine
        """
        self.llm_provider = llm_provider
        self.model = model
        self.system_instruction = system_instruction
        
        # Setup LLM
        self.llm = self._setup_llm(llm_provider, model, api_key, base_url)
        
        # Setup Tools
        self.registry = ToolRegistry()
        if tools:
            for tool in tools:
                # Tools should be decorated with @tool
                if hasattr(tool, '_tool_name'):
                    self.registry.register(
                        name=tool._tool_name,
                        description=tool._tool_description,
                        function=tool,
                        args_model=tool._args_model
                    )
                else:
                    logger.warning(f"Tool {tool.__name__} not decorated with @tool, skipping")
        
        self.executor = AsyncToolExecutor(self.registry)
        
        # Setup Redirect Prompt
        if redirect_prompt is None:
            redirect_prompt = f"{system_instruction}\n\nResponda de forma breve e direta."
            
        # Resolve Strategies
        resolved_contract_strat = None
        if isinstance(contract_strategy, str):
            if contract_strategy.lower() == "simple":
                resolved_contract_strat = SimpleTextContractStrategy(self.llm)
            elif contract_strategy.lower() == "epistemic":
                resolved_contract_strat = EpistemicContractStrategy(self.llm)
            else:
                logger.warning(f"Unknown contract strategy '{contract_strategy}', using default")
        else:
            resolved_contract_strat = contract_strategy
            
        resolved_synthesis_strat = None
        if isinstance(synthesis_strategy, str):
            if synthesis_strategy.lower() == "concat":
                resolved_synthesis_strat = ConcatenationSynthesisStrategy()
            elif synthesis_strategy.lower() == "llm":
                resolved_synthesis_strat = LLMSynthesisStrategy(self.llm)
            else:
                logger.warning(f"Unknown synthesis strategy '{synthesis_strategy}', using default")
        else:
            resolved_synthesis_strat = synthesis_strategy
        
        # Initialize Engine
        self.engine = AsyncAgentEngine(
            llm=self.llm, # llm instance
            registry=self.registry, # tool registry
            executor=self.executor, # tool executor
            system_instruction=system_instruction, # system prompt
            tool_choice=None,  # Auto-select tools
            redirect_system_prompt=redirect_prompt, # redirect prompt
            skip_validation=skip_validation, # skip tool validation
            validation_fn=validation_fn, # validation function for tools
            enable_parallel_planning=enable_parallel_planning, # enable parallel planning (creates multiple forks)
            planning_system_prompt=planning_system_prompt, # planning system prompt
            contract_strategy=resolved_contract_strat,   # contract strategy for forks
            synthesis_strategy=resolved_synthesis_strat, # synthesis strategy for forks
            enable_intent_analysis=enable_intent_analysis, # enable intent analysis and creates ToDo list
            intent_analysis_llm=self.llm, # intent analysis llm
            max_parallel_branches=max_parallel_branches, # max parallel branches (if enable_parallel_planning)
            max_global_requests=max_global_requests, # max global requests for LLM calls
            post_router_hook=post_router_hook, # post router hook to modify router decision
            initial_state=initial_state,  # Param name is initial_state, not initial_state_name
            **kwargs
        )
        
        logger.info(f"✅ Agent initialized: {llm_provider}/{model}")
        logger.info(f"   Tools: {len(self.registry.tools)} registered")
        logger.info(f"   Parallel: {enable_parallel_planning}, Intent: {enable_intent_analysis}")
        logger.info(f"   Custom functions: validation_fn={validation_fn is not None}, planning_prompt={planning_system_prompt is not None}, post_hook={post_router_hook is not None}")
    
    def _setup_llm(
        self,
        provider: str,
        model: str,
        api_key: Optional[str],
        base_url: Optional[str]
    ) -> AsyncLLMClient:
        """Setup LLM client based on provider."""
        if provider.lower() == "openrouter":
            api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not found in environment")
            
            return AsyncOpenRouterProvider(
                api_key=api_key,
                model=model
            )
        
        elif provider.lower() == "ollama":
            # TODO: Implement Ollama client
            raise NotImplementedError("Ollama provider not yet implemented")
        
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    async def run(
        self,
        query: str,
        chat_history: Optional[List[Dict]] = None
    ) -> AgentResponse:
        """
        Execute query and return complete response.
        
        Args:
            query: User query
            chat_history: Optional chat history
            
        Returns:
            AgentResponse with content, metadata, and token usage
        """
        # Create context
        monitor = SafetyMonitor(max_requests=self.engine.max_global_requests)
        context = AsyncExecutionContext(user_query=query, safety_monitor=monitor)
        
        await context.set_memory("system_instruction", self.system_instruction)
        await context.set_memory("chat_history", chat_history or [])
        
        # Run engine
        await self.engine.dispatch(context)
        
        # Extract response
        final_answer = await context.get_memory("final_answer", "")
        token_usage = await context.get_memory("total_usage", {})
        tool_calls = context.tool_calls or []
        
        return AgentResponse(
            content=final_answer,
            metadata={
                "query": query,
                "timestamp": datetime.now().isoformat()
            },
            token_usage=token_usage,
            tool_calls=tool_calls
        )
    
    async def stream(
        self,
        query: str,
        chat_history: Optional[List[Dict]] = None
    ) -> AsyncIterator[str]:
        """
        Stream response tokens as they are generated.
        
        Args:
            query: User query
            chat_history: Optional chat history
            
        Yields:
            Response tokens
        """
        # Create context
        monitor = SafetyMonitor(max_requests=self.engine.max_global_requests)
        context = AsyncExecutionContext(user_query=query, safety_monitor=monitor)
        
        await context.set_memory("system_instruction", self.system_instruction)
        await context.set_memory("chat_history", chat_history or [])
        await context.set_memory("enable_streaming", True)
        
        # Run engine
        await self.engine.dispatch(context)
        
        # Store context for metadata access
        self.last_context = context
        
        # Stream from context
        stream = await context.get_memory("answer_stream")
        if stream:
            async for token in stream:
                yield token
        else:
            # Fallback: return complete answer
            final_answer = await context.get_memory("final_answer", "")
            yield final_answer
    
    def add_tool(self, tool: Callable) -> None:
        """
        Add a tool to the agent's registry.
        
        Args:
            tool: Tool function decorated with @tool
        """
        self.registry.register(tool)
        logger.info(f"✅ Tool added: {tool.__name__}")
    
    def add_tools(self, tools: List[Callable]) -> None:
        """
        Add multiple tools to the agent's registry.
        
        Args:
            tools: List of tool functions
        """
        for tool in tools:
            self.add_tool(tool)
