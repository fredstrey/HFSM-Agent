"""
State Factory
=============

Factory pattern for creating states with dependency injection.
"""

from typing import TYPE_CHECKING
from finitestatemachineAgent.hfsm_agent import (
    HierarchicalState,
    RouterState,
    ToolState,
    ValidationState,
    AnswerState,
    RetryState,
    FailState
)

if TYPE_CHECKING:
    from core.agent_config import AgentConfig


class StateFactory:
    """
    Factory for creating state instances with all dependencies injected.
    
    Centralizes state creation logic and makes it easier to modify
    state initialization without changing AgentEngine.
    """
    
    def __init__(self, llm, registry, executor, config: 'AgentConfig'):
        """
        Initialize factory with shared dependencies.
        
        Args:
            llm: LLM client instance
            registry: Tool registry
            executor: Tool executor
            config: Agent configuration
        """
        self.llm = llm
        self.registry = registry
        self.executor = executor
        self.config = config
    
    def create_router_state(self, parent: HierarchicalState) -> RouterState:
        """Create RouterState with dependencies."""
        return RouterState(parent, self.llm, self.registry)
    
    def create_tool_state(self, parent: HierarchicalState) -> ToolState:
        """Create ToolState with dependencies."""
        return ToolState(parent, self.executor)
    
    def create_validation_state(self, parent: HierarchicalState) -> ValidationState:
        """Create ValidationState with custom prompt."""
        state = ValidationState(parent, self.llm)
        # TODO: Inject custom validation prompt from config
        return state
    
    def create_answer_state(self, parent: HierarchicalState) -> AnswerState:
        """Create AnswerState with dependencies."""
        return AnswerState(parent, self.llm)
    
    def create_retry_state(self, parent: HierarchicalState) -> RetryState:
        """Create RetryState with dependencies."""
        return RetryState(parent)
    
    def create_fail_state(self, parent: HierarchicalState) -> FailState:
        """Create FailState with dependencies."""
        return FailState(parent)
    
    def create_state(self, state_type: str, parent: HierarchicalState) -> HierarchicalState:
        """
        Generic state creation method.
        
        Args:
            state_type: Type of state to create (e.g., "RouterState")
            parent: Parent state in hierarchy
            
        Returns:
            Initialized state instance
            
        Raises:
            ValueError: If state_type is unknown
        """
        creators = {
            "RouterState": self.create_router_state,
            "ToolState": self.create_tool_state,
            "ValidationState": self.create_validation_state,
            "AnswerState": self.create_answer_state,
            "RetryState": self.create_retry_state,
            "FailState": self.create_fail_state
        }
        
        creator = creators.get(state_type)
        if not creator:
            raise ValueError(f"Unknown state type: {state_type}")
        
        return creator(parent)
