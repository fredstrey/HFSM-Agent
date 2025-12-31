"""
State Transitions
==================

Defines valid state transitions and provides validation.
"""

from enum import Enum
from typing import List, Tuple, Set
import logging

logger = logging.getLogger(__name__)


class StateTransition(Enum):
    """
    Valid state transitions in the HFSM.
    
    Each transition is a tuple of (from_state, to_state).
    """
    # Router transitions
    ROUTER_TO_TOOL = ("RouterState", "ToolState")
    ROUTER_TO_ANSWER = ("RouterState", "AnswerState")
    
    # Tool execution flow
    TOOL_TO_VALIDATION = ("ToolState", "ValidationState")
    
    # Validation outcomes
    VALIDATION_TO_ROUTER = ("ValidationState", "RouterState")
    VALIDATION_TO_RETRY = ("ValidationState", "RetryState")
    
    # Retry logic
    RETRY_TO_ROUTER = ("RetryState", "RouterState")
    RETRY_TO_FAIL = ("RetryState", "FailState")
    
    # Terminal states (no outgoing transitions)
    # AnswerState and FailState are terminal
    
    @property
    def from_state(self) -> str:
        """Get source state."""
        return self.value[0]
    
    @property
    def to_state(self) -> str:
        """Get destination state."""
        return self.value[1]


class TransitionValidator:
    """
    Validates state transitions against allowed transitions.
    
    Prevents invalid state transitions that could lead to
    undefined behavior or infinite loops.
    
    Example:
        >>> validator = TransitionValidator()
        >>> validator.validate("RouterState", "ToolState")  # True
        >>> validator.validate("AnswerState", "RouterState")  # False
    """
    
    # Set of all valid transitions
    VALID_TRANSITIONS: Set[Tuple[str, str]] = {t.value for t in StateTransition}
    
    @classmethod
    def validate(cls, from_state: str, to_state: str) -> bool:
        """
        Check if transition is valid.
        
        Args:
            from_state: Source state name
            to_state: Destination state name
            
        Returns:
            True if transition is allowed, False otherwise
        """
        transition = (from_state, to_state)
        is_valid = transition in cls.VALID_TRANSITIONS
        
        if not is_valid:
            logger.warning(
                f"Invalid transition attempted: {from_state} -> {to_state}"
            )
        
        return is_valid
    
    @classmethod
    def get_allowed_transitions(cls, from_state: str) -> List[str]:
        """
        Get list of allowed destination states from given state.
        
        Args:
            from_state: Source state name
            
        Returns:
            List of allowed destination state names
        """
        return [
            to_state 
            for (frm, to_state) in cls.VALID_TRANSITIONS 
            if frm == from_state
        ]
    
    @classmethod
    def is_terminal_state(cls, state_name: str) -> bool:
        """
        Check if state is terminal (no outgoing transitions).
        
        Args:
            state_name: State name to check
            
        Returns:
            True if state has no outgoing transitions
        """
        return len(cls.get_allowed_transitions(state_name)) == 0
    
    @classmethod
    def validate_or_raise(cls, from_state: str, to_state: str):
        """
        Validate transition and raise exception if invalid.
        
        Args:
            from_state: Source state name
            to_state: Destination state name
            
        Raises:
            ValueError: If transition is not allowed
        """
        if not cls.validate(from_state, to_state):
            allowed = cls.get_allowed_transitions(from_state)
            raise ValueError(
                f"Invalid transition: {from_state} -> {to_state}. "
                f"Allowed transitions from {from_state}: {allowed}"
            )
