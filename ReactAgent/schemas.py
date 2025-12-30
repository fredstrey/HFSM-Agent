from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import BaseModel, Field

class ReActDecision(str, Enum):
    """Possible decisions from ReAct analysis"""
    CONTINUE = "continue"  # Information is sufficient, proceed to response
    RETRY_WITH_REFINEMENT = "retry_with_refinement"  # Refine query and try again
    CALL_DIFFERENT_TOOL = "call_different_tool"  # Try a different tool
    INSUFFICIENT_DATA = "insufficient_data"  # No more attempts, use what we have

class ReActAnalysis(BaseModel):
    """Structured analysis from ReAct agent"""
    decision: ReActDecision
    reasoning: str
    refined_query: Optional[str] = None
    suggested_tool: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

class AgentResponse(BaseModel):
    """Base response model for the agent"""
    answer: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
