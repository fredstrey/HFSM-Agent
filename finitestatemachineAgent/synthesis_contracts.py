"""
Synthesis Contracts
===================

Domain-agnostic contracts for semantic synthesis.
Engine knows WHAT (contracts), not HOW (synthesis).
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SynthesisRequest:
    """
    Domain-agnostic request for synthesis.
    
    Engine passes this to synthesis strategy without knowing domain.
    
    Example:
        request = SynthesisRequest(
            task_description="Compare leasing vs debêntures",
            fork_outputs=["Output 1: ...", "Output 2: ..."],
            constraints=["max 500 words"],
            output_format="markdown"
        )
    """
    task_description: str  # Original user query
    fork_outputs: List[str]  # Text outputs from forks
    constraints: List[str] = field(default_factory=list)  # Optional constraints
    output_format: str = "markdown"  # Desired format


@dataclass
class SynthesisResult:
    """
    Domain-agnostic synthesis result.
    
    Engine validates structure (not content).
    
    Guardrails:
    - answer must not be empty
    - answer must be within token limits
    - required fields must be present
    
    NOT validated:
    - answer quality
    - factual accuracy
    """
    answer: str  # Synthesized response
    confidence: Optional[float] = None  # Optional confidence score (0.0 to 1.0)
    gaps: Optional[List[str]] = None  # Information gaps identified
    inconsistencies: Optional[List[str]] = None  # Conflicts found
