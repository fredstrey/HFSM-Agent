"""
Synthesis Strategy Interface
=============================

Abstract interface for synthesis strategies.
Engine depends on this interface, never on implementation.
"""

from abc import ABC, abstractmethod
from finitestatemachineAgent.synthesis_contracts import SynthesisRequest, SynthesisResult


class SynthesisStrategy(ABC):
    """
    Abstract synthesis strategy.
    
    Implementations can use:
    - LLM (semantic synthesis)
    - Rules (deterministic synthesis)
    - Hybrid approaches
    
    Engine doesn't care HOW synthesis happens.
    """
    
    @abstractmethod
    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        """
        Synthesize multiple outputs into one coherent response.
        
        Args:
            request: Synthesis request with task and outputs
            
        Returns:
            Synthesis result with answer and metadata
            
        Raises:
            ValueError: If synthesis fails
        """
        pass
