"""
LLM Synthesis Strategy
=======================

LLM-driven implementation of synthesis strategy.

Engine doesn't see:
- Prompt engineering
- Model selection
- Temperature
- Token limits
"""

import logging
from finitestatemachineAgent.synthesis_strategy import SynthesisStrategy
from finitestatemachineAgent.synthesis_contracts import SynthesisRequest, SynthesisResult

logger = logging.getLogger("AsyncAgentEngine")


class LLMSynthesisStrategy(SynthesisStrategy):
    """
    LLM-driven synthesis implementation.
    
    Consolidates multiple research outputs into one coherent answer.
    """
    
    def __init__(self, llm, temperature: float = 0.3):
        """
        Initialize LLM synthesis strategy.
        
        Args:
            llm: LLM client
            temperature: Sampling temperature
        """
        self.llm = llm
        self.temperature = temperature
    
    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        """
        Synthesize outputs using LLM.
        
        Args:
            request: Synthesis request
            
        Returns:
            Synthesis result with consolidated answer
            
        Raises:
            ValueError: If synthesis fails
        """
        logger.info(f"🧬 [LLMSynthesis] Synthesizing {len(request.fork_outputs)} outputs")
        
        # Build synthesis prompt
        system_prompt = """You are a synthesis expert. Consolidate multiple research outputs into one coherent answer.

EPISTEMIC SYNTHESIS:

1. EVIDENCE PRIORITIZATION:
   - Prefer claims with "retrieved" evidence (from documents/tools)
   - Use "general_knowledge" claims to fill conceptual gaps
   - Use "inferred" claims when necessary for coherence

2. CONDITIONAL LANGUAGE:
   - For retrieved evidence: Use assertive language ("is", "are", "does")
   - For general knowledge: Use "typically", "generally", "in most cases"
   - For inferred: Use "likely", "suggests", "appears to"

3. UNCERTAINTY HANDLING:
   - Absence of evidence ≠ evidence of absence
   - State what is uncertain and WHY
   - Do NOT invent facts, but DO provide context

4. TRANSPARENCY:
   - Present conflicts without choosing sides
   - Acknowledge gaps in knowledge
   - Use natural language (no technical jargon)

DO NOT mention:
- "evidence types"
- "contracts"
- "forks"
- "variants"
- "retrieved/general_knowledge/inferred"

Present information as if you researched it directly."""
        
        # Format outputs
        outputs_text = "\n\n---\n\n".join(
            f"Research Finding {i+1}:\n{output}" 
            for i, output in enumerate(request.fork_outputs)
        )
        
        # Add constraints if any
        constraints_text = ""
        if request.constraints:
            constraints_text = f"\n\nConstraints:\n" + "\n".join(f"- {c}" for c in request.constraints)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User's question: {request.task_description}"},
            {"role": "user", "content": f"Research findings:\n\n{outputs_text}{constraints_text}"},
            {"role": "user", "content": f"Synthesize these findings into one coherent answer in {request.output_format} format:"}
        ]
        
        # Call LLM
        try:
            response = await self.llm.chat(messages)
            
            # Track usage if available
            usage = response.get('usage', {})
            if usage:
                logger.info(f"📊 [LLMSynthesis] Token usage: {usage.get('total_tokens', 0)} tokens")
            
            answer = response["content"]
            
            # Validate non-empty
            if not answer or len(answer.strip()) == 0:
                raise ValueError("LLM returned empty synthesis")
            
            logger.info(f"✅ [LLMSynthesis] Synthesis complete ({len(answer)} chars)")
            
            # Parse response (best effort - could be enhanced to extract metadata)
            return SynthesisResult(
                answer=answer,
                confidence=None,  # Could be extracted from response
                gaps=None,  # Could be extracted from response
                inconsistencies=None  # Could be extracted from response
            )
            
        except Exception as e:
            logger.error(f"❌ [LLMSynthesis] Failed: {e}")
            raise ValueError(f"Synthesis failed: {e}")
