"""
Fork Contract Strategies
========================

Defines strategies for extracting contracts from fork execution.
Allows developers to choose between complex epistemic extraction (JSON)
or simple text summarization.
"""

from abc import ABC, abstractmethod
import logging
import json
from typing import Any
from core.context_async import AsyncExecutionContext
from finitestatemachineAgent.fork_contracts import ForkResult, Claim, EvidenceType

logger = logging.getLogger("AsyncAgentEngine")

class ForkContractStrategy(ABC):
    """Base strategy for extracting contracts from fork execution."""
    
    def __init__(self, llm):
        self.llm = llm

    @abstractmethod
    async def extract(self, context: AsyncExecutionContext, branch_id: str, branch_goal: str) -> Any:
        """Extract contract from context."""
        pass


class EpistemicContractStrategy(ForkContractStrategy):
    """
    Current Default: Extracts structured claims with epistemic evidence.
    Returns: ForkResult (Pydantic model)
    """
    
    async def extract(self, context: AsyncExecutionContext, branch_id: str, branch_goal: str) -> ForkResult:
        logger.info("📊 [ForkContract] Strategy: Epistemic Extraction")
        
        research_notes = await context.get_memory("research_notes", "")
        
        # Build claim extraction prompt
        system_instruction = """You are a claim extraction worker with EPISTEMIC REASONING.

Your job:
1. Analyze research results from tools
2. Extract VERIFIABLE CLAIMS in structured format
3. Mark what you covered and what you're uncertain about

Output ONLY valid JSON in this exact format:
{
  "claims": [
    {"key": "topic.subtopic", "value": "actual value", "evidence": [{"type": "retrieved", "source": "tool:tool_name"}], "confidence": 0.9}
  ],
  "coverage": ["topic1", "topic2"],
  "uncertain_topics": [{"topic": "topic3", "reason": "not_found_in_retrieval"}]
}

EPISTEMIC RULES:
1. EVIDENCE TYPES (required):
   - "retrieved": From tool results/documents
   - "general_knowledge": From your training
   - "inferred": Logical deduction

2. CONFIDENCE:
   - Retrieved: 0.7-1.0
   - General knowledge: 0.4-0.7
   - Inferred: 0.3-0.6

DO NOT summarize. ONLY extract verifiable facts."""
        
        # Prepare tool results
        tool_results_text = ""
        if context.tool_calls:
            for call in context.tool_calls:
                tool_name = call.get("tool_name", "unknown")
                result = call.get("result", "")
                tool_results_text += f"\nTool: {tool_name}\nResult: {str(result)[:500]}\n"
        
        if research_notes:
            tool_results_text += f"\nDirect notes: {research_notes}\n"
        
        # Combine into single user message
        user_content = f"Research goal: {branch_goal}\n\nTool results:\n{tool_results_text}\n\nExtract claims as JSON:"
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]
        
        # Call LLM
        response = await self.llm.chat(messages)
        content = response.get("content", "").strip()
        
        try:
            # Parse JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            contract_data = json.loads(content.strip())
            
            # Map to Pydantic
            claims = [Claim(**c) for c in contract_data.get("claims", [])]
            
            # Calibrate confidence
            for claim in claims:
                for ev in claim.evidence:
                    if ev.type == EvidenceType.GENERAL_KNOWLEDGE:
                        claim.confidence = min(claim.confidence, 0.7)
                    elif ev.type == EvidenceType.INFERRED:
                        claim.confidence = min(claim.confidence, 0.6)

            return ForkResult(
                branch_id=branch_id,
                claims=claims,
                coverage=contract_data.get("coverage", []),
                uncertain_topics=contract_data.get("uncertain_topics", []),
                confidence=sum(c.confidence for c in claims) / len(claims) if claims else 0.0
            )

        except Exception as e:
            logger.error(f"❌ [EpistemicStrategy] Failed: {e}")
            return ForkResult(branch_id=branch_id, confidence=0.0)


class SimpleTextContractStrategy(ForkContractStrategy):
    """
    New Strategy: Simply summarizes the findings in text.
    Returns: ForkResult (wrapping text summary in a claim for compatibility)
    """
    
    async def extract(self, context: AsyncExecutionContext, branch_id: str, branch_goal: str) -> ForkResult:
        logger.info("📊 [ForkContract] Strategy: Simple Text Summary")
        
        research_notes = await context.get_memory("research_notes", "")
        
        system_instruction = """You are a research assistant.
Summarize the findings for the given research goal based on the tool results.
Be concise and direct. Focus on the answer."""

        tool_results_text = ""
        if context.tool_calls:
            for call in context.tool_calls:
                tool_name = call.get("tool_name", "unknown")
                result = call.get("result", "")
                tool_results_text += f"\nTool: {tool_name}\nResult: {str(result)[:500]}\n"
        
        if research_notes:
            tool_results_text += f"\nDirect notes: {research_notes}\n"
            
        # Combine into single user message
        user_content = f"Research Goal: {branch_goal}\n\nEvidence:\n{tool_results_text}\n\nSummary:"
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]
        
        response = await self.llm.chat(messages)
        summary = response.get("content", "").strip()
        
        # Wrap in ForkResult for compatibility
        # We create a single 'summary' claim containing the text
        claim = Claim(
            key="summary",
            value=summary,
            evidence=[{"type": EvidenceType.RETRIEVED, "source": "llm_summary"}],
            confidence=1.0
        )
        
        return ForkResult(
            branch_id=branch_id,
            claims=[claim],
            coverage=[branch_goal],
            uncertain_topics=[],
            confidence=1.0
        )
