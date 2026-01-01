"""
Research Fork State
====================

Specialized state for fork execution that bypasses the full Router logic.
Forks execute with minimal context and a focused research goal.
"""

import logging
import json
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger("AsyncAgentEngine")

# Import base classes to avoid circular dependency
from core.context_async import AsyncExecutionContext


class Transition:
    """Transition object for state changes."""
    def __init__(self, to: str, reason: str = "", metadata: dict = None):
        self.to = to
        self.reason = reason
        self.metadata = metadata or {}


class AsyncHierarchicalState(ABC):
    """Base class for async hierarchical states."""
    def __init__(self, parent: Optional['AsyncHierarchicalState'] = None):
        self.parent = parent
    
    @abstractmethod
    async def handle(self, context: AsyncExecutionContext):
        pass
    
    def find_state_by_type(self, type_name: str):
        """Traverse hierarchy to find state."""
        if self.parent:
            return self.parent.find_state_by_type(type_name)
        raise Exception(f"State provider for {type_name} not found")



class ResearchForkState(AsyncHierarchicalState):
    """
    Entry point for fork execution.
    
    Instead of going through RouterState (which would add unnecessary LLM calls),
    forks start here with a focused research goal.
    
    Flow: ResearchForkState -> ToolState (if tools needed) -> ForkSummaryState
    """
    def __init__(self, parent, llm, registry, tool_choice=None):
        super().__init__(parent)
        self.llm = llm
        self.registry = registry
        self.tool_choice = tool_choice
    
    async def handle(self, context: AsyncExecutionContext):
        # Get branch goal from memory (set by ForkDispatchState)
        branch_goal = await context.get_memory("branch_goal", "")
        branch_id = await context.get_memory("branch_id", "unknown")
        
        if not branch_goal:
            logger.error(f"❌ [ResearchFork:{branch_id}] No branch goal found")
            return Transition(to="ForkSummaryState", reason="Missing branch goal")
        
        # 🔥 Enhanced logging for observability
        logger.info("=" * 70)
        logger.info(f"🔬 [ResearchFork:{branch_id}] STARTING FORK EXECUTION")
        logger.info(f"   📋 Task: {branch_goal}")
        logger.info(f"   🎯 Branch ID: {branch_id}")
        logger.info("=" * 70)
        
        # Build focused research prompt
        system_instruction = """You are a research worker executing a specific research task.

Your job:
1. Analyze the research goal
2. Select appropriate tools to gather information
3. Execute efficiently

Do NOT ask questions. Do NOT engage in conversation. Focus on the task."""
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Research goal: {branch_goal}"}
        ]
        
        # Safety check
        await context.increment_llm_call()
        
        # Call LLM with tools
        response = await self.llm.chat_with_tools(
            messages=messages,
            tools=self.registry.to_openai_format(),
            tool_choice=self.tool_choice
        )
        
        # Track usage
        usage = response.get('usage', {})
        if usage:
            await context.accumulate_usage(usage)
        
        # Check for tool calls
        if response.get("tool_calls"):
            logger.info(f"🔧 [ResearchFork:{branch_id}] {len(response['tool_calls'])} tool(s) selected")
            
            # Store tool calls
            for call in response["tool_calls"]:
                await context.add_tool_call(
                    tool_name=call["function"]["name"],
                    arguments=json.loads(call["function"]["arguments"]),
                    result=None
                )
            
            # Execute tools
            return Transition(to="ToolState", reason="Tools selected for research")
        
        # No tools needed - store direct research notes
        elif response.get("content"):
            logger.info(f"📝 [ResearchFork:{branch_id}] Direct research notes provided")
            await context.set_memory("research_notes", response["content"])
            return Transition(to="ForkContractState", reason="Direct research completed")
        
        else:
            logger.warning(f"⚠️ [ResearchFork:{branch_id}] No tools or content generated")
            return Transition(to="ForkContractState", reason="Empty research result")


class ForkContractState(AsyncHierarchicalState):
    """
    Extracts structured contracts from fork execution results.
    
    Replaces ForkSummaryState with deterministic claim extraction.
    Output: ForkResult contract (not text summary)
    """
    def __init__(self, parent, llm):
        super().__init__(parent)
        self.llm = llm
    
    async def handle(self, context: AsyncExecutionContext):
        logger.info("📊 [ForkContract] Extracting claims from research...")
        
        branch_id = await context.get_memory("branch_id", "unknown")
        branch_goal = await context.get_memory("branch_goal", "")
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

1. INFERENCE PERMISSION:
   - You MAY use general knowledge when appropriate
   - You MAY make reasonable inferences
   - Prefer claims over uncertainty

2. EVIDENCE TYPES (required):
   - "retrieved": From tool results/documents
   - "general_knowledge": From your training (widely known facts)
   - "inferred": Logical deduction

3. UNCERTAINTY (use sparingly):
   - Only when you cannot provide ANY answer
   - Specify reason: "not_found_in_retrieval", "insufficient_context", "ambiguous_query"
   - Do NOT omit when inference is possible

4. CONFIDENCE:
   - Retrieved: 0.7-1.0
   - General knowledge: 0.4-0.7
   - Inferred: 0.3-0.6

DO NOT summarize. ONLY extract verifiable facts."""
        
        # Prepare tool results for analysis
        tool_results_text = ""
        evidence_list = []
        
        if context.tool_calls:
            for call in context.tool_calls:
                tool_name = call["tool_name"]
                result = call.get("result", "")
                evidence_list.append(f"tool:{tool_name}")
                tool_results_text += f"\nTool: {tool_name}\nResult: {str(result)[:500]}\n"
        
        # Add research notes if available
        if research_notes:
            tool_results_text += f"\nDirect notes: {research_notes}\n"
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Research goal: {branch_goal}"},
            {"role": "user", "content": f"Tool results:\n{tool_results_text}"},
            {"role": "user", "content": "Extract claims as JSON:"}
        ]
        
        # Safety check
        await context.increment_llm_call()
        
        # Call LLM for claim extraction
        response = await self.llm.chat(messages)
        
        # Track usage
        usage = response.get('usage', {})
        if usage:
            await context.accumulate_usage(usage)
        
        # 🔥 DEBUG: Log LLM response
        logger.info(f"🔍 [ForkContract:{branch_id}] LLM Response:")
        logger.info(f"   Content length: {len(response.get('content', ''))}")
        logger.info(f"   First 300 chars: {response.get('content', '')[:300]}")
        
        # Parse JSON response
        try:
            content = response["content"]
            
            # Extract JSON if wrapped in markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            contract_data = json.loads(content.strip())
            
            # 🔥 DEBUG: Log parsed contract
            logger.info(f"📋 [ForkContract:{branch_id}] Parsed contract data:")
            logger.info(f"   Claims: {len(contract_data.get('claims', []))}")
            logger.info(f"   Coverage: {contract_data.get('coverage', [])}")
            logger.info(f"   Omissions: {contract_data.get('omissions', [])}")
            
            # Build ForkResult contract
            from finitestatemachineAgent.fork_contracts import ForkResult, Claim, EvidenceType
            
            claims = [
                Claim(**claim_dict) 
                for claim_dict in contract_data.get("claims", [])
            ]
            
            # 🔥 EPISTEMIC: Calibrate confidence by evidence type
            for claim in claims:
                for ev in claim.evidence:
                    if ev.type == EvidenceType.GENERAL_KNOWLEDGE:
                        # Cap confidence for general knowledge
                        claim.confidence = min(claim.confidence, 0.7)
                        logger.debug(f"📉 [ForkContract:{branch_id}] Capped confidence for general_knowledge: {claim.key}")
                    elif ev.type == EvidenceType.INFERRED:
                        # Cap confidence for inferred claims
                        claim.confidence = min(claim.confidence, 0.6)
                        logger.debug(f"📉 [ForkContract:{branch_id}] Capped confidence for inferred: {claim.key}")
            
            fork_result = ForkResult(
                branch_id=branch_id,
                claims=claims,
                coverage=contract_data.get("coverage", []),
                uncertain_topics=contract_data.get("uncertain_topics", []),
                confidence=sum(c.confidence for c in claims) / len(claims) if claims else 0.0
            )
            
            # Store contract (not summary)
            await context.set_memory("fork_contract", fork_result.model_dump())
            
            logger.info(f"✅ [ForkContract:{branch_id}] Contract created:")
            logger.info(f"   📊 Claims: {len(claims)}")
            for claim in claims:
                logger.info(f"      - {claim.key}: {claim.value} (confidence: {claim.confidence})")
            logger.info(f"   ✅ Coverage: {len(fork_result.coverage)} topics")
            logger.info(f"   ❓ Uncertain: {len(fork_result.uncertain_topics)} topics")
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ [ForkContract:{branch_id}] Failed to parse JSON: {e}")
            logger.error(f"   Response: {response['content'][:200]}")
            
            # Fallback: create minimal contract
            from finitestatemachineAgent.fork_contracts import ForkResult
            fork_result = ForkResult(
                branch_id=branch_id,
                claims=[],
                coverage=[],
                omissions=[branch_goal],
                confidence=0.0
            )
            await context.set_memory("fork_contract", fork_result.model_dump())
        
        except Exception as e:
            logger.error(f"❌ [ForkContract:{branch_id}] Unexpected error: {e}")
            
            # Fallback: create minimal contract
            from finitestatemachineAgent.fork_contracts import ForkResult
            fork_result = ForkResult(
                branch_id=branch_id,
                claims=[],
                coverage=[],
                omissions=[branch_goal],
                confidence=0.0
            )
            await context.set_memory("fork_contract", fork_result.model_dump())
        
        # Terminal state for fork
        return None


# Keep old name as alias for backward compatibility
ForkSummaryState = ForkContractState
