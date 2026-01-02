"""
Finance.AI Utilities
=====================

Helper functions and configurations for the Finance.AI agent.
Includes validation, planning enhancements, and hooks.
"""

import logging
from finitestatemachineAgent.hfsm_agent_async import Transition

logger = logging.getLogger(__name__)


async def tools_validation(context, tool_name, result):
    """
    Custom validation logic for Finance.AI tools.
    
    Args:
        context: Execution context
        tool_name: Name of the tool that was executed
        result: Result returned by the tool
        
    Returns:
        True if the tool result is valid, False otherwise
    """
    if tool_name in ("get_stock_price", "compare_stocks"):
        # For stock tools, check if result has success=True
        return isinstance(result, dict) and result.get("success") == True
    
    elif tool_name == "search_documents":
        # For document search, check if we have results
        return isinstance(result, dict) and result.get("results") and len(result.get("results", [])) > 0
    
    # Default: accept any non-None result
    return result is not None


def enhance_rag_planning_prompt(default_prompt, context):
    """
    Enhances default planning prompt with Finance.AI-specific divide-and-conquer strategy.
    
    Instructs LLM to break complex financial queries into smaller independent tasks
    for parallel execution.
    
    Args:
        default_prompt: Default planning prompt from engine
        context: Execution context
        
    Returns:
        Enhanced prompt with RAG-specific instructions
    """
    enhancement = """

ESTRATÉGIA DIVIDIR E CONQUISTAR (FINANCE.AI):

Para consultas financeiras complexas, você DEVE quebrar em sub-tarefas independentes:

1. **Comparações de Ativos**: 
   - Se comparar múltiplos ativos (ex: "Compare PETR4, VALE3 e ITUB4")
   - Crie um branch para cada ativo
   - Cada branch pesquisa um ativo específico

2. **Análises Multi-Tópico**:
   - Se a pergunta envolve múltiplos conceitos (ex: "Explique Selic, Copom e inflação")
   - Crie um branch para cada conceito
   - Cada branch pesquisa um conceito específico

3. **Consultas Compostas**:
   - Se combina dados + conceitos (ex: "Qual o preço do PETR4 e o que é dividend yield?")
   - Branch 1: Buscar preço do ativo
   - Branch 2: Buscar conceito teórico

REGRAS IMPORTANTES:
- Só paralelizar se as sub-tarefas forem INDEPENDENTES
- Cada branch deve ter um objetivo claro e específico
- Máximo de 3 branches
- Para consultas simples (1 ativo, 1 conceito), use strategy: "single"

EXEMPLOS:

Query: "Compare NVDA e TSLA"
→ strategy: "parallel_research"
→ branches: [
    {"id": "nvda", "goal": "Pesquisar preço e dados da NVDA"},
    {"id": "tsla", "goal": "Pesquisar preço e dados da TSLA"}
]

Query: "Qual o preço do PETR4?"
→ strategy: "single" (consulta simples, não precisa paralelizar)

Query: "Explique Selic, Copom e CDI"
→ strategy: "parallel_research"
→ branches: [
    {"id": "selic", "goal": "Pesquisar conceito de Selic"},
    {"id": "copom", "goal": "Pesquisar conceito de Copom"},
    {"id": "cdi", "goal": "Pesquisar conceito de CDI"}
]"""
    
    return default_prompt + enhancement


async def enforce_tool_usage(context, transition):
    """
    Finance.AI-specific hook: Reject direct answers, force tool usage.
    
    This keeps the engine domain-agnostic while allowing Finance.AI
    to enforce its own rules about always using tools for financial data.
    
    Args:
        context: Execution context
        transition: Proposed transition
        
    Returns:
        Modified transition if needed, or None to keep original
    """
    # Skip enforcement in fork contexts (forks have their own flow)
    is_fork = await context.get_memory("branch_id") is not None
    if is_fork:
        return None
    
    # Check if IntentAnalysis classified this as a simple query
    intent_analysis = await context.get_memory("intent_analysis", {})
    complexity = intent_analysis.get("complexity", "simple")
    needs_tools = intent_analysis.get("needs_tools", False)
    
    # Skip enforcement for simple queries that don't need tools
    if complexity == "simple" and not needs_tools:
        logger.info("🚀 [Finance.AI] Allowing direct answer for simple query")
        return None
    
    if transition.to == "AnswerState" and transition.reason == "Direct answer generation":
        # LLM tried to answer directly without tools - unacceptable for Finance.AI
        retry_count = await context.get_memory("rag_tool_retry", 0)
        
        if retry_count < 2:
            await context.set_memory("rag_tool_retry", retry_count + 1)
            logger.info(f"🔄 [Finance.AI] Forcing tool usage (attempt {retry_count + 1}/2)")
            
            # Override transition to retry
            return Transition(to="RetryState", reason="Finance.AI requires tool usage")
        else:
            logger.error("❌ [Finance.AI] LLM refusing to use tools after retries")
            # Let it fail to RetryState
            return Transition(to="RetryState", reason="Tool usage required")
    
    # Reset retry count on successful tool usage
    if transition.to == "ToolState":
        await context.set_memory("rag_tool_retry", 0)
    
    return None  # Keep original transition


async def extract_metadata(context):
    """
    Extract sources_used and confidence from tool results.
    
    This is Finance.AI-specific logic for metadata extraction.
    Should be called after answer generation to populate metadata.
    
    Args:
        context: Execution context with tool_calls
        
    Returns:
        dict with sources_used and confidence
    """
    sources_used = []
    has_data = False
    
    # Include merged tool calls from parallel execution
    merged_tools = await context.get_memory("merged_tool_calls", [])
    all_calls = (context.tool_calls or []) + merged_tools
    
    for call in all_calls:
        tool_name = call.get("tool_name")
        result = call.get("result", {})
        
        if tool_name == "search_documents" and isinstance(result, dict):
            for doc in result.get("results", []):
                meta = doc.get("metadata", {})
                src = meta.get("source")
                if src and src not in sources_used:
                    sources_used.append(src)
            has_data = True
        
        elif tool_name in ("get_stock_price", "compare_stocks"):
            if isinstance(result, dict) and result.get("success"):
                source = f"yfinance:{tool_name}"
                if source not in sources_used:
                    sources_used.append(source)
                has_data = True
    
    # Calculate confidence
    confidence = "high" if has_data else "low"
    
    # Store in context memory for API access
    await context.set_memory("sources_used", sources_used)
    await context.set_memory("confidence", confidence)
    
    return {
        "sources_used": sources_used,
        "confidence": confidence
    }
