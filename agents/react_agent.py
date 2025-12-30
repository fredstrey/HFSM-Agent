"""
ReAct Agent - Reasoning + Acting

This agent analyzes tool execution results and decides whether to:
- Continue with current results
- Retry with refined query
- Call different tools
- Stop after max iterations

It implements a reasoning loop that improves query quality and result completeness.
"""
from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
from providers.openrouter import OpenRouterProvider


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


class ReActAgent:
    """
    ReAct Agent that analyzes tool results and decides next actions
    
    This agent implements a reasoning loop that:
    1. Analyzes tool execution results
    2. Determines if information is sufficient
    3. Refines queries if needed
    4. Suggests alternative tools
    5. Decides when to stop iterating
    
    Example:
        ```python
        react_agent = ReActAgent(model="xiaomi/mimo-v2-flash:free")
        
        analysis = react_agent.analyze_and_decide(
            original_query="Compare tech stocks",
            current_query="Compare tech stocks",
            tool_results={"tool": "search_documents", "result": {...}},
            iteration=1,
            max_iterations=3
        )
        
        if analysis.decision == ReActDecision.RETRY_WITH_REFINEMENT:
            # Use analysis.refined_query for next iteration
            next_query = analysis.refined_query
        ```
    """
    
    def __init__(self, model: str = "xiaomi/mimo-v2-flash:free"):
        """
        Initialize ReAct Agent
        
        Args:
            model: LLM model for reasoning (default: xiaomi/mimo-v2-flash:free)
        """
        self.llm = OpenRouterProvider(
            model=model,
            temperature=0.3  # Low temperature for consistent reasoning
        )
        self.model_name = model
        print(f"✅ ReActAgent inicializado com modelo: {model}")
    
    def analyze_and_decide(
        self,
        original_query: str,
        current_query: str,
        tool_results: Dict[str, Any],
        accumulated_context: List[str],
        iteration: int,
        max_iterations: int
    ) -> ReActAnalysis:
        """
        Analyze tool results and decide next action
        
        Args:
            original_query: User's original query
            current_query: Current query being processed (may be refined)
            tool_results: Results from tool execution
            accumulated_context: Context accumulated from all iterations
            iteration: Current iteration number (1-indexed)
            max_iterations: Maximum allowed iterations
            
        Returns:
            ReActAnalysis with decision and reasoning
        """
        print(f"\n🧠 [ReActAgent] Iteração {iteration}/{max_iterations}")
        print(f"   Query original: '{original_query}'")
        print(f"   Query atual: '{current_query}'")
        
        # Build analysis prompt
        prompt = self._build_analysis_prompt(
            original_query=original_query,
            current_query=current_query,
            tool_results=tool_results,
            accumulated_context=accumulated_context,
            iteration=iteration,
            max_iterations=max_iterations
        )
        
        # Get LLM analysis
        llm_response = self.llm.chat([
            {"role": "user", "content": prompt}
        ])
        
        # Parse response
        analysis = self._parse_analysis(llm_response, iteration, max_iterations)
        
        # Log decision
        self._log_decision(analysis)
        
        return analysis
    
    def _build_analysis_prompt(
        self,
        original_query: str,
        current_query: str,
        tool_results: Dict[str, Any],
        accumulated_context: List[str],
        iteration: int,
        max_iterations: int
    ) -> str:
        """Build prompt for ReAct analysis"""
        
        # Format tool results
        tool_name = tool_results.get("tool_name", "unknown")
        result_data = tool_results.get("result", {})
        
        # Extract accumulated answers
        accumulated_answers_list = result_data.get("accumulated_answers", [])
        accumulated_summary = "\n".join([f"Iteração {i+1}: {ans[:200]}..." if len(ans) > 200 else f"Iteração {i+1}: {ans}" 
                                         for i, ans in enumerate(accumulated_answers_list)]) if accumulated_answers_list else "Nenhuma resposta acumulada ainda"
        
        # Format accumulated context
        context_summary = "\n".join(accumulated_context) if accumulated_context else "Nenhum contexto acumulado ainda"
        
        prompt = f"""Você é um agente de raciocínio que analisa resultados de ferramentas e decide se precisa de mais informações.

QUERY ORIGINAL DO USUÁRIO: {original_query}

QUERY ATUAL (pode ter sido refinada): {current_query}

ITERAÇÃO: {iteration}/{max_iterations}

FERRAMENTA EXECUTADA: {tool_name}

RESULTADO DA ITERAÇÃO ATUAL:
{result_data.get('current_answer', result_data)}

RESPOSTAS ACUMULADAS DE TODAS AS ITERAÇÕES:
{accumulated_summary}

CONTEXTO ACUMULADO DE ITERAÇÕES ANTERIORES:
{context_summary}

TAREFA:
Analise se os resultados obtidos são SUFICIENTES para responder COMPLETAMENTE à query original do usuário.

⚠️ ATENÇÃO: Verifique as RESPOSTAS ACUMULADAS acima para ver o que JÁ FOI RESPONDIDO em iterações anteriores!

⚠️ ATENÇÃO ESPECIAL PARA QUERIES MÚLTIPLAS:
- Se a query original contém MÚLTIPLAS PERGUNTAS (ex: "preço da AAPL E quem define Selic?")
- Verifique se TODAS as partes foram respondidas
- Se apenas PARTE da query foi respondida, decida CALL_DIFFERENT_TOOL ou RETRY_WITH_REFINEMENT
- NÃO decida CONTINUE se alguma parte da query ainda não foi respondida

DECISÕES POSSÍVEIS:

1. CONTINUE - Use APENAS se:
   - TODAS as partes da query original foram respondidas
   - Temos informações suficientes para gerar uma resposta COMPLETA
   - A ferramenta retornou dados válidos e relevantes para TODA a query
   - Se a query tem múltiplas partes, TODAS devem estar respondidas

2. RETRY_WITH_REFINEMENT - Use se:
   - A query está vaga ou incompleta
   - Podemos melhorar a query para obter melhores resultados
   - Ainda temos iterações disponíveis ({max_iterations - iteration} restantes)
   - Exemplo: "Compare tech stocks" → "Compare AAPL, MSFT, GOOGL"

3. CALL_DIFFERENT_TOOL - Use se:
   - A ferramenta errada foi chamada
   - Outra ferramenta seria mais apropriada
   - A query tem múltiplas partes e apenas algumas foram respondidas
   - Exemplo: Query "preço AAPL e quem define Selic?" → chamou get_stock_price (respondeu AAPL) mas falta search_documents (Selic)

4. INSUFFICIENT_DATA - Use se:
   - Já tentamos {iteration} vezes e não conseguimos dados úteis
   - Não há mais iterações disponíveis
   - Os dados simplesmente não existem

FORMATO DE RESPOSTA (responda EXATAMENTE neste formato):

DECISION: [CONTINUE|RETRY_WITH_REFINEMENT|CALL_DIFFERENT_TOOL|INSUFFICIENT_DATA]
REASONING: [Explique brevemente por que tomou essa decisão]
REFINED_QUERY: [Se RETRY_WITH_REFINEMENT, forneça a query melhorada. Caso contrário, deixe vazio]
SUGGESTED_TOOL: [Se CALL_DIFFERENT_TOOL, sugira a ferramenta. Caso contrário, deixe vazio]
CONFIDENCE: [0.0 a 1.0 - quão confiante está nesta decisão]

EXEMPLOS:

Exemplo 1 - Informação suficiente:
DECISION: CONTINUE
REASONING: A ferramenta get_stock_price retornou o preço da TSLA ($475.19) que responde completamente à pergunta do usuário.
REFINED_QUERY: 
SUGGESTED_TOOL: 
CONFIDENCE: 0.95

Exemplo 2 - Query múltipla PARCIALMENTE respondida:
Query: "Qual o preço da AAPL e quem define a taxa Selic?"
Resultado: Retornou apenas preço da AAPL
DECISION: CALL_DIFFERENT_TOOL
REASONING: A query tem duas partes: (1) preço da AAPL - RESPONDIDA, (2) quem define Selic - NÃO RESPONDIDA. Preciso chamar search_documents para responder a segunda parte.
REFINED_QUERY: Quem define a taxa Selic?
SUGGESTED_TOOL: search_documents
CONFIDENCE: 0.90

Exemplo 3 - Query vaga precisa refinamento:
DECISION: RETRY_WITH_REFINEMENT
REASONING: A query "Compare tech stocks" é muito vaga. Precisamos especificar quais ações comparar.
REFINED_QUERY: Compare os preços das ações AAPL, MSFT, GOOGL e NVDA
SUGGESTED_TOOL: 
CONFIDENCE: 0.85

Exemplo 4 - Ferramenta errada:
DECISION: CALL_DIFFERENT_TOOL
REASONING: O usuário quer comparar ações mas chamamos get_stock_price que retorna apenas uma ação.
REFINED_QUERY: 
SUGGESTED_TOOL: compare_stocks
CONFIDENCE: 0.90

Exemplo 5 - Sem dados após múltiplas tentativas:
DECISION: INSUFFICIENT_DATA
REASONING: Já tentamos 3 vezes e não encontramos documentos relevantes sobre este tópico.
REFINED_QUERY: 
SUGGESTED_TOOL: 
CONFIDENCE: 0.70

RESPONDA AGORA:"""
        
        return prompt
    
    def _parse_analysis(
        self,
        llm_response: str,
        iteration: int,
        max_iterations: int
    ) -> ReActAnalysis:
        """Parse LLM response into structured analysis"""
        
        # Default values
        decision = ReActDecision.INSUFFICIENT_DATA
        reasoning = "Não foi possível analisar a resposta"
        refined_query = None
        suggested_tool = None
        confidence = 0.5
        
        try:
            # Parse line by line
            lines = llm_response.strip().split("\n")
            
            for line in lines:
                line = line.strip()
                
                if line.startswith("DECISION:"):
                    decision_str = line.replace("DECISION:", "").strip().upper()
                    if "CONTINUE" in decision_str:
                        decision = ReActDecision.CONTINUE
                    elif "RETRY" in decision_str:
                        decision = ReActDecision.RETRY_WITH_REFINEMENT
                    elif "DIFFERENT" in decision_str or "CALL_DIFFERENT" in decision_str:
                        decision = ReActDecision.CALL_DIFFERENT_TOOL
                    elif "INSUFFICIENT" in decision_str:
                        decision = ReActDecision.INSUFFICIENT_DATA
                
                elif line.startswith("REASONING:"):
                    reasoning = line.replace("REASONING:", "").strip()
                
                elif line.startswith("REFINED_QUERY:"):
                    refined_query = line.replace("REFINED_QUERY:", "").strip()
                    if not refined_query:
                        refined_query = None
                
                elif line.startswith("SUGGESTED_TOOL:"):
                    suggested_tool = line.replace("SUGGESTED_TOOL:", "").strip()
                    if not suggested_tool:
                        suggested_tool = None
                
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.replace("CONFIDENCE:", "").strip())
                        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                    except:
                        confidence = 0.5
            
            # Force INSUFFICIENT_DATA if at max iterations
            if iteration >= max_iterations and decision != ReActDecision.CONTINUE:
                decision = ReActDecision.INSUFFICIENT_DATA
                reasoning = f"Atingido limite de {max_iterations} iterações. {reasoning}"
        
        except Exception as e:
            print(f"⚠️ [ReActAgent] Erro ao parsear resposta: {e}")
            print(f"   Resposta LLM: {llm_response}")
        
        return ReActAnalysis(
            decision=decision,
            reasoning=reasoning,
            refined_query=refined_query,
            suggested_tool=suggested_tool,
            confidence=confidence
        )
    
    def _log_decision(self, analysis: ReActAnalysis):
        """Log the decision for debugging"""
        
        decision_emoji = {
            ReActDecision.CONTINUE: "✅",
            ReActDecision.RETRY_WITH_REFINEMENT: "🔄",
            ReActDecision.CALL_DIFFERENT_TOOL: "🔧",
            ReActDecision.INSUFFICIENT_DATA: "❌"
        }
        
        emoji = decision_emoji.get(analysis.decision, "❓")
        
        print(f"\n{emoji} [ReActAgent] Decisão: {analysis.decision.value}")
        print(f"   Raciocínio: {analysis.reasoning}")
        print(f"   Confiança: {analysis.confidence:.2f}")
        
        if analysis.refined_query:
            print(f"   Query refinada: '{analysis.refined_query}'")
        
        if analysis.suggested_tool:
            print(f"   Ferramenta sugerida: {analysis.suggested_tool}")
