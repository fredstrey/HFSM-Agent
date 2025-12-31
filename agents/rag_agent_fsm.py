from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Import FSM Agent
from finitestatemachineAgent.fsm_agent import StateMachineAgent, AgentResponse
from core import ExecutionContext

# Import RAG Tools
import tools.rag_tools_v3 as rag_tools
from tools.rag_schemas import RAGResponse

class RAGAgentFSM:
    """
    RAG Agent implemented with Finite State Machine architecture.
    Specialized in Finance and Economics.
    """
    
    def __init__(
        self,
        embedding_manager,
        model: str = "xiaomi/mimo-v2-flash:free",
        max_steps: int = 15
    ):
        """
        Initialize RAG FSM Agent
        """
        # Initialize RAG tools with the embedding manager
        rag_tools.initialize_rag_tools(embedding_manager)
        
        # Tools to use
        tools = [
            rag_tools.search_documents,
            rag_tools.get_stock_price,
            rag_tools.compare_stocks,
            rag_tools.redirect
        ]
        
        # RAG System Instruction
        system_instruction = """
VOCÊ É UM ASSISTENTE FINANCEIRO (RAG AGENT).
Sua missão é ajudar com dúvidas sobre finanças, economia e investimentos.

⚠️ REGRAS DE EXECUÇÃO:
1. SEMPRE VERIFIQUE O "HISTÓRICO DE EXECUÇÃO" ANTES DE CHAMAR UMA FERRAMENTA.
   - Se já tem a resposta, não chame ferramenta novamente.
2. CHAME APENAS UMA FERRAMENTA POR VEZ.
3. Se a pergunta for sobre um ativo específico, use 'get_stock_price'.
4. Se for sobre conceitos ou relatórios (BACEN, inflação, etc), use 'search_documents'.
5. Se for comparar ações, use 'compare_stocks'.
"""
        
        # Initialize FSM Agent
        self.fsm_agent = StateMachineAgent(
            model=model,
            tools=tools,
            max_steps=max_steps,
            system_instruction=system_instruction
        )
        
        self.embedding_manager = embedding_manager
        print(f"✅ RAGAgentFSM inicializado com modelo: {model}")

    def run(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        context: Optional[ExecutionContext] = None
    ):
        """
        Run the agent processing loop.
        
        Args:
            query: User query
            chat_history: Optional list of chat history messages
            context: Optional execution context
            
        Returns:
            Tuple[RAGResponse, ExecutionContext]
        """
        # Enhance query with chat history if provided
        enhanced_query = query
        if chat_history:
            history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]]) # Last 5 messages
            enhanced_query = f"""
Histórico recente da conversa:
{history_str}

Pergunta atual:
{query}
"""
        
        # Run FSM (returns AgentResponse, ExecutionContext)
        response, context = self.fsm_agent.run(enhanced_query)
        
        # Finalize response with structure (sources, confidence)
        return self._finalize_response(response.answer, context)

    def _finalize_response(self, content: str, context: ExecutionContext) -> tuple[RAGResponse, ExecutionContext]:
        """
        Structure the response deterministically based on tool results.
        Replicates logic from RAGAgentV2.
        """
        sources_used = []
        scores = []
        has_stock_data = False
        
        # Iterate over tool calls to extract sources and scores
        if context.tool_calls:
            for call in context.tool_calls:
                tool_name = call.get("tool_name")
                result = call.get("result", {})
                
                # Handle search_documents
                if tool_name == "search_documents":
                    # Check if result is a dict (success) or str (error/direct)
                    if isinstance(result, dict):
                        # Extract results list
                        results_list = result.get("results", [])
                        for doc in results_list:
                            meta = doc.get("metadata", {})
                            source = meta.get("source")
                            if source and source not in sources_used:
                                sources_used.append(source)
                            
                            if "score" in doc:
                                scores.append(doc["score"])
                
                # Handle stock tools
                elif tool_name in ["get_stock_price", "compare_stocks"]:
                    if isinstance(result, dict) and result.get("success", False):
                        sources_used.append(f"yfinance:{tool_name}")
                        has_stock_data = True

        # Calculate Confidence
        confidence = "medium" # Default
        
        if has_stock_data:
            confidence = "high"
        elif scores:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            
            if max_score > 0.7 or (avg_score > 0.6 and len(scores) >= 2):
                confidence = "high"
            elif avg_score >= 0.5:
                confidence = "medium"
            else:
                confidence = "low"
        elif not sources_used:
            confidence = "low"
            
        print(f"🎯 Finalized Response - Sources: {len(sources_used)}, Confidence: {confidence}")

        rag_response = RAGResponse(
            answer=content,
            sources_used=sources_used,
            confidence=confidence
        )
        
        return rag_response, context
