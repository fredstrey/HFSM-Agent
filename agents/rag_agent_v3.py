from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

from ReactAgent import ReactAgent, ExecutionContext, AgentResponse
import tools.rag_tools_v3 as rag_tools
from tools.rag_schemas import RAGResponse

class RAGAgentV3(ReactAgent):
    """
    RAG Agent v3 - Refactored using the Generic ReactAgent framework.
    
    Specialized in Finance and Economics with tool-calling capabilities.
    """
    
    def __init__(
        self,
        embedding_manager,
        model: str = "xiaomi/mimo-v2-flash:free",
        max_iterations: int = 3
    ):
        """
        Initialize RAG Agent v3
        
        Args:
            embedding_manager: EmbeddingManager for document search
            model: The LLM model name
            max_iterations: Max reasoning loops
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
        
        # Build RAG system prompt
        system_prompt = self._build_rag_system_prompt()
        
        # Call parent constructor
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            response_model=RAGResponse,
            max_iterations=max_iterations,
            tools=tools
        )
        
        self.embedding_manager = embedding_manager
        print(f"✅ RAGAgentV3 inicializado com modelo: {model}")

    def run(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        context: Optional[ExecutionContext] = None
    ) -> Tuple[RAGResponse, ExecutionContext]:
        """
        Execute RAG agent run.
        """
        # The generic ReactAgent.run() handles the ReAct loop and tool execution.
        # It also handles structured validation if response_model is provided.
        response, context = super().run(query, chat_history, context)
        
        # Ensure the response is of type RAGResponse (ReactAgent._finalize_response handles this)
        return response, context

    def _build_rag_system_prompt(self) -> str:
        """Build RAG-specific system prompt"""
        return """VOCÊ É UM ASSISTENTE FINANCEIRO. SEMPRE USE UMA TOOL.

⚠️ REGRA CRÍTICA: CHAME APENAS **UMA TOOL POR VEZ**
- NÃO chame múltiplas tools simultaneamente
- Chame UMA tool, analise o resultado, depois decida se precisa de outra
- Se a pergunta tem múltiplas partes, responda UMA PARTE por vez

TOOLS DISPONÍVEIS:
1. search_documents - Conceitos financeiros, economia, BACEN, taxas, inflação
2. get_stock_price - Preço de UMA ação específica
3. compare_stocks - COMPARAR DUAS OU MAIS AÇÕES
4. redirect - Assuntos não financeiros

REGRAS:
- SEMPRE chame UMA tool por vez
- NUNCA responda diretamente na primeira interação sem usar tools (a menos que seja uma saudação simples)
- Se a pergunta tem múltiplas partes, responda uma parte por vez"""
