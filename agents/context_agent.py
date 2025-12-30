"""
Context Agent - Extracts user intent from chat history

This agent analyzes the conversation history and current query
to extract the user's true intent, making it easier for the
FunctionCaller to select the correct tool.
"""
from typing import List, Dict, Optional
from providers.openrouter import OpenRouterProvider


class ContextAgent:
    """
    Agent that extracts user intent from chat history
    
    Uses a lightweight LLM to:
    - Analyze conversation history
    - Extract user's true intent
    - Resolve pronouns and references
    - Provide clear, standalone query
    
    Example:
        ```python
        context_agent = ContextAgent(model="xiaomi/mimo-v2-flash:free")
        
        intent = context_agent.extract_intent(
            current_query="E da AAPL?",
            chat_history=[
                {"role": "user", "content": "Qual o preço da TSLA?"},
                {"role": "assistant", "content": "TSLA está a $475"}
            ]
        )
        # Returns: "Qual o preço da ação AAPL?"
        ```
    """
    
    def __init__(self, model: str = "xiaomi/mimo-v2-flash:free"):
        """
        Initialize Context Agent
        
        Args:
            model: LLM model for intent extraction (default: xiaomi/mimo-v2-flash:free)
        """
        self.llm = OpenRouterProvider(
            model=model,
            temperature=0.1  # Low temperature for consistency
        )
        self.model_name = model
    
    def extract_intent(
        self,
        current_query: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Extract user intent from query and history
        
        Args:
            current_query: Current user query
            chat_history: Previous conversation messages
            
        Returns:
            Clear, standalone query representing user's intent
        """
        # If no history, return query as-is
        if not chat_history or len(chat_history) == 0:
            return current_query
        
        # Build prompt for intent extraction
        prompt = self._build_intent_prompt(current_query, chat_history)
        
        # Get intent from LLM
        response = self.llm.chat([
            {"role": "user", "content": prompt}
        ])
        
        # Extract and clean intent
        intent = self._clean_intent(response)
        
        print(f"🎯 [ContextAgent] Intent extraído: '{intent}'")
        
        return intent
    
    def _build_intent_prompt(
        self,
        current_query: str,
        chat_history: List[Dict[str, str]]
    ) -> str:
        """
        Build prompt for intent extraction
        
        Args:
            current_query: Current query
            chat_history: Chat history
            
        Returns:
            Prompt for LLM
        """
        # Format history (last 3 interactions max)
        history_text = ""
        recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
        
        for msg in recent_history:
            role = "Usuário" if msg["role"] == "user" else "Assistente"
            history_text += f"{role}: {msg['content']}\n"
        
        prompt = f"""Analise a pergunta e retorne no formato: TOOL: intenção

HISTÓRICO:
{history_text}

PERGUNTA: {current_query}

TOOLS:
- search_documents (conceitos financeiros, economia, BACEN, taxas)
- get_stock_price (preço de 1 ação)
- compare_stocks (comparar ações)
- redirect (assuntos não financeiros)

EXEMPLOS:

"O que é Selic?"
→ search_documents: O que é taxa Selic?

"pra que serve o BACEN?"
→ search_documents: Qual a função do Banco Central?

"Preço TSLA"
→ get_stock_price: Qual o preço da ação TSLA?

"Compare TSLA e AAPL"
→ compare_stocks: Compare TSLA e AAPL

"Receita de bolo"
→ redirect: Redirecione o usuário

IMPORTANTE: Responda APENAS no formato "TOOL: intenção". NÃO inclua raciocínio, explicações ou /think.

RESPONDA NO FORMATO: TOOL: intenção"""
        
        return prompt
    
    def _clean_intent(self, response: str) -> str:
        """
        Clean and validate extracted intent
        
        Args:
            response: LLM response
            
        Returns:
            Cleaned intent
        """
        # Remove common prefixes
        intent = response.strip()
        
        # Remove /think tokens and everything after them
        if "/think" in intent:
            intent = intent.split("/think")[0].strip()
        
        prefixes_to_remove = [
            "Intent:",
            "Intenção:",
            "A intenção é:",
            "O usuário quer:",
            "Pergunta:",
        ]
        
        for prefix in prefixes_to_remove:
            if intent.startswith(prefix):
                intent = intent[len(prefix):].strip()
        
        # Remove quotes
        intent = intent.strip('"\'')
        
        # If empty or too short, return original
        if len(intent) < 3:
            return response.strip()
        
        return intent
