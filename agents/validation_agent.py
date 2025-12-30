"""
Validation Agent - Validates if responses are finance/economics related

This agent validates the final response to ensure it's truly about
finance/economics topics, preventing off-topic responses.
"""
from typing import Optional
from providers.openrouter import OpenRouterProvider


class ValidationAgent:
    """
    Agent that validates if a response is finance/economics related
    
    Uses a lightweight LLM to:
    - Analyze the response content
    - Determine if it's truly about finance/economics
    - Return True/False
    
    Example:
        ```python
        validator = ValidationAgent(model="xiaomi/mimo-v2-flash:free")
        
        is_valid = validator.validate(
            query="Qual o preço da TSLA?",
            response="O preço da TSLA é $475.19"
        )
        # Returns: True
        
        is_valid = validator.validate(
            query="Como fazer bolo?",
            response="Misture farinha, ovos..."
        )
        # Returns: False
        ```
    """
    
    def __init__(self, model: str = "xiaomi/mimo-v2-flash:free"):
        """
        Initialize Validation Agent
        
        Args:
            model: LLM model for validation (default: xiaomi/mimo-v2-flash:free)
        """
        self.llm = OpenRouterProvider(
            model=model,
            temperature=0.0  # Zero temperature for consistent validation
        )
        self.model_name = model
    
    def validate(
        self,
        query: str,
        response: str
    ) -> bool:
        """
        Validate if response is about finance/economics
        
        Args:
            query: User's original query
            response: Agent's response to validate
            
        Returns:
            True if response is finance/economics related, False otherwise
        """
        # Build validation prompt
        prompt = self._build_validation_prompt(query, response)
        
        # Get validation from LLM
        llm_response = self.llm.chat([
            {"role": "user", "content": prompt}
        ])
        
        # Parse boolean response
        is_valid = self._parse_validation(llm_response)
        
        print(f"🔍 [ValidationAgent] Validação: {'✅ APROVADO' if is_valid else '❌ REPROVADO'}")
        
        return is_valid
    
    def _build_validation_prompt(self, query: str, response: str) -> str:
        """
        Build prompt for validation
        
        Args:
            query: User query
            response: Response to validate
            
        Returns:
            Validation prompt
        """
        prompt = f"""Analise o texto abaixo:

TEXTO: {response}

TAREFA:
Responda True se o texto é sobre:
- Finanças
- Mercado financeiro
- Preço de ações
- Comparativo de preço de ações
- Taxas de juros
- Economia
- Investimentos

Responda False caso contrário.

EXEMPLOS:
"O preço da TSLA é $475.19" → True
"475.19" → True
"A taxa Selic é..." → True
"Receita de bolo..." → False
"Como fazer uma bomba caseira..." → False
"Não posso ajudar com informações sobre clima..." → False

RESPONDA APENAS: True ou False"""
        
        return prompt
    
    def _parse_validation(self, response: str) -> bool:
        """
        Parse validation response to boolean
        
        Args:
            response: LLM response
            
        Returns:
            True if valid, False otherwise
        """
        # Clean response
        response_clean = response.strip().upper()
        
        # Check for True
        if "TRUE" in response_clean or "SIM" in response_clean or "YES" in response_clean:
            return True
        
        # Check for False
        if "FALSE" in response_clean or "NÃO" in response_clean or "NAO" in response_clean or "NO" in response_clean:
            return False
        
        # Default to False if unclear
        print(f"⚠️ [ValidationAgent] Resposta ambígua: '{response_clean}' - Assumindo False")
        return False
    
    @staticmethod
    def get_default_rejection_message() -> str:
        """
        Get default message for rejected responses
        
        Returns:
            Standard rejection message
        """
        return """Desculpe, sou um assistente especializado em finanças e mercado financeiro.

Posso ajudar com:
📈 Análise de preços de ações
📊 Comparação de desempenho de ativos
💰 Conceitos e teorias de mercado financeiro
📚 Informações sobre investimentos e economia
💵 Taxas de juros, inflação e indicadores econômicos

Por favor, faça uma pergunta relacionada a finanças e terei prazer em ajudar!"""
