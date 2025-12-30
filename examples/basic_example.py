"""
Exemplo básico de uso do framework com decorators
"""
from pydantic import BaseModel, Field
from rag_tools import tool, FunctionAgent
from rag_tools.providers import OllamaProvider


# ==========================================
# 1. DEFINIR RESPONSE MODEL
# ==========================================

class Response(BaseModel):
    """Modelo de resposta do agente"""
    answer: str = Field(..., description="Resposta para o usuário")
    calculation_used: bool = Field(default=False, description="Se usou cálculo")
    confidence: str = Field(default="medium", description="Nível de confiança")


# ==========================================
# 2. DEFINIR TOOLS COM DECORATOR
# ==========================================

@tool(name="calculate", description="Realiza cálculos matemáticos")
def calculator(expression: str) -> dict:
    """
    Calcula uma expressão matemática
    
    Args:
        expression: Expressão para calcular (ex: "2 + 2")
    
    Returns:
        Resultado do cálculo
    """
    try:
        result = eval(expression)
        return {
            "expression": expression,
            "result": result,
            "success": True
        }
    except Exception as e:
        return {
            "expression": expression,
            "error": str(e),
            "success": False
        }


@tool(name="get_weather", description="Obtém informações do clima")
def get_weather(city: str, unit: str = "celsius") -> dict:
    """
    Retorna o clima de uma cidade (simulado)
    
    Args:
        city: Nome da cidade
        unit: Unidade de temperatura (celsius ou fahrenheit)
    
    Returns:
        Informações do clima
    """
    # Simulação
    return {
        "city": city,
        "temperature": 25 if unit == "celsius" else 77,
        "unit": unit,
        "condition": "ensolarado",
        "humidity": 60
    }


# ==========================================
# 3. CRIAR E USAR AGENTE
# ==========================================

def main():
    """Função principal"""
    
    print("=" * 70)
    print("🎯 Exemplo Básico - Framework com Decorators")
    print("=" * 70)
    
    # Provider Ollama
    provider = OllamaProvider(
        model="gemma3:1b",
        temperature=0.3
    )
    
    # Verifica disponibilidade
    if not provider.is_available():
        print("❌ Ollama não está disponível!")
        return
    
    print("✅ Ollama conectado!")
    
    # Cria agente
    agent = FunctionAgent(
        llm_provider=provider,
        response_model=Response,
        system_prompt="Você é um assistente útil que pode fazer cálculos e consultar o clima."
    )
    
    print(f"✅ Agente criado com {len(agent.registry.list())} tools registradas")
    print(f"   Tools: {', '.join(agent.registry.list())}")
    
    # ==========================================
    # TESTES
    # ==========================================
    
    # Teste 1: Cálculo
    print("\n" + "=" * 70)
    print("🔹 Teste 1: Pergunta com cálculo")
    print("-" * 70)
    
    query1 = "Quanto é 15 multiplicado por 8?"
    response1 = agent.run(query1)
    
    print(f"\n✅ Resposta:")
    print(f"   {response1.answer}")
    print(f"   Usou cálculo: {response1.calculation_used}")
    print(f"   Confiança: {response1.confidence}")
    
    # Teste 2: Clima
    print("\n" + "=" * 70)
    print("🔹 Teste 2: Pergunta sobre clima")
    print("-" * 70)
    
    agent.reset()
    query2 = "Como está o clima em São Paulo?"
    response2 = agent.run(query2)
    
    print(f"\n✅ Resposta:")
    print(f"   {response2.answer}")
    print(f"   Confiança: {response2.confidence}")
    
    # Teste 3: Sem tool
    print("\n" + "=" * 70)
    print("🔹 Teste 3: Pergunta simples (sem tool)")
    print("-" * 70)
    
    agent.reset()
    query3 = "Olá, como você está?"
    response3 = agent.run(query3)
    
    print(f"\n✅ Resposta:")
    print(f"   {response3.answer}")
    
    print("\n" + "=" * 70)
    print("✅ Exemplo concluído!")
    print("=" * 70)


if __name__ == "__main__":
    main()
