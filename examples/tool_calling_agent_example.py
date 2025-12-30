"""
Example: Using Generic ToolCallingAgent

This example shows how to create a custom agent with specific tools
using the generic ToolCallingAgent base class.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field
from typing import List
from core import tool, ToolCallingAgent


# Define custom response model
class CalculatorResponse(BaseModel):
    """Custom response with calculation metadata"""
    answer: str
    sources_used: List[str] = Field(default_factory=list)
    confidence: str = "high"
    calculation_performed: bool = False
    expression_evaluated: str = ""


# Define custom tools
@tool()
def calculator(expression: str) -> dict:
    """
    Calculate mathematical expression
    
    Args:
        expression: Math expression to evaluate (e.g., "2+2", "10*5")
        
    Returns:
        Result of calculation
    """
    try:
        result = eval(expression)
        return {
            "result": result,
            "expression": expression
        }
    except Exception as e:
        return {
            "error": str(e),
            "expression": expression
        }


@tool()
def get_time() -> dict:
    """
    Get current time
    
    Returns:
        Current time information
    """
    from datetime import datetime
    now = datetime.now()
    return {
        "time": now.strftime("%H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "timestamp": now.isoformat()
    }


def main():
    """Example usage of generic ToolCallingAgent"""
    
    # Create agent with custom tools and custom response model
    agent = ToolCallingAgent(
        tools=[calculator, get_time],
        tool_caller_model="gemma3:1b",
        response_model="gemma3:1b",
        response_class=CalculatorResponse,  # Custom response model
        system_prompt="""Você é um assistente útil que pode fazer cálculos e informar a hora.

TOOLS DISPONÍVEIS:
- calculator: Para cálculos matemáticos
- get_time: Para informar a hora atual

REGRAS:
1. SEMPRE use uma tool para responder
2. Para perguntas sobre cálculos, use calculator
3. Para perguntas sobre hora/data, use get_time
4. Não invente informações

Escolha a tool apropriada e forneça os argumentos."""
    )
    
    # Test 1: Calculator
    print("\n" + "="*70)
    print("TESTE 1: Calculadora com Response Customizado")
    print("="*70)
    
    response, context = agent.run(
        query="Quanto é 15 * 7?"
    )
    
    print(f"\nResposta: {response.answer}")
    print(f"Fontes: {response.sources_used}")
    print(f"Cálculo realizado: {response.calculation_performed}")
    print(f"Expressão: {response.expression_evaluated}")
    
    # Test 2: Time
    print("\n" + "="*70)
    print("TESTE 2: Hora atual")
    print("="*70)
    
    response, context = agent.run(
        query="Que horas são?"
    )
    
    print(f"\nResposta: {response.answer}")
    
    # Test 3: With chat history
    print("\n" + "="*70)
    print("TESTE 3: Com histórico")
    print("="*70)
    
    response, context = agent.run(
        query="E quanto é o dobro disso?",
        chat_history=[
            {"role": "user", "content": "Quanto é 15 * 7?"},
            {"role": "assistant", "content": "15 * 7 = 105"}
        ]
    )
    
    print(f"\nResposta: {response.answer}")


if __name__ == "__main__":
    main()
