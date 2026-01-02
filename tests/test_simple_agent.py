"""
Test: Simple Workflow Agent
============================

Testa se o agente simples com initial_state="AnswerState" funciona.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from finitestatemachineAgent.agent import Agent
import dotenv
dotenv.load_dotenv()

async def test_simple_agent():
    """Testa agente simples que começa direto no AnswerState."""
    
    print("🧪 Testing Simple Workflow Agent...")
    print("=" * 50)
    
    # Criar agente simples
    simple_agent = Agent(
        llm_provider="openrouter",
        model="xiaomi/mimo-v2-flash:free",
        system_instruction="You are a helpful assistant. Answer concisely.",
        initial_state="AnswerState",  # 🔥 Começa direto na resposta
        enable_intent_analysis=False,
        enable_parallel_planning=False
    )
    
    # Teste 1: Query simples
    print("\n📝 Test 1: Simple math question")
    response = await simple_agent.run("What is 2+2?")
    print(f"Response: {response.content}")
    print(f"Tokens: {response.token_usage}")
    
    # Teste 2: Streaming
    print("\n📝 Test 2: Streaming response")
    print("Response: ", end="")
    async for token in simple_agent.stream("Tell me a short joke"):
        print(token, end="", flush=True)
    print()
    
    print("\n✅ Tests completed!")


if __name__ == "__main__":
    asyncio.run(test_simple_agent())
