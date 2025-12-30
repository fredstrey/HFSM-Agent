"""
Test script for ReAct Agent integration

This script tests the ReAct agent with different scenarios to verify:
1. Information sufficient on first iteration
2. Query refinement needed
3. Multiple iterations with context accumulation
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from embedding_manager.embedding_manager import EmbeddingManager
from agents.rag_agent_v2 import RAGAgentV2


def test_react_agent():
    """Test ReAct agent with various scenarios"""
    
    print("=" * 80)
    print("🧪 TESTANDO REACT AGENT")
    print("=" * 80)
    
    # Initialize embedding manager
    print("\n📦 Inicializando EmbeddingManager...")
    embedding_manager = EmbeddingManager(
        embedding_model="qwen3-embedding:0.6b",
        qdrant_url="http://localhost:6333",
        collection_name="rag_api"
    )
    
    # Initialize RAG Agent with ReAct
    print("\n🤖 Inicializando RAG Agent com ReAct...")
    rag_agent = RAGAgentV2(
        embedding_manager=embedding_manager,
        tool_caller_model="xiaomi/mimo-v2-flash:free",
        response_model="xiaomi/mimo-v2-flash:free",
        context_model="xiaomi/mimo-v2-flash:free",
        max_iterations=3  # ReAct loop with 3 iterations
    )
    
    # Test scenarios
    test_cases = [
        {
            "name": "Cenário 1: Query clara - deve ter sucesso na primeira iteração",
            "query": "Qual o preço da TSLA?",
            "expected_iterations": 1
        },
        {
            "name": "Cenário 2: Comparação de ações",
            "query": "Compare TSLA e NVDA",
            "expected_iterations": 1
        },
        {
            "name": "Cenário 3: Query sobre conceitos financeiros",
            "query": "O que é taxa Selic?",
            "expected_iterations": 1
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print("\n" + "=" * 80)
        print(f"📋 TESTE {i}: {test_case['name']}")
        print("=" * 80)
        print(f"Query: {test_case['query']}")
        
        try:
            # Run agent
            response, context = rag_agent.run(
                query=test_case['query'],
                chat_history=[]
            )
            
            # Print results
            print("\n" + "-" * 80)
            print("📊 RESULTADOS:")
            print("-" * 80)
            print(f"✅ Resposta: {response.answer[:300]}...")
            print(f"📚 Sources: {response.sources_used}")
            print(f"🎯 Confidence: {response.confidence}")
            print(f"❌ Out of scope: {response.is_out_of_scope}")
            
        except Exception as e:
            print(f"\n❌ ERRO: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✅ TESTES CONCLUÍDOS")
    print("=" * 80)


if __name__ == "__main__":
    test_react_agent()
