"""
Test script for RAG Agent V2

Tests the new RAG Agent implementation that uses
the generic ToolCallingAgent base class.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from embedding_manager.embedding_manager import EmbeddingManager
from agents.rag_agent_v2 import RAGAgentV2


def main():
    """Test RAG Agent V2"""
    
    print("\n" + "="*70)
    print("🧪 TESTANDO RAG AGENT V2")
    print("="*70)
    
    # Initialize embedding manager
    print("\n📊 Inicializando Embedding Manager...")
    embedding_manager = EmbeddingManager(
        embedding_model="qwen3-embedding:0.6b",
        qdrant_url="http://localhost:6333",
        collection_name="rag_api"
    )
    
    # Create RAG Agent V2
    print("🤖 Criando RAG Agent V2...")
    agent = RAGAgentV2(
        embedding_manager=embedding_manager,
        tool_caller_model="gemma3:1b",
        response_model="gemma3:1b"
    )
    
    # Test 1: Document search
    print("\n" + "="*70)
    print("TESTE 1: Busca em documentos")
    print("="*70)
    
    response, context = agent.run(
        query="O que é a taxa Selic?"
    )
    
    print(f"\n📝 Resposta: {response.answer}")
    print(f"📚 Fontes: {response.sources_used}")
    print(f"✅ Confiança: {response.confidence}")
    print(f"🚫 Fora do escopo: {response.is_out_of_scope}")
    
    # Test 2: Stock price
    print("\n" + "="*70)
    print("TESTE 2: Preço de ação")
    print("="*70)
    
    response, context = agent.run(
        query="Qual o preço da TSLA?"
    )
    
    print(f"\n📝 Resposta: {response.answer}")
    print(f"📚 Fontes: {response.sources_used}")
    
    # Test 3: With chat history
    print("\n" + "="*70)
    print("TESTE 3: Com histórico de conversa")
    print("="*70)
    
    response, context = agent.run(
        query="E da AAPL?",
        chat_history=[
            {"role": "user", "content": "Qual o preço da TSLA?"},
            {"role": "assistant", "content": "O preço da TSLA é $475.19 USD."}
        ]
    )
    
    print(f"\n📝 Resposta: {response.answer}")
    print(f"📚 Fontes: {response.sources_used}")
    
    # Test 4: Out of scope
    print("\n" + "="*70)
    print("TESTE 4: Pergunta fora do escopo")
    print("="*70)
    
    response, context = agent.run(
        query="Como fazer um bolo de chocolate?"
    )
    
    print(f"\n📝 Resposta: {response.answer}")
    print(f"🚫 Fora do escopo: {response.is_out_of_scope}")
    
    print("\n" + "="*70)
    print("✅ TESTES CONCLUÍDOS")
    print("="*70)


if __name__ == "__main__":
    main()
