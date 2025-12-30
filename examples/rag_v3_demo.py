import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.rag_agent_v3 import RAGAgentV3
from embedding_manager.embedding_manager import EmbeddingManager

def main():
    # Initialize EmbeddingManager (requires Qdrant running or mock)
    # For demo purposes, we'll assume it's correctly configured
    try:
        embedding_manager = EmbeddingManager()
        print("✅ EmbeddingManager carregado.")
    except Exception as e:
        print(f"❌ Erro ao carregar EmbeddingManager (verifique o Qdrant): {e}")
        return

    # Instantiate Agent V3
    agent = RAGAgentV3(
        embedding_manager=embedding_manager,
        model="xiaomi/mimo-v2-flash:free"
    )

    print("\n=== Demo RAG Agent V3 ===\n")

    # Example 1: Finance concept
    query_1 = "O que é a taxa Selic?"
    print(f"User: {query_1}")
    response, _ = agent.run(query_1)
    print(f"Agent: {response.answer}")
    
    if hasattr(response, "sources_used"):
        print(f"Sources: {response.sources_used}")
    else:
        print("Note: Response is generic.")
        if hasattr(response, "metadata") and response.metadata:
             print(f"Metadata: {response.metadata}")

    print("\n" + "="*50 + "\n")

    # Example 2: Stock price
    query_2 = "Qual o valor da ação da Apple (AAPL)?"
    print(f"User: {query_2}")
    response, _ = agent.run(query_2)
    print(f"Agent: {response.answer}")
    
    if hasattr(response, "sources_used"):
         print(f"Sources: {response.sources_used}")
    else:
         print("Note: Response is generic.")

    print("\n" + "="*50 + "\n")

    # Example 3: Iteration Limit Test (simulate hitting limit)
    # We use a complex query but limit iterations to 1 so it fails to finish normally
    print("Testando limite de iterações (Max=1)...")
    agent_limited = RAGAgentV3(
        embedding_manager=embedding_manager,
        model="xiaomi/mimo-v2-flash:free",
        max_iterations=1 
    )
    query_3 = "quem define taxa selic? Qual o preço da ação da AAPL?"
    print(f"User: {query_3}")
    response, _ = agent_limited.run(query_3)
    print(f"Agent: {response.answer}")
    
    # Check if generic response (on error) or RAGResponse
    if hasattr(response, "sources_used"):
        print(f"Sources: {response.sources_used}")
    else:
        print("Note: Response is generic (no sources metadata).")
        if hasattr(response, "metadata") and response.metadata:
             print(f"Metadata: {response.metadata}")

if __name__ == "__main__":
    main()
