import sys
import os
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from agents.rag_agent_fsm import RAGAgentFSM
from embedding_manager.embedding_manager import EmbeddingManager

def test_rag_fsm():
    print("Initializing RAG FSM Agent Test...")
    
    try:
        # Initialize Embedding Manager (assuming Qdrant is running as per user)
        # We need a dummy usage of it, tools will use it
        embedding_manager = EmbeddingManager()
        
        agent = RAGAgentFSM(
            embedding_manager=embedding_manager,
            model="xiaomi/mimo-v2-flash:free", # Use a reliable model
            max_steps=10
        )
        
        # Test Query 1: Simple Stock Price (should use tool)
        query = "Qual o preço da ação da Apple (AAPL)?"
        print(f"\n--- Test Query 1: {query} ---")
        response, context = agent.run(query)
        print(f"Final Answer: {response.answer}")
        print(f"Confidence: {response.confidence}")
        print(f"Sources: {response.sources_used}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag_fsm()
