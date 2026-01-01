"""
Test IntentAnalysis State directly
"""
import asyncio
import sys
import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.rag_agent_hfsm_async import AsyncRAGAgentFSM
from embedding_manager.embedding_manager import EmbeddingManager

async def test_intent_analysis():
    print("🧪 Testing IntentAnalysis State...")
    
    # Initialize components
    embedding_manager = EmbeddingManager(
        embedding_model="qwen3-embedding:0.6b",
        qdrant_url="http://localhost:6333",
        collection_name="rag_api"
    )
    
    # Create agent
    agent = AsyncRAGAgentFSM(
        embedding_manager=embedding_manager,
        model="xiaomi/mimo-v2-flash:free"
    )
    
    # Test query
    query = "Como classificar Leasing, Debêntures e ACC?"
    
    print(f"\n📝 Query: {query}\n")
    print("=" * 80)
    
    # Run stream
    tokens = []
    async for token in agent.run_stream(query):
        tokens.append(token)
        print(token, end="", flush=True)
    
    print("\n" + "=" * 80)
    print(f"\n✅ Total tokens: {len(tokens)}")

if __name__ == "__main__":
    asyncio.run(test_intent_analysis())
