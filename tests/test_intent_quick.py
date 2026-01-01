"""
Quick test to verify IntentAnalysisState logs
"""
import asyncio
import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.rag_custom_states import IntentAnalysisState
from providers.llm_client_async import AsyncLLMClient
from core.context_async import AsyncExecutionContext
from core.safety_monitor import SafetyMonitor

async def test():
    print("🧪 Testing IntentAnalysisState...")
    
    # Create state
    llm = AsyncLLMClient(model="xiaomi/mimo-v2-flash:free")
    state = IntentAnalysisState(parent=None, llm=llm)
    
    # Create context
    monitor = SafetyMonitor(max_requests=10)
    context = AsyncExecutionContext(user_query="Como funciona o leasing?", safety_monitor=monitor)
    await context.set_memory("system_instruction", "Você é um assistente financeiro.")
    await context.set_memory("chat_history", [])
    
    print("\n📝 Calling IntentAnalysis.handle()...\n")
    
    # Call handle
    result = await state.handle(context)
    
    print(f"\n✅ Result: {result}")
    print(f"📊 Intent analyzed: {await context.get_memory('intent_analyzed')}")
    print(f"📝 Todo list: {await context.get_memory('todo_list')}")

if __name__ == "__main__":
    asyncio.run(test())
