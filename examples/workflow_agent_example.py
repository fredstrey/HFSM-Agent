"""
Simple Workflow Agent Example
==============================
Testa funcionamento de initial_state="AnswerState".
"""
import sys
import asyncio
from pathlib import Path

# Add root folder to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from finitestatemachineAgent.agent import Agent

async def main():
    print("🧪 Testing Workflow Agent (initial_state)...")
    
    try:
        # Inicializa agente com initial_state e dummy key para passar validação
        agent = Agent(
            llm_provider="openrouter",
            model="xiaomi/mimo-v2-flash:free",
            api_key="sk-dummy-key", 
            initial_state="AnswerState",
            enable_intent_analysis=False,
            enable_parallel_planning=False
        )
        
        # 1. Verifica configuração interna
        actual_state = getattr(agent.engine, "initial_state_name", None)
        expected = "AnswerState"
        
        print(f"🔹 Configured: initial_state='{expected}'")
        print(f"🔹 Actual in Engine: '{actual_state}'")
        
        if actual_state == expected:
            print("✅ SUCCESS: Engine accepted initial_state parameter.")
        else:
            print(f"❌ FAILED: Param not passed to engine correctly.")
            
        # 2. Tenta executar (opcional)
        print("Attempting execution (may fail auth)...")
        try:
            await asyncio.wait_for(agent.run("Hello"), timeout=5)
        except Exception as e:
            if "API_KEY" in str(e) or "environment" in str(e) or "401" in str(e):
                 print(f"⚠️ Run aborted due to Auth (Expected): {e}")
            else:
                 print(f"⚠️ Run failed: {e}")

    except Exception as e:
        print(f"❌ Error: {e}")

    print("✅ Test execution finished.")

if __name__ == "__main__":
    asyncio.run(main())
