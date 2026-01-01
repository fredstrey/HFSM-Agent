"""
Async HFSM Agent End-to-End Test
=================================

Test the complete async HFSM agent with real tools and LLM.
"""

import sys
import os
import asyncio
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finitestatemachineAgent.hfsm_agent_async import AsyncAgentEngine
from providers.llm_client_async import AsyncLLMClient
from core.executor_async import AsyncToolExecutor
from core.registry import ToolRegistry
from core.decorators import tool
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Test Tools
# =============================================================================

@tool(name="get_weather", description="Get weather for a city")
def get_weather(city: str) -> Dict[str, Any]:
    """Mock weather tool."""
    return {
        "success": True,
        "city": city,
        "temperature": 25,
        "condition": "Sunny"
    }


@tool(name="calculate", description="Perform calculation")
def calculate(expression: str) -> Dict[str, Any]:
    """Mock calculator tool."""
    try:
        result = eval(expression)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# Tests
# =============================================================================

async def test_simple_query():
    """Test simple query without tools."""
    print("\n" + "="*60)
    print("Test 1: Simple Query (No Tools)")
    print("="*60)
    
    # Setup
    registry = ToolRegistry()
    executor = AsyncToolExecutor(registry)
    llm = AsyncLLMClient(model="xiaomi/mimo-v2-flash:free")
    
    agent = AsyncAgentEngine(
        llm=llm,
        registry=registry,
        executor=executor,
        system_instruction="You are a helpful assistant."
    )
    
    # Run
    print("\nQuery: What is 2+2?")
    print("Response: ", end="", flush=True)
    
    async for token in agent.run_stream("What is 2+2?"):
        print(token, end="", flush=True)
    
    print("\n✅ Simple query completed")


async def test_tool_calling():
    """Test query that requires tool calling."""
    print("\n" + "="*60)
    print("Test 2: Tool Calling")
    print("="*60)
    
    # Setup with tools
    registry = ToolRegistry()
    registry.register(
        name=get_weather._tool_name,
        description=get_weather._tool_description,
        function=get_weather,
        args_model=get_weather._args_model
    )
    registry.register(
        name=calculate._tool_name,
        description=calculate._tool_description,
        function=calculate,
        args_model=calculate._args_model
    )
    
    executor = AsyncToolExecutor(registry)
    llm = AsyncLLMClient(model="xiaomi/mimo-v2-flash:free")
    
    agent = AsyncAgentEngine(
        llm=llm,
        registry=registry,
        executor=executor,
        system_instruction="You are a helpful assistant with access to tools."
    )
    
    # Run
    print("\nQuery: What's the weather in Paris?")
    print("Response: ", end="", flush=True)
    
    async for token in agent.run_stream("What's the weather in Paris?"):
        print(token, end="", flush=True)
    
    print("\n✅ Tool calling completed")


async def test_concurrent_queries():
    """Test multiple concurrent agent runs."""
    print("\n" + "="*60)
    print("Test 3: Concurrent Queries")
    print("="*60)
    
    # Setup
    registry = ToolRegistry()
    registry.register(
        name=calculate._tool_name,
        description=calculate._tool_description,
        function=calculate,
        args_model=calculate._args_model
    )
    
    executor = AsyncToolExecutor(registry)
    llm = AsyncLLMClient(model="xiaomi/mimo-v2-flash:free")
    
    agent = AsyncAgentEngine(
        llm=llm,
        registry=registry,
        executor=executor,
        system_instruction="You are a math assistant."
    )
    
    # Run 3 queries concurrently
    queries = [
        "What is 10 + 20?",
        "What is 5 * 6?",
        "What is 100 / 4?"
    ]
    
    print(f"\nRunning {len(queries)} queries concurrently...")
    
    async def run_query(q, idx):
        print(f"\n  Query {idx+1}: {q}")
        print(f"  Response {idx+1}: ", end="", flush=True)
        async for token in agent.run_stream(q):
            print(token, end="", flush=True)
        print()
    
    import time
    start = time.time()
    
    await asyncio.gather(*[run_query(q, i) for i, q in enumerate(queries)])
    
    elapsed = time.time() - start
    
    print(f"\n✅ All {len(queries)} queries completed in {elapsed:.2f}s")
    print(f"   (Sequential would take ~{len(queries) * 3}s)")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all tests."""
    print("\n🧪 Async HFSM Agent End-to-End Test\n")
    
    try:
        await test_simple_query()
        await test_tool_calling()
        await test_concurrent_queries()
        
        print("\n" + "="*60)
        print("✅ All async HFSM tests passed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
