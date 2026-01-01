"""
Async Components Test
======================

Test async providers, executor, and context before full HFSM migration.
"""

import sys
import os
import asyncio
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from providers.llm_client_async import AsyncLLMClient
from core.executor_async import AsyncToolExecutor
from core.context_async import AsyncExecutionContext
from core.registry import ToolRegistry
from core.decorators import tool
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Test Tools (Sync and Async)
# =============================================================================

@tool(name="sync_tool", description="Synchronous test tool")
def sync_tool(message: str) -> Dict[str, Any]:
    """Sync tool for testing."""
    import time
    time.sleep(0.1)  # Simulate work
    return {"success": True, "message": f"Sync: {message}"}


@tool(name="async_tool", description="Asynchronous test tool")
async def async_tool(message: str) -> Dict[str, Any]:
    """Async tool for testing."""
    await asyncio.sleep(0.1)  # Simulate async work
    return {"success": True, "message": f"Async: {message}"}

# Mark async tool
async_tool._is_async = True


# =============================================================================
# Test Functions
# =============================================================================

async def test_async_llm():
    """Test async LLM client."""
    print("\n" + "="*60)
    print("Test 1: Async LLM Client")
    print("="*60)
    
    llm = AsyncLLMClient(model="xiaomi/mimo-v2-flash:free")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one sentence."}
    ]
    
    print("\nCalling async LLM...")
    response = await llm.chat(messages)
    
    print(f"✅ Response: {response['content'][:100]}...")
    print(f"   Tokens: {response['usage'].get('total_tokens', 'N/A')}")


async def test_async_executor():
    """Test async tool executor."""
    print("\n" + "="*60)
    print("Test 2: Async Tool Executor")
    print("="*60)
    
    # Setup registry
    registry = ToolRegistry()
    registry.register(
        name=sync_tool._tool_name,
        description=sync_tool._tool_description,
        function=sync_tool,
        args_model=sync_tool._args_model
    )
    registry.register(
        name=async_tool._tool_name,
        description=async_tool._tool_description,
        function=async_tool,
        args_model=async_tool._args_model
    )
    
    executor = AsyncToolExecutor(registry)
    
    # Test parallel execution
    tool_calls = [
        {"tool_name": "sync_tool", "arguments": {"message": "test1"}},
        {"tool_name": "async_tool", "arguments": {"message": "test2"}},
        {"tool_name": "sync_tool", "arguments": {"message": "test3"}},
    ]
    
    print(f"\nExecuting {len(tool_calls)} tools concurrently...")
    import time
    start = time.time()
    
    results = await executor.execute_parallel(tool_calls)
    
    elapsed = time.time() - start
    
    print(f"✅ All tools completed in {elapsed:.2f}s")
    for i, result in enumerate(results):
        print(f"   Tool {i+1}: {result.get('message', result)}")
    
    # Should be ~0.1s (concurrent) not ~0.3s (sequential)
    if elapsed < 0.2:
        print(f"🚀 Concurrency working! (would take ~0.3s sequentially)")


async def test_async_context():
    """Test async execution context."""
    print("\n" + "="*60)
    print("Test 3: Async Execution Context")
    print("="*60)
    
    context = AsyncExecutionContext(user_query="test")
    
    # Test concurrent updates
    async def worker(worker_id: int):
        for i in range(3):
            await context.set_memory(f"worker_{worker_id}_{i}", f"value_{i}")
            await context.add_tool_call(
                tool_name=f"tool_{worker_id}",
                arguments={"arg": i},
                result=f"result_{i}"
            )
    
    print("\nRunning 3 concurrent workers...")
    await asyncio.gather(
        worker(1),
        worker(2),
        worker(3)
    )
    
    print(f"✅ All workers completed")
    print(f"   Total tool calls: {len(context.tool_calls)}")
    print(f"   Total memory items: {len(context.memory)}")


async def test_async_streaming():
    """Test async streaming."""
    print("\n" + "="*60)
    print("Test 4: Async Streaming")
    print("="*60)
    
    llm = AsyncLLMClient(model="xiaomi/mimo-v2-flash:free")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count from 1 to 5."}
    ]
    
    print("\nStreaming response:")
    print("   ", end="", flush=True)
    
    async for token in llm.chat_stream(messages):
        print(token, end="", flush=True)
    
    print("\n✅ Streaming completed")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all async tests."""
    print("\n🧪 Async Components Test Suite\n")
    
    try:
        await test_async_llm()
        await test_async_executor()
        await test_async_context()
        await test_async_streaming()
        
        print("\n" + "="*60)
        print("✅ All async tests passed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
