"""
Example: Using AgentConfig and New Async Features
================================================
Demonstrates the new configuration system, circuit breaker (concept),
thread safety (via asyncio), and resource management in the ASYNC framework.
"""

import sys
import os
import asyncio
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finitestatemachineAgent.hfsm_agent_async import AsyncAgentEngine
from core.agent_config import AgentConfig, FAST_CONFIG, ROBUST_CONFIG
from core.executor_async import AsyncToolExecutor
from core.registry import ToolRegistry
from core.decorators import tool
from providers.llm_client_async import AsyncLLMClient
from core.context_async import AsyncExecutionContext
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# 1. Define Tools
# =============================================================================

@tool(name="get_data", description="Get some data (may fail randomly)")
async def get_data(query: str) -> Dict[str, Any]:
    """Mock async tool that sometimes fails."""
    import random
    # Simulate async I/O
    await asyncio.sleep(0.1)
    
    if random.random() < 0.3:  # 30% failure rate
        raise Exception("Random failure!")
    return {"success": True, "data": f"Data for: {query}"}


# =============================================================================
# 2. Example 1: Async Configuration & Context
# =============================================================================

async def example_custom_config():
    """Demonstrate configuration in async context."""
    print("\n" + "="*60)
    print("Example 1: Async Configuration")
    print("="*60)
    
    # Create custom config (illustrative)
    config = AgentConfig(
        max_retries=5,
        tool_timeout=10.0,
        max_workers=10  # Higher concurrency in async!
    )
    
    print(f"\nConfiguration:")
    print(f"  Max Retries: {config.max_retries}")
    print(f"  Max Workers: {config.max_workers} (Async tasks)")
    
    # Setup agent components
    registry = ToolRegistry()
    registry.register(
        name=get_data._tool_name,
        description=get_data._tool_description,
        function=get_data,
        args_model=get_data._args_model
    )
    
    executor = AsyncToolExecutor(registry)
    llm = AsyncLLMClient(model="xiaomi/mimo-v2-flash:free")
    
    # Initialize Async Engine
    # Note: In async, we don't need a context manager for cleanup (usually),
    # but we can implement close() methods if needed for HTTP clients.
    agent = AsyncAgentEngine(
        llm=llm,
        registry=registry,
        executor=executor,
        system_instruction="You are a helpful assistant.",
        tool_choice="auto"
    )
    
    print("\n✅ Async Agent created")
    
    # Simulate a run
    context = AsyncExecutionContext(
        user_query="get data for analysis",
        max_iterations=5
    )
    
    # Inject config into context memory (how HFSM accesses it)
    await context.set_memory("max_retries", config.max_retries)
    
    print(f"   Context configured with max_retries={config.max_retries}")
    
    # We won't actually run the full agent here to keep it fast,
    # but this shows setup.
    print("✅ Async setup complete")


# =============================================================================
# 3. Example 2: Async Concurrency & Safety
# =============================================================================

async def example_async_safety():
    """Demonstrate async-safe execution context."""
    print("\n" + "="*60)
    print("Example 2: Async ExecutionContext Safety")
    print("="*60)
    
    context = AsyncExecutionContext(user_query="test")
    
    async def worker(worker_id: int):
        """Simulate concurrent context updates."""
        for i in range(5):
            # Async safe operations with await
            await context.set_memory(f"worker_{worker_id}_item_{i}", f"value_{i}")
            
            await context.add_tool_call(
                tool_name=f"tool_{worker_id}",
                arguments={"arg": i},
                result=f"result_{i}"
            )
            # Yield control
            await asyncio.sleep(0.01)
    
    print("\nStarting 3 concurrent workers...")
    
    # Run concurrently
    await asyncio.gather(
        worker(1),
        worker(2),
        worker(3)
    )
    
    print(f"✅ All workers completed")
    print(f"   Total tool calls: {len(context.tool_calls)}")
    print(f"   Total memory items: {len(context.memory)}")
    print("   No race conditions! (Asyncio is single-threaded but concurrent)")


# =============================================================================
# Run Examples
# =============================================================================

async def main():
    print("\n🎯 Async AgentConfig and Features Demo\n")
    
    try:
        await example_custom_config()
        await example_async_safety()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
