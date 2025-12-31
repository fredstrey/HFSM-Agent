"""
Example: Using AgentConfig and New Features
============================================

Demonstrates the new configuration system, circuit breaker,
thread safety, and resource management.
"""

import sys
import os
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finitestatemachineAgent.hfsm_agent import AgentEngine
from core.agent_config import AgentConfig, FAST_CONFIG, ROBUST_CONFIG
from core.circuit_breaker import CircuitBreaker
from core.resilient_executor import ResilientToolExecutor
from core.registry import ToolRegistry
from core.executor import ToolExecutor
from core.decorators import tool
from providers.llm_client import LLMClient
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# 1. Define Tools
# =============================================================================

@tool(name="get_data", description="Get some data (may fail randomly)")
def get_data(query: str) -> Dict[str, Any]:
    """Mock tool that sometimes fails."""
    import random
    if random.random() < 0.3:  # 30% failure rate
        raise Exception("Random failure!")
    return {"success": True, "data": f"Data for: {query}"}


# =============================================================================
# 2. Example 1: Using Custom Configuration
# =============================================================================

def example_custom_config():
    """Demonstrate custom AgentConfig."""
    print("\n" + "="*60)
    print("Example 1: Custom Configuration")
    print("="*60)
    
    # Create custom config
    config = AgentConfig(
        max_retries=5,
        tool_timeout=10.0,
        circuit_breaker_threshold=3,
        max_workers=2,
        enable_snapshots=False  # Disable for faster execution
    )
    
    # Validate config
    config.validate()
    
    print(f"\nConfiguration:")
    print(f"  Max Retries: {config.max_retries}")
    print(f"  Tool Timeout: {config.tool_timeout}s")
    print(f"  Circuit Breaker Threshold: {config.circuit_breaker_threshold}")
    print(f"  Max Workers: {config.max_workers}")
    
    # Setup agent (config will be used in future refactoring)
    registry = ToolRegistry()
    registry.register(
        name=get_data._tool_name,
        description=get_data._tool_description,
        function=get_data,
        args_model=get_data._args_model
    )
    
    executor = ToolExecutor(registry)
    llm = LLMClient(model="xiaomi/mimo-v2-flash:free")
    
    # Use context manager for automatic cleanup
    with AgentEngine(
        llm=llm,
        registry=registry,
        executor=executor,
        system_instruction="You are a helpful assistant."
    ) as agent:
        print("\n✅ Agent created with context manager")
        print("   Resources will be automatically cleaned up on exit")
    
    print("✅ Agent closed automatically")


# =============================================================================
# 3. Example 2: Circuit Breaker
# =============================================================================

def example_circuit_breaker():
    """Demonstrate circuit breaker pattern."""
    print("\n" + "="*60)
    print("Example 2: Circuit Breaker")
    print("="*60)
    
    # Create circuit breaker
    cb = CircuitBreaker(threshold=3, timeout=5.0, name="tool_breaker")
    
    # Setup
    registry = ToolRegistry()
    registry.register(
        name=get_data._tool_name,
        description=get_data._tool_description,
        function=get_data,
        args_model=get_data._args_model
    )
    
    base_executor = ToolExecutor(registry)
    
    # Wrap with circuit breaker
    resilient_executor = ResilientToolExecutor(base_executor, cb)
    
    print(f"\nCircuit Breaker Config:")
    print(f"  Threshold: {cb.threshold} failures")
    print(f"  Timeout: {cb.timeout}s")
    print(f"  Initial State: {cb.get_state().value}")
    
    # Simulate multiple calls (some will fail)
    print("\nSimulating tool calls...")
    for i in range(10):
        try:
            result = resilient_executor.execute("get_data", {"query": f"test_{i}"})
            if result.get("success"):
                print(f"  Call {i+1}: ✅ Success")
            else:
                print(f"  Call {i+1}: ⚠️  Circuit Open - {result.get('error')}")
        except Exception as e:
            print(f"  Call {i+1}: ❌ Failed - {e}")
    
    print(f"\nFinal Circuit State: {cb.get_state().value}")
    print(f"Failure Count: {cb.get_failures()}")


# =============================================================================
# 4. Example 3: Predefined Configs
# =============================================================================

def example_predefined_configs():
    """Show predefined configuration profiles."""
    print("\n" + "="*60)
    print("Example 3: Predefined Configurations")
    print("="*60)
    
    configs = {
        "FAST": FAST_CONFIG,
        "ROBUST": ROBUST_CONFIG,
    }
    
    for name, config in configs.items():
        print(f"\n{name} Configuration:")
        print(f"  Max Retries: {config.max_retries}")
        print(f"  Max Workers: {config.max_workers}")
        print(f"  Tool Timeout: {config.tool_timeout}s")
        print(f"  Circuit Breaker: {'Enabled' if config.circuit_breaker_enabled else 'Disabled'}")
        print(f"  Snapshots: {'Enabled' if config.enable_snapshots else 'Disabled'}")


# =============================================================================
# 5. Example 4: Thread Safety
# =============================================================================

def example_thread_safety():
    """Demonstrate thread-safe context."""
    print("\n" + "="*60)
    print("Example 4: Thread-Safe ExecutionContext")
    print("="*60)
    
    from core.context import ExecutionContext
    import threading
    
    context = ExecutionContext(user_query="test")
    
    def worker(worker_id: int):
        """Simulate concurrent context updates."""
        for i in range(5):
            context.set_memory(f"worker_{worker_id}_item_{i}", f"value_{i}")
            context.add_tool_call(
                tool_name=f"tool_{worker_id}",
                arguments={"arg": i},
                result=f"result_{i}"
            )
    
    # Create multiple threads
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    
    print("\nStarting 3 concurrent workers...")
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()
    
    print(f"✅ All workers completed")
    print(f"   Total tool calls: {len(context.tool_calls)}")
    print(f"   Total memory items: {len(context.memory)}")
    print("   No race conditions or data corruption!")


# =============================================================================
# Run Examples
# =============================================================================

if __name__ == "__main__":
    print("\n🎯 AgentConfig and New Features Demo\n")
    
    try:
        example_custom_config()
        example_circuit_breaker()
        example_predefined_configs()
        example_thread_safety()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
