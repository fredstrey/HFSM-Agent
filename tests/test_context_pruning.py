"""
Test Context Pruning
====================

Test async context pruning functionality.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from finitestatemachineAgent import Agent
from finitestatemachineAgent.context_pruning import AsyncContextPruner

def test_default_no_pruning():
    """Test that pruning is disabled by default."""
    print("Test 1: Default behavior (no pruning)")
    agent = Agent(model="xiaomi/mimo-v2-flash:free")
    
    # Check that pruner is not created
    has_pruner = hasattr(agent.engine, 'pruner')
    print(f"  Has pruner attribute: {has_pruner}")
    
    # Check hierarchy
    policy_is_root = agent.engine.policy == agent.engine.root
    print(f"  Policy points to root (no middleware): {policy_is_root}")
    
    if not has_pruner and policy_is_root:
        print("  ✅ PASS: Pruning disabled by default\n")
    else:
        print("  ❌ FAIL: Pruning should be disabled by default\n")


def test_pruning_enabled():
    """Test that pruning can be enabled."""
    print("Test 2: Pruning enabled")
    agent = Agent(
        model="xiaomi/mimo-v2-flash:free",
        enable_context_pruning=True,
        pruner_keep_recent=3,
        pruner_max_length=150
    )
    
    # Check that pruner is created
    has_pruner = hasattr(agent.engine, 'pruner')
    print(f"  Has pruner attribute: {has_pruner}")
    
    if has_pruner:
        print(f"  Pruner keep_recent: {agent.engine.pruner.keep_recent}")
        print(f"  Pruner max_length: {agent.engine.pruner.max_length}")
        
        # Check hierarchy
        policy_type = type(agent.engine.policy).__name__
        print(f"  Policy type: {policy_type}")
        
        if policy_type == "AsyncContextPolicyState" and agent.engine.pruner.keep_recent == 3:
            print("  ✅ PASS: Pruning enabled correctly\n")
        else:
            print("  ❌ FAIL: Pruning configuration incorrect\n")
    else:
        print("  ❌ FAIL: Pruner not created\n")


def test_custom_pruner():
    """Test custom pruner injection."""
    print("Test 3: Custom pruner")
    
    class MyCustomPruner(AsyncContextPruner):
        async def prune(self, context):
            print("    Custom pruner called!")
            await super().prune(context)
    
    custom_pruner = MyCustomPruner(keep_recent=2, max_length=100)
    
    agent = Agent(
        model="xiaomi/mimo-v2-flash:free",
        enable_context_pruning=True,
        context_pruner=custom_pruner
    )
    
    # Check that custom pruner is used
    is_custom = isinstance(agent.engine.pruner, MyCustomPruner)
    print(f"  Using custom pruner: {is_custom}")
    print(f"  Custom pruner keep_recent: {agent.engine.pruner.keep_recent}")
    
    if is_custom and agent.engine.pruner.keep_recent == 2:
        print("  ✅ PASS: Custom pruner injected correctly\n")
    else:
        print("  ❌ FAIL: Custom pruner not working\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Async Context Pruning")
    print("=" * 60 + "\n")
    
    test_default_no_pruning()
    test_pruning_enabled()
    test_custom_pruner()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
