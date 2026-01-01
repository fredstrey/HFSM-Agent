"""
Test Contract-Based Forking
=============================

Verifies that contract extraction and deterministic merging work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from finitestatemachineAgent.fork_contracts import ForkResult, Claim, MergedContract
from finitestatemachineAgent.hfsm_agent_async import MergeState, ExecutionState
from core.context_async import AsyncExecutionContext, SafetyMonitor


async def test_contract_merge_consensus():
    """Test that merge detects consensus when all forks agree."""
    print("\n🧪 Test 1: Consensus Detection")
    print("=" * 60)
    
    # Create 2 forks with same claim
    monitor = SafetyMonitor(max_requests=50)
    
    fork1 = AsyncExecutionContext(user_query="", safety_monitor=monitor)
    await fork1.set_memory("branch_id", "fork_1")
    await fork1.set_memory("fork_contract", ForkResult(
        branch_id="fork_1",
        claims=[
            Claim(key="leasing.supervisor", value="BCB", evidence=["tool:search"], confidence=0.9)
        ],
        coverage=["leasing"],
        omissions=[]
    ).model_dump())
    
    fork2 = AsyncExecutionContext(user_query="", safety_monitor=monitor)
    await fork2.set_memory("branch_id", "fork_2")
    await fork2.set_memory("fork_contract", ForkResult(
        branch_id="fork_2",
        claims=[
            Claim(key="leasing.supervisor", value="BCB", evidence=["tool:web"], confidence=0.95)
        ],
        coverage=["leasing"],
        omissions=[]
    ).model_dump())
    
    # Merge
    merge_state = MergeState(ExecutionState(None))
    merged = merge_state._contract_merge([fork1, fork2])
    
    # Verify
    assert "leasing.supervisor" in merged["resolved"], "Should have resolved claim"
    assert len(merged["conflicts"]) == 0, "Should have no conflicts"
    
    print(f"✅ Consensus detected correctly")
    print(f"   - Resolved: {merged['resolved']['leasing.supervisor']['value']}")


async def test_contract_merge_conflict():
    """Test that merge detects conflicts when forks disagree."""
    print("\n🧪 Test 2: Conflict Detection")
    print("=" * 60)
    
    # Create 2 forks with different claims
    monitor = SafetyMonitor(max_requests=50)
    
    fork1 = AsyncExecutionContext(user_query="", safety_monitor=monitor)
    await fork1.set_memory("branch_id", "fork_1")
    await fork1.set_memory("fork_contract", ForkResult(
        branch_id="fork_1",
        claims=[
            Claim(key="spread.applies", value="Yes", evidence=["tool:search"], confidence=0.8)
        ],
        coverage=["spread"],
        omissions=[]
    ).model_dump())
    
    fork2 = AsyncExecutionContext(user_query="", safety_monitor=monitor)
    await fork2.set_memory("branch_id", "fork_2")
    await fork2.set_memory("fork_contract", ForkResult(
        branch_id="fork_2",
        claims=[
            Claim(key="spread.applies", value="No", evidence=["tool:web"], confidence=0.7)
        ],
        coverage=["spread"],
        omissions=[]
    ).model_dump())
    
    # Merge
    merge_state = MergeState(ExecutionState(None))
    merged = merge_state._contract_merge([fork1, fork2])
    
    # Verify
    assert "spread.applies" in merged["conflicts"], "Should have conflict"
    assert len(merged["resolved"]) == 0, "Should have no resolved claims"
    assert len(merged["conflicts"]["spread.applies"]) == 2, "Should have 2 conflicting values"
    
    print(f"✅ Conflict detected correctly")
    print(f"   - Conflicting values: {[c['value'] for c in merged['conflicts']['spread.applies']]}")


async def test_contract_merge_mixed():
    """Test merge with both consensus and conflicts."""
    print("\n🧪 Test 3: Mixed Consensus and Conflicts")
    print("=" * 60)
    
    # Create 2 forks with mixed claims
    monitor = SafetyMonitor(max_requests=50)
    
    fork1 = AsyncExecutionContext(user_query="", safety_monitor=monitor)
    await fork1.set_memory("branch_id", "fork_1")
    await fork1.set_memory("fork_contract", ForkResult(
        branch_id="fork_1",
        claims=[
            Claim(key="topic.a", value="Agreed", evidence=["tool:search"], confidence=0.9),
            Claim(key="topic.b", value="Value1", evidence=["tool:search"], confidence=0.8)
        ],
        coverage=["topic"],
        omissions=[]
    ).model_dump())
    
    fork2 = AsyncExecutionContext(user_query="", safety_monitor=monitor)
    await fork2.set_memory("branch_id", "fork_2")
    await fork2.set_memory("fork_contract", ForkResult(
        branch_id="fork_2",
        claims=[
            Claim(key="topic.a", value="Agreed", evidence=["tool:web"], confidence=0.95),
            Claim(key="topic.b", value="Value2", evidence=["tool:web"], confidence=0.7)
        ],
        coverage=["topic"],
        omissions=[]
    ).model_dump())
    
    # Merge
    merge_state = MergeState(ExecutionState(None))
    merged = merge_state._contract_merge([fork1, fork2])
    
    # Verify
    assert "topic.a" in merged["resolved"], "Should have resolved topic.a"
    assert "topic.b" in merged["conflicts"], "Should have conflict on topic.b"
    
    print(f"✅ Mixed merge handled correctly")
    print(f"   - Resolved: topic.a = {merged['resolved']['topic.a']['value']}")
    print(f"   - Conflict: topic.b has {len(merged['conflicts']['topic.b'])} values")


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🚀 Contract-Based Forking Tests")
    print("=" * 60)
    
    try:
        await test_contract_merge_consensus()
        await test_contract_merge_conflict()
        await test_contract_merge_mixed()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
