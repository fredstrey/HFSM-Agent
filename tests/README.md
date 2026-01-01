# 🎓 Development Guide: Agents and Tools

Welcome to the **Finance.AI / Fred.AI Framework** examples.
This guide teaches you how to extend the system by creating new Agents, States (HFSM), and Custom Tools.

## 🏗️ Framework Structure

The system uses a **HFSM (Hierarchical Finite State Machine)**. Unlike simple graphs (LangGraph) or linear chains, our state machine allows for robust hierarchy and governance.

### Key Concepts

1.  **AgentEngine**: The "brain" that manages the execution loop.
2.  **HierarchicalState**: Base class for all states.
    *   `handle(context)`: Processes logic and returns the *next* state.
    *   `on_enter(context)`: Executed when entering the state.
3.  **ExecutionContext**: Shared memory (variables, history, metrics).
4.  **Tools**: Python functions decorated with `@tool`.

---

## 🛠️ Creating Tools

Tools are standard Python functions. Use the `@tool` decorator to automatically register them in the LLM schema.

```python
from core.decorators import tool
from typing import Dict, Any

@tool(
    name="check_verification",
    description="Checks if a user is verified in the system."
)
def check_verification(user_id: str) -> Dict[str, Any]:
    # Real logic here (DB, API, etc.)
    is_verified = user_id.startswith("ADM")
    
    return {
        "success": True,
        "verified": is_verified,
        "role": "admin" if is_verified else "user"
    }
```

---

## 🧠 Creating Custom States

To create new behavior (e.g., an approval flow), create a subclass of `HierarchicalState`.

```python
from finitestatemachineAgent.hfsm_agent import HierarchicalState, ExecutionContext

class ApprovalState(HierarchicalState):
    def handle(self, context: ExecutionContext):
        print("🚦 Analyzing approval...")
        
        # Decision logic
        user_data = context.get_memory("user_data")
        
        if user_data.get("verified"):
            # Transition to next state (e.g., SuccessState)
            return self.parent.find_state_by_type("SuccessState")
        else:
            # Transition to failure or retry
            return self.parent.find_state_by_type("FailState")
```

---

## 🚀 Complete Example

See the `demo_custom_agent.py` file in this folder for an executable example that implements:

1.  **Mock Tools** (Weather, Flight Booking).
2.  **Custom States** (`WeatherCheckState`, `BookingState`).
3.  **Engine Assembly** and execution of a conversational flow.

---

## 🚀 Complete Example

See the `customer_support_agent.py` file in this folder for a **production-ready** example that implements:

1.  **Domain-Specific Tools**: Mock customer support tools (`check_order_status`, `estimate_delivery`, `search_faq`).
2.  **Agent Wrapper Class**: `CustomerSupportAgent` that encapsulates the entire setup (following the `rag_agent_hfsm.py` pattern).
3.  **Clean Interface**: Simple `run_stream(query)` method for external use.
4.  **Best Practices**: Proper initialization, tool registration, and system instruction design.

### How to Run the Complete Example

```bash
python examples/customer_support_age# Finance.AI Examples

This directory contains learning resources and practical examples for using the Finance.AI framework.

**⭐ All examples have been updated to the new Async Architecture (v3.0)!**

## 📚 Learning Path

Recommended order for learning the framework:

### 1. [test_async.py](./test_async.py)
**Concept**: Async Components
- How `AsyncLLMClient` works
- How `AsyncToolExecutor` runs tools concurrently
- How `AsyncExecutionContext` handles thread safety

### 2. [agent_config_demo.py](./agent_config_demo.py)
**Concept**: Configuration & Resilience
- Thread-safe memory management
- Async concurrency patterns
- Timeout handling

### 3. [customer_support_agent.py](./customer_support_agent.py)
**Concept**: Building a Complete Agent
- Creating domain-specific async tools (`@tool`)
- Wrapping `AsyncAgentEngine`
- Streaming responses with `run_stream()`

### 4. [demo_custom_agent.py](./demo_custom_agent.py)
**Concept**: Advanced Customization
- Creating custom async states (`AsyncHierarchicalState`)
- Modifying state transitions
- Implementing middleware logic (e.g., Visa Check)

### 5. [test_async_hfsm.py](./test_async_hfsm.py)
**Concept**: Testing
- Unit testing async agents
- Mocking async tools
- Verifying FSM transitions

## 🏃 Running Examples

Since all examples are async, run them using python directly:

```bash
# 1. Basic Async Components
python examples/test_async.py

# 2. Config & Resilience
python examples/agent_config_demo.py

# 3. Customer Support Agent
python examples/customer_support_agent.py

# 4. Custom States (Travel Agent)
python examples/demo_custom_agent.py
```
