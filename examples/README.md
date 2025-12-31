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
python examples/customer_support_agent.py
```

This will run 3 test queries demonstrating:
- Order status lookup
- Delivery estimation
- FAQ search

---

## 🎯 When to Use Each Example

| Example | Use Case |
| :--- | :--- |
| `customer_support_agent.py` | **Learn the full pattern**: Building a complete, production-ready agent from scratch. |
| `demo_custom_agent.py` | **Learn advanced customization**: Creating custom states and modifying the execution flow. |

---

## 📚 Next Steps

After understanding these examples, you can:
1.  Replace the mock tools with real API calls (database, external services, etc.)
2.  Add more sophisticated validation logic in custom states
3.  Implement persistence and resume capabilities
4.  Deploy your agent via FastAPI (see `api/api.py` for reference)

