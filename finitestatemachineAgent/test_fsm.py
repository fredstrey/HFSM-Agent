import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from FiniteStateMachineAgent.fsm_agent import StateMachineAgent
from pydantic import BaseModel, Field

# Mock Tool
class CalculateArgs(BaseModel):
    operation: str = Field(..., description="Operation: add, subtract, multiply")
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")

def calculate(operation: str, a: float, b: float) -> str:
    """Performs basic math calculations"""
    if operation == "add":
        return str(a + b)
    elif operation == "subtract":
        return str(a - b)
    elif operation == "multiply":
        return str(a * b)
    else:
        return "Unknown operation"

# Add metadata to function (simulating @tool decorator)
calculate._tool_name = "calculate"
calculate._tool_description = "Performs basic math calculations (add, subtract, multiply)"
calculate._args_model = CalculateArgs


def main():
    print("Initializing FSM Agent...")
    try:
        # Use a known capable model or default
        agent = StateMachineAgent(
            model="xiaomi/mimo-v2-flash:free", 
            tools=[calculate],
            max_steps=10
        )
        
        query = "Quanto é 50 vezes 25? E depois some 100 ao resultado."
        print(f"\nUser Query: {query}")
        
        response = agent.run(query)
        
        print("\nFinal Response:")
        print(response.answer)
        
        print("\nMetadata (Tool Calls):")
        for i, call in enumerate(response.metadata.get("tool_calls", [])):
            print(f"{i+1}. {call['tool_name']} -> {call['result']}")
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
