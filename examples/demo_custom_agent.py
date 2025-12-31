"""
Example: Travel Agent with Custom States
==========================================
This example demonstrates ADVANCED customization of the HFSM framework.
It builds a complete Travel Agent with custom states that modify the execution flow.

Key Learning Points:
1. How to create a complete agent (like customer_support_agent.py)
2. How to add custom states that intercept and modify behavior
3. How to integrate custom states into the normal execution flow
4. When and why to use custom states vs just tools
"""

import sys
import os
from typing import List, Dict, Optional, Generator
from datetime import datetime, timedelta

# Add root path to import core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finitestatemachineAgent.hfsm_agent import AgentEngine, HierarchicalState
from finitestatemachineAgent import hfsm_agent
from core.context import ExecutionContext
from core.decorators import tool
from core.registry import ToolRegistry
from core.executor import ToolExecutor
from providers.llm_client import LLMClient


# =============================================================================
# 1. DOMAIN-SPECIFIC TOOLS (Travel Domain)
# =============================================================================

# Mock databases
WEATHER_DB = {
    "PARIS": {"temp": 15, "condition": "Cloudy", "advisory": None},
    "LONDON": {"temp": 12, "condition": "Rainy", "advisory": "Bring umbrella"},
    "RIO": {"temp": 30, "condition": "Sunny", "advisory": None},
    "TOKYO": {"temp": 8, "condition": "Clear", "advisory": None},
}

VISA_REQUIREMENTS = {
    "PARIS": {"US": False, "BR": False, "CN": True},
    "LONDON": {"US": False, "BR": False, "CN": True},
    "RIO": {"US": False, "BR": False, "CN": False},
    "TOKYO": {"US": False, "BR": False, "CN": False},
}


@tool(
    name="check_weather",
    description="Check weather forecast for a destination city."
)
def check_weather(city: str) -> Dict[str, any]:
    """Get weather information for a city."""
    print(f"🌤️  [TOOL] Checking weather in: {city}")
    
    weather = WEATHER_DB.get(city.upper())
    
    if not weather:
        return {
            "success": False,
            "error": f"Weather data not available for {city}"
        }
    
    return {
        "success": True,
        "city": city,
        "temperature": weather["temp"],
        "condition": weather["condition"],
        "advisory": weather["advisory"]
    }


@tool(
    name="check_visa_requirements",
    description="Check if a visa is required for travel from one country to a destination."
)
def check_visa_requirements(destination: str, nationality: str) -> Dict[str, any]:
    """Check visa requirements."""
    print(f"🛂 [TOOL] Checking visa: {nationality} → {destination}")
    
    dest_reqs = VISA_REQUIREMENTS.get(destination.upper())
    
    if not dest_reqs:
        return {
            "success": False,
            "error": f"Visa information not available for {destination}"
        }
    
    nat_code = nationality.upper()[:2]  # US, BR, CN, etc.
    visa_required = dest_reqs.get(nat_code, True)  # Default to required if unknown
    
    return {
        "success": True,
        "destination": destination,
        "nationality": nationality,
        "visa_required": visa_required,
        "message": f"Visa {'required' if visa_required else 'not required'} for {nationality} citizens traveling to {destination}"
    }


@tool(
    name="book_flight",
    description="Book a flight to a destination on a specific date."
)
def book_flight(destination: str, date: str) -> Dict[str, any]:
    """Book a flight (mock)."""
    print(f"✈️  [TOOL] Booking flight to {destination} on {date}")
    
    return {
        "success": True,
        "booking_id": f"FLT-{destination[:3].upper()}-{datetime.now().strftime('%Y%m%d')}",
        "destination": destination,
        "date": date,
        "status": "Confirmed",
        "price": 450.00
    }


# =============================================================================
# 2. CUSTOM STATES (Advanced Feature)
# =============================================================================

class VisaCheckState(HierarchicalState):
    """
    A custom state that automatically checks visa requirements
    when a travel-related query is detected.
    
    This demonstrates how to add business logic BEFORE the normal tool execution.
    """
    
    def handle(self, context: ExecutionContext) -> Optional[HierarchicalState]:
        """
        Intercept travel queries and inject visa reminder into system prompt.
        """
        query_lower = context.user_query.lower()
        
        # Check if this is a travel-related query
        travel_keywords = ["travel", "trip", "visit", "go to", "viajar"]
        is_travel_query = any(keyword in query_lower for keyword in travel_keywords)
        
        if is_travel_query:
            print("🚦 [VisaCheckState] Travel query detected - injecting visa reminder")
            
            # Modify the system instruction to remind about visa
            current_sys = context.get_memory("system_instruction", "")
            if "VISA" not in current_sys:
                new_sys = current_sys + "\n\nIMPORTANT: Always check visa requirements using the check_visa_requirements tool before booking flights."
                context.set_memory("system_instruction", new_sys)
        
        # Continue to normal Router flow
        return self.parent.find_state_by_type("RouterState")


# =============================================================================
# 3. AGENT WRAPPER CLASS (Production Pattern with Custom States)
# =============================================================================

class TravelAgent:
    """
    A complete Travel Agent with custom state integration.
    
    This demonstrates:
    - Full agent setup (like CustomerSupportAgent)
    - Integration of custom states into the execution flow
    - How to modify ALLOWED_TRANSITIONS for custom states
    """
    
    def __init__(self, model: str = "xiaomi/mimo-v2-flash:free"):
        """Initialize the Travel Agent with custom states."""
        print("🤖 Initializing Travel Agent with Custom States...")
        
        # 1. Create Tool Registry
        registry = ToolRegistry()
        
        # 2. Register all tools
        tools_list = [
            check_weather,
            check_visa_requirements,
            book_flight
        ]
        
        for tool_func in tools_list:
            if hasattr(tool_func, '_tool_name'):
                registry.register(
                    name=tool_func._tool_name,
                    description=tool_func._tool_description,
                    function=tool_func,
                    args_model=tool_func._args_model
                )
        
        # 3. Create Executor and LLM
        executor = ToolExecutor(registry)
        llm = LLMClient(model=model)
        
        # 4. Define System Instruction
        system_instruction = """
You are a professional Travel Agent assistant.

Your responsibilities:
1. Help users plan trips and vacations
2. Check weather conditions for destinations
3. Verify visa requirements
4. Book flights when requested

IMPORTANT RULES:
- Always check weather before recommending a destination
- Use tools to get accurate, real-time information
- Be helpful, friendly, and detail-oriented
"""
        
        # 5. Create the HFSM Agent Engine
        self.agent = AgentEngine(
            llm=llm,
            registry=registry,
            executor=executor,
            system_instruction=system_instruction
        )
        
        # 6. CUSTOM STATE INTEGRATION
        # Create and register our custom state
        self.visa_check_state = VisaCheckState(self.agent.root)
        self.agent.register_state("VisaCheckState", self.visa_check_state)
        
        # 7. Modify transition map to inject our custom state
        # We want: Start -> VisaCheckState -> RouterState (instead of Start -> RouterState)
        hfsm_agent.ALLOWED_TRANSITIONS["Start"] = ["VisaCheckState"]
        hfsm_agent.ALLOWED_TRANSITIONS["VisaCheckState"] = ["RouterState"]
        
        print("✅ Travel Agent initialized with custom VisaCheckState!\n")
    
    def run_stream(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> tuple[Generator[str, None, None], ExecutionContext]:
        """
        Process a travel query with custom state handling.
        
        Args:
            query: The user's travel question or request
            chat_history: Previous conversation history (optional)
        
        Returns:
            Tuple of (token_generator, execution_context)
        """
        # We need to start from our custom state instead of RouterState
        # So we'll use the internal _run_loop with our custom starting point
        
        # Setup context (similar to run_stream)
        context = ExecutionContext(user_query=query)
        context.set_memory("system_instruction", self.agent.system_instruction)
        context.set_memory("chat_history", chat_history or [])
        context.set_memory("retry_count", 0)
        context.set_memory("max_retries", 2)
        context.set_memory("pending_tool_calls", [])
        
        # Inject dependencies
        context.set_memory("llm", self.agent.llm)
        context.set_memory("registry", self.agent.registry)
        context.set_memory("executor", self.agent.executor)
        
        # Start from our custom state
        return self.agent._run_loop(self.visa_check_state, context)


# =============================================================================
# 4. DEMO USAGE
# =============================================================================

def run_demo():
    """Demonstrates the Travel Agent with custom states."""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize the agent
    agent = TravelAgent()
    
    # Example queries
    test_queries = [
        "I want to travel to Paris next week. What's the weather like?",
        "Do I need a visa to visit Tokyo? I'm from Brazil.",
        "Book me a flight to London for tomorrow.",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: {query}")
        print('='*70)
        
        # Run the agent
        token_stream, context = agent.run_stream(query)
        
        # Stream the response
        print("\n🤖 Agent Response:")
        full_response = ""
        for token in token_stream:
            print(token, end="", flush=True)
            full_response += token
        
        print("\n")


if __name__ == "__main__":
    run_demo()
