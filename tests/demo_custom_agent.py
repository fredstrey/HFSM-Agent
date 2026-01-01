"""
Example: Async Travel Agent with Custom States
==============================================
This example demonstrates ADVANCED customization of the Async HFSM framework.
It builds a complete Travel Agent with custom async states that modify execution flow.

Key Learning Points:
1. HOW TO CREATE CUSTOM ASYNC STATES
2. How to integrate them into the AsyncAgentEngine
3. How to modify the state transitions dynamically
"""

import sys
import os
import asyncio
from typing import List, Dict, Optional, Generator, Any
from datetime import datetime, timedelta

# Add root path to import core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finitestatemachineAgent.hfsm_agent_async import AsyncAgentEngine, AsyncHierarchicalState
from core.context_async import AsyncExecutionContext
from core.decorators import tool
from core.registry import ToolRegistry
from core.executor_async import AsyncToolExecutor
from providers.llm_client_async import AsyncLLMClient
from dotenv import load_dotenv

load_dotenv()


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


@tool(name="check_weather", description="Check weather forecast for a destination city.")
async def check_weather(city: str) -> Dict[str, Any]:
    """Get weather information (async)."""
    print(f"🌤️  [TOOL] Checking weather in: {city}")
    await asyncio.sleep(0.2)
    weather = WEATHER_DB.get(city.upper())
    if not weather:
        return {"success": False, "error": f"Weather data not available for {city}"}
    return {"success": True, **weather, "city": city}


@tool(name="check_visa_requirements", description="Check if visa is required.")
async def check_visa_requirements(destination: str, nationality: str) -> Dict[str, Any]:
    """Check visa requirements (async)."""
    print(f"🛂 [TOOL] Checking visa: {nationality} -> {destination}")
    await asyncio.sleep(0.2)
    dest_reqs = VISA_REQUIREMENTS.get(destination.upper())
    if not dest_reqs:
        return {"success": False, "error": f"Visa info not available for {destination}"}
    
    nat_code = nationality.upper()[:2]
    visa_required = dest_reqs.get(nat_code, True)
    
    return {
        "success": True,
        "destination": destination,
        "visa_required": visa_required,
        "message": f"Visa {'REQUIRED' if visa_required else 'NOT required'} for {nationality} -> {destination}"
    }


# =============================================================================
# 2. CUSTOM ASYNC STATE
# =============================================================================

class VisaCheckState(AsyncHierarchicalState):
    """
    Custom state that proactively checks visa requirements if a destination is mentioned.
    This runs BEFORE the RouterState.
    """
    
    async def handle(self, context: AsyncExecutionContext):
        print("🔍 [VisaCheckState] Analyzing query for travel intent...")
        
        query = context.user_query.lower()
        
        # Simple keyword detection (in production, use LLM or specific tool)
        destinations = ["paris", "london", "rio", "tokyo"]
        found_dest = next((city for city in destinations if city in query), None)
        
        if found_dest:
            print(f"   Destination detected: {found_dest.upper()}")
            
            # Check if we already checked visa (avoid infinite loops)
            if await context.get_memory("visa_checked"):
                print("   Visa already checked, proceeding to Router.")
                return self.find_state_by_type("RouterState")
            
            # Inject a system instruction to remind user about visas
            current_sys = await context.get_memory("system_instruction", "")
            new_sys = current_sys + f"\nIMPORTANT: The user is traveling to {found_dest}. ALways check visa requirements first!"
            await context.set_memory("system_instruction", new_sys)
            await context.set_memory("visa_checked", True)
            
            print("   ⚠️  Injected visa reminder into System Prompt.")
        
        # Always transition to RouterState
        return self.find_state_by_type("RouterState")


# =============================================================================
# 3. SETUP & RUN
# =============================================================================

async def main():
    # 1. Setup Registry
    registry = ToolRegistry()
    registry.register_tool(check_weather)
    registry.register_tool(check_visa_requirements)
    
    # 2. Setup Engine components
    llm = AsyncLLMClient(model="xiaomi/mimo-v2-flash:free")
    executor = AsyncToolExecutor(registry)
    
    system_instruction = "You are a travel assistant. Help users plan trips."
    
    engine = AsyncAgentEngine(
        llm=llm,
        registry=registry,
        executor=executor,
        system_instruction=system_instruction
    )
    
    # 3. REGISTER CUSTOM STATE
    # Initialize implementation with parent (engine.root)
    
    visa_state = VisaCheckState(engine.root)
    engine.states["VisaCheckState"] = visa_state
    
    # 4. MODIFY START STATE
    # The engine defaults to starting at 'router_state'.
    engine.router_state = visa_state
    
    print("✅ Custom State 'VisaCheckState' registered and set as entry point.")
    
    # 5. RUN
    print("\nTest Run: Travel to Paris")
    print("🤖 Agent: ", end="", flush=True)
    async for token in engine.run_stream("I want to go to Paris. I am Brazilian."):
        print(token, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    asyncio.run(main())
