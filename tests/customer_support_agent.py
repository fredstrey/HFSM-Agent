"""
Example: Complete Async Customer Support Agent
==============================================
This example demonstrates how to build a production-ready ASYNC agent using the HFSM framework.
It follows the same pattern as rag_agent_hfsm_async.py but with a different domain.

Key Learning Points:
1. How to create domain-specific async tools
2. How to wrap AsyncAgentEngine in a custom class
3. How to initialize and run an async agent
"""

import sys
import os
import asyncio
from typing import List, Dict, Optional, AsyncGenerator, Any
from datetime import datetime, timedelta
import random

# Add root path to import core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finitestatemachineAgent.hfsm_agent_async import AsyncAgentEngine
from core.context_async import AsyncExecutionContext
from core.decorators import tool
from core.registry import ToolRegistry
from core.executor_async import AsyncToolExecutor
from providers.llm_client_async import AsyncLLMClient
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# 1. DOMAIN-SPECIFIC TOOLS (Mock implementations)
# =============================================================================

# Mock database
ORDERS_DB = {
    "ORD-001": {"customer": "John Doe", "product": "Laptop", "status": "shipped", "date": "2024-12-20"},
    "ORD-002": {"customer": "Jane Smith", "product": "Mouse", "status": "delivered", "date": "2024-12-15"},
    "ORD-003": {"customer": "Bob Wilson", "product": "Keyboard", "status": "processing", "date": "2024-12-28"},
}

@tool(
    name="check_order_status",
    description="Check the current status of a customer order by order ID."
)
async def check_order_status(order_id: str) -> Dict[str, Any]:
    """Retrieves order status from the database (simulates async DB)."""
    print(f"🔍 [TOOL] Checking order: {order_id}")
    await asyncio.sleep(0.5)  # Simulate DB latency
    
    order = ORDERS_DB.get(order_id.upper())
    
    if not order:
        return {
            "success": False,
            "error": f"Order {order_id} not found in system"
        }
    
    return {
        "success": True,
        "order_id": order_id,
        "customer": order["customer"],
        "product": order["product"],
        "status": order["status"],
        "order_date": order["date"]
    }


@tool(
    name="estimate_delivery",
    description="Estimate delivery date for an order based on its current status."
)
async def estimate_delivery(order_id: str) -> Dict[str, Any]:
    """Calculates estimated delivery date."""
    print(f"📦 [TOOL] Estimating delivery for: {order_id}")
    await asyncio.sleep(0.5)
    
    order = ORDERS_DB.get(order_id.upper())
    
    if not order:
        return {"success": False, "error": f"Order {order_id} not found"}
    
    # Mock logic based on status
    status = order["status"]
    if status == "delivered":
        return {"success": True, "status": "delivered", "estimated_delivery": "Already delivered"}
    elif status == "shipped":
        # Estimate 2 days from now
        est_date = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        return {"success": True, "status": "shipped", "estimated_delivery": est_date}
    elif status == "processing":
        # Estimate 5 days from now
        est_date = (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d")
        return {"success": True, "status": "processing", "estimated_delivery": est_date}
    else:
        return {"success": False, "error": f"Unknown status: {status}"}


# =============================================================================
# 2. CUSTOMER SUPPORT AGENT CLASS
# =============================================================================

class CustomerSupportAgent:
    """
    A specialized agent for customer support.
    Wraps the core AsyncAgentEngine with domain-specific tools and prompts.
    """
    
    def __init__(self, model: str = "xiaomi/mimo-v2-flash:free"):
        # 1. Initialize Registry
        self.registry = ToolRegistry()
        
        # 2. Register Tools
        self.register_tools()
        
        # 3. Initialize Compute (LLM & Executor)
        self.llm = AsyncLLMClient(model=model)
        self.executor = AsyncToolExecutor(self.registry)
        
        # 4. Define System Instruction
        system_instruction = """
        You are a helpful Customer Support Agent for an electronics store.
        Your goal is to assist customers with their orders using the available tools.
        
        Guidelines:
        - ALWAYS start by asking for the Order ID if not provided.
        - Use 'check_order_status' to find order details.
        - Use 'estimate_delivery' if the customer asks when it will arrive.
        - Be polite, professional, and concise.
        - If you cannot find an order, apologize and ask them to check the ID.
        """
        
        # 5. Initialize Engine
        self.engine = AsyncAgentEngine(
            llm=self.llm,
            registry=self.registry,
            executor=self.executor,
            system_instruction=system_instruction
            # tool_choice="auto" is default
        )
        
        print("🤖 Customer Support Agent (Async) initialized!")

    def register_tools(self):
        """Register all domain tools."""
        tools = [check_order_status, estimate_delivery]
        for t in tools:
            self.registry.register(
                name=t._tool_name,
                description=t._tool_description,
                function=t,
                args_model=t._args_model
            )

    async def run(self, query: str):
        """Run the agent on a query (non-streaming)."""
        print(f"\n💬 User: {query}")
        print("Thinking...")
        
        response = await self.engine.run(query)
        print(f"🤖 Agent: {response}")
        return response

    async def run_stream(self, query: str):
        """Run the agent with streaming response."""
        print(f"\n💬 User: {query}")
        print("Thinking...", end="", flush=True)
        
        print("\n🤖 Agent: ", end="", flush=True)
        async for token in self.engine.run_stream(query):
            print(token, end="", flush=True)
        print("\n")


# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================

async def main():
    agent = CustomerSupportAgent()
    
    # query 1: simple status check
    await agent.run_stream("Where is my laptop? Order ID is ORD-001")
    
    # query 2: multi-step (check status -> estimate)
    # "When will my keyboard arrive?" -> Needs to find order first (ORD-003)
    # But usually user provides ID. Let's try:
    await agent.run_stream("When will order ORD-003 arrive?")

if __name__ == "__main__":
    asyncio.run(main())
