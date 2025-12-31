"""
Example: Complete Customer Support Agent
==========================================
This example demonstrates how to build a production-ready agent using the HFSM framework.
It follows the same pattern as rag_agent_hfsm.py but with a different domain (customer support).

Key Learning Points:
1. How to create domain-specific tools with the @tool decorator
2. How to wrap AgentEngine in a custom class for your use case
3. How to initialize tools and register them properly
4. How to customize the system instruction for your domain
"""

import sys
import os
from typing import List, Dict, Optional, Generator
from datetime import datetime, timedelta
import random

# Add root path to import core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finitestatemachineAgent.hfsm_agent import AgentEngine
from core.context import ExecutionContext
from core.decorators import tool
from core.registry import ToolRegistry
from core.executor import ToolExecutor
from providers.llm_client import LLMClient


# =============================================================================
# 1. DOMAIN-SPECIFIC TOOLS (Mock implementations for learning)
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
def check_order_status(order_id: str) -> Dict[str, any]:
    """
    Retrieves order status from the database.
    
    Args:
        order_id: The order ID (e.g., ORD-001)
    
    Returns:
        Dictionary with order details or error message
    """
    print(f"🔍 [TOOL] Checking order: {order_id}")
    
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
def estimate_delivery(order_id: str) -> Dict[str, any]:
    """
    Calculates estimated delivery date.
    
    Args:
        order_id: The order ID
    
    Returns:
        Dictionary with delivery estimate
    """
    print(f"📦 [TOOL] Estimating delivery for: {order_id}")
    
    order = ORDERS_DB.get(order_id.upper())
    
    if not order:
        return {
            "success": False,
            "error": f"Order {order_id} not found"
        }
    
    # Mock logic based on status
    status = order["status"]
    today = datetime.now()
    
    if status == "delivered":
        estimate = "Already delivered"
    elif status == "shipped":
        estimate = (today + timedelta(days=2)).strftime("%Y-%m-%d")
    elif status == "processing":
        estimate = (today + timedelta(days=5)).strftime("%Y-%m-%d")
    else:
        estimate = "Unknown"
    
    return {
        "success": True,
        "order_id": order_id,
        "current_status": status,
        "estimated_delivery": estimate
    }


@tool(
    name="search_faq",
    description="Search the FAQ database for answers to common questions."
)
def search_faq(query: str) -> Dict[str, any]:
    """
    Searches FAQ database (mock implementation).
    
    Args:
        query: The user's question
    
    Returns:
        Dictionary with FAQ results
    """
    print(f"📚 [TOOL] Searching FAQ for: '{query}'")
    
    # Mock FAQ database
    faqs = {
        "return": "You can return items within 30 days of purchase. Contact support@example.com to initiate a return.",
        "shipping": "Standard shipping takes 5-7 business days. Express shipping is 2-3 days.",
        "payment": "We accept credit cards, PayPal, and bank transfers.",
        "warranty": "All products come with a 1-year manufacturer warranty."
    }
    
    # Simple keyword matching
    query_lower = query.lower()
    for keyword, answer in faqs.items():
        if keyword in query_lower:
            return {
                "success": True,
                "question": query,
                "answer": answer,
                "source": "FAQ Database"
            }
    
    return {
        "success": False,
        "message": "No FAQ found for this question. Please contact human support."
    }


# =============================================================================
# 2. AGENT WRAPPER CLASS (Production Pattern)
# =============================================================================

class CustomerSupportAgent:
    """
    A complete customer support agent built on the HFSM framework.
    
    This class demonstrates the production pattern:
    - Encapsulates tool initialization
    - Wraps AgentEngine with domain-specific configuration
    - Provides a clean interface for external use (API, CLI, etc.)
    """
    
    def __init__(self, model: str = "xiaomi/mimo-v2-flash:free"):
        """
        Initialize the Customer Support Agent.
        
        Args:
            model: LLM model to use (default: free tier for demo)
        """
        print("🤖 Initializing Customer Support Agent...")
        
        # 1. Create Tool Registry
        registry = ToolRegistry()
        
        # 2. Register all tools
        tools_list = [
            check_order_status,
            estimate_delivery,
            search_faq
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
        
        # 4. Define System Instruction (domain-specific)
        system_instruction = """
You are a helpful Customer Support Agent for an e-commerce company.

Your responsibilities:
1. Help customers track their orders
2. Answer questions about shipping and delivery
3. Provide information from the FAQ database
4. Be polite, professional, and concise

IMPORTANT RULES:
- Always use tools to get accurate information (don't make up order details)
- If you can't find information, suggest contacting human support
- Be empathetic and understanding with customer concerns
"""
        
        # 5. Create the HFSM Agent Engine
        self.agent = AgentEngine(
            llm=llm,
            registry=registry,
            executor=executor,
            system_instruction=system_instruction
        )
        
        print("✅ Agent initialized successfully!\n")
    
    def run_stream(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> tuple[Generator[str, None, None], ExecutionContext]:
        """
        Process a customer query and stream the response.
        
        Args:
            query: The customer's question or request
            chat_history: Previous conversation history (optional)
        
        Returns:
            Tuple of (token_generator, execution_context)
        """
        # Delegate to the underlying AgentEngine
        token_stream, context = self.agent.run_stream(query, chat_history=chat_history)
        
        return token_stream, context


# =============================================================================
# 3. DEMO USAGE (How to use the agent)
# =============================================================================

def run_demo():
    """
    Demonstrates how to use the CustomerSupportAgent.
    This is what you'd call from your API, CLI, or other interface.
    """
    # Initialize the agent
    agent = CustomerSupportAgent()
    
    # Example queries
    test_queries = [
        "What's the status of order ORD-001?",
        "When will my order ORD-003 arrive?",
        "How do I return an item?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print('='*60)
        
        # Run the agent
        token_stream, context = agent.run_stream(query)
        
        # Stream the response
        print("\n🤖 Agent Response:")
        full_response = ""
        for token in token_stream:
            print(token, end="", flush=True)
            full_response += token
        
        print("\n")
        
        # You can access context for debugging/logging
        # print(f"Debug - Tool calls made: {len(context.tool_calls)}")


if __name__ == "__main__":
    # Make sure to load environment variables if needed
    from dotenv import load_dotenv
    load_dotenv()
    
    run_demo()
