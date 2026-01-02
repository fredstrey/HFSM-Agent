"""
Finance.AI Agent
=================

Specialized financial market assistant using the high-level Agent API.

This demonstrates how to create a domain-specific agent with:
- Custom persona (Finance.AI)
- Domain-specific tools (stock prices, comparison, document search)
- Parallel execution for complex queries
- Intent analysis for efficient routing
"""

import os
import sys
from datetime import datetime
from typing import AsyncIterator

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finitestatemachineAgent import Agent
from agents import finance_ai_utils, finance_ai_tools

class FinanceAI:
    """
    Finance.AI - Specialized Financial Market Assistant
    
    Built on top of the HFSM Agent framework with:
    - Real-time stock data
    - Document search for economic concepts
    - Parallel research capabilities
    - Intent-based routing
    """
    
    def __init__(
        self,
        llm_provider: str = "openrouter",
        model: str = "xiaomi/mimo-v2-flash:free",
        api_key: str = None
    ):
        """
        Initialize Finance.AI agent.
        
        Args:
            llm_provider: LLM provider (default: openrouter)
            model: Model to use
            api_key: API key (defaults to OPENROUTER_API_KEY env var)
        """
        # Define Finance.AI Persona
        self.persona = """Você é o **Finance.AI**, um assistente especializado em Mercado Financeiro.

OBJETIVO:
Fornecer análises precisas, dados de mercado e explicações educativas sobre economia e investimentos.

DIRETRIZES:
1. **Especialidade**: Foco total em finanças (Ações, FIIs, Selic, IPCA, Economia Global).
2. **Ferramentas**: Você tem acesso a dados em tempo real. USE-OS. Não invente dados.
3. **Tom**: Profissional, analítico e direto.
4. **Limites**: Se o usuário perguntar sobre temas não-financeiros (esportes, culinária), redirecione para finanças ou encerre educadamente.

Seja útil para o investidor."""

        # Define Redirect Prompt (for simple queries)
        self.redirect_prompt = f"""Você é o **Finance.AI**, um assistente especializado em Mercado Financeiro.
Data de hoje: {datetime.now().strftime('%d/%m/%Y')}

DIRETRIZES DE RESPOSTA RÁPIDA:
1. **Saudações**: Responda cordialmente e ofereça ajuda com análises de ações (ex: PETR4, VALE3), conceitos econômicos (Selic, IPCA) ou notícias.
2. **Fora de Escopo**: Se a pergunta não for sobre finanças, explique educadamente que seu foco é o mercado financeiro.
3. Seja breve e não invente dados."""

        # Initialize Agent with Finance.AI configuration
        self.agent = Agent(
            # LLM Config
            llm_provider=llm_provider,
            model=model,
            api_key=api_key,
            
            # Persona
            system_instruction=self.persona,
            redirect_prompt=self.redirect_prompt,
            
            # Finance Tools
            tools=[
                finance_ai_tools.get_stock_price,
                finance_ai_tools.compare_stocks,
                finance_ai_tools.search_documents
            ],
            
            # Features

            # enable forks for parallel research
            enable_parallel_planning=True,

            # enable planning, intent analysis and creates ToDo list for tasks
            enable_intent_analysis=True,

            # maximum number of parallel branches when using forks
            max_parallel_branches=3,
            
            # Safety
            max_global_requests=50,

            # after tool is used, goes to validation state to validate the result
            skip_validation=False,
            
            # Finance.AI-specific customizations

            # custom tools validations
            # if no function is defined, it will use llm to validates as default
            validation_fn=finance_ai_utils.tools_validation,
            
            # custom planning system prompt
            # you can use this to override the basic prompt for parallel planning using your own prompt
            planning_system_prompt=finance_ai_utils.enhance_rag_planning_prompt,
            
            # custom post router hook, forces tool usage after validation node
            # you can use this after router decision, to intercept and alter router decision (ex: enforce tool usage)     
            post_router_hook=finance_ai_utils.enforce_tool_usage,

            contract_strategy="epistemic",   # contract strategy for forks (simple, epistemic, your-custom-strategy)
            synthesis_strategy="llm", # synthesis strategy for forks (llm, concat, your-custom-strategy)

        )
    
    async def run(self, query: str, chat_history: list = None):
        """
        Execute a query and return the complete response.
        
        Args:
            query: User query
            chat_history: Optional chat history
            
        Returns:
            AgentResponse with content and metadata
        """
        return await self.agent.run(query, chat_history)
    
    async def stream(self, query: str, chat_history: list = None) -> AsyncIterator[str]:
        """
        Stream response tokens as they are generated.
        
        Args:
            query: User query
            chat_history: Optional chat history
            
        Yields:
            Response tokens
        """
        async for token in self.agent.stream(query, chat_history):
            yield token
        
        # Extract metadata after streaming completes
        if hasattr(self.agent, 'last_context'):
            await finance_ai_utils.extract_metadata(self.agent.last_context)