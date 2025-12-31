from typing import List, Dict, Optional, Generator
from finitestatemachineAgent.hfsm_agent import AgentEngine
from core.context import ExecutionContext

import tools.rag_tools as rag_tools
from tools.rag_schemas import RAGResponse
from core.registry import ToolRegistry
from core.executor import ToolExecutor
from providers.llm_client import LLMClient


class RAGAgentFSMStreaming:
    def __init__(
        self,
        embedding_manager,
        model: str = "google/gemini-2.0-flash-exp:free"
    ):
        rag_tools.initialize_rag_tools(embedding_manager)

        registry = ToolRegistry()
        tools_list = [
            rag_tools.search_documents,
            rag_tools.get_stock_price,
            rag_tools.compare_stocks,
            rag_tools.redirect
        ]

        for tool_func in tools_list:
             if hasattr(tool_func, '_tool_name'):
                registry.register(
                    name=tool_func._tool_name,
                    description=tool_func._tool_description,
                    function=tool_func,
                    args_model=tool_func._args_model
                )

        executor = ToolExecutor(registry)
        llm = LLMClient(model=model)

        system_instruction = """
Você é o Finance.AI, um assistente financeiro especialista.

REGRAS CRITICAS:
1. Para conceitos econômicos, definições e contexto (ex: Selic, Copom, Inflação, PIB), SEMPRE use 'search_documents'. NUNCA use 'redirect' para temas econômicos.
2. Para cotações e performance de ativos (ex: PETR4, NVDA, comparações), SEMPRE use 'get_stock_price' ou 'compare_stocks'.
3. Use 'redirect' APENAS para assuntos totalmente fora de finanças (ex: futebol, receitas, piadas).
"""

        self.agent = AgentEngine(
            llm=llm,
            registry=registry,
            executor=executor,
            system_instruction=system_instruction
        )

    def run_stream(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> tuple[Generator[str, None, None], ExecutionContext]:

        token_stream, context = self.agent.run_stream(query, chat_history=chat_history)

        def wrapped_gen():
            answer = []
            for token in token_stream:
                answer.append(token)
                yield token

            final_answer = "".join(answer)
            self._finalize_response(final_answer, context)

        return wrapped_gen(), context

    def _enhance_query(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]]
    ) -> str:
        # Not used in current implementation but kept for compatibility
        return query

    def _finalize_response(
        self,
        content: str,
        context: ExecutionContext
    ):
        """
        Calculate metrics and store in context memory
        """

        sources_used = []
        scores = []
        has_stock_data = False

        for call in context.tool_calls or []:
            tool_name = call.get("tool_name")
            result = call.get("result", {})

            if tool_name == "search_documents" and isinstance(result, dict):
                for doc in result.get("results", []):
                    meta = doc.get("metadata", {})
                    src = meta.get("source")
                    if src and src not in sources_used:
                        sources_used.append(src)
                    if "score" in doc:
                        scores.append(doc["score"])

            elif tool_name in ("get_stock_price", "compare_stocks"):
                if isinstance(result, dict) and result.get("success"):
                    sources_used.append(f"yfinance:{tool_name}")
                    has_stock_data = True

        confidence = self._calculate_confidence(
            has_stock_data,
            scores,
            sources_used
        )

        # Store in context.memory so API can access it
        context.set_memory("sources_used", sources_used)
        context.set_memory("confidence", confidence)
        context.set_memory("final_answer", content)

    def _calculate_confidence(
        self,
        has_stock_data: bool,
        scores: List[float],
        sources_used: List[str]
    ) -> str:
        if has_stock_data:
            return "high"

        if scores:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)

            if max_score > 0.7 or (avg_score > 0.6 and len(scores) >= 2):
                return "high"
            if avg_score >= 0.5:
                # To be conservative, ensure at least one strong result
                if max_score > 0.6:
                     return "medium"
                return "low"
            return "low"

        return "low" if not sources_used else "medium"
