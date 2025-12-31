"""
RAG Agent v2 - Using Generic ToolCallingAgent Base Class

This is a refactored version of the RAG Agent that inherits from
the generic ToolCallingAgent, providing cleaner architecture and
better separation of concerns.
"""
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

from core import ToolCallingAgent, AgentResponse
from core import ExecutionContext
import tools.rag_tools as rag_tools
from .context_agent import ContextAgent
from .validation_agent import ValidationAgent
from .react_agent import ReActAgent, ReActDecision

class RAGResponse(BaseModel):
    """Custom response model for RAG Agent"""
    answer: str
    sources_used: List[str] = Field(default_factory=list)
    confidence: str = "high"
    is_out_of_scope: bool = False


class RAGAgentV2(ToolCallingAgent):
    """
    RAG Agent specialized in Finance and Economics
    
    Features:
    - Document search with similarity threshold
    - Stock price lookup and comparison
    - Out-of-scope detection
    - Chat history support
    - Auto-query substitution
    
    Example:
        ```python
        agent = RAGAgentV2(
            embedding_manager=embedding_manager,
            tool_caller_model="functiongemma:270m"
        )
        
        response, context = agent.run(
            query="O que é a taxa Selic?",
            chat_history=[...]
        )
        ```
    """
    
    def __init__(
        self,
        embedding_manager,
        tool_caller_model: str = "xiaomi/mimo-v2-flash:free",
        response_model: str = "xiaomi/mimo-v2-flash:free",
        context_model: str = "xiaomi/mimo-v2-flash:free",
        max_iterations: int = 3
    ):
        """
        Initialize RAG Agent
        
        Args:
            embedding_manager: EmbeddingManager instance for document search
            tool_caller_model: Model for tool calling (default: xiaomi/mimo-v2-flash:free)
            response_model: Model for response generation (default: xiaomi/mimo-v2-flash:free)
            context_model: Model for intent extraction (default: xiaomi/mimo-v2-flash:free)
            max_iterations: Maximum tool calling iterations (default: 3)
        """
        # Initialize RAG tools
        rag_tools.initialize_rag_tools(embedding_manager)
        
        # Get tools from registry
        tools = [
            rag_tools.search_documents,
            rag_tools.get_stock_price,
            rag_tools.compare_stocks,
            rag_tools.redirect
        ]
        
        # Call parent constructor
        super().__init__(
            tools=tools,
            tool_caller_model=tool_caller_model,
            response_model=response_model,
            response_class=RAGResponse,
            system_prompt=self._build_rag_system_prompt(),
            max_iterations=max_iterations
        )
        
        self.embedding_manager = embedding_manager
        
        # Initialize Context Agent for intent extraction
        self.context_agent = ContextAgent(model=context_model)
        print(f"✅ ContextAgent inicializado com modelo: {context_model}")
        
        # Initialize Validation Agent for response validation
        self.validation_agent = ValidationAgent(model=context_model)
        print(f"✅ ValidationAgent inicializado com modelo: {context_model}")
        
        # Initialize ReAct Agent for reasoning loop
        self.react_agent = ReActAgent(model=context_model)
        self.max_react_iterations = max_iterations
        print(f"✅ ReActAgent inicializado com max_iterations: {max_iterations}")
    
    def run(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        context: Optional[ExecutionContext] = None
    ) -> Tuple[RAGResponse, ExecutionContext]:
        """
        Execute RAG agent with ReAct reasoning loop
        
        Args:
            query: User query
            chat_history: Optional chat history
            context: Optional existing ExecutionContext
            
        Returns:
            Tuple of (RAGResponse, ExecutionContext)
        """
        # Store original query
        original_query = query
        current_query = query
        
        # Initialize context if not provided
        if context is None:
            context = ExecutionContext(user_query=original_query)
        
        # Accumulated context from all iterations
        accumulated_context: List[str] = []
        accumulated_answers: List[str] = []  # Store answers from each iteration
        all_sources: List[str] = []
        
        # ReAct loop - max 3 iterations
        for iteration in range(1, self.max_react_iterations + 1):
            print(f"\n{'='*70}")
            print(f"🔄 [RAG Agent] Iteração {iteration}/{self.max_react_iterations}")
            print(f"{'='*70}")
            
            # Extract intent using ContextAgent
            print(f"\n🔍 Query para esta iteração: '{current_query}'")
            context_response = self.context_agent.extract_intent(
                current_query=current_query,
                chat_history=chat_history or []
            )
            print(f"🎯 ContextAgent resposta: '{context_response}'")
            
            # Parse tool suggestion and intent from ContextAgent
            if ":" in context_response:
                parts = context_response.split(":", 1)
                suggested_tool = parts[0].strip().lower()
                extracted_intent = parts[1].strip()
                
                if not extracted_intent or extracted_intent.lower() in ['search_documents', 'get_stock_price', 'compare_stocks', 'redirect']:
                    extracted_intent = current_query
                    print(f"   Tool sugerida: {suggested_tool}")
                    print(f"   Intent: usando query atual '{current_query}'")
                else:
                    print(f"   Tool sugerida: {suggested_tool}")
                    print(f"   Intent extraído: {extracted_intent}")
            else:
                extracted_intent = context_response
                suggested_tool = None
                print(f"   Intent extraído: {extracted_intent}")
            
            # Use extracted intent for tool calling
            query_for_tools = extracted_intent
            
            # Call parent run to execute tools
            print(f"\n🔧 Executando ferramentas...")
            response, iteration_context = super().run(
                query=query_for_tools,
                chat_history=chat_history,
                context=ExecutionContext(user_query=current_query)  # Fresh context for each iteration
            )
            
            # Accumulate context from this iteration
            if iteration_context.retrieved_documents:
                for doc in iteration_context.retrieved_documents:
                    content = doc.get("content", "")
                    if content and content not in accumulated_context:
                        accumulated_context.append(content)
            
            # Accumulate sources
            for source in response.sources_used:
                if source not in all_sources:
                    all_sources.append(source)
            
            # Accumulate answer from this iteration
            if response.answer and not response.is_out_of_scope:
                accumulated_answers.append(response.answer)
                print(f"   📝 Resposta acumulada da iteração {iteration}")
            
            # Prepare tool results for ReAct analysis
            tool_results = {
                "tool_name": "multiple_tools",
                "result": {
                    "current_answer": response.answer,
                    "sources": response.sources_used,
                    "confidence": response.confidence,
                    "is_out_of_scope": response.is_out_of_scope,
                    "retrieved_documents": len(iteration_context.retrieved_documents) if iteration_context.retrieved_documents else 0,
                    "accumulated_answers": accumulated_answers,  # Pass all previous answers
                    "iteration": iteration
                }
            }
            
            # ReAct: Analyze and decide
            print(f"\n🧠 Analisando resultados com ReActAgent...")
            react_analysis = self.react_agent.analyze_and_decide(
                original_query=original_query,
                current_query=current_query,
                tool_results=tool_results,
                accumulated_context=accumulated_context,
                iteration=iteration,
                max_iterations=self.max_react_iterations
            )
            
            # Act on ReAct decision
            if react_analysis.decision == ReActDecision.CONTINUE:
                print(f"\n✅ ReAct decidiu CONTINUAR - informação suficiente")
                context.retrieved_documents = iteration_context.retrieved_documents
                context.has_context = True
                break
            
            elif react_analysis.decision == ReActDecision.RETRY_WITH_REFINEMENT:
                if iteration < self.max_react_iterations:
                    print(f"\n🔄 ReAct decidiu REFINAR - tentando novamente")
                    if react_analysis.refined_query:
                        current_query = react_analysis.refined_query
                        print(f"   Nova query: '{current_query}'")
                    else:
                        print(f"   ⚠️ Sem query refinada, usando query atual")
                else:
                    print(f"\n⚠️ Limite de iterações atingido - usando dados acumulados")
                    context.retrieved_documents = iteration_context.retrieved_documents
                    context.has_context = True
                    break
            
            elif react_analysis.decision == ReActDecision.CALL_DIFFERENT_TOOL:
                if iteration < self.max_react_iterations:
                    print(f"\n🔧 ReAct sugeriu ferramenta diferente")
                    if react_analysis.suggested_tool:
                        print(f"   Ferramenta sugerida: {react_analysis.suggested_tool}")
                    # Use refined query if provided for NEXT iteration
                    if react_analysis.refined_query:
                        current_query = react_analysis.refined_query
                        print(f"   📝 Próxima iteração usará query: '{current_query}'")
                    # Continue to next iteration with new query
                else:
                    print(f"\n⚠️ Limite de iterações atingido - usando dados acumulados")
                    context.retrieved_documents = iteration_context.retrieved_documents
                    context.has_context = True
                    break
            
            elif react_analysis.decision == ReActDecision.INSUFFICIENT_DATA:
                print(f"\n❌ ReAct decidiu que dados são insuficientes")
                context.retrieved_documents = iteration_context.retrieved_documents
                context.has_context = True
                context.is_out_of_scope = response.is_out_of_scope
                break
        
        # After ReAct loop, regenerate response with accumulated data
        if len(accumulated_answers) > 1:
            # Multiple iterations with different answers - synthesize them intelligently
            print(f"\n📝 Sintetizando resposta final de {len(accumulated_answers)} iterações")
            
            # Build synthesis prompt
            answers_text = "\n\n".join([f"Resposta {i+1}: {ans}" for i, ans in enumerate(accumulated_answers)])
            synthesis_prompt = f"""Você recebeu múltiplas respostas parciais para a pergunta do usuário. Sintetize-as em UMA resposta concisa e completa, SEM REDUNDÂNCIA.

PERGUNTA ORIGINAL: {original_query}

RESPOSTAS PARCIAIS:
{answers_text}

INSTRUÇÕES:
- Combine as informações de todas as respostas em UMA resposta coesa
- NÃO repita a mesma informação múltiplas vezes
- Seja direto e objetivo
- Mantenha todas as informações relevantes
- Use formatação clara (bullets, números, etc. se apropriado)

RESPOSTA FINAL SINTETIZADA:"""
            
            synthesized_answer = self.response_generator.chat([
                {"role": "user", "content": synthesis_prompt}
            ])
            
            response.answer = synthesized_answer
            response.sources_used = all_sources
        elif len(accumulated_context) > 1 and len(accumulated_answers) == 1:
            # Multiple iterations but only one answer - regenerate with all context
            print(f"\n📝 Gerando resposta final com contexto acumulado de {iteration} iterações")
            final_answer = self._generate_response(original_query, accumulated_context)
            response.answer = final_answer
            response.sources_used = all_sources
        
        # Update context with accumulated data
        context.has_context = len(accumulated_context) > 0
        
        # Validate response (only for search_documents, not stock tools)
        is_stock_tool = any("yfinance" in src for src in response.sources_used)
        
        if not is_stock_tool and not response.is_out_of_scope:
            print("\n🔍 Validando resposta final...")
            is_valid = self.validation_agent.validate(
                query=original_query,
                response=response.answer
            )
            
            if not is_valid:
                print("❌ Resposta reprovada - usando mensagem padrão")
                response.answer = self.validation_agent.get_default_rejection_message()
                response.is_out_of_scope = True
                response.confidence = "low"
        
        # Debug: Print final response
        print(f"\n📤 [DEBUG] Resposta final do RAG Agent:")
        print(f"   Iterações executadas: {iteration}")
        print(f"   Answer: {response.answer[:200]}..." if len(response.answer) > 200 else f"   Answer: {response.answer}")
        print(f"   Sources: {response.sources_used}")
        print(f"   Confidence: {response.confidence}")
        print(f"   Out of scope: {response.is_out_of_scope}")
        
        return response, context
    
    def _build_rag_system_prompt(self) -> str:
        """Build RAG-specific system prompt"""
        return """VOCÊ É UM ASSISTENTE FINANCEIRO. SEMPRE USE UMA TOOL.

⚠️ REGRA CRÍTICA: CHAME APENAS **UMA TOOL POR VEZ**
- NÃO chame múltiplas tools simultaneamente
- Chame UMA tool, analise o resultado, depois decida se precisa de outra
- Se a pergunta tem múltiplas partes, responda UMA PARTE por vez

TOOLS DISPONÍVEIS:
1. search_documents - Conceitos financeiros, economia, BACEN, taxas, inflação
2. get_stock_price - Preço de UMA ação específica
3. compare_stocks - COMPARAR DUAS OU MAIS AÇÕES (use esta para comparações!)
4. redirect - Assuntos não financeiros

EXEMPLOS:

"O que é Selic?"
→ search_documents({"query": "O que é taxa Selic?"})

"Preço da TSLA"
→ get_stock_price({"ticker": "TSLA"})

"Compare TSLA e NVDA"
→ compare_stocks({"tickers": ["TSLA", "NVDA"]})

"Qual a melhor entre AAPL e MSFT?"
→ compare_stocks({"tickers": ["AAPL", "MSFT"]})

"Receita de bolo"
→ redirect()

"Preço da AAPL e quem define Selic?" (MÚLTIPLAS PARTES)
→ PRIMEIRA CHAMADA: get_stock_price({"ticker": "AAPL"})
→ (aguardar resultado, depois em nova iteração chamar search_documents)

REGRAS:
- SEMPRE chame UMA tool por vez
- NUNCA responda diretamente
- Para comparar ações, use compare_stocks
- Para uma ação só, use get_stock_price
- Se pergunta tem múltiplas partes, responda uma parte por vez"""
    
    def _process_tool_result(
        self,
        tool_name: str,
        result: Any,
        context: ExecutionContext,
        retrieved_context: List[str],
        sources_used: List[str]
    ) -> Dict[str, Any]:
        """
        Process RAG tool results
        
        Handles:
        - Document search results
        - Stock price data
        - Out-of-scope detection
        """
        print(f"✅ Tool executada: {tool_name}")
        
        # Get actual result from wrapper
        tool_result = result.get("result", {})
        
        # Handle search_documents
        if tool_name == "search_documents":
            # Check if out of scope
            if tool_result.get("out_of_scope", False):
                print("⚠️  Pergunta fora do escopo")
                context.is_out_of_scope = True
                context.has_context = True
                retrieved_context.append("OUT_OF_SCOPE_FINANCE")
                return {"should_break": True}
            
            # Process search results
            results = tool_result.get("results", [])
            total_found = tool_result.get("total_found", 0)
            
            print(f"   Documentos encontrados: {total_found}")
            
            if total_found > 0:
                # Add documents to context
                for doc in results:
                    content = doc.get("content", "")
                    score = doc.get("score", 0.0)
                    doc_metadata = doc.get("metadata", {})
                    
                    if content:
                        retrieved_context.append(content)
                        context.add_document(
                            content=content,
                            score=score,
                            metadata=doc_metadata
                        )
                    
                    # Track sources
                    source = doc_metadata.get("source", "")
                    if source and source not in sources_used:
                        sources_used.append(source)
                
                context.has_context = True
            else:
                # No documents found - mark as out of scope
                print("⚠️  Nenhum documento relevante")
                context.is_out_of_scope = True
                context.has_context = True
                retrieved_context.append("OUT_OF_SCOPE_FINANCE")
            
            return {"should_break": True}
        
        # Handle stock tools
        elif tool_name in ["get_stock_price", "compare_stocks"]:
            # Check if tool was successful
            if not tool_result.get("success", True):
                # Tool failed - use error message
                error_message = tool_result.get("message", "Não consegui obter dados da ação. Verifique se o ticker está correto.")
                print(f"   ❌ Erro na ferramenta: {error_message}")
                retrieved_context.append(error_message)
                context.has_context = True
                return {"should_break": True}
            
            print("   Dados financeiros obtidos")
            
            # Add result to context
            retrieved_context.append(str(tool_result))
            sources_used.append(f"yfinance:{tool_name}")
            context.has_context = True
            
            return {"should_break": True}
        
        # Handle redirect
        elif tool_name == "redirect":
            print("⚠️  Pergunta redirecionada (fora do escopo)")
            context.is_out_of_scope = True
            context.has_context = True
            retrieved_context.append("OUT_OF_SCOPE_FINANCE")
            return {"should_break": True}
        
        return {"should_break": False}
    
    def _build_response(
        self,
        answer: str,
        sources_used: List[str],
        context: ExecutionContext
    ) -> RAGResponse:
        """
        Build RAG-specific response with dynamic confidence
        
        Confidence calculation:
        - high: similarity > 0.7 OR multiple sources OR stock data
        - medium: similarity 0.5-0.7 OR single source
        - low: similarity < 0.5 OR out of scope
        """
        # Calculate confidence
        confidence = self._calculate_confidence(context, sources_used)
        
        return RAGResponse(
            answer=answer,
            sources_used=sources_used,
            confidence=confidence,
            is_out_of_scope=context.is_out_of_scope
        )
    
    def _calculate_confidence(
        self,
        context: ExecutionContext,
        sources_used: List[str]
    ) -> str:
        """
        Calculate confidence level based on context
        
        Args:
            context: Execution context with retrieved documents
            sources_used: List of sources
            
        Returns:
            Confidence level: "low", "medium", or "high"
        """
        # Out of scope = low confidence
        if context.is_out_of_scope:
            return "low"
        
        # No documents = low confidence
        if not context.retrieved_documents:
            # Unless it's stock data (yfinance)
            if any("yfinance" in src for src in sources_used):
                return "high"
            return "low"
        
        # Calculate average similarity score
        scores = [
            doc.get("score", 0)
            for doc in context.retrieved_documents
            if "score" in doc
        ]
        
        if scores:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            
            # High confidence: high similarity or multiple good sources
            if max_score > 0.7 or (avg_score > 0.6 and len(scores) >= 2):
                return "high"
            
            # Medium confidence: moderate similarity
            elif avg_score >= 0.5:
                return "medium"
            
            # Low confidence: low similarity
            else:
                return "low"
        
        # Default to medium if no scores but has sources
        if sources_used:
            return "medium"
        
        return "low"
    
    def _generate_response(self, query: str, context: List[str]) -> str:
        """
        Generate RAG response with out-of-scope handling
        """
        # Check if out of scope
        if context and "OUT_OF_SCOPE_FINANCE" in context:
            return """Desculpe, sou um assistente especializado em finanças e mercado financeiro. 

Posso ajudar com:
📈 Análise de preços de ações
📊 Comparação de desempenho de ativos
💰 Conceitos e teorias de mercado financeiro
📚 Informações sobre investimentos e economia

Por favor, faça uma pergunta relacionada a finanças e terei prazer em ajudar!"""
        
        # Normal RAG response
        if not context:
            context_text = "Nenhum contexto disponível."
        else:
            context_text = "\n\n".join(context)
        
        prompt = f"""Com base no contexto abaixo, responda a pergunta do usuário de forma clara e objetiva.

CONTEXTO:
{context_text}

PERGUNTA: {query}

RESPOSTA (seja direto e use informações do contexto):"""
        
        response = self.response_generator.chat([
            {"role": "user", "content": prompt}
        ])
        
        return response
