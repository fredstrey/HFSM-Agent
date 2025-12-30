"""
Agente RAG com tool calling usando FunctionGemma
"""
import json
from typing import List, Dict, Any, Optional

from providers.function_caller import FunctionCallerProvider
from providers.ollama import OllamaProvider
from tools.rag_schemas import RAGResponse
from tools import rag_tools
from embedding_manager.embedding_manager import EmbeddingManager
from core.registry import ToolRegistry
from core.executor import ToolExecutor
from core.execution_context import ExecutionContext


class RAGAgent:
    """Agente RAG que usa FunctionGemma para tool calling e gemma3 para respostas"""
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        tool_caller_model: str = "gemma3:1b",
        response_model: str = "gemma3:1b",
        system_prompt: Optional[str] = None,
        max_iterations: int = 3,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ):
        """
        Inicializa agente RAG
        
        Args:
            embedding_manager: Gerenciador de embeddings
            tool_caller_model: Modelo para tool calling (FunctionGemma)
            response_model: Modelo para gerar respostas (gemma3:1b)
            system_prompt: Prompt do sistema (opcional)
            max_iterations: Máximo de iterações
            chat_history: Histórico de chat (opcional)
        """
        # Provider para tool calling
        self.tool_caller = FunctionCallerProvider(
            model=tool_caller_model,
            temperature=0.1
        )
        
        # Provider para respostas finais
        self.response_generator = OllamaProvider(
            model=response_model,
            temperature=0.3
        )
        
        # Inicializa RAG tools com embedding manager
        rag_tools.initialize_rag_tools(embedding_manager)
        
        # Registry e Executor
        self.registry = ToolRegistry()
        self.executor = ToolExecutor(self.registry)
        
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_iterations = max_iterations
        self.conversation_history: List[Dict[str, str]] = chat_history or []
    
    def run(self, query: str) -> RAGResponse:
        """
        Executa o agente RAG
        
        Args:
            query: Pergunta do usuário
            
        Returns:
            RAGResponse com resposta e metadados
        """
        print(f"\n🤖 RAG Agent processando: {query}")
        print("=" * 70)
        
        # Cria contexto de execução
        context = ExecutionContext(
            user_query=query,
            max_iterations=self.max_iterations,
            chat_history=self.conversation_history
        )
        
        # Inicializa contexto e histórico (mantido para compatibilidade)
        retrieved_context = []
        sources_used = []
        
        # Mensagens para tool calling
        # Constrói prompt do sistema com histórico se existir
        system_content = self._build_tool_calling_prompt()
        
        # Adiciona histórico ao prompt do sistema se existir
        if self.conversation_history:
            print(f"📜 [DEBUG] Adicionando {len(self.conversation_history)} mensagens ao contexto do prompt")
            
            # Para modelos pequenos, usar apenas última interação
            last_user_msg = None
            last_assistant_msg = None
            
            # Pega última interação (últimas 2 mensagens)
            if len(self.conversation_history) >= 2:
                last_user_msg = self.conversation_history[-2]
                last_assistant_msg = self.conversation_history[-1]
            
            if last_user_msg and last_assistant_msg:
                history_text = f"""

📜 CONTEXTO DA CONVERSA ANTERIOR:
Pergunta anterior: "{last_user_msg['content']}"
Resposta anterior: "{last_assistant_msg['content']}"

⚠️ REGRA IMPORTANTE:
Se a pergunta atual se refere à pergunta anterior, use a MESMA tool que seria usada para a pergunta anterior.
Exemplos:
- Anterior: "preço da TSLA" → Atual: "E da AAPL?" → Use: get_stock_price
- Anterior: "taxa Selic" → Atual: "quem define?" → Use: search_documents

Pergunta ATUAL do usuário:"""
                system_content += history_text
        else:
            print("📜 [DEBUG] Nenhum histórico para adicionar")
        
        tool_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]
        
        # Loop de tool calling
        for iteration in range(1, self.max_iterations + 1):
            context.current_iteration = iteration
            print(f"\n🔄 Iteração {iteration}/{self.max_iterations}")
            print(f"📋 Query original: {context.user_query}")
            # Chama FunctionGemma para decidir qual tool usar
            response = self.tool_caller.chat(
                messages=tool_messages,
                tools=self.registry.to_openai_format()
            )
            
            print(f"💬 FunctionGemma: {response.get('content', '')[:150]}...")
            
            # Verifica se há tool call
            tool_call = self.tool_caller.parse_tool_call(response)
            
            if tool_call:
                tool_name = tool_call.get('name')
                print(f"🔧 Tool call detectada: {tool_name}")
                
                # Executa tool
                try:
                    # Extrai nome e argumentos da tool
                    tool_name = tool_call.get("name")
                    arguments = tool_call.get("arguments", {})
                    
                    print(f"🔧 Tool call detectada: {tool_name}")
                    
                    # Injeta user_query no contexto se for search_documents
                    if tool_name == "search_documents" and "query" in arguments:
                        # Preserva a query original do usuário
                        original_query = arguments.get("query", "")
                        print(f"   Query da tool: '{original_query}'")
                        print(f"   Query do usuário: '{context.user_query}'")
                        # Usa a query original se a tool inventou algo diferente
                        if original_query != context.user_query:
                            print(f"   ⚠️  Substituindo query inventada pela original")
                            arguments["query"] = context.user_query
                    
                    # Registra tool call no contexto
                    context.add_tool_call(tool_name, arguments)
                    
                    # Executa tool
                    result = self.executor.execute(tool_name, arguments)
                    
                    print(f"✅ Tool executada com sucesso")
                    
                    # Extrai o resultado real da tool (executor envolve em {"success": True, "result": {...}})
                    tool_result = result.get("result", {})
                    
                    # Se for search_documents, extrai contexto
                    if tool_name == "search_documents" and result.get("success"):
                        # Verifica se retornou 0 documentos (fora do escopo)
                        if tool_result.get("total_found", 0) == 0:
                            print("⚠️  Nenhum documento relevante - marcando como fora do escopo")
                            context.mark_out_of_scope()
                            retrieved_context.append("OUT_OF_SCOPE_FINANCE")
                        else:
                            for doc in tool_result.get("results", []):
                                # Adiciona ao contexto
                                context.add_document(
                                    content=doc["content"],
                                    score=doc.get("score", 0),
                                    metadata=doc.get("metadata", {})
                                )
                                # Mantém compatibilidade com código legado
                                retrieved_context.append(doc["content"])
                                if "metadata" in doc and "source" in doc["metadata"]:
                                    sources_used.append(doc["metadata"]["source"])
                            print(f"   Documentos encontrados: {tool_result.get('total_found', 0)}")
                    
                    # Se for redirect, marca como fora do escopo
                    elif tool_name == "redirect" and result.get("success"):
                        if tool_result.get("redirected"):
                            print("⚠️  Pergunta redirecionada (fora do escopo)")
                            retrieved_context.append("OUT_OF_SCOPE_FINANCE")
                    
                    # Se for finance tool, adiciona ao contexto
                    elif tool_name in ["get_stock_price", "compare_stocks"] and result.get("success"):
                        summary = tool_result.get("summary", str(tool_result))
                        retrieved_context.append(summary)
                        sources_used.append(f"yfinance:{tool_name}")
                        print(f"   Dados financeiros obtidos")
                    
                    # Adiciona resultado ao contexto
                    tool_messages.append({
                        "role": "assistant",
                        "content": response.get("content", "")
                    })
                    tool_messages.append({
                        "role": "user",
                        "content": f"Resultado: {json.dumps(result, ensure_ascii=False)}\n\nAgora você tem o contexto necessário. Não chame mais tools."
                    })
                    
                    # Se obteve resultados, sai do loop
                    if result.get("success") and (retrieved_context or tool_name in ["get_stock_price", "compare_stocks", "redirect"]):
                        print("✅ Contexto obtido, finalizando tool calling")
                        break
                    
                except Exception as e:
                    print(f"❌ Erro ao executar tool: {str(e)}")
                    tool_messages.append({
                        "role": "user",
                        "content": f"Erro: {str(e)}. Informe ao usuário que houve um erro."
                    })
                    break
            else:
                # Sem tool call - força retry
                print("⚠️  Nenhuma tool call detectada - forçando retry")
                tool_messages.append({
                    "role": "system",
                    "content": "Você não chamou nenhuma tool. Escolha UMA das 4 tools disponíveis para responder a pergunta do usuário: search_documents (para buscar docs), get_stock_price (1 ação), compare_stocks (2+ ações) ou redirect (pergunta fora do escopo). Chame a tool apropriada AGORA.\n\npergunta do usuário: {query}"
                })
                # Continua o loop para tentar novamente (não break)
        
        # Gera resposta final com gemma3:1b
        print("\n📝 Gerando resposta final com gemma3:1b...")
        final_answer = self._generate_final_response(
            query=query,
            context=retrieved_context
        )
        
        # Remove duplicatas de sources
        sources_used = list(set(sources_used))
        
        # Cria resposta validada
        response = RAGResponse(
            answer=final_answer,
            sources_used=sources_used,
            confidence="high" if retrieved_context else "medium"
        )
        
        print(f"\n✅ Resposta gerada!")
        print(f"   Fontes usadas: {len(sources_used)}")
        print(f"   Confiança: {response.confidence}")
        print(f"   Contexto da execução: {context}")
        
        # Atualiza histórico de conversa
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response.answer})
        
        return response, context
    
    def _generate_final_response(self, query: str, context: List[str]) -> str:
        """
        Gera resposta final usando gemma3:1b
        
        Args:
            query: Pergunta original
            context: Lista de contextos recuperados
            
        Returns:
            Resposta gerada
        """
        # Verifica se está fora do escopo (pergunta não financeira)
        if context and "OUT_OF_SCOPE_FINANCE" in context:
            return """Desculpe, sou um assistente especializado em finanças e mercado financeiro. 

Posso ajudar com:
📈 Análise de preços de ações
📊 Comparação de desempenho de ativos
💰 Conceitos e teorias de mercado financeiro
📚 Informações sobre investimentos e economia

Por favor, faça uma pergunta relacionada a finanças e terei prazer em ajudar!"""
        
        if context:
            context_text = "\n\n".join([f"Contexto {i+1}:\n{ctx}" 
                                       for i, ctx in enumerate(context)])
            prompt = f"""Com base no contexto abaixo, responda a pergunta do usuário.

CONTEXTO:
{context_text}

PERGUNTA: {query}

Responda de forma clara e objetiva, usando as informações do contexto."""
        else:
            prompt = f"""Responda a pergunta do usuário com base no seu conhecimento.

PERGUNTA: {query}

Responda de forma clara e objetiva."""
        
        messages = [
            {"role": "system", "content": "Você é um assistente financeiro especializado que responde perguntas sobre mercado financeiro, investimentos e economia."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.response_generator.chat(messages)
        return response
    
    def reset(self):
        """Reseta histórico do agente"""
        self.conversation_history = []
    
    def _build_tool_calling_prompt(self) -> str:
        """Constrói prompt para tool calling"""
        return """Você é um assistente de FINANÇAS e ECONOMIA. Você DEVE SEMPRE escolher uma tool.

🛠️ TOOLS DISPONÍVEIS:

1. **search_documents** - Busca em documentos sobre finanças/economia
   Use para: conceitos, teoria, Selic, inflação, PIB, mercado de capitais, etc.
   IMPORTANTE: Use a pergunta ORIGINAL do usuário como query

2. **get_stock_price** - Preço de UMA ação

3. **compare_stocks** - Compara MÚLTIPLAS ações

4. **redirect** - Pergunta fora do escopo
   Use para: perguntas NÃO relacionadas a finanças/economia

⚠️ REGRAS:
- SEMPRE escolha uma tool (obrigatório)
- Economia/finanças → search_documents
- 1 ação → get_stock_price
- 2+ ações → compare_stocks
- Outra → redirect

Escolha a tool apropriada agora."""
    
    def _default_system_prompt(self) -> str:
        """System prompt padrão"""
        return "Você é um assistente RAG que busca informações relevantes e gera respostas precisas."
