"""
Generic Tool Calling Agent
Base class for agents that use tool calling with LLMs
"""
import json
import threading
from typing import List, Callable, Optional, Dict, Any, Tuple
from pydantic import BaseModel

from core.registry import ToolRegistry
from core.execution_context import ExecutionContext
from core.executor import ToolExecutor
from core.agent_response import AgentResponse
from providers.openrouter import OpenRouterProvider
from providers.openrouter_function_caller import OpenRouterFunctionCaller


class ToolCallingAgent:
    """
    Generic agent that can call tools using function calling
    
    This is a base class that provides:
    - Tool registration and management
    - Function calling loop
    - Response generation
    - Execution context tracking
    
    Can be extended for specific use cases (e.g., RAG, Q&A, etc.)
    """
    
    # Class-level registry (shared across all instances)
    _registry = ToolRegistry()
    
    # Class-level semaphore to ensure sequential tool execution (max 1 concurrent)
    _tool_execution_semaphore = threading.Semaphore(1)
    
    def __init__(
        self,
        tools: List[Callable],
        tool_caller_model: str = "xiaomi/mimo-v2-flash:free",
        response_model: str = "xiaomi/mimo-v2-flash:free",
        response_class: type = None,  # Custom response BaseModel
        system_prompt: Optional[str] = None,
        max_iterations: int = 3
    ):
        """
        Initialize Tool Calling Agent
        
        Args:
            tools: List of tool functions to register
            tool_caller_model: Model for tool calling (default: xiaomi/mimo-v2-flash:free)
            response_model: Model for response generation (default: xiaomi/mimo-v2-flash:free)
            response_class: Custom Pydantic BaseModel for responses (default: AgentResponse)
            system_prompt: Custom system prompt (optional)
            max_iterations: Maximum tool calling iterations (default: 3)
        """
        self.tools = tools
        self.tool_caller_model_name = tool_caller_model
        self.response_model_name = response_model
        self.response_class = response_class or AgentResponse
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        
        # Initialize providers
        self.tool_caller = OpenRouterFunctionCaller(
            model=tool_caller_model,
            temperature=0.3
        )
        
        self.response_generator = OpenRouterProvider(
            model=response_model,
            temperature=0.7
        )
        
        # Tool management
        self.registry = ToolRegistry()
        self.executor = ToolExecutor(self.registry)
        
        # Register tools
        for tool_func in tools:
            # Check if tool was already registered via @tool decorator
            if hasattr(tool_func, '_tool_name'):
                # Tool already registered, just verify it's in registry
                tool_name = tool_func._tool_name
                if not self.registry.get(tool_name):
                    # Re-register using metadata from decorator
                    self.registry.register(
                        name=tool_name,
                        description=tool_func._tool_description,
                        function=tool_func,
                        args_model=tool_func._args_model
                    )
            else:
                # Tool not decorated, try to register manually
                # This requires the function to have proper type hints
                raise ValueError(
                    f"Tool {tool_func.__name__} must be decorated with @tool decorator. "
                    f"Example: @tool\\ndef {tool_func.__name__}(...):"
                )
        
        # Configuration
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_iterations = max_iterations
        self.response_class = response_class or AgentResponse
        
        # Store model names for reference
        self.tool_caller_model_name = tool_caller_model
        self.response_model_name = response_model
    
    def run(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        context: Optional[ExecutionContext] = None
    ) -> Tuple[AgentResponse, ExecutionContext]:
        """
        Execute agent with tool calling
        
        Args:
            query: User query
            chat_history: Optional chat history
            context: Optional existing ExecutionContext
            
        Returns:
            Tuple of (AgentResponse, ExecutionContext)
        """
        print(f"\n🤖 Agent processando: {query}")
        print("=" * 70)
        
        # Create or update context
        if context is None:
            context = ExecutionContext(
                user_query=query,
                max_iterations=self.max_iterations,
                chat_history=chat_history or []
            )
        
        # Build tool calling messages
        tool_messages = self._build_tool_messages(query, chat_history or [])
        
        # Tool calling loop
        retrieved_context = []
        sources_used = []
        
        for iteration in range(1, self.max_iterations + 1):
            context.current_iteration = iteration
            print(f"\n🔄 Iteração {iteration}/{self.max_iterations}")
            
            # Call tool caller
            response = self.tool_caller.call_with_tools(
                messages=tool_messages,
                tools=self.registry.to_openai_format()
            )
            
            # Truncate for display
            content = response.get("content", "")
            tool_calls_preview = response.get("tool_calls", [])
            response_preview = str(response)[:200] + "..." if len(str(response)) > 200 else str(response)
            print(f"💬 Tool Caller: {response_preview}")
            
            # Parse tool calls
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                print("⚠️  Nenhuma tool call detectada")
                if iteration < self.max_iterations:
                    tool_messages.append({
                        "role": "system",
                        "content": (
                            "ERRO CRÍTICO: Você NÃO chamou nenhuma tool.\n\n"
                            "RESPONDA EXCLUSIVAMENTE com UMA chamada de tool.\n"
                            "NÃO escreva texto.\n"
                            "NÃO explique.\n"
                            "NÃO faça perguntas.\n\n"
                            "Formato OBRIGATÓRIO:\n"
                            "{ \"name\": \"<tool_name>\", \"arguments\": { ... } }\n\n"
                            "Tools permitidas:\n"
                            "- search_documents\n"
                            "- get_stock_price\n"
                            "- compare_stocks\n"
                            "- redirect\n\n"
                            f"Pergunta do usuário: {query}"
                        )
                    })
                    continue
                break
            
            # Execute tools (HARDCODED: only execute FIRST tool for ReAct loop)
            for idx, tool_call in enumerate(tool_calls):
                tool_name = tool_call["name"]
                arguments = tool_call.get("arguments", {})
                
                print(f"🔧 Executando tool: {tool_name}")
                
                # If multiple tools were called, warn and skip the rest
                if len(tool_calls) > 1:
                    print(f"   ⚠️ Múltiplas tools detectadas ({len(tool_calls)}), executando apenas a primeira")
                
                # Acquire semaphore to ensure only 1 tool executes at a time
                self._tool_execution_semaphore.acquire()
                try:
                    result = self.executor.execute(tool_name, arguments)
                    
                    # Process result (subclass can override)
                    processed = self._process_tool_result(
                        tool_name, result, context, retrieved_context, sources_used
                    )
                    
                    if processed.get("should_break", False):
                        break
                        
                except Exception as e:
                    print(f"❌ Erro ao executar tool: {str(e)}")
                    tool_messages.append({
                        "role": "user",
                        "content": f"Erro: {str(e)}. Informe ao usuário."
                    })
                    break
                finally:
                    # Always release semaphore
                    self._tool_execution_semaphore.release()
                
                # HARDCODED: Break after first tool to allow ReAct to reason
                break
            
            # Check if we should continue
            if context.has_context or context.is_out_of_scope:
                break
        
        # Generate final response
        final_answer = self._generate_response(query, retrieved_context)
        
        # Build response using custom class
        response = self._build_response(
            answer=final_answer,
            sources_used=list(set(sources_used)),
            context=context
        )
        
        print(f"\n✅ Resposta gerada!")
        print(f"   Fontes: {len(sources_used)}")
        
        return response, context
    
    def _build_response(
        self,
        answer: str,
        sources_used: List[str],
        context: ExecutionContext
    ):
        """
        Build response using custom response class
        
        Subclasses can override this to add custom fields
        
        Args:
            answer: Generated answer
            sources_used: List of sources
            context: Execution context
            
        Returns:
            Instance of response_class
        """
        # Default AgentResponse only needs answer
        if self.response_class == AgentResponse:
            return self.response_class(answer=answer)
        
        # Try to instantiate custom class with common fields
        try:
            return self.response_class(
                answer=answer,
                sources_used=sources_used,
                confidence="high"
            )
        except TypeError:
            # If custom class has different fields, try with just answer
            try:
                return self.response_class(answer=answer)
            except Exception as e:
                raise ValueError(
                    f"Could not instantiate {self.response_class.__name__}. "
                    f"Override _build_response() method for custom response fields. Error: {e}"
                )
    
    def _build_tool_messages(
        self,
        query: str,
        chat_history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Build messages for tool calling with history
        
        Args:
            query: Current user query
            chat_history: Chat history
            
        Returns:
            List of messages for tool caller
        """
        system_content = self.system_prompt
        
        # Add chat history context if exists
        if chat_history and len(chat_history) >= 2:
            last_user = chat_history[-2]
            last_assistant = chat_history[-1]
            
            history_context = f"""

📜 CONTEXTO DA CONVERSA ANTERIOR:
Pergunta anterior: "{last_user['content']}"
Resposta anterior: "{last_assistant['content']}"

⚠️ Use o contexto acima para entender a pergunta atual.

Pergunta ATUAL do usuário:"""
            system_content += history_context
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]
    
    def _parse_tool_calls(self, response) -> List[Dict[str, Any]]:
        """
        Parse tool calls from LLM response
        
        Args:
            response: LLM response (can be string or dict)
            
        Returns:
            List of tool calls
        """
        # Handle dict response from FunctionCaller
        if isinstance(response, dict):
            tool_calls = response.get("tool_calls")
            if tool_calls:
                # Extract function calls
                parsed_calls = []
                for call in tool_calls:
                    if "function" in call:
                        func = call["function"]
                        arguments = func.get("arguments", {})
                        
                        # Parse arguments if it's a JSON string
                        if isinstance(arguments, str):
                            print(f"🔍 Parsing arguments string: {arguments[:100]}...")
                            try:
                                # First parse to get the actual data structure
                                parsed = json.loads(arguments)
                                
                                # Check if any values are still JSON strings (double-encoded)
                                if isinstance(parsed, dict):
                                    for key, value in parsed.items():
                                        if isinstance(value, str):
                                            # Try to parse again in case it's double-encoded
                                            try:
                                                parsed[key] = json.loads(value)
                                                print(f"   ✅ Double-decoded {key}: {parsed[key]}")
                                            except:
                                                # Not JSON, keep as-is
                                                pass
                                
                                arguments = parsed
                                print(f"   ✅ Final parsed arguments: {arguments}")
                            except Exception as e:
                                print(f"   ❌ Failed to parse arguments: {e}")
                                arguments = {}
                        
                        parsed_calls.append({
                            "name": func.get("name"),
                            "arguments": arguments
                        })
                return parsed_calls
            return []
        
        # Handle string response (legacy)
        if not isinstance(response, str):
            return []
            
        # Try to find JSON in response
        try:
            # Look for JSON patterns
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                
                # Handle different formats
                if isinstance(data, dict):
                    if "name" in data:
                        return [data]
                    elif "tool_calls" in data:
                        return data["tool_calls"]
                elif isinstance(data, list):
                    return data
        except:
            pass
        
        return []
    
    def _process_tool_result(
        self,
        tool_name: str,
        result: Any,
        context: ExecutionContext,
        retrieved_context: List[str],
        sources_used: List[str]
    ) -> Dict[str, Any]:
        """
        Process tool execution result
        
        Subclasses can override this to add custom logic
        
        Args:
            tool_name: Name of executed tool
            result: Tool execution result
            context: Execution context
            retrieved_context: List to append context to
            sources_used: List to append sources to
            
        Returns:
            Dict with processing info (e.g., should_break)
        """
        print(f"✅ Tool executada: {tool_name}")
        context.has_context = True
        return {"should_break": True}
    
    def _generate_response(self, query: str, context: List[str]) -> str:
        """
        Generate final response using context
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Generated response
        """
        if not context:
            context_text = "Nenhum contexto disponível."
        else:
            context_text = "\n\n".join(context)
        
        prompt = f"""Com base no contexto abaixo, responda a pergunta do usuário.

CONTEXTO:
{context_text}

PERGUNTA: {query}

RESPOSTA:"""
        
        response = self.response_generator.chat([
            {"role": "user", "content": prompt}
        ])
        
        return response
    
    def _default_system_prompt(self) -> str:
        """
        Default system prompt for tool calling
        
        Subclasses should override this
        """
        return """Você é um assistente útil que usa tools para responder perguntas.

REGRAS:
1. SEMPRE use uma tool para responder
2. Escolha a tool mais apropriada para a pergunta
3. Não invente informações

Escolha uma tool e forneça os argumentos necessários."""
