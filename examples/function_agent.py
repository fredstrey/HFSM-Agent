"""
Agente genérico com function calling
"""
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from ..core.registry import ToolRegistry
from ..core.executor import ToolExecutor


class FunctionAgent:
    """Agente genérico que usa tools registradas via decorator"""
    
    def __init__(
        self,
        llm_provider,
        response_model: type[BaseModel],
        system_prompt: Optional[str] = None,
        max_iterations: int = 5
    ):
        """
        Inicializa agente
        
        Args:
            llm_provider: Provider do LLM (deve ter método chat())
            response_model: Modelo Pydantic para validar resposta final
            system_prompt: Prompt do sistema (opcional)
            max_iterations: Máximo de iterações
        """
        self.llm_provider = llm_provider
        self.response_model = response_model
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_iterations = max_iterations
        
        self.registry = ToolRegistry()
        self.executor = ToolExecutor(self.registry)
        self.memory: List[Dict[str, Any]] = []
    
    def run(self, query: str) -> BaseModel:
        """
        Executa o agente
        
        Args:
            query: Pergunta do usuário
            
        Returns:
            Resposta validada pelo response_model
        """
        print(f"\n🤖 Agente processando: {query}")
        print("=" * 70)
        
        # Prepara mensagens
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": query}
        ]
        
        tool_results = None
        
        # Loop de iterações
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n🔄 Iteração {iteration}/{self.max_iterations}")
            
            # Chama LLM
            response = self.llm_provider.chat(messages, stream=False)
            print(f"💬 LLM: {response[:200]}...")
            
            # Tenta executar tool call
            result = self.executor.execute_from_llm_response(response)
            
            if result and result["success"]:
                print(f"✅ Tool executada: {result['tool_name']}")
                tool_results = result["result"]
                
                # Adiciona resultado ao contexto
                messages.append({
                    "role": "assistant",
                    "content": response
                })
                messages.append({
                    "role": "user",
                    "content": f"Resultado da tool: {json.dumps(tool_results)}\n\nAgora gere a resposta final em JSON."
                })
                
            elif result and not result["success"]:
                print(f"❌ Erro na tool: {result['error']}")
                messages.append({
                    "role": "user",
                    "content": f"Erro: {result['error']}. Tente novamente ou gere resposta final."
                })
                
            else:
                # Sem tool call, tenta validar como resposta final
                validated = self._validate_response(response)
                if validated:
                    print("✅ Resposta final validada!")
                    return validated
                
                # Não validou, pede para tentar novamente
                if iteration < self.max_iterations:
                    messages.append({
                        "role": "user",
                        "content": "Resposta inválida. Responda em JSON no formato correto."
                    })
        
        # Última tentativa de validação
        print("\n⚠️  Máximo de iterações atingido")
        return self._validate_response(response, fallback=True)
    
    def reset(self):
        """Reseta memória do agente"""
        self.memory = []
    
    def _build_system_prompt(self) -> str:
        """Constrói system prompt com tools disponíveis"""
        tools_desc = []
        for tool_name in self.registry.list():
            tool_data = self.registry.get(tool_name)
            tools_desc.append(f"- {tool_name}: {tool_data['description']}")
        
        tools_text = "\n".join(tools_desc) if tools_desc else "Nenhuma tool disponível"
        
        schema = self.response_model.model_json_schema()
        
        return f"""{self.system_prompt}

TOOLS DISPONÍVEIS:
{tools_text}

Para usar uma tool, responda no formato:
tool_name({{"arg1": "value1", "arg2": "value2"}})

FORMATO DE RESPOSTA FINAL (JSON):
{json.dumps(schema, indent=2)}

Responda SEMPRE em JSON válido no formato especificado."""
    
    def _default_system_prompt(self) -> str:
        """System prompt padrão"""
        return """Você é um assistente útil que pode usar tools para responder perguntas.

Analise a pergunta e decida qual tool deve ser chamada e os parâmetros utilizados"""
    
    def _validate_response(self, text: str, fallback: bool = False) -> Optional[BaseModel]:
        """Valida resposta como JSON"""
        try:
            # Extrai JSON
            start = text.find("{")
            end = text.rfind("}") + 1
            
            if start == -1 or end == 0:
                raise ValueError("Nenhum JSON encontrado")
            
            json_text = text[start:end]
            payload = json.loads(json_text)
            
            # Valida com Pydantic
            return self.response_model(**payload)
            
        except Exception as e:
            if fallback:
                # Cria resposta básica
                return self.response_model(
                    answer=text,
                    **{k: [] if "list" in str(v) else "unknown" 
                       for k, v in self.response_model.model_fields.items() 
                       if k != "answer"}
                )
            return None
