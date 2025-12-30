"""
Provider FunctionGemma para tool calling
"""
import requests
import json
import re
from typing import List, Dict, Any, Optional


class FunctionCallerProvider:
    """Provider para FunctionGemma com suporte a tool calling"""
    
    def __init__(
        self,
        model: str = "alibayram/Qwen3-30B-A3B-Instruct-2507",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        stream: bool = False
    ):
        """
        Inicializa provider FunctionGemma
        
        Args:
            model: Nome do modelo FunctionGemma
            base_url: URL base do Ollama
            temperature: Temperatura para geração
            stream: Se deve usar streaming
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.stream = stream
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Chat completion com suporte a tools
        
        Args:
            messages: Lista de mensagens
            tools: Lista de tools disponíveis (formato OpenAI)
            stream: Override do streaming
            **kwargs: Parâmetros adicionais
            
        Returns:
            Dicionário com resposta e possíveis tool calls
        """
        use_stream = stream if stream is not None else self.stream
        
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": use_stream,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature)
            }
        }
        
        # Adiciona tools se fornecidas
        if tools:
            payload["tools"] = tools
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            if use_stream:
                # Streaming não implementado nesta versão
                return {"content": "", "tool_calls": None}
            else:
                data = response.json()
                message = data.get("message", {})
                
                return {
                    "content": message.get("content", ""),
                    "tool_calls": message.get("tool_calls", None)
                }
                
        except Exception as e:
            raise Exception(f"Erro ao chamar FunctionGemma: {str(e)}")
    
    def parse_tool_call(self, response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extrai tool call da resposta do FunctionGemma
        
        Args:
            response: Resposta do chat()
            
        Returns:
            Dicionário com tool call ou None
        """
        # Verifica se há tool_calls na resposta
        tool_calls = response.get("tool_calls")
        
        if tool_calls and len(tool_calls) > 0:
            # Retorna primeira tool call
            tool_call = tool_calls[0]
            tool_name = tool_call.get("function", {}).get("name", "")
            
            # Limpa o nome da tool (remove parênteses e espaços)
            tool_name = tool_name.strip().rstrip('(').rstrip(')').strip()
            
            parsed = {
                "name": tool_name,
                "arguments": tool_call.get("function", {}).get("arguments", {})
            }
            print(f"🔧 [DEBUG] Tool call parsed: {parsed}")
            return parsed
        
        # Fallback: tenta parsear do conteúdo de texto
        content = response.get("content", "")
        
        # Padrão: tool_name({"arg": "value"})
        pattern = r'(\w+)\s*\(\s*(\{[^}]+\})\s*\)'
        match = re.search(pattern, content)
        
        if match:
            tool_name = match.group(1)
            try:
                arguments = json.loads(match.group(2))
                return {
                    "name": tool_name,
                    "arguments": arguments
                }
            except json.JSONDecodeError:
                pass
        
        return None
    
    def is_available(self) -> bool:
        """Verifica se Ollama está disponível"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
