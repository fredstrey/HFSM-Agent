"""
Registry singleton para tools
"""
from typing import Dict, List, Callable, Any, Type, Optional
from pydantic import BaseModel


class ToolRegistry:
    """Registry singleton para gerenciar tools"""
    
    _instance = None
    _tools: Dict[str, Dict[str, Any]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance
    
    def register(
        self,
        name: str,
        description: str,
        function: Callable,
        args_model: Type[BaseModel],
        return_type: Type = Any
    ):
        """
        Registra uma tool
        
        Args:
            name: Nome da tool
            description: Descrição
            function: Função a ser executada
            args_model: Modelo Pydantic para argumentos
            return_type: Tipo de retorno
        """
        self._tools[name] = {
            "name": name,
            "description": description,
            "function": function,
            "args_model": args_model,
            "return_type": return_type,
            "schema": self._generate_schema(name, description, args_model)
        }
        
        print(f"✅ Tool registrada: {name}")
    
    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """Retorna uma tool pelo nome"""
        return self._tools.get(name)
    
    def list(self) -> List[str]:
        """Lista nomes de todas as tools"""
        return list(self._tools.keys())
    
    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Retorna todas as tools"""
        return self._tools.copy()
    
    def to_openai_format(self) -> List[Dict[str, Any]]:
        """
        Converte tools para formato OpenAI/Ollama
        
        Returns:
            Lista de tools no formato esperado pelo LLM
        """
        tools = []
        for tool_data in self._tools.values():
            tools.append({
                "type": "function",
                "function": tool_data["schema"]
            })
        return tools
    
    def _generate_schema(
        self,
        name: str,
        description: str,
        args_model: Type[BaseModel]
    ) -> Dict[str, Any]:
        """Gera schema JSON para a tool"""
        schema = args_model.model_json_schema()
        
        return {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", [])
            }
        }
    
    def clear(self):
        """Limpa todas as tools (útil para testes)"""
        self._tools.clear()
