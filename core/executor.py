"""
Executor de tools com validação automática
"""
import json
from typing import Dict, Any, Optional
from pydantic import ValidationError
from .registry import ToolRegistry


class ToolExecutor:
    """Executor de tools com validação Pydantic"""
    
    def __init__(self, registry: Optional[ToolRegistry] = None):
        """
        Inicializa executor
        
        Args:
            registry: Registry de tools (usa singleton se não especificado)
        """
        self.registry = registry or ToolRegistry()
    
    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa uma tool com validação automática
        
        Args:
            tool_name: Nome da tool
            arguments: Argumentos para a tool
            
        Returns:
            Resultado da execução
        """
        # Busca tool
        tool_data = self.registry.get(tool_name)
        if not tool_data:
            raise ValueError(f"Tool '{tool_name}' não encontrada")
        
        function = tool_data["function"]
        args_model = tool_data["args_model"]
        
        try:
            # Valida argumentos com Pydantic
            validated_args = args_model(**arguments)
            
            # Executa função
            result = function(**validated_args.model_dump())
            
            # Retorna resultado
            return {
                "success": True,
                "result": result,
                "tool_name": tool_name
            }
            
        except ValidationError as e:
            # Erro de validação
            return {
                "success": False,
                "error": f"Erro de validação: {str(e)}",
                "tool_name": tool_name
            }
        
        except Exception as e:
            # Erro na execução
            return {
                "success": False,
                "error": f"Erro na execução: {str(e)}",
                "tool_name": tool_name
            }
    
    def execute_from_llm_response(self, llm_response: str) -> Optional[Dict[str, Any]]:
        """
        Extrai e executa tool call da resposta do LLM
        
        Args:
            llm_response: Resposta do LLM contendo tool call
            
        Returns:
            Resultado da execução ou None se não houver tool call
        """
        tool_call = self._extract_tool_call(llm_response)
        if not tool_call:
            return None
        
        return self.execute(tool_call["name"], tool_call["arguments"])
    
    def _extract_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extrai tool call do texto
        
        Formato esperado: tool_name({"arg1": "value1", ...})
        """
        for tool_name in self.registry.list():
            if tool_name in text:
                try:
                    # Encontra início do JSON
                    start = text.find(tool_name) + len(tool_name)
                    json_start = text.find("{", start)
                    if json_start == -1:
                        continue
                    
                    # Encontra fim do JSON
                    brace_count = 0
                    json_end = json_start
                    for i in range(json_start, len(text)):
                        if text[i] == '{':
                            brace_count += 1
                        elif text[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    
                    # Parse JSON
                    args_json = text[json_start:json_end]
                    arguments = json.loads(args_json)
                    
                    return {
                        "name": tool_name,
                        "arguments": arguments
                    }
                except:
                    continue
        
        return None
