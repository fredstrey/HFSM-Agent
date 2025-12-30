"""
Provider Ollama para o framework
"""
import requests
from typing import List, Dict, Any, Optional


class OllamaProvider:
    """Provider para Ollama"""
    
    def __init__(
        self,
        model: str = "gemma3:1b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        stream: bool = False
    ):
        """
        Inicializa provider
        
        Args:
            model: Nome do modelo
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
        stream: Optional[bool] = None,
        **kwargs
    ) -> str:
        """
        Chat completion
        
        Args:
            messages: Lista de mensagens
            stream: Override do streaming
            **kwargs: Parâmetros adicionais
            
        Returns:
            Resposta do modelo
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
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            if use_stream:
                # Streaming não implementado nesta versão simples
                return ""
            else:
                data = response.json()
                return data["message"]["content"]
                
        except Exception as e:
            raise Exception(f"Erro ao chamar Ollama: {str(e)}")
    
    def is_available(self) -> bool:
        """Verifica se Ollama está disponível"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
