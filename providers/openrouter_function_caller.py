"""
OpenRouter Function Caller Provider

Specialized provider for function calling using OpenRouter API.
"""
import os
import json
import requests
from typing import List, Dict, Optional


class OpenRouterFunctionCaller:
    """
    Function calling provider using OpenRouter
    
    Handles tool/function calling with OpenRouter's API
    """
    
    def __init__(
        self,
        model: str = "xiaomi/mimo-v2-flash:free",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.3,
        stream: bool = False
    ):
        """
        Initialize OpenRouter Function Caller
        
        Args:
            model: Model name
            api_key: OpenRouter API key
            base_url: API base URL
            temperature: Sampling temperature
            stream: Enable streaming
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key required")
        
        self.base_url = base_url
        self.temperature = temperature
        self.stream = stream
    
    def call_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict]
    ) -> Dict:
        """
        Call LLM with function calling support
        
        Args:
            messages: Chat messages
            tools: Tool definitions in OpenAI format
            
        Returns:
            Response dict with content and tool_calls
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "tools": tools,
            "stream": False
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",  # Optional
            "X-Title": "RAG Agent"  # Optional
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            message = data["choices"][0]["message"]
            
            result = {
                "content": message.get("content", ""),
                "tool_calls": message.get("tool_calls")
            }
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Erro ao chamar OpenRouter: {e}")
            raise
