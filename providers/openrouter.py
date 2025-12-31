"""
OpenRouter Provider - LLM provider using OpenRouter API

Supports chat completion with streaming and reasoning tokens.
"""
import os
import requests
from typing import List, Dict, Optional


class OpenRouterProvider:
    """
    LLM provider using OpenRouter API
    
    Supports:
    - Chat completion
    - Streaming responses
    - Reasoning tokens tracking
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
        Initialize OpenRouter provider
        
        Args:
            model: Model name (default: xiaomi/mimo-v2-flash:free)
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            base_url: API base URL
            temperature: Sampling temperature
            stream: Enable streaming
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY env var or pass api_key parameter.")
        
        self.base_url = base_url
        self.temperature = temperature
        self.stream = stream
        
    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        stream: Optional[bool] = None
    ) -> str:
        """
        Send chat completion request
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions
            stream: Override instance stream setting
            
        Returns:
            Response content as string
        """
        use_stream = stream if stream is not None else self.stream
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": use_stream
        }
        
        if tools:
            payload["tools"] = tools
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        if use_stream:
            return self._stream_chat(payload, headers)
        else:
            return self._sync_chat(payload, headers)
    
    def _sync_chat(self, payload: Dict, headers: Dict) -> Dict:
        """Synchronous chat request"""
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers
        )
        if not response.ok:
            print(f"OpenRouter Error: {response.text}")
        response.raise_for_status()
        
        data = response.json()
        return {
            "content": data["choices"][0]["message"]["content"],
            "usage": data.get("usage", {})
        }
    
    def _stream_chat(self, payload: Dict, headers: Dict) -> Dict:
        """Streaming chat request"""
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            stream=True
        )
        response.raise_for_status()
        
        full_response = ""
        usage = {}
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        import json
                        data = json.loads(data_str)
                        
                        # Extract content
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_response += content
                        
                        # Track usage if available
                        if "usage" in data:
                            usage = data["usage"]
                    except:
                        pass
        
        return {
            "content": full_response,
            "usage": usage
        }
    
    def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict]
    ) -> Dict:
        """
        Chat with function calling support
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
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        
        data = response.json()
        message = data["choices"][0]["message"]
        
        result = {
            "content": message.get("content", ""),
            "tool_calls": message.get("tool_calls"),
            "usage": data.get("usage", {})
        }
        
        return result
