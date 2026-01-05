"""
OpenRouter Provider - Async Version
====================================

Async implementation using httpx for better concurrency.
"""

import os
import httpx
import json
from typing import List, Dict, Optional, AsyncIterator


class AsyncOpenRouterProvider:
    """
    Async LLM provider using OpenRouter API with httpx.
    
    Supports:
    - Async chat completion
    - Async streaming responses
    - Concurrent requests
    """
    
    def __init__(
        self,
        model: str = "xiaomi/mimo-v2-flash:free",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.3,
        timeout: float = 30.0
    ):
        """
        Initialize async OpenRouter provider.
        
        Args:
            model: Model name
            api_key: OpenRouter API key
            base_url: API base URL
            temperature: Sampling temperature
            timeout: Request timeout in seconds
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. "
                "Set OPENROUTER_API_KEY env var or pass api_key parameter."
            )
        
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Async chat completion.
        
        Args:
            messages: List of message dicts
            tools: Optional tool definitions
            
        Returns:
            Dict with 'content' and 'usage'
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False
        }
        
        if tools:
            payload["tools"] = tools
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/fredstrey/react_agent",
            "X-Title": "Finance.AI"
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers
            )
            
            if not response.is_success:
                print(f"OpenRouter Error: {response.text}")
            
            response.raise_for_status()
            
            data = response.json()
            return {
                "content": data["choices"][0]["message"]["content"],
                "usage": data.get("usage", {})
            }
    
    async def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict],
        tool_choice: Optional[str] = None
    ) -> Dict:
        """
        Async chat with function calling.
        
        Args:
            messages: List of message dicts
            tools: Tool definitions
            tool_choice: Optional tool choice ("auto", "required", "none")
            
        Returns:
            Dict with 'content', 'tool_calls', and 'usage'
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "tools": tools,
            "stream": False
        }
        
        # Only add tool_choice if explicitly set (not None)
        # Some models don't support this parameter
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/fredstrey/react_agent",
            "X-Title": "Finance.AI"
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers
            )
            
            # 🔥 Fallback Check
            if response.status_code in [404, 400]:
                error_text = response.text.lower()
                if "tool" in error_text or "endpoint" in error_text or "model does not support" in error_text:
                     print(f"⚠️ [OpenRouter] Native tool calling failed ({response.status_code}). Switching to ReAct fallback.")
                     return await self._chat_with_tools_fallback(messages, tools)

            if not response.is_success:
                print(f"OpenRouter Error: {response.text}")
                
            response.raise_for_status()
            
            data = response.json()
            message = data["choices"][0]["message"]
            
            return {
                "content": message.get("content", ""),
                "tool_calls": message.get("tool_calls"),
                "usage": data.get("usage", {})
            }

    async def _chat_with_tools_fallback(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict]
    ) -> Dict:
        """
        Fallback ReAct-style implementation for models without native tool support.
        """
        # 1. Format tools for prompt
        tools_desc = self._format_tools_for_prompt(tools)
        
        # 2. Inject system instructions
        system_injection = f"""
Thinking Process:
1. Analyze the user's request.
2. Decide if you need to use a tool.
3. If YES, output a JSON object EXACTLY like this:
```json
{{
    "tool": "tool_name",
    "arguments": {{ "arg_name": "value" }}
}}
```
4. If NO, answer directly.

AVAILABLE TOOLS:
{tools_desc}
"""
        # Append to last system message or create new one
        messages_copy = [m.copy() for m in messages]
        
        # Find system message to append to, or prepend a new one
        system_msg_idx = next((i for i, m in enumerate(messages_copy) if m["role"] == "system"), -1)
        if system_msg_idx >= 0:
            messages_copy[system_msg_idx]["content"] += "\n" + system_injection
        else:
            messages_copy.insert(0, {"role": "system", "content": system_injection})
            
        # 3. Call standard chat (no tools param)
        response = await self.chat(messages_copy)
        content = response["content"]
        
        # 4. Parse output for JSON tool call
        tool_calls = []
        clean_content = content
        
        try:
            # Look for JSON block
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "{" in content and "}" in content:
                # Naive extraction
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
            else:
                json_str = ""
                
            if json_str:
                tool_data = json.loads(json_str)
                if "tool" in tool_data and "arguments" in tool_data:
                    # Construct tool call object compatible with OpenAI format
                    tool_calls.append({
                        "id": f"call_fallback_{os.urandom(4).hex()}",
                        "type": "function",
                        "function": {
                            "name": tool_data["tool"],
                            "arguments": json.dumps(tool_data["arguments"])
                        }
                    })
                    # Remove the JSON from content to avoid double printing
                    clean_content = content.replace(json_str, "").replace("```json", "").replace("```", "").strip()
                    
        except Exception:
            # Parsing failed, treat as normal message
            pass
            
        return {
            "content": clean_content if not tool_calls else None, # If tool call, content usually null in OpenAI standards, but here we can keep reasoning if needed. strict: None if tool call.
            "tool_calls": tool_calls if tool_calls else None,
            "usage": response.get("usage", {})
        }

    def _format_tools_for_prompt(self, tools: List[Dict]) -> str:
        """Convert tool definitions to a readable string."""
        desc = []
        for t in tools:
            func = t.get("function", {})
            name = func.get("name", "unknown")
            description = func.get("description", "")
            params = json.dumps(func.get("parameters", {}), indent=2)
            desc.append(f"- {name}: {description}\n  Parameters: {params}")
        return "\n\n".join(desc)
    
    async def chat_stream(
        self,
        messages: List[Dict[str, str]]
    ) -> AsyncIterator[str]:
        """
        Async streaming chat completion.
        
        Args:
            messages: List of message dicts
            
        Yields:
            Response tokens as they arrive
            
        Note:
            Usage information is sent as the last chunk with a special marker.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
            "stream_options": {"include_usage": True}  # Request usage in stream
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/fredstrey/react_agent",
            "X-Title": "Finance.AI"
        }
        
        usage_data = None
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line and line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            
                            # Check for usage information
                            if "usage" in data:
                                usage_data = data["usage"]
                            
                            # Yield content tokens
                            delta = data["choices"][0]["delta"]
                            if "content" in delta:
                                yield delta["content"]
                        except json.JSONDecodeError:
                            continue
        
        # Yield usage with special marker for AnswerState to capture

        if usage_data:
            yield {"__usage__": usage_data}
