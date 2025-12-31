from typing import List, Dict, Iterator, Any
from providers.openrouter import OpenRouterProvider


class LLMClient:
    """
    Adapter to standardize LLM interface used by agents
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.3,
    ):
        self.provider = OpenRouterProvider(
            model=model,
            temperature=temperature,
            stream=False
        )

    def chat(self, messages: List[Dict[str, str]]) -> Dict:
        """
        Returns: { "content": str, "usage": dict }
        """
        return self.provider.chat(messages)

    def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict]
    ) -> Dict:
        return self.provider.chat_with_tools(messages, tools)

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        context: Any = None
    ) -> Iterator[str]:
        """
        Streaming em forma de generator (token por token)
        Optionally updates context with usage stats.
        """
        stream_provider = OpenRouterProvider(
            model=self.provider.model,
            temperature=self.provider.temperature,
            stream=True
        )

        response_dict = stream_provider.chat(messages, stream=True)

        if isinstance(response_dict, dict):
            content = response_dict.get("content", "")
            
            # Update usage if context provided
            if context and "usage" in response_dict:
                usage = response_dict["usage"]
                total_usage = context.get_memory("total_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
                total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                total_usage["total_tokens"] += usage.get("total_tokens", 0)
                context.set_memory("total_usage", total_usage)
                
        else:
            content = response_dict

        for char in content:
            yield char
