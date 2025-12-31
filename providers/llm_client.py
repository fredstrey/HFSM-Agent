from typing import List, Dict, Iterator
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

    def chat(self, messages: List[Dict[str, str]]) -> str:
        return self.provider.chat(messages)

    def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict]
    ) -> Dict:
        return self.provider.chat_with_tools(messages, tools)

    def chat_stream(
        self,
        messages: List[Dict[str, str]]
    ) -> Iterator[str]:
        """
        Streaming em forma de generator (token por token)
        """
        stream_provider = OpenRouterProvider(
            model=self.provider.model,
            temperature=self.provider.temperature,
            stream=True
        )

        response = stream_provider.chat(messages, stream=True)

        for char in response:
            yield char
