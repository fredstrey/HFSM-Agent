from __future__ import annotations

import json
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator, Optional

from core.context import ExecutionContext

class AgentState(str, Enum):
    ROUTER = "router"
    TOOL = "tool"
    VALIDATION = "validation"
    RETRY = "retry"
    ANSWER = "answer"
    FAIL = "fail"


class SubFSM:
    def run(self, context: ExecutionContext):
        raise NotImplementedError


class RouterFSM(SubFSM):
    def __init__(self, llm, registry):
        self.llm = llm
        self.registry = registry

    def run(self, context: ExecutionContext) -> AgentState:
        system_instruction = context.get_memory("system_instruction", "")
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": context.user_query},
        ]

        for call in context.tool_calls:
            # Reconstruct the assistant message that triggered this tool
            tool_call_id = f"call_{call['tool_name']}_{call.get('iteration', 0)}" # Simple deterministic ID
            
            messages.append({
                "role": "assistant",
                "content": None, 
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": call["tool_name"],
                            "arguments": json.dumps(call["arguments"], ensure_ascii=False)
                        }
                    }
                ]
            })

            # Safely handle result serialization
            try:
                content_str = json.dumps(call.get("result", ""), ensure_ascii=False)
            except:
                content_str = str(call.get("result", ""))
                
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": call["tool_name"],
                "content": content_str
            })

        response = self.llm.chat_with_tools(
            messages=messages,
            tools=self.registry.to_openai_format()
        )

        if response.get("tool_calls"):
            context.set_memory("pending_tool_calls", response["tool_calls"])
            return AgentState.TOOL

        context.set_memory("last_llm_content", response.get("content", ""))
        return AgentState.ANSWER


class ToolFSM(SubFSM):
    def __init__(self, executor, max_workers: int = 4):
        self.executor = executor
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def run(self, context: ExecutionContext) -> AgentState:
        futures_map = {}
        pending_calls = context.get_memory("pending_tool_calls", [])

        for call in pending_calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"])
            except:
                args = {}
            
            print(f"🛠️ [ToolFSM] Executing: {name}")
            future = self.pool.submit(self.executor.execute, name, args)
            futures_map[future] = call

        for future in as_completed(futures_map):
            call = futures_map[future]
            try:
                result_map = future.result()
                result = result_map.get("result") if result_map.get("success") else result_map.get("error")
            except Exception as e:
                result = str(e)
                
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"])
            except:
                args = {}
                
            context.add_tool_call(name, args, result)

        context.set_memory("pending_tool_calls", [])
        return AgentState.VALIDATION


class ValidationFSM(SubFSM):
    def __init__(self, llm):
        self.llm = llm

    def run(self, context: ExecutionContext) -> AgentState:
        prompt = f"""
Você é um validador lógico.
Verifique se as informações coletadas são suficientes para responder.

PERGUNTA:
{context.user_query}

DADOS:
{json.dumps(context.tool_calls, ensure_ascii=False, default=str)}

Responda apenas:
{{"valid": true}} ou {{"valid": false}}
"""

        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            # Clean response
            clean_resp = response.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean_resp)
            is_valid = result.get("valid", False)
        except:
            is_valid = False

        return AgentState.ANSWER if is_valid else AgentState.RETRY


class RetryFSM(SubFSM):
    def run(self, context: ExecutionContext) -> AgentState:
        retry_count = context.get_memory("retry_count", 0)
        max_retries = context.get_memory("max_retries", 2)
        
        retry_count += 1
        context.set_memory("retry_count", retry_count)
        
        if retry_count > max_retries:
            return AgentState.FAIL

        context.user_query = f"Refine melhor:\n{context.user_query}"
        return AgentState.ROUTER


class AnswerFSM(SubFSM):
    def __init__(self, llm):
        self.llm = llm

    def run(self, context: ExecutionContext) -> Generator[str, None, None]:
        system_instruction = context.get_memory("system_instruction", "")
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": context.user_query},
        ]

        # Add collected tool outputs
        for call in context.tool_calls:
            # Reconstruct the assistant message that triggered this tool
            tool_call_id = f"call_{call['tool_name']}_{call.get('iteration', 0)}"
            
            messages.append({
                "role": "assistant",
                "content": None, 
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": call["tool_name"],
                            "arguments": json.dumps(call["arguments"], ensure_ascii=False)
                        }
                    }
                ]
            })

            # Safely handle result
            try:
                content_str = json.dumps(call.get("result", ""), ensure_ascii=False)
            except:
                content_str = str(call.get("result", ""))
                
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": call["tool_name"],
                "content": content_str
            })

        # Append a final system instruction forcing synthesis
        messages.append({
            "role": "system",
            "content": "Based on the tool results above, provide a clear and direct answer to the user's question. Do NOT call any more tools. Just answer."
        })

        yield from self.llm.chat_stream(messages)


class AgentEngine:
    def __init__(
        self,
        llm,
        registry,
        executor,
        system_instruction: str = ""
    ):
        self.system_instruction = system_instruction

        self.router = RouterFSM(llm, registry)
        self.tool = ToolFSM(executor)
        self.validation = ValidationFSM(llm)
        self.retry = RetryFSM()
        self.answer = AnswerFSM(llm)

    def run_stream(
        self,
        query: str,
        chat_history: Optional[list] = None
    ) -> tuple[Generator[str, None, None], ExecutionContext]:

        context = ExecutionContext(
            user_query=query,
        )
        # Initialize memory
        context.set_memory("system_instruction", self.system_instruction)
        context.set_memory("chat_history", chat_history or [])
        context.set_memory("retry_count", 0)
        context.set_memory("max_retries", 2)
        context.set_memory("pending_tool_calls", [])

        state = AgentState.ROUTER

        while True:
            print(f"🔄 [FSM] State: {state}")
            
            if state == AgentState.ROUTER:
                state = self.router.run(context)

            elif state == AgentState.TOOL:
                state = self.tool.run(context)

            elif state == AgentState.VALIDATION:
                print("🔍 [FSM] Validating...")
                state = self.validation.run(context)

            elif state == AgentState.RETRY:
                print("⚠️ [FSM] Retrying...")
                state = self.retry.run(context)

            elif state == AgentState.ANSWER:
                print("✅ [FSM] Answering...")
                return self.answer.run(context), context

            elif state == AgentState.FAIL:
                print("❌ [FSM] Failed.")
                def fail_gen():
                    yield "Erro: Não foi possível obter as informações necessárias após várias tentativas."
                return fail_gen(), context

