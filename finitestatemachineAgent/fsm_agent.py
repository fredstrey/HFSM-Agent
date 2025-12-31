import sys
import os
import json
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# Ensure we can import modules from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.context import ExecutionContext
from core.registry import ToolRegistry
from core.executor import ToolExecutor
from core.schemas import AgentResponse
from providers.openrouter import OpenRouterProvider

class AgentState(str, Enum):
    ROUTE = "route"
    CALL_TOOL = "call_tool"
    ANSWER = "answer"
    FAIL = "fail"

class Decision(BaseModel):
    call_tool: bool
    tool_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    ready_to_answer: bool = False

class StateMachineAgent:
    def __init__(self, model: str, tools: list, max_steps: int = 10, system_instruction: str = ""):
        self.llm = OpenRouterProvider(model=model, temperature=0)
        self.registry = ToolRegistry() # Singleton instance
        self.executor = ToolExecutor(self.registry)
        self.max_steps = max_steps
        self.system_instruction = system_instruction
        
        # Register tools provided to this agent instance
        for tool in tools:
            # Safely get tool metadata, assuming standard decorating/attribute pattern
            name = getattr(tool, "_tool_name", tool.__name__)
            description = getattr(tool, "_tool_description", tool.__doc__)
            args_model = getattr(tool, "_args_model", None)
            
            if args_model:
                self.registry.register(
                    name=name,
                    description=description,
                    function=tool,
                    args_model=args_model
                )
            else:
                print(f"Warning: Tool {name} has no args_model, skipping registration.")

    def run(self, query: str) -> tuple[AgentResponse, ExecutionContext]:
        state = AgentState.ROUTE
        context = ExecutionContext(user_query=query)
        steps = 0
        
        tool_decision = None

        print(f"--- Starting FSM Agent Run for: '{query}' ---")

        while state not in {AgentState.ANSWER, AgentState.FAIL}:
            context.current_iteration = steps # Sync iteration count

            if steps >= self.max_steps:
                print(f"Max steps {self.max_steps} reached.")
                state = AgentState.FAIL
                break

            if state == AgentState.ROUTE:
                print(f"[Step {steps}] STATE: ROUTE - Reasoning...")
                decision = self._decide(query, context)
                print(f"[Step {steps}] DECISION: {decision}")

                if decision.call_tool and decision.tool_name:
                    state = AgentState.CALL_TOOL
                    tool_decision = decision
                elif decision.ready_to_answer:
                    state = AgentState.ANSWER
                else:
                    print("[Step {steps}] ERROR: Ambiguous decision (neither call_tool nor ready_to_answer).")
                    state = AgentState.FAIL

            elif state == AgentState.CALL_TOOL:
                print(f"[Step {steps}] STATE: CALL_TOOL - Executing {tool_decision.tool_name}...")
                
                result_map = self.executor.execute(
                    tool_decision.tool_name,
                    tool_decision.arguments or {}
                )
                
                success = result_map.get("success", False)
                result = result_map.get("result") if success else result_map.get("error")

                context.add_tool_call(
                    tool_decision.tool_name,
                    tool_decision.arguments or {},
                    result
                )
                print(f"[Step {steps}] RESULT: {str(result)[:200]}...")
                state = AgentState.ROUTE

            steps += 1

        if state == AgentState.ANSWER:
            print(f"[Step {steps}] STATE: ANSWER - Generating final response...")
            answer = self._answer(query, context)
            print("--- FSM Agent Run Completed: SUCCESS ---")
            return AgentResponse(answer=answer, metadata={"tool_calls": context.tool_calls}), context

        print("--- FSM Agent Run Completed: FAIL ---")
        return AgentResponse(
            answer="Não foi possível responder com as informações disponíveis (Limite de passos ou falha interna).",
            metadata={"tool_calls": context.tool_calls}
        ), context

    def _tool_context(self, context: ExecutionContext) -> str:
        if not context.tool_calls:
            return "Nenhuma ferramenta chamada ainda."
        
        history = ""
        for call in context.tool_calls:
            history += f"\n[Iteração {call.get('iteration')}]\n"
            history += f"Ferramenta: {call.get('tool_name')}\n"
            history += f"Argumentos: {json.dumps(call.get('arguments'), ensure_ascii=False)}\n"
            history += f"Resultado: {str(call.get('result'))}\n"
            history += "-" * 30
        return history

    def _decide(self, query: str, context: ExecutionContext) -> Decision:
        tools_desc = []
        for name in self.registry.list():
            meta = self.registry.get(name)
            if meta:
                # Include full schema in the description
                schema = meta['args_model'].model_json_schema()
                # Clean up schema for brevity if needed, or just dump it
                props = schema.get("properties", {})
                required = schema.get("required", [])
                
                tool_info = f"- {name}: {meta['description']}\n  Argumentos (JSON Schema): {json.dumps(props, ensure_ascii=False)}\n  Obrigatórios: {required}"
                tools_desc.append(tool_info)
        
        tools_str = "\n\n".join(tools_desc)

        prompt = f"""
You are the 'Brain' of a Finite State Machine (FSM) Agent.
Your responsibility is to decide the logical NEXT STEP to resolve the user's request.

CONTEXT / PERSONA:
{self.system_instruction}

USER QUERY:
"{query}"

AVAILABLE TOOLS:
{tools_str}

EXECUTION HISTORY (Accumulated Context):
{self._tool_context(context)}

DECISION INSTRUCTIONS:
1. Check the History. If the necessary information is ALREADY there, or if the question is trivial/general knowledge, set "ready_to_answer": true.
2. If external information is needed, choose the correct tool and set "call_tool": true, with "tool_name".
3. GENERATE THE "arguments" STRICTLY ACCORDING TO THE PROVIDED JSON SCHEMA FOR THE TOOL. DO NOT INVENT ARGUMENTS.
4. If the last tool failed or gave an error, try another approach or stop.

IMPORTANT: Respond ONLY with a valid JSON object. No text before or after.
Expected JSON format:
{{
  "call_tool": false,
  "tool_name": null,
  "arguments": null,
  "ready_to_answer": true
}}
OR
{{
  "call_tool": true,
  "tool_name": "tool_name",
  "arguments": {{ "arg": "value" }},
  "ready_to_answer": false
}}
"""
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # We assume the LLM provider returns a string
            response_text = self.llm.chat(messages)
            
            # Simple cleanup for markdown code blocks if the model adds them
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            
            cleaned_text = cleaned_text.strip()
            
            return Decision.model_validate_json(cleaned_text)
        except Exception as e:
            print(f"Erro ao interpretar decisão do LLM: {e}")
            # print(f"Resposta Crua: {response_text}") # Avoid UnboundLocalError
            return Decision(call_tool=False, ready_to_answer=False)

    def _answer(self, query: str, context: ExecutionContext) -> str:
        prompt = f"""
You are a helpful and precise assistant.
{self.system_instruction}

Answer the user's question using the information collected.

USER QUERY: {query}

CONTEXTO:
{self._tool_context(context)}

Answer in the same language as the query, citing the data obtained when relevant.
"""
        return self.llm.chat([{"role": "user", "content": prompt}])
