import json
from typing import List, Dict, Any, Optional, Tuple, Type
from pydantic import BaseModel

from providers.openrouter import OpenRouterProvider
from providers.openrouter_function_caller import OpenRouterFunctionCaller
from core.registry import ToolRegistry
from core.executor import ToolExecutor
from core.context import ExecutionContext
from core.schemas import ReActDecision, ReActAnalysis, AgentResponse

class ReactAgent:
    """
    Generic ReAct Agent that orchestrates reasoning, tool calling, and response generation.
    """
    
    def __init__(
        self,
        model: str,
        system_prompt: Optional[str] = None,
        response_model: Optional[Type[BaseModel]] = None,
        max_iterations: int = 5,
        tools: List[Any] = None
    ):
        """
        Initialize the Generic ReAct Agent
        
        Args:
            model: The LLM model name to use
            system_prompt: Custom system instructions
            response_model: Optional Pydantic model for structured output
            max_iterations: Max reasoning loops
            tools: List of @tool decorated functions
        """
        self.model_name = model
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.response_model = response_model
        self.max_iterations = max_iterations
        
        # Initialize providers
        self.llm = OpenRouterProvider(model=model, temperature=0.7)
        self.function_caller = OpenRouterFunctionCaller(model=model, temperature=0.1)
        
        # Tool management
        self.registry = ToolRegistry()
        self.executor = ToolExecutor(self.registry)
        
        # Register tools if provided
        if tools:
            for tool_func in tools:
                if hasattr(tool_func, "_tool_name"):
                    # Use existing metadata to register in THIS registry
                    self.registry.register(
                        name=tool_func._tool_name,
                        description=tool_func._tool_description,
                        function=tool_func,
                        args_model=tool_func._args_model
                    )
                else:
                    print(f"Warning: function {tool_func.__name__} is not decorated with @tool")
        
    def run(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        context: Optional[ExecutionContext] = None
    ) -> Tuple[Any, ExecutionContext]:
        """
        Run the ReAct loop to answer the user query
        """
        history = chat_history or []
        if context is None:
            context = ExecutionContext(user_query=query, max_iterations=self.max_iterations)
        
        print(f"\n🚀 Starting Generic ReAct Agent for query: '{query}'")
        
        for iteration in range(1, self.max_iterations + 1):
            context.current_iteration = iteration
            print(f"🔄 Iteration {iteration}/{self.max_iterations}")
            
            # 1. Reasoning & Acting (get next action)
            messages = self._prepare_messages(query, history, context)
            
            response = self.function_caller.call_with_tools(
                messages=messages,
                tools=self.registry.to_openai_format()
            )
            
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # No tool call, assume the LLM provided a direct answer or final thought
                content = response.get("content", "")
                if content:
                    print("✅ LLM provided a direct answer.")
                    return self._finalize_response(content, context)
                else:
                    print("⚠️ LLM returned empty response without tool calls. Retrying...")
                    continue

            # 2. Execution (Action)
            # For simplicity, we execute the first tool call in the ReAct loop
            tool_call = tool_calls[0]
            tool_name = tool_call["name"]
            arguments = tool_call["arguments"]
            
            print(f"🔧 Calling Tool: {tool_name}({arguments})")
            result = self.executor.execute(tool_name, arguments)
            
            # 3. Observation
            context.add_tool_call(tool_name, arguments, result)
            
            if not result.get("success"):
                print(f"❌ Tool Error: {result.get('error')}")
            else:
                print(f"📝 Tool Result obtained.")

            # 4. ReAct Analysis (Explicit Reasoning Step)
            if self.max_iterations > 1:
                print("🧠 Analisando progresso com ReAct...")
                analysis = self._analyze_progress(query, tool_name, arguments, result, context)
                print(f"🤔 Decisão ReAct: {analysis.decision} | Razão: {analysis.reasoning}")
                
                if analysis.decision == "continue":
                     # Information is sufficient
                     print("✅ Informação suficiente. Gerando resposta final.")
                     # Synthesize final answer from ALL context
                     final_content = self._synthesize_final_answer(query, context)
                     return self._finalize_response(final_content, context)
                
                elif analysis.decision in ["retry_with_refinement", "call_different_tool"]:
                     # Update query for next iteration if provided
                     if analysis.refined_query:
                         print(f"🔄 Refinando query para: '{analysis.refined_query}'")
                         query = analysis.refined_query
                         # Add this refinement to history so LLM knows why it's changing
                         messages.append({"role": "assistant", "content": f"Thought: {analysis.reasoning}. I need to refine the query."})
                         messages.append({"role": "user", "content": f"Proceed with refined query: {analysis.refined_query}"})
                     else:
                         print("⚠️ Sem query refinada, continuando com a mesma...")
                
                elif analysis.decision == "insufficient_data":
                     print("❌ Dados insuficientes/impossível responder.")
                     final_content = self._synthesize_final_answer(query, context)
                     return self._finalize_response(final_content, context)

        # If max iterations reached without final answer, try to synthesize from context
        print(f"\n⚠️ Max iterations ({self.max_iterations}) reached. Synthesizing final answer from context...")
        final_content = self._synthesize_final_answer(query, context)
        return self._finalize_response(final_content, context)

    def _synthesize_final_answer(self, query: str, context: ExecutionContext) -> str:
        """
        Synthesize a final answer using ALL accumulated context
        """
        observations = []
        if context.tool_calls:
            for tc in context.tool_calls:
                res = tc["result"]
                observations.append(f"Tool: {tc['tool_name']}\nArguments: {tc['arguments']}\nResult: {res.get('result', res.get('error'))}")
        
        if not observations:
            return "No information gathered."
            
        obs_text = "\n\n".join(observations)
        
        synthesis_prompt = f"""You are a helpful assistant. Synthesize a comprehensive answer to the user's query based on the accumulated tool results.
        
USER QUERY: {context.user_query} (Current Refinement: {query})

GATHERED INFORMATION:
{obs_text}

INSTRUCTIONS:
- Answer the ORIGINAL user query fully.
- Combine information from ALL tool calls.
- If the query had multiple parts, ensure ALL parts are answered.
- If information is missing for any part, explicitly state what is missing.
- Do NOT mention "tools" or "iterations" in the final answer, just provide the specific information.

FINAL ANSWER:"""
        
        try:
             response = self.llm.chat([
                 {"role": "system", "content": self.system_prompt},
                 {"role": "user", "content": synthesis_prompt}
             ])
             return response
        except Exception as e:
             return f"Error synthesizing answer: {e}"

    def _prepare_messages(self, query: str, history: List[Dict[str, str]], context: ExecutionContext) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add history
        messages.extend(history[-5:]) # Last 5 messages
        
        # Add current context summary if any tools were called
        if context.tool_calls:
            observation_text = "OBSERVATIONS FROM PREVIOUS TOOL CALLS:\n"
            for tc in context.tool_calls:
                res = tc["result"]
                # Provide a clearer observation for the LLM
                observation_text += f"Thought: I called {tc['tool_name']} with {tc['arguments']}\n"
                observation_text += f"Observation: {res.get('result', res.get('error'))}\n\n"
            
            messages.append({"role": "system", "content": observation_text})
        
        messages.append({"role": "user", "content": query})
        return messages

    def _parse_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not response or not isinstance(response, dict):
            return []
            
        tool_calls = response.get("tool_calls") or []
        parsed = []
        for tc in tool_calls:
            if "function" in tc:
                func = tc["function"]
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {}
                parsed.append({"name": func["name"], "arguments": args})
        return parsed

    def _finalize_response(self, content: str, context: ExecutionContext) -> Tuple[Any, ExecutionContext]:
        """Finalize and validate the response"""
        if self.response_model:
            print(f"🎯 Validating response against {self.response_model.__name__}...")
            # Get required fields from the model
            schema_fields = list(self.response_model.model_fields.keys())
            
            format_prompt = f"""You are a formatter. Based on the following information, generate a JSON object matching this schema.
            
REQUIRED FIELDS: {schema_fields}

You should populate the fields based on the thought process and tool results below, and the final answer.

CONTEXT FROM TOOLS:
{self._get_tool_context_string(context)}

FINAL ANSWER CONTENT:
{content}

Respond ONLY with the raw JSON object. No explanation, no markdown and no other text."""
            
            structured_response = self.llm.chat([{"role": "user", "content": format_prompt}])
            try:
                # Basic JSON cleaning
                clean_json = structured_response.strip()
                if "```json" in clean_json:
                    clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_json:
                    clean_json = clean_json.split("```")[1].split("```")[0].strip()
                
                parsed_data = json.loads(clean_json)
                final_answer = self.response_model(**parsed_data)
                return final_answer, context
            except Exception as e:
                print(f"❌ Validation failed: {e}. Returning raw content in metadata.")
                return AgentResponse(answer=content, metadata={"validation_error": str(e)}), context
        
        return AgentResponse(answer=content), context

    def _analyze_progress(self, original_query: str, tool_name: str, tool_args: Any, tool_result: Any, context: ExecutionContext) -> ReActAnalysis:
        """
        Analyze the result of a tool call to decide the next step.
        """
        # Create a simplified view of context for the analyzer
        prompt = f"""You are a ReAct manager. Analyze the following interaction and determine the next step.
        
ORIGINAL USER QUERY: "{original_query}"

LAST TOOL CALLED: {tool_name}
ARGUMENTS: {tool_args}
RESULT: {str(tool_result)[:2000]} # Truncated if too long

ACCUMULATED CONTEXT:
{self._get_tool_context_string(context)}

DECISION OPTIONS:
- "continue": ALL parts of the ORIGINAL query are fully answered. STOP tools and generate the final answer.
- "retry_with_refinement": The tool failed OR only answered PART of the query. We need to continue with a REFINED query for the missing part.
- "call_different_tool": The tool was not helpful, try a different one.
- "insufficient_data": We have tried everything and cannot find the answer. Stop.

CRITICAL RULES:
1. If the query consists of multiple questions (e.g. "What is X AND what is Y?"), you MUST NOT choose "continue" until BOTH are answered.
2. If only one part is answered, choose "retry_with_refinement" and refine the query to ask for the MISSING part.
3. "continue" means FINISH/STOP. Do not use it if you plan to do more work.

Respond with a JSON object matching the ReActAnalysis schema:
{{
    "decision": "continue/retry_with_refinement/call_different_tool/insufficient_data",
    "reasoning": "Explanation of why...",
    "refined_query": "New query if retrying...",
    "suggested_tool": "Name of tool if switching..."
}}"""

        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            
            # Clean JSON
            clean_json = response.strip()
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0].strip()
                
            data = json.loads(clean_json)
            return ReActAnalysis(**data)
        except Exception as e:
            print(f"⚠️ Error in ReAct analysis: {e}. Defaulting to 'retry' with same query.")
            return ReActAnalysis(decision="retry_with_refinement", reasoning=f"Error analyzing: {e}", refined_query=original_query)

    def _get_tool_context_string(self, context: ExecutionContext) -> str:
        """Format tool calls for the prompt"""
        if not context.tool_calls:
            return "No tools were used."
        
        output = []
        for tc in context.tool_calls:
            res = tc["result"]
            output.append(f"Tool '{tc['tool_name']}' with args {tc['arguments']} returned: {res.get('result', res.get('error'))}")
        return "\n".join(output)
