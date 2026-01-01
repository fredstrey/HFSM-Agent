"""
Custom States for RAG Agent
============================

Domain-specific states that extend the HFSM without modifying the engine.
"""

import logging
import json
from finitestatemachineAgent.hfsm_agent_async import AsyncHierarchicalState, Transition

logger = logging.getLogger(__name__)


class IntentAnalysisState(AsyncHierarchicalState):
    """
    Analyzes chat history and user intent before routing.
    
    This state:
    1. Receives chat history + current query
    2. Analyzes user intent and context
    3. Prepares a structured todo list
    4. Enhances the query with context
    5. Routes to RouterState with prepared context
    """
    
    def __init__(self, parent, llm):
        super().__init__(parent)
        self.llm = llm
    
    async def handle(self, context):
        logger.info("=" * 80)
        logger.info("🧠 [IntentAnalysis] STARTING - Analyzing user intent...")
        logger.info(f"📝 [IntentAnalysis] Query: {context.user_query[:100]}...")
        logger.info("=" * 80)
        
        try:
            # Get chat history and system instruction
            chat_history = await context.get_memory("chat_history", [])
            system_instruction = await context.get_memory("system_instruction", "")
            current_query = context.user_query
            
            logger.info(f"[IntentAnalysis] Chat history length: {len(chat_history)}")
            logger.info(f"[IntentAnalysis] Building prompt...")
            
            # Build analysis prompt
            messages = [{
                "role": "system",
                "content": f"""{system_instruction}

TAREFA ESPECIAL: Análise de Intenção

Você deve analisar o histórico da conversa e a pergunta atual para:
1. Extrair a intenção real do usuário
2. Identificar contexto relevante do histórico
3. Criar uma todo list estruturada para responder

IMPORTANTE: Retorne APENAS o JSON, sem texto adicional, sem markdown, sem explicações.

Formato JSON obrigatório:
{{
    "intent": "descrição clara da intenção",
    "context_from_history": ["ponto1", "ponto2"],
    "enhanced_query": "pergunta reformulada com contexto",
    "todo_list": ["tarefa1", "tarefa2", "tarefa3"],
    "language": "pt"
}}"""
            }]
            
            # Add chat history
            if chat_history:
                messages.extend(chat_history[-5:])  # Last 5 messages for context
            
            # Add current query
            messages.append({
                "role": "user",
                "content": f"Pergunta atual: {current_query}\n\nRetorne APENAS o JSON da análise, sem texto adicional."
            })
            
            logger.info(f"[IntentAnalysis] Calling LLM with {len(messages)} messages...")
            
        except Exception as e:
            logger.error(f"❌ [IntentAnalysis] FAILED in setup: {e}")
            import traceback
            logger.error(traceback.format_exc())
            await context.set_memory("intent_analyzed", True)
            await context.set_memory("user_language", "pt")
            return Transition(to="RouterState", reason="Intent analysis failed in setup")
        
        try:
            # Call LLM for intent analysis
            response = await self.llm.chat(messages)  # 🔥 Fixed: only takes messages
            
            # 🔥 Extract JSON from response (handle markdown code blocks)
            content = response.get("content", "").strip()
            
            # 🔥 DEBUG: Log raw response BEFORE processing
            logger.info(f"[IntentAnalysis] Raw LLM response (first 300 chars): {content[:300]}...")
            
            # Remove markdown code blocks if present
            if content.startswith("```"):
                # Extract JSON from ```json ... ``` or ``` ... ```
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])  # Remove first and last line
                logger.info(f"[IntentAnalysis] Removed markdown, cleaned content: {content[:200]}...")
            
            analysis = json.loads(content)
            
            # Store analysis in context
            await context.set_memory("intent_analysis", analysis)
            await context.set_memory("todo_list", analysis.get("todo_list", []))
            await context.set_memory("user_language", analysis.get("language", "pt"))
            await context.set_memory("intent_analyzed", True)  # 🔥 Mark as analyzed
            
            # Enhance the query with context
            enhanced_query = analysis.get("enhanced_query", current_query)
            context.user_query = enhanced_query
            
            logger.info(f"✅ [IntentAnalysis] Intent: {analysis.get('intent', 'unknown')}")
            logger.info(f"🌍 [IntentAnalysis] Language: {analysis.get('language', 'unknown')}")
            logger.info(f"📝 [IntentAnalysis] Todo list ({len(analysis.get('todo_list', []))} items):")
            for i, task in enumerate(analysis.get('todo_list', []), 1):
                logger.info(f"   {i}. {task}")
            
            # Log context from history if available
            context_items = analysis.get('context_from_history', [])
            if context_items:
                logger.info(f"📚 [IntentAnalysis] Context from history:")
                for item in context_items:
                    logger.info(f"   - {item}")
            
            logger.info(f"🔄 [IntentAnalysis] Enhanced query: {enhanced_query}")
            
        except Exception as e:
            logger.error(f"❌ [IntentAnalysis] Failed to parse analysis: {e}")
            
            # 🔥 CRITICAL: Set flag even on error to prevent infinite loop
            await context.set_memory("intent_analyzed", True)
            await context.set_memory("user_language", "pt")  # Default
            
            # Log raw response for debugging
            try:
                logger.debug(f"Raw LLM response: {response.get('content', 'N/A')[:500]}")
            except:
                pass
        
        # Route to normal flow (RouterState)
        return Transition(to="RouterState", reason="Intent analyzed, ready for routing")


class ResponseValidatorState(AsyncHierarchicalState):
    """
    Validates answer quality and language after AnswerState.
    
    This state:
    1. Checks if answer is in the same language as the query
    2. Validates completeness based on todo list
    3. Either approves (TerminalState) or retries
    """
    
    def __init__(self, parent, llm):
        super().__init__(parent)
        self.llm = llm
    
    async def handle(self, context):
        logger.info("🔍 [Validator] Checking answer quality...")
        
        answer = await context.get_memory("final_answer", "")
        user_language = await context.get_memory("user_language", "pt")
        todo_list = await context.get_memory("todo_list", [])
        
        # Build validation prompt
        messages = [{
            "role": "system",
            "content": f"""Você é um validador de respostas.

Verifique se a resposta:
1. Está no idioma correto: {user_language}
2. Responde todos os itens da todo list
3. É completa e precisa

Todo list:
{json.dumps(todo_list, indent=2, ensure_ascii=False)}

Retorne JSON:
{{
    "valid": true/false,
    "reason": "motivo se inválida",
    "language_correct": true/false,
    "completeness_score": 0.0-1.0
}}"""
        }, {
            "role": "user",
            "content": f"Resposta a validar:\n\n{answer}"
        }]
        
        try:
            response = await self.llm.chat(messages, context)
            validation = json.loads(response["content"])
            
            if validation.get("valid", False):
                logger.info(f"✅ [Validator] Answer is valid (completeness: {validation.get('completeness_score', 0):.2f})")
                return Transition(to="TerminalState", reason="Answer validated")
            else:
                reason = validation.get("reason", "Unknown")
                logger.warning(f"❌ [Validator] Invalid answer: {reason}")
                return Transition(to="RetryState", reason=f"Validation failed: {reason}")
                
        except Exception as e:
            logger.error(f"❌ [Validator] Validation failed: {e}")
            # Accept answer if validation fails
            return Transition(to="TerminalState", reason="Validation error, accepting answer")
