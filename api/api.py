"""
API FastAPI with ReAct Agent
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse
import json
import uuid
import sys
import os
from typing import AsyncGenerator
from dotenv import load_dotenv
from api_utils import _sync_to_async_generator

# Load environment variables
load_dotenv()

# Adiciona pasta raiz ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api_schemas import ChatRequest, ChatResponse, ProcessPDFRequest, ProcessPDFResponse
from embedding_manager.embedding_manager import EmbeddingManager
from agents.rag_agent_hfsm import RAGAgentFSMStreaming
from agents.rag_agent_v3 import RAGAgentV3
from pdf_pipeline.pdf_processor import PDFProcessor

# Initialize FastAPI
app = FastAPI(
    title="ReAct Agent API",
    description="API de chat com ReAct Agent usando FunctionGemma",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global components
print("🚀 Initializing components...")

# Embedding manager
embedding_manager = EmbeddingManager(
    embedding_model="qwen3-embedding:0.6b",
    qdrant_url="http://localhost:6333",
    collection_name="rag_api"
)

# Initialize collection if necessary
try:
    embedding_manager.initialize_collection(recreate=False)
    print("✅ Qdrant collection initialized")
except Exception as e:
    print(f"⚠️  Warning initializing collection: {e}")

# PDF Processor
pdf_processor = PDFProcessor(embedding_manager)
print("\n✅ PDF Processor initialized!")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Agent API",
        "version": "2.0.0",
        "models": {
            "tool_caller": "gemma3:1b",
            "embeddings": "qwen3-embedding:0.6b",
            "response_generator": "gemma3:1b"
        },
        "endpoints": {
            "/stream": "POST - Chat with streaming",
            "/chat": "POST - Chat without streaming",
            "/health": "GET - Health check",
            "/documents": "POST - Add documents",
            "/process_pdf": "POST - Process PDF"
        }
    }


@app.get("/health")
async def health():
    """Health check"""
    try:
        # Check providers
        tool_caller_ok = rag_agent.tool_caller.is_available()
        response_gen_ok = rag_agent.response_generator.is_available()
        
        # Check collection
        collection_info = embedding_manager.get_collection_info()
        
        return {
            "status": "healthy" if (tool_caller_ok and response_gen_ok) else "degraded",
            "components": {
                "functiongemma": tool_caller_ok,
                "qwen3": response_gen_ok,
                "qdrant": "error" not in collection_info
            },
            "collection": {
                "name": collection_info.get("name", embedding_manager.collection_name),
                "documents": collection_info.get("points_count", 0)
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/stream")
async def stream_chat(request: ChatRequest):
    """
    Chat endpoint with streaming using a Hierarchical Finite State Machine (HFSM) Agent with native tool-calling capabilities.
    """

    conversation_id = request.conversation_id or str(uuid.uuid4())

    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            # -----------------------------
            # 1. Process chat history
            # -----------------------------
            chat_history = []
            if request.chat_history:
                history_dicts = [msg.model_dump() for msg in request.chat_history]
                chat_history = history_dicts[-6:]  # last 3 turns

            # -----------------------------
            # 2. Initialize STREAMING agent
            # -----------------------------
            rag_agent = RAGAgentFSMStreaming(
                embedding_manager=embedding_manager,
                model="xiaomi/mimo-v2-flash:free"
            )

            # -----------------------------
            # 3. Run FSM in threadpool
            # -----------------------------
            def run_agent():
                return rag_agent.run_stream(
                    query=request.message,
                    chat_history=chat_history
                )

            token_stream, context = await run_in_threadpool(run_agent)

            # -----------------------------
            # 4. Stream tokens
            # -----------------------------
            async for token in _sync_to_async_generator(token_stream):
                yield json.dumps({
                    "type": "token",
                    "content": token
                })

            # -----------------------------
            # 5. Final metadata event
            # -----------------------------
            yield json.dumps({
                "type": "metadata",
                "content": "",
                "metadata": {
                    "conversation_id": conversation_id,
                    "sources_used": context.get_memory("sources_used"),
                    "confidence": context.get_memory("confidence"),
                    "usage": context.get_memory("total_usage"),
                    "context": context.model_dump(mode="json")
                }
            })

        except Exception as e:
            yield json.dumps({
                "type": "error",
                "content": str(e),
                "metadata": {}
            })

    return EventSourceResponse(generate_stream())

@app.post("/strem_fsm")
async def stream_fsm(request: ChatRequest):
    """
    Endpoint chat with streaming via Server-Sent Events

    This endpoint uses the Finite State Machine (FSM) agent with streaming.
    
    Args:
        request: ChatRequest with user message
        
    Returns:
        EventSourceResponse with streaming chunks
    """
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        """Generate streaming chunks"""
        try:
            # Process chat history (limits to last 3 interactions = 6 messages)
            chat_history = []
            if request.chat_history:
                print(f"📜 [DEBUG] Chat history received: {len(request.chat_history)} messages")
                # Convert ChatMessage to dict
                history_dicts = [msg.model_dump() for msg in request.chat_history]
                # Limita às últimas 6 mensagens (3 interações user+assistant)
                chat_history = history_dicts[-6:]
                print(f"📜 [DEBUG] Chat history processado: {len(chat_history)} mensagens")
            else:
                print("📜 [DEBUG] Nenhum chat history recebido")
            
            # Create new instance of RAG Agent V2 with history
            rag_agent = RAGAgentFSM(
                embedding_manager=embedding_manager,
                model="xiaomi/mimo-v2-flash:free",
                max_steps=10 
            )
            
            
            
            # Execute RAG Agent V2 with history
            # Run blocking synchronous code in a threadpool to avoid blocking the event loop
            response, contexto = await run_in_threadpool(
                rag_agent.run,
                query=request.message,
                chat_history=chat_history
            )
            
            # Response chunk
            yield json.dumps({
                "type": "system",
                "content": response.answer
            })
            
            # Final chunk
            yield json.dumps({
                "type": "metadata",
                "content": "",
                "metadata": {
                    "conversation_id": conversation_id,
                    "sources_used": response.sources_used,
                    "confidence": response.confidence,
                    "context": contexto.model_dump(mode='json')
                }
            })
            
        except Exception as e:
            # Error chunk
            yield json.dumps({
                "type": "error",
                "content": str(e),
                "metadata": {}
            })
    
    return EventSourceResponse(generate_stream())

@app.post("/stream_react")
async def stream_react(request: ChatRequest):
    """
    Endpoint chat with ReAct Agent
    
    Args:
        request: ChatRequest with user message
        
    Returns:
        EventSourceResponse with streaming chunks
    """
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        """Generate streaming chunks"""
        try:
            # Process chat history
            chat_history = []
            if request.chat_history:
                history_dicts = [msg.model_dump() for msg in request.chat_history]
                chat_history = history_dicts[-6:]

            # Create new instance of RAG Agent V3
            rag_agent = RAGAgentV3(
                embedding_manager=embedding_manager,
                model="xiaomi/mimo-v2-flash:free",
                max_iterations=5 
            )
            
            # Execute ReAct Agent
            def run_agent():
                return rag_agent.run(
                    query=request.message,
                    chat_history=chat_history
                )

            # blocked run (ReAct is currently sync and non-streaming)
            response, context = await run_in_threadpool(run_agent)
            
            # Response chunk
            yield json.dumps({
                "type": "system",
                "content": response.answer
            })
            
            # Final chunk
            yield json.dumps({
                "type": "metadata",
                "content": "",
                "metadata": {
                    "conversation_id": conversation_id,
                    "sources_used": response.sources_used,
                    "confidence": response.confidence,
                    "context": context.model_dump(mode='json')
                }
            })
            
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "content": str(e),
                "metadata": {}
            })
    
    return EventSourceResponse(generate_stream())

@app.post("/documents")
async def add_documents(documents: list[str], metadatas: list[dict] = None):
    """
    Add documents to knowledge base
    
    Args:
        documents: List of document texts
        metadatas: Optional list of metadata
        
    Returns:
        Status of the operation
    """
    try:
        embedding_manager.add_documents(documents, metadatas)
        
        collection_info = embedding_manager.get_collection_info()
        
        return {
            "status": "success",
            "documents_added": len(documents),
            "total_documents": collection_info.get("points_count", 0)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_pdf", response_model=ProcessPDFResponse)
async def process_pdf(request: ProcessPDFRequest):
    """
    Process a PDF and add to knowledge base
    
    Args:
        request: ProcessPDFRequest with PDF path and parameters
        
    Returns:
        ProcessPDFResponse with processing statistics
    """
    try:
        # Check if file exists
        if not os.path.exists(request.pdf_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.pdf_path}")
        
        # Processa PDF
        stats = pdf_processor.process_pdf(
            pdf_path=request.pdf_path,
            max_tokens=request.max_tokens
        )
        
        return ProcessPDFResponse(
            status="success",
            pdf_file=stats["pdf_file"],
            total_chunks=stats["total_chunks"],
            total_tokens=stats["total_tokens"],
            avg_tokens_per_chunk=stats["avg_tokens_per_chunk"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("🚀 Starting RAG Agent API...")
    print("=" * 70)
    print("\n📍 Endpoints disponíveis:")
    print("   • http://localhost:8000/")
    print("   • http://localhost:8000/stream (POST)")
    print("   • http://localhost:8000/chat (POST)")
    print("   • http://localhost:8000/documents (POST)")
    print("   • http://localhost:8000/process_pdf (POST)")
    print("   • http://localhost:8000/health")
    print("\n🤖 Models:")
    print("   • Tool Caller: xiaomi/mimo-v2-flash:free (OpenRouter)")
    print("   • Embeddings: qwen3-embedding:0.6b")
    print("   • Response: xiaomi/mimo-v2-flash:free (OpenRouter)")
    print("\n" + "=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
