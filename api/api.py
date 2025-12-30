"""
API FastAPI com RAG Agent usando FunctionGemma
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import json
import uuid
import sys
import os
from typing import AsyncGenerator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Adiciona pasta raiz ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api_schemas import ChatRequest, ChatResponse, ProcessPDFRequest, ProcessPDFResponse
from embedding_manager.embedding_manager import EmbeddingManager
from agents.rag_agent_v2 import RAGAgentV2
from pdf_pipeline.pdf_processor import PDFProcessor

# Inicializa FastAPI
app = FastAPI(
    title="RAG Agent API",
    description="API de chat com RAG Agent usando FunctionGemma",
    version="2.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa componentes globais
print("🚀 Inicializando componentes...")

# Embedding manager
embedding_manager = EmbeddingManager(
    embedding_model="qwen3-embedding:0.6b",
    qdrant_url="http://localhost:6333",
    collection_name="rag_api"
)

# Inicializa coleção se necessário
try:
    embedding_manager.initialize_collection(recreate=False)
    print("✅ Coleção Qdrant inicializada")
except Exception as e:
    print(f"⚠️  Aviso ao inicializar coleção: {e}")

# PDF Processor
pdf_processor = PDFProcessor(embedding_manager)
print("\n✅ PDF Processor inicializado!")


@app.get("/")
async def root():
    """Endpoint raiz"""
    return {
        "message": "RAG Agent API",
        "version": "2.0.0",
        "models": {
            "tool_caller": "gemma3:1b",
            "embeddings": "qwen3-embedding:0.6b",
            "response_generator": "gemma3:1b"
        },
        "endpoints": {
            "/stream": "POST - Chat com streaming",
            "/chat": "POST - Chat sem streaming",
            "/health": "GET - Health check",
            "/documents": "POST - Adicionar documentos",
            "/process_pdf": "POST - Processar PDF"
        }
    }


@app.get("/health")
async def health():
    """Health check"""
    try:
        # Verifica providers
        tool_caller_ok = rag_agent.tool_caller.is_available()
        response_gen_ok = rag_agent.response_generator.is_available()
        
        # Verifica coleção
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
    Endpoint de chat com streaming via Server-Sent Events
    
    Args:
        request: ChatRequest com mensagem do usuário
        
    Returns:
        EventSourceResponse com chunks de streaming
    """
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        """Gera chunks de streaming"""
        try:
            # Processa histórico de chat (limita às últimas 3 interações = 6 mensagens)
            chat_history = []
            if request.chat_history:
                print(f"📜 [DEBUG] Chat history recebido: {len(request.chat_history)} mensagens")
                # Converte ChatMessage para dict
                history_dicts = [msg.model_dump() for msg in request.chat_history]
                # Limita às últimas 6 mensagens (3 interações user+assistant)
                chat_history = history_dicts[-6:]
                print(f"📜 [DEBUG] Chat history processado: {len(chat_history)} mensagens")
            else:
                print("📜 [DEBUG] Nenhum chat history recebido")
            
            # Cria nova instância do RAG Agent V2 com histórico
            rag_agent = RAGAgentV2(
                embedding_manager=embedding_manager,
                tool_caller_model="xiaomi/mimo-v2-flash:free",
                response_model="xiaomi/mimo-v2-flash:free",
                context_model="xiaomi/mimo-v2-flash:free"
            )
            
            
            # Executa RAG Agent V2 com histórico
            response, contexto = rag_agent.run(
                query=request.message,
                chat_history=chat_history
            )
            
            # Chunk de resposta
            yield json.dumps({
                "type": "system",
                "content": response.answer
            })
            
            # Chunk final
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
            # Chunk de erro
            yield json.dumps({
                "type": "error",
                "content": str(e),
                "metadata": {}
            })
    
    return EventSourceResponse(generate_stream())

@app.post("/documents")
async def add_documents(documents: list[str], metadatas: list[dict] = None):
    """
    Adiciona documentos à base de conhecimento
    
    Args:
        documents: Lista de textos dos documentos
        metadatas: Lista opcional de metadados
        
    Returns:
        Status da operação
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
    Processa um PDF e adiciona à base de conhecimento
    
    Args:
        request: ProcessPDFRequest com caminho do PDF e parâmetros
        
    Returns:
        ProcessPDFResponse com estatísticas do processamento
    """
    try:
        # Verifica se arquivo existe
        if not os.path.exists(request.pdf_path):
            raise HTTPException(status_code=404, detail=f"Arquivo não encontrado: {request.pdf_path}")
        
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
    print("🚀 Iniciando servidor RAG Agent API...")
    print("=" * 70)
    print("\n📍 Endpoints disponíveis:")
    print("   • http://localhost:8000/")
    print("   • http://localhost:8000/stream (POST)")
    print("   • http://localhost:8000/chat (POST)")
    print("   • http://localhost:8000/documents (POST)")
    print("   • http://localhost:8000/process_pdf (POST)")
    print("   • http://localhost:8000/health")
    print("\n🤖 Modelos:")
    print("   • Tool Caller: xiaomi/mimo-v2-flash:free (OpenRouter)")
    print("   • Embeddings: qwen3-embedding:0.6b")
    print("   • Response: xiaomi/mimo-v2-flash:free (OpenRouter)")
    print("\n" + "=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
