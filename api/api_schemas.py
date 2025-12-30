from pydantic import BaseModel, Field
from typing import List, Optional


class ChatMessage(BaseModel):
    """Mensagem de chat"""
    role: str = Field(..., description="Papel: user ou assistant")
    content: str = Field(..., description="Conteúdo da mensagem")


class ChatRequest(BaseModel):
    """Request para endpoint de chat"""
    message: str = Field(..., description="Mensagem do usuário")
    conversation_id: Optional[str] = Field(None, description="ID da conversação")
    stream: bool = Field(default=True, description="Se deve usar streaming")
    chat_history: Optional[List[ChatMessage]] = Field(default=None, description="Histórico de chat (últimas 3 interações)")


class ChatStreamChunk(BaseModel):
    """Chunk de streaming"""
    type: str = Field(..., description="Tipo: thinking, tool_call, response, done")
    content: str = Field(default="", description="Conteúdo do chunk")
    metadata: Optional[dict] = Field(None, description="Metadados adicionais")


class ChatResponse(BaseModel):
    """Resposta completa do chat"""
    answer: str = Field(..., description="Resposta final")
    sources_used: List[str] = Field(default_factory=list, description="Fontes usadas")
    confidence: Optional[str] = Field(None, description="Nível de confiança")
    conversation_id: str = Field(..., description="ID da conversação")


class ProcessPDFRequest(BaseModel):
    """Request para processar PDF"""
    pdf_path: str = Field(..., description="Caminho para o arquivo PDF")
    max_tokens: int = Field(default=500, description="Máximo de tokens por chunk")


class ProcessPDFResponse(BaseModel):
    """Resposta do processamento de PDF"""
    status: str = Field(..., description="Status do processamento")
    pdf_file: str = Field(..., description="Nome do arquivo PDF")
    total_chunks: int = Field(..., description="Total de chunks criados")
    total_tokens: int = Field(..., description="Total de tokens processados")
    avg_tokens_per_chunk: float = Field(..., description="Média de tokens por chunk")
