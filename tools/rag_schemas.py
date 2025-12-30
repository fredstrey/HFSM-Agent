from pydantic import BaseModel, Field
from typing import List, Optional


# =========================
# SCHEMAS PARA TOOLS
# =========================

class SearchArgs(BaseModel):
    """Argumentos para busca semântica em documentos"""
    query: str = Field(..., description="Texto da consulta para buscar documentos relevantes")
    top_k: int = Field(default=3, description="Número de documentos mais relevantes a retornar")


class DocumentChunk(BaseModel):
    """Representa um chunk de documento retornado pela busca"""
    content: str = Field(..., description="Conteúdo do chunk do documento")
    score: float = Field(..., description="Score de similaridade (0-1)")
    metadata: dict = Field(default_factory=dict, description="Metadados do documento")


class SearchResponse(BaseModel):
    """Resposta da tool de busca semântica"""
    query: str = Field(..., description="Query original")
    results: List[DocumentChunk] = Field(..., description="Lista de documentos encontrados")
    total_found: int = Field(..., description="Total de documentos encontrados")


# =========================
# SCHEMA PARA RESPOSTA FINAL
# =========================

class RAGResponse(BaseModel):
    """Resposta final validada do agente RAG"""
    answer: str = Field(..., description="Resposta à pergunta do usuário")
    sources_used: List[str] = Field(
        default_factory=list,
        description="Lista de fontes/documentos usados para gerar a resposta"
    )
    confidence: Optional[str] = Field(
        None,
        description="Nível de confiança na resposta (high/medium/low)"
    )


# =========================
# SCHEMAS PARA PDF PROCESSING
# =========================

class ProcessPDFRequest(BaseModel):
    """Request para processar PDF"""
    pdf_path: str = Field(..., description="Caminho para o arquivo PDF")
    max_tokens: int = Field(default=500, description="Máximo de tokens por chunk")
    overlap_tokens: int = Field(default=50, description="Tokens de sobreposição entre chunks")


class ProcessPDFResponse(BaseModel):
    """Resposta do processamento de PDF"""
    status: str = Field(..., description="Status do processamento")
    pdf_file: str = Field(..., description="Nome do arquivo PDF")
    total_chunks: int = Field(..., description="Total de chunks criados")
    total_tokens: int = Field(..., description="Total de tokens processados")
    avg_tokens_per_chunk: float = Field(..., description="Média de tokens por chunk")
