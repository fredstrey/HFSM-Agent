"""
Contexto de execução para RAG Agent
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ExecutionContext(BaseModel):
    """
    Contexto de execução que mantém estado durante o processamento
    """
    # Query original do usuário (imutável)
    user_query: str = Field(..., description="Pergunta original do usuário")
    
    # Histórico de chat
    chat_history: List[Dict[str, str]] = Field(default_factory=list, description="Histórico de conversa")
    
    # Metadados da execução
    conversation_id: Optional[str] = Field(None, description="ID da conversação")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp da execução")
    
    # Estado da execução
    current_iteration: int = Field(default=0, description="Iteração atual do loop")
    max_iterations: int = Field(default=3, description="Máximo de iterações")
    
    # Contexto recuperado
    retrieved_documents: List[Dict[str, Any]] = Field(default_factory=list, description="Documentos recuperados")
    sources_used: List[str] = Field(default_factory=list, description="Fontes utilizadas")
    
    # Tool calls realizadas
    tool_calls_history: List[Dict[str, Any]] = Field(default_factory=list, description="Histórico de tool calls")
    
    # Flags de estado
    is_out_of_scope: bool = Field(default=False, description="Se a pergunta está fora do escopo")
    has_context: bool = Field(default=False, description="Se contexto foi recuperado")
    
    class Config:
        arbitrary_types_allowed = True
    
    def add_tool_call(self, tool_name: str, arguments: Dict[str, Any], result: Any = None):
        """Adiciona uma tool call ao histórico"""
        self.tool_calls_history.append({
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "iteration": self.current_iteration
        })
    
    def add_document(self, content: str, score: float, metadata: Dict[str, Any]):
        """Adiciona um documento recuperado"""
        self.retrieved_documents.append({
            "content": content,
            "score": score,
            "metadata": metadata
        })
        self.has_context = True
        
        # Adiciona fonte se disponível
        if "source" in metadata:
            source = metadata["source"]
            if source not in self.sources_used:
                self.sources_used.append(source)
    
    def mark_out_of_scope(self):
        """Marca a pergunta como fora do escopo"""
        self.is_out_of_scope = True
    
    def get_context_summary(self) -> str:
        """Retorna um resumo do contexto"""
        return f"""
Contexto de Execução:
- Query: {self.user_query}
- Iteração: {self.current_iteration}/{self.max_iterations}
- Documentos recuperados: {len(self.retrieved_documents)}
- Fontes: {len(self.sources_used)}
- Tool calls: {len(self.tool_calls_history)}
- Fora do escopo: {self.is_out_of_scope}
"""
