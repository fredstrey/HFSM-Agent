"""
Exemplo de RAG usando o framework
"""
from pydantic import BaseModel, Field
from typing import List
from rag_tools import tool, FunctionAgent
from rag_tools.providers import OllamaProvider
from rag_tools.rag_schemas import EmbeddingManager


# ==========================================
# RESPONSE MODEL
# ==========================================

class RAGResponse(BaseModel):
    """Resposta do agente RAG"""
    answer: str = Field(..., description="Resposta para o usuário")
    sources_used: List[str] = Field(default_factory=list, description="Fontes consultadas")
    confidence: str = Field(default="medium", description="Nível de confiança")


# ==========================================
# TOOLS COM DECORATOR
# ==========================================

# Inicializa embedding manager globalmente
embedding_manager = EmbeddingManager(
    embedding_model="qwen3-embedding:0.6b",
    collection_name="rag_demo"
)


@tool(name="search_documents", description="Busca documentos relevantes na base de conhecimento")
def search_documents(query: str, top_k: int = 3) -> dict:
    """
    Busca documentos similares à query
    
    Args:
        query: Termos de busca
        top_k: Número de resultados
    
    Returns:
        Documentos encontrados
    """
    try:
        results = embedding_manager.search(query, top_k)
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        return {
            "query": query,
            "error": str(e),
            "results": []
        }


# ==========================================
# MAIN
# ==========================================

def main():
    """Função principal"""
    
    print("=" * 70)
    print("🎯 Exemplo RAG - Framework com Decorators")
    print("=" * 70)
    
    # Provider
    provider = OllamaProvider(model="gemma3:1b", temperature=0.3)
    
    if not provider.is_available():
        print("❌ Ollama não está disponível!")
        return
    
    print("✅ Ollama conectado!")
    
    # Cria agente
    agent = FunctionAgent(
        llm_provider=provider,
        response_model=RAGResponse,
        system_prompt="Você é um assistente que responde perguntas sobre Python usando uma base de conhecimento."
    )
    
    print(f"✅ Agente RAG criado com {len(agent.registry.list())} tools")
    
    # Testes
    queries = [
        "Quem criou o Python?",
        "O que é o pip?",
        "Quais são os frameworks web em Python?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 70}")
        print(f"🔹 Teste {i}/{len(queries)}: {query}")
        print("-" * 70)
        
        agent.reset()
        response = agent.run(query)
        
        print(f"\n✅ Resposta:")
        print(f"   {response.answer}")
        if response.sources_used:
            print(f"   📚 Fontes: {', '.join(response.sources_used)}")
        print(f"   🎯 Confiança: {response.confidence}")
    
    print(f"\n{'=' * 70}")
    print("✅ Exemplo RAG concluído!")
    print("=" * 70)


if __name__ == "__main__":
    main()
