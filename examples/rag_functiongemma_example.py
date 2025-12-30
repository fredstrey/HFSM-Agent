"""
Exemplo completo de RAG Agent com FunctionGemma
"""
from rag_tools import EmbeddingManager, RAGAgent


def main():
    """Função principal"""
    
    print("=" * 70)
    print("🎯 RAG Agent - FunctionGemma + qwen3-embedding + gemma3:1b")
    print("=" * 70)
    
    # ==========================================
    # 1. CONFIGURAR EMBEDDING MANAGER
    # ==========================================
    
    print("\n📦 Configurando Embedding Manager...")
    embedding_manager = EmbeddingManager(
        embedding_model="qwen3-embedding:0.6b",
        qdrant_url="http://localhost:6333",
        collection_name="rag_documents"
    )
    
    # Inicializa coleção (recria para exemplo limpo)
    embedding_manager.initialize_collection(recreate=True)
    
    # ==========================================
    # 2. ADICIONAR DOCUMENTOS DE EXEMPLO
    # ==========================================
    
    print("\n📚 Adicionando documentos de exemplo...")
    
    documents = [
        "Python é uma linguagem de programação de alto nível, interpretada e de propósito geral. Foi criada por Guido van Rossum e lançada em 1991.",
        "Machine Learning é um subcampo da inteligência artificial que permite que sistemas aprendam e melhorem a partir da experiência sem serem explicitamente programados.",
        "RAG (Retrieval-Augmented Generation) é uma técnica que combina recuperação de informações com geração de texto para criar respostas mais precisas e fundamentadas.",
        "Ollama é uma ferramenta que permite executar modelos de linguagem grandes localmente em seu computador, sem necessidade de conexão com a internet.",
        "Qdrant é um banco de dados vetorial de código aberto otimizado para busca por similaridade e aplicações de IA.",
        "FunctionGemma é um modelo especializado em tool calling, capaz de decidir quando e como usar ferramentas disponíveis para responder perguntas.",
        "Embeddings são representações vetoriais de texto que capturam o significado semântico das palavras e frases em um espaço de alta dimensão.",
        "A temperatura em modelos de linguagem controla a aleatoriedade das respostas. Valores baixos (0.1-0.3) geram respostas mais determinísticas, enquanto valores altos (0.7-1.0) geram respostas mais criativas."
    ]
    
    metadatas = [
        {"source": "python_docs.txt", "topic": "programming"},
        {"source": "ml_guide.txt", "topic": "ai"},
        {"source": "rag_paper.txt", "topic": "ai"},
        {"source": "ollama_docs.txt", "topic": "tools"},
        {"source": "qdrant_docs.txt", "topic": "database"},
        {"source": "functiongemma_docs.txt", "topic": "ai"},
        {"source": "embeddings_guide.txt", "topic": "ai"},
        {"source": "llm_params.txt", "topic": "ai"}
    ]
    
    embedding_manager.add_documents(documents, metadatas)
    
    # Verifica coleção
    info = embedding_manager.get_collection_info()
    print(f"\n✅ Coleção criada:")
    print(f"   Nome: {info.get('name', embedding_manager.collection_name)}")
    print(f"   Documentos: {info.get('points_count', 0)}")
    
    # ==========================================
    # 3. CRIAR RAG AGENT
    # ==========================================
    
    print("\n🤖 Criando RAG Agent...")
    agent = RAGAgent(
        embedding_manager=embedding_manager,
        tool_caller_model="gemma3:1b",
        response_model="gemma3:1b"
    )
    
    print("✅ RAG Agent criado!")
    print("   Tool Caller: gemma3:1b")
    print("   Embeddings: qwen3-embedding:0.6b")
    print("   Response Generator: gemma3:1b")
    
    # ==========================================
    # 4. TESTES
    # ==========================================
    
    # Teste 1: Pergunta que requer busca
    print("\n" + "=" * 70)
    print("🔹 Teste 1: Pergunta sobre RAG (deve buscar documentos)")
    print("-" * 70)
    
    query1 = "O que é RAG e como funciona?"
    response1 = agent.run(query1)
    
    print(f"\n✅ Resposta:")
    print(f"   {response1.answer}")
    print(f"   Fontes: {', '.join(response1.sources_used) if response1.sources_used else 'Nenhuma'}")
    print(f"   Confiança: {response1.confidence}")
    
    # Teste 2: Pergunta sobre FunctionGemma
    print("\n" + "=" * 70)
    print("🔹 Teste 2: Pergunta sobre FunctionGemma")
    print("-" * 70)
    
    agent.reset()
    query2 = "Para que serve o FunctionGemma?"
    response2 = agent.run(query2)
    
    print(f"\n✅ Resposta:")
    print(f"   {response2.answer}")
    print(f"   Fontes: {', '.join(response2.sources_used) if response2.sources_used else 'Nenhuma'}")
    print(f"   Confiança: {response2.confidence}")
    
    # Teste 3: Pergunta sobre embeddings
    print("\n" + "=" * 70)
    print("🔹 Teste 3: Pergunta sobre embeddings")
    print("-" * 70)
    
    agent.reset()
    query3 = "O que são embeddings e para que servem?"
    response3 = agent.run(query3)
    
    print(f"\n✅ Resposta:")
    print(f"   {response3.answer}")
    print(f"   Fontes: {', '.join(response3.sources_used) if response3.sources_used else 'Nenhuma'}")
    print(f"   Confiança: {response3.confidence}")
    
    # Teste 4: Pergunta casual (não deve buscar)
    print("\n" + "=" * 70)
    print("🔹 Teste 4: Pergunta casual (não deve buscar documentos)")
    print("-" * 70)
    
    agent.reset()
    query4 = "Olá, como você está?"
    response4 = agent.run(query4)
    
    print(f"\n✅ Resposta:")
    print(f"   {response4.answer}")
    print(f"   Fontes: {', '.join(response4.sources_used) if response4.sources_used else 'Nenhuma'}")
    print(f"   Confiança: {response4.confidence}")
    
    print("\n" + "=" * 70)
    print("✅ Exemplo concluído!")
    print("=" * 70)


if __name__ == "__main__":
    main()
