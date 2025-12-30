import ollama
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid


class EmbeddingManager:
    """Gerenciador de embeddings usando Ollama e Qdrant"""
    
    def __init__(
        self,
        embedding_model: str = "qwen3-embedding:0.6b",
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "rag_api"
    ):
        """
        Inicializa o gerenciador de embeddings
        
        Args:
            embedding_model: Modelo para gerar embeddings
            qdrant_url: URL do servidor Qdrant
            collection_name: Nome da coleção no Qdrant
        """
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        
        # Inicializa cliente Qdrant
        self.qdrant_client = QdrantClient(url=qdrant_url)
        
        # Dimensão do embedding (será detectada automaticamente)
        self.embedding_dim: Optional[int] = None
        
    def _get_embedding(self, text: str) -> List[float]:
        """
        Gera embedding para um texto usando Ollama
        
        Args:
            text: Texto para gerar embedding
            
        Returns:
            Lista de floats representando o embedding
        """
        try:
            response = ollama.embed(
                model=self.embedding_model,
                input=text
            )
            
            # Ollama retorna embeddings como lista
            embedding = response['embeddings'][0]
            
            # Detecta dimensão na primeira vez
            if self.embedding_dim is None:
                self.embedding_dim = len(embedding)
                print(f"📊 Dimensão do embedding detectada: {self.embedding_dim}")
            
            return embedding
            
        except Exception as e:
            raise Exception(f"Erro ao gerar embedding: {str(e)}")
    
    def initialize_collection(self, recreate: bool = False):
        """
        Inicializa a coleção no Qdrant
        
        Args:
            recreate: Se True, recria a coleção (apaga dados existentes)
        """
        # Gera um embedding de teste para detectar dimensão
        if self.embedding_dim is None:
            test_embedding = self._get_embedding("test")
        
        # Verifica se coleção existe
        collections = self.qdrant_client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)
        
        if collection_exists and recreate:
            print(f"🗑️  Deletando coleção existente: {self.collection_name}")
            self.qdrant_client.delete_collection(self.collection_name)
            collection_exists = False
        
        if not collection_exists:
            print(f"📦 Criando coleção: {self.collection_name}")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
        else:
            print(f"✅ Coleção já existe: {self.collection_name}")
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """
        Adiciona documentos à coleção
        
        Args:
            documents: Lista de textos dos documentos
            metadatas: Lista opcional de metadados para cada documento
        """
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        if len(documents) != len(metadatas):
            raise ValueError("Número de documentos e metadados deve ser igual")
        
        print(f"📝 Adicionando {len(documents)} documentos...")
        
        points = []
        for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
            # Gera embedding
            embedding = self._get_embedding(doc)
            
            # Cria ponto
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "content": doc,
                    **metadata
                }
            )
            points.append(point)
            
            if (i + 1) % 10 == 0:
                print(f"  Processados {i + 1}/{len(documents)} documentos...")
        
        # Insere no Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"✅ {len(documents)} documentos adicionados com sucesso!")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Busca documentos similares à query
        
        Args:
            query: Texto da consulta
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de dicionários com content, score e metadata
        """
        # DEBUG
        print(f"🔎 [DEBUG] EmbeddingManager.search() chamado:")
        print(f"   collection_name: {self.collection_name}")
        print(f"   query: '{query}'")
        print(f"   top_k: {top_k}")
        
        # Gera embedding da query
        query_embedding = self._get_embedding(query)
        print(f"   embedding gerado: dim={len(query_embedding)}")
        
        # Busca no Qdrant usando query_points
        from qdrant_client.models import SearchRequest
        
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k
        ).points
        
        print(f"   resultados do Qdrant: {len(results)} pontos")
        
        # DEBUG: Mostra primeiro resultado
        if results:
            print(f"   [DEBUG] Primeiro resultado:")
            print(f"      payload keys: {list(results[0].payload.keys())}")
            print(f"      score: {results[0].score}")
        
        # Formata resultados
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result.payload.get("content", ""),
                "score": result.score,
                "metadata": {k: v for k, v in result.payload.items() if k != "content"}
            })
        
        print(f"   [DEBUG] Formatted results: {len(formatted_results)} items")
        if formatted_results:
            print(f"   [DEBUG] Primeiro item formatado tem content: {bool(formatted_results[0]['content'])}")
        
        return formatted_results
    
    def get_collection_info(self) -> Dict:
        """
        Retorna informações sobre a coleção
        
        Returns:
            Dicionário com informações da coleção
        """
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            return {"error": str(e)}
