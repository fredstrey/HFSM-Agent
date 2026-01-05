import ollama
from typing import List, Dict, Optional, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, SparseVectorParams, SparseIndexParams, SparseVector
import uuid
from fastembed import SparseTextEmbedding

class EmbeddingManager:
    """Embedding manager using Ollama and Qdrant with Hybrid Search and Optional Reranking"""
    
    def __init__(
        self,
        embedding_model: str = "qwen3-embedding:0.6b",
        reranker_model: str = "dengcao/Qwen3-Reranker-0.6B:Q8_0",
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "rag_api",
        filter: Optional[str] = "Constituição da República Federativa do Brasil",
        enable_rerank: bool = False  # Disabled by default (slow on CPU)
    ):
        """
        Initialize the embedding manager
        
        Args:
            embedding_model: Model for generating dense embeddings
            reranker_model: Model for reranking results (only loaded if enable_rerank=True)
            qdrant_url: Qdrant server URL
            collection_name: Collection name in Qdrant
            filter: Default source filter for searches
            enable_rerank: Enable reranking (requires more memory/CPU)
        """
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model if enable_rerank else None
        self.collection_name = collection_name
        self.enable_rerank = enable_rerank
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=qdrant_url)
        
        # Initialize Sparse Embedding Model (BM25)
        print("🔧 Initializing BM25 sparse model...")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        print("✅ BM25 model ready!")
        
        # Embedding dimension (will be detected automatically)
        self.embedding_dim: Optional[int] = None
        self.filter: Optional[str] = filter
        
    def _get_embedding(self, text: str) -> List[float]:
        """Generate dense embedding using Ollama"""
        try:
            response = ollama.embed(
                model=self.embedding_model,
                input=text
            )
            embedding = response['embeddings'][0]
            
            if self.embedding_dim is None:
                self.embedding_dim = len(embedding)
                print(f"📊 Embedding dimension detected: {self.embedding_dim}")
            
            return embedding
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")

    def _get_sparse_embedding(self, text: str) -> SparseVector:
        """Generate sparse embedding (BM25) using fastembed"""
        try:
            sparse_embedding = list(self.sparse_model.embed([text]))[0]
            return SparseVector(
                indices=sparse_embedding.indices.tolist(),
                values=sparse_embedding.values.tolist()
            )
        except Exception as e:
            raise Exception(f"Error generating sparse embedding: {str(e)}")

    def initialize_collection(self, recreate: bool = False):
        """Initialize the collection in Qdrant with Dense and Sparse support"""
        try:
            if self.embedding_dim is None:
                print("🔍 Detecting embedding dimension...")
                self._get_embedding("test")
                print(f"✅ Dimension: {self.embedding_dim}")
                
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if collection_exists and recreate:
                print(f"🗑️  Deleting existing collection: {self.collection_name}")
                self.qdrant_client.delete_collection(self.collection_name)
                collection_exists = False
            
            if not collection_exists:
                print(f"📦 Creating collection: {self.collection_name} with Hybrid Search support")
                try:
                    self.qdrant_client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config={
                            "semantic": VectorParams(
                                size=self.embedding_dim,
                                distance=Distance.COSINE
                            )
                        },
                        sparse_vectors_config={
                            "bm25": SparseVectorParams(
                                index=SparseIndexParams(on_disk=True)
                            )
                        }
                    )
                    print("✅ Collection created successfully!")
                except Exception as e:
                    print(f"❌ Error creating collection with sparse vectors: {e}")
                    print("⚠️ Trying without sparse vectors...")
                    self.qdrant_client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config={
                            "semantic": VectorParams(
                                size=self.embedding_dim,
                                distance=Distance.COSINE
                            )
                        }
                    )
                    print("✅ Collection created (dense only)")
            else:
                print(f"✅ Collection already exists: {self.collection_name}")
        except Exception as e:
            print(f"❌ Fatal error in initialize_collection: {e}")
            import traceback
            traceback.print_exc()
            raise
            
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """Add documents with both Dense and Sparse vectors"""
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        print(f"📝 Adding {len(documents)} documents...")
        
        points = []
        for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
            dense_vector = self._get_embedding(doc)
            sparse_vector = self._get_sparse_embedding(doc)
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "semantic": dense_vector,
                    "bm25": sparse_vector
                },
                payload={
                    "content": doc,
                    **metadata
                }
            )
            points.append(point)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(documents)} documents...")
        
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"✅ {len(documents)} documents added successfully!")

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range using min-max normalization"""
        if not scores or len(scores) == 1:
            return [1.0] * len(scores)
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Rerank documents using Ollama (only if enable_rerank=True)"""
        if not self.enable_rerank:
            return documents
            
        print(f"🔄 Reranking {len(documents)} documents with {self.reranker_model}...")
        
        # Unload embedding model to free memory
        try:
            print(f"💾 Unloading embedding model...")
            ollama.generate(model=self.embedding_model, prompt="", keep_alive=0)
        except:
            pass
        
        scored_docs = []
        total_docs = len(documents)
        
        for idx, doc in enumerate(documents, 1):
            print(f"   📄 Reranking document {idx}/{total_docs}...")
            
            prompt = f"Query: {query}\nDocument: {doc['content']}\nRelevant:"
            
            try:
                response = ollama.generate(
                    model=self.reranker_model,
                    prompt=prompt,
                    stream=False
                )
                output_text = response['response'].strip()
                
                try:
                    score = float(output_text)
                except ValueError:
                    score = 1.0 if "yes" in output_text.lower() else 0.0 if "no" in output_text.lower() else 0.5
                
                doc['rerank_score'] = score
                scored_docs.append(doc)
                print(f"      ✅ Score: {score:.4f}")
                
            except Exception as e:
                print(f"⚠️ Error reranking: {e}")
                doc['rerank_score'] = 0.0
                scored_docs.append(doc)
        
        scored_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
        return scored_docs

    def search(self, query: str, top_k: int = 5, filter: Optional[str] = None) -> List[Dict]:
        """Hybrid Search (Dense + Sparse) with optional Reranking"""
        if filter is None:
            filter = self.filter

        print(f"🔎 Hybrid Search for: '{query}'")
        
        query_dense = self._get_embedding(query)
        query_sparse = self._get_sparse_embedding(query)
        
        query_filter = None
        if filter:
            query_filter = Filter(must=[FieldCondition(key="source", match=MatchValue(value=filter))])

        limit_candidates = top_k * 3
        
        # Search dense and sparse
        dense_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_dense,
            using="semantic",
            limit=limit_candidates,
            query_filter=query_filter
        ).points
        
        sparse_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_sparse,
            using="bm25",
            limit=limit_candidates,
            query_filter=query_filter
        ).points

        # Normalize scores
        dense_scores = [res.score for res in dense_results]
        sparse_scores = [res.score for res in sparse_results]
        
        normalized_dense = self._normalize_scores(dense_scores)
        normalized_sparse = self._normalize_scores(sparse_scores)
        
        dense_score_map = {dense_results[i].id: normalized_dense[i] for i in range(len(dense_results))}
        sparse_score_map = {sparse_results[i].id: normalized_sparse[i] for i in range(len(sparse_results))}
        
        # Combine results
        combined = {}
        
        for res in dense_results:
            if res.id not in combined:
                combined[res.id] = {
                    "payload": res.payload,
                    "scores": {"dense": 0.0, "sparse": 0.0}
                }
            combined[res.id]["scores"]["dense"] = dense_score_map[res.id]
        
        for res in sparse_results:
            if res.id not in combined:
                combined[res.id] = {
                    "payload": res.payload,
                    "scores": {"dense": 0.0, "sparse": 0.0}
                }
            combined[res.id]["scores"]["sparse"] = sparse_score_map[res.id]
        
        # Calculate hybrid score
        candidates = []
        for id, data in combined.items():
            hybrid_score = (data["scores"]["dense"] * 0.7) + (data["scores"]["sparse"] * 0.3)
            candidates.append({
                "content": data["payload"].get("content", ""),
                "metadata": {k: v for k, v in data["payload"].items() if k != "content"},
                "hybrid_score": hybrid_score,
                "dense_score": data["scores"]["dense"],
                "sparse_score": data["scores"]["sparse"]
            })
            
        candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
        candidates = candidates[:limit_candidates]
        
        # Optional reranking
        if self.enable_rerank:
            print(f"🔄 Reranking enabled")
            reranked_results = self._rerank(query, candidates)
            return reranked_results[:top_k]
        else:
            print(f"⚡ Reranking disabled - returning hybrid results")
            for c in candidates:
                c['rerank_score'] = c['hybrid_score']
            return candidates[:top_k]
    
    def get_collection_info(self) -> Dict:
        """Returns information about the collection"""
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "name": info.config.params.vectors,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            return {"error": str(e)}
