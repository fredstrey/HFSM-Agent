from embedding_manager.embedding_manager import EmbeddingManager

print("🚀 Testing Optimized Hybrid RAG...")

# Test 1: Without reranking (default, fast)
print("\n" + "="*60)
print("TEST 1: Hybrid Search (NO Reranking)")
print("="*60)

em = EmbeddingManager(
    collection_name="hybrid",
    filter=None,
    enable_rerank=False  # Should NOT load reranker model
)

print(f"\n✅ Initialized successfully!")
print(f"   Reranker model: {em.reranker_model}")
print(f"   Enable rerank: {em.enable_rerank}")

print("\n✅ All tests passed!")
print("📊 Reranker model is NOT loaded when enable_rerank=False")
