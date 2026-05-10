"""Quick interactive search test using hybrid retrieval.

Usage:
    python -m scripts.test_search
"""
import pickle

from src.config import settings
from src.embeddings.embedder import Embedder
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import VectorStore


def main():
    # Load components
    embedder = Embedder()
    store = VectorStore()

    # Load persisted chunks for BM25
    chunks_path = settings.processed_dir / "chunks.pkl"
    if not chunks_path.exists():
        print(f"❌ Chunks not found at {chunks_path}")
        print("   Run `python -m scripts.index` first.\n")
        return

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    bm25 = BM25Retriever()
    bm25.fit(chunks)

    hybrid = HybridRetriever(
        vector_store=store,
        bm25_retriever=bm25,
        embedder=embedder,
    )

    print(f"\n🔍 Hybrid retrieval ready: {store.count()} chunks indexed.")
    print("   Combines semantic search (embeddings) + BM25 (keywords) with RRF.\n")
    print("Enter queries to test. Type 'quit' to exit.\n")

    while True:
        query = input("\n❓ Query: ").strip()
        if query.lower() in ("quit", "exit", "q", ""):
            break

        results = hybrid.search(query, top_k=3)

        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} (RRF score: {r['rrf_score']:.4f}) ---")
            print(f"Source: {r['metadata']['source']}")
            print(f"Title: {r['metadata']['title']}")
            print(f"Content: {r['content'][:300]}...")
        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
