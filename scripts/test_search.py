"""Quick interactive search test against the vector store.

Use this to manually evaluate retrieval quality before adding the
generation step (Day 6).

Usage:
  python -m scripts.test_search
"""
from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore


def main():
  embedder = Embedder()
  store = VectorStore()

  print(f"\n🔍 Vector store has {store.count()} chunks indexed.\n")

  if store.count() == 0:
      print("❌ No chunks indexed.")
      print("   Run `python -m scripts.ingest` then `python -m scripts.index` first.\n")
      return

  print("Enter queries to test semantic search. Type 'quit' to exit.\n")

  while True:
      query = input("\n❓ Query: ").strip()
      if query.lower() in ("quit", "exit", "q", ""):
          break

      query_embedding = embedder.embed_text(query)
      results = store.search(query_embedding, top_k=3)

      for i, r in enumerate(results, 1):
          print(f"\n--- Result {i} (distance: {r['distance']:.4f}) ---")
          print(f"Source: {r['metadata']['source']}")
          print(f"Title: {r['metadata']['title']}")
          print(f"Content: {r['content'][:300]}...")
      print("\n" + "=" * 60)


if __name__ == "__main__":
  main()
