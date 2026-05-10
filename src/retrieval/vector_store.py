"""Vector store implementation using ChromaDB.

Why ChromaDB for this project:
- Zero infrastructure overhead (file-based persistence)
- Easy local development and testing
- Built-in metadata filtering
- Free and open source
- Sufficient performance for thousands of chunks

For production at scale (millions of chunks), migration paths:
- Pinecone: managed, low latency, paid
- Weaviate: hybrid search built-in, self-hosted or cloud
- pgvector: PostgreSQL extension, good if already using Postgres
- Qdrant: high performance, self-hosted or cloud
"""
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from src.ingestion.chunker import Chunk
from src.logger import logger


class VectorStore:
  """Manages chunk embeddings in ChromaDB with cosine similarity."""

  def __init__(
      self,
      persist_dir: Path | None = None,
      collection_name: str | None = None,
  ):
      persist_dir = persist_dir or settings.chroma_persist_dir
      collection_name = collection_name or settings.collection_name

      persist_dir.mkdir(parents=True, exist_ok=True)

      self.client = chromadb.PersistentClient(
          path=str(persist_dir),
          settings=ChromaSettings(anonymized_telemetry=False),
      )

      # Cosine space is standard for sentence-transformers embeddings.
      # ChromaDB returns "distance" not "similarity": lower = more similar.
      self.collection = self.client.get_or_create_collection(
          name=collection_name,
          metadata={"hnsw:space": "cosine"},
      )

      logger.info(
          "vector_store_initialized",
          persist_dir=str(persist_dir),
          collection=collection_name,
          existing_count=self.collection.count(),
      )

  def add_chunks(
      self, chunks: list[Chunk], embeddings: list[list[float]]
  ) -> None:
      """Add chunks with their pre-computed embeddings to the collection."""
      if len(chunks) != len(embeddings):
          raise ValueError(
              f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
          )

      ids = [chunk.chunk_id for chunk in chunks]
      documents = [chunk.content for chunk in chunks]
      metadatas = [chunk.metadata for chunk in chunks]

      self.collection.add(
          ids=ids,
          embeddings=embeddings,
          documents=documents,
          metadatas=metadatas,
      )

      logger.info("chunks_added_to_store", count=len(chunks))

  def search(
      self,
      query_embedding: list[float],
      top_k: int = 5,
      filter_metadata: dict | None = None,
  ) -> list[dict]:
      """Search for the top_k most similar chunks.

      Args:
          query_embedding: The embedded query vector.
          top_k: Number of results to return.
          filter_metadata: Optional metadata filter (e.g., {"title": "AWS Lambda"}).

      Returns:
          List of dicts with chunk_id, content, metadata, and distance.
          Lower distance = more similar (cosine distance, not similarity).
      """
      results = self.collection.query(
          query_embeddings=[query_embedding],
          n_results=top_k,
          where=filter_metadata,
      )

      formatted = []
      if results["ids"] and results["ids"][0]:
          for i in range(len(results["ids"][0])):
              formatted.append({
                  "chunk_id": results["ids"][0][i],
                  "content": results["documents"][0][i],
                  "metadata": results["metadatas"][0][i],
                  "distance": results["distances"][0][i],
              })

      logger.info(
          "search_completed",
          top_k=top_k,
          results_count=len(formatted),
      )

      return formatted

  def count(self) -> int:
      """Return number of chunks in the store."""
      return self.collection.count()

  def reset(self) -> None:
      """Delete all chunks. Useful for re-indexing from scratch."""
      collection_name = self.collection.name
      self.client.delete_collection(collection_name)
      self.collection = self.client.get_or_create_collection(
          name=collection_name,
          metadata={"hnsw:space": "cosine"},
      )
      logger.warning("vector_store_reset")
