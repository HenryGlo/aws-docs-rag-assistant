"""End-to-end RAG pipeline: retrieve -> generate -> respond.

This is the main entry point for query handling. Used by:
- scripts/ask.py (interactive CLI)
- src/api/main.py (FastAPI endpoint, Day 7)
- src/evaluation/evaluator.py (RAGAs evaluation, Day 9)

The pipeline is initialized once and reused per query for efficiency
(avoids reloading models and rebuilding indexes on every request).
"""
import pickle
from dataclasses import dataclass

from src.config import settings
from src.embeddings.embedder import Embedder
from src.generation.generator import AnswerGenerator
from src.logger import logger
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import VectorStore


@dataclass
class RAGResponse:
  """Complete response from the RAG pipeline."""

  question: str
  answer: str
  sources: list[dict]
  retrieved_chunks: list[dict]
  metrics: dict


class RAGPipeline:
  """End-to-end RAG: retrieve -> generate -> respond.

  Initialization loads:
  - Embedding model (~80 MB)
  - Vector store connection
  - Persisted chunks for BM25
  - Anthropic API client

  Reuse the same instance across multiple queries for efficiency.
  """

  def __init__(self):
      logger.info("initializing_rag_pipeline")

      self.embedder = Embedder()
      self.vector_store = VectorStore()

      # Build BM25 from persisted chunks (saved by scripts/index.py)
      chunks_path = settings.processed_dir / "chunks.pkl"
      if not chunks_path.exists():
          raise FileNotFoundError(
              f"Chunks not found at {chunks_path}. "
              f"Run `python -m scripts.index` first."
          )
      with open(chunks_path, "rb") as f:
          chunks = pickle.load(f)

      self.bm25 = BM25Retriever()
      self.bm25.fit(chunks)

      self.retriever = HybridRetriever(
          vector_store=self.vector_store,
          bm25_retriever=self.bm25,
          embedder=self.embedder,
      )

      self.generator = AnswerGenerator()

      logger.info(
          "rag_pipeline_initialized",
          indexed_chunks=self.vector_store.count(),
      )

  def query(self, question: str, top_k: int | None = None) -> RAGResponse:
      """Run a complete RAG query end-to-end.

      Args:
          question: User's natural language question.
          top_k: Number of chunks to retrieve. Defaults to settings.top_k.

      Returns:
          RAGResponse with the answer, sources, retrieved chunks, and metrics.
      """
      top_k = top_k or settings.top_k

      # 1. Retrieve relevant chunks (hybrid: semantic + BM25 + RRF)
      retrieved = self.retriever.search(question, top_k=top_k)

      # 2. Generate answer using Claude with the retrieved context
      result = self.generator.generate(question, retrieved)

      return RAGResponse(
          question=question,
          answer=result.answer,
          sources=result.sources,
          retrieved_chunks=retrieved,
          metrics={
              "input_tokens": result.input_tokens,
              "output_tokens": result.output_tokens,
              "model": result.model,
              "num_sources": len(retrieved),
          },
    )
