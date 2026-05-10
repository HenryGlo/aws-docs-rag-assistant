"""Hybrid retrieval combining semantic search and BM25.

Uses Reciprocal Rank Fusion (RRF) to merge results from both retrievers.

Why RRF over weighted score fusion:
- Score normalization is hard: semantic distances and BM25 scores have
different scales, distributions, and interpretations.
- RRF only uses ranks, not raw scores. Robust and parameter-light.
- Empirically performs as well or better than tuned weighted fusion.

Reference:
  Cormack, Clarke, Buettcher. "Reciprocal Rank Fusion outperforms
  Condorcet and individual rank learning methods." (SIGIR 2009)

Formula:
  rrf_score(d) = Σ 1 / (k + rank_i(d))

Where rank_i(d) is the rank of document d in retriever i's results,
and k is a constant (typically 60) that dampens contributions from
lower-ranked results.
"""
from src.embeddings.embedder import Embedder
from src.logger import logger
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_store import VectorStore


class HybridRetriever:
  """Combines semantic search and BM25 with Reciprocal Rank Fusion.

  Retrieves more candidates than top_k from each retriever (typically 3x)
  so the fusion has enough signal to merge effectively.
  """

  def __init__(
      self,
      vector_store: VectorStore,
      bm25_retriever: BM25Retriever,
      embedder: Embedder,
      rrf_k: int = 60,
  ):
      self.vector_store = vector_store
      self.bm25 = bm25_retriever
      self.embedder = embedder
      self.rrf_k = rrf_k

  def search(self, query: str, top_k: int = 5) -> list[dict]:
      """Run both retrievers and merge results with RRF.

      Returns top_k chunks with rrf_score added to each result.
      """
      # Pull more candidates than top_k from each retriever.
      # 3x is a heuristic balance: enough overlap for good fusion,
      # not so much that we waste computation.
      candidates_per_retriever = top_k * 3

      # Semantic search (embeddings + cosine similarity in ChromaDB)
      query_embedding = self.embedder.embed_text(query)
      semantic_results = self.vector_store.search(
          query_embedding=query_embedding,
          top_k=candidates_per_retriever,
      )

      # Keyword search (BM25)
      bm25_results = self.bm25.search(
          query=query,
          top_k=candidates_per_retriever,
      )

      # Merge with Reciprocal Rank Fusion
      merged = self._reciprocal_rank_fusion(
          result_lists=[semantic_results, bm25_results],
          top_k=top_k,
      )

      logger.info(
          "hybrid_search_complete",
          query_length=len(query),
          semantic_count=len(semantic_results),
          bm25_count=len(bm25_results),
          merged_count=len(merged),
      )

      return merged

  def _reciprocal_rank_fusion(
      self,
      result_lists: list[list[dict]],
      top_k: int,
  ) -> list[dict]:
      """Merge ranked results from multiple retrievers using RRF.

      For each chunk, sum 1/(k + rank) across all retrievers it appears in.
      Higher score = better consensus across retrievers.
      """
      scores: dict[str, float] = {}
      chunk_data: dict[str, dict] = {}

      for results in result_lists:
          for rank, result in enumerate(results):
              chunk_id = result["chunk_id"]
              # rank is 0-indexed in our list, but RRF formula uses 1-indexed
              contribution = 1.0 / (self.rrf_k + rank + 1)
              scores[chunk_id] = scores.get(chunk_id, 0.0) + contribution

              # Keep first occurrence's data (semantic results have distance,
              # BM25 results have bm25_score; first one wins)
              if chunk_id not in chunk_data:
                  chunk_data[chunk_id] = result

      # Sort by RRF score descending
      sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

      merged = []
      for chunk_id in sorted_ids[:top_k]:
          entry = chunk_data[chunk_id].copy()
          entry["rrf_score"] = scores[chunk_id]
          merged.append(entry)

      return merged
