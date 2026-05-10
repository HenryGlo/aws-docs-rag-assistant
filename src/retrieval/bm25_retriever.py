"""BM25 keyword-based retrieval.

BM25 (Best Matching 25) is a probabilistic ranking function that scores
documents based on term frequency and inverse document frequency. It's the
standard baseline for keyword search and complements semantic embeddings.

Why BM25 in addition to semantic search:
- Exact matching for technical terms (VPC, IAM, API names)
- No model training or embeddings needed (fast, deterministic)
- Strong baseline that often catches what embeddings miss
- Robust on out-of-domain queries (rare technical jargon)

For larger corpora (>100k documents), use a persistent inverted index
like Elasticsearch, OpenSearch, or Tantivy instead of in-memory BM25.
"""
import re

from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk
from src.logger import logger


class BM25Retriever:
  """Keyword-based retrieval using BM25Okapi.

  Build the index once with `fit()`, then call `search()` for each query.
  The index lives in memory; the chunks list is what makes search work.
  """

  def __init__(self):
      self.bm25: BM25Okapi | None = None
      self.chunks: list[Chunk] = []

  @staticmethod
  def _tokenize(text: str) -> list[str]:
      """Tokenize text: lowercase, alphanumeric only, split on whitespace.

      Simple tokenization works well for English technical content. For
      multilingual or domain-specific corpora, consider:
      - Snowball stemmer for English root forms
      - spaCy for proper sentence/token segmentation
      - Custom tokenizers for code (preserves operators, special chars)
      """
      text = text.lower()
      text = re.sub(r"[^a-z0-9\s]", " ", text)
      return text.split()

  def fit(self, chunks: list[Chunk]) -> None:
      """Build the BM25 index from chunks. Call once before search."""
      self.chunks = chunks
      tokenized_corpus = [self._tokenize(chunk.content) for chunk in chunks]
      self.bm25 = BM25Okapi(tokenized_corpus)
      logger.info("bm25_index_built", num_chunks=len(chunks))

  def search(self, query: str, top_k: int = 5) -> list[dict]:
      """Search for top_k chunks using BM25 scoring.

      Returns chunks ranked by BM25 score (higher score = more relevant).
      """
      if not self.bm25:
          raise RuntimeError("BM25 index not built. Call fit() first.")

      tokenized_query = self._tokenize(query)
      scores = self.bm25.get_scores(tokenized_query)

      # Get indices of top_k highest scores
      top_indices = sorted(
          range(len(scores)),
          key=lambda i: scores[i],
          reverse=True,
      )[:top_k]

      results = []
      for idx in top_indices:
          chunk = self.chunks[idx]
          results.append({
              "chunk_id": chunk.chunk_id,
              "content": chunk.content,
              "metadata": chunk.metadata,
              "bm25_score": float(scores[idx]),
          })

      return results
