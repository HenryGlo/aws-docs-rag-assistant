"""Unit tests for hybrid retrieval and RRF fusion."""
from unittest.mock import Mock

from src.retrieval.hybrid_retriever import HybridRetriever


class TestReciprocalRankFusion:
  """Tests for the RRF merging logic."""

  def _make_retriever(self):
      """Create a HybridRetriever with mocked dependencies for unit testing."""
      return HybridRetriever(
          vector_store=Mock(),
          bm25_retriever=Mock(),
          embedder=Mock(),
          rrf_k=60,
      )

  def test_rrf_combines_two_lists(self):
      """RRF should merge two ranked lists into one."""
      retriever = self._make_retriever()

      list_a = [
          {"chunk_id": "a", "content": "A", "metadata": {}},
          {"chunk_id": "b", "content": "B", "metadata": {}},
      ]
      list_b = [
          {"chunk_id": "b", "content": "B", "metadata": {}},
          {"chunk_id": "c", "content": "C", "metadata": {}},
      ]

      merged = retriever._reciprocal_rank_fusion([list_a, list_b], top_k=3)

      assert len(merged) == 3
      chunk_ids = [r["chunk_id"] for r in merged]
      # 'b' appears in both lists, should rank highest
      assert chunk_ids[0] == "b"

  def test_rrf_includes_score_in_result(self):
      """Each merged result should have rrf_score attached."""
      retriever = self._make_retriever()

      results = [{"chunk_id": "x", "content": "X", "metadata": {}}]
      merged = retriever._reciprocal_rank_fusion([results], top_k=1)

      assert "rrf_score" in merged[0]
      assert merged[0]["rrf_score"] > 0

  def test_rrf_respects_top_k(self):
      """Returned list should not exceed top_k."""
      retriever = self._make_retriever()

      list_a = [
          {"chunk_id": f"a{i}", "content": f"A{i}", "metadata": {}}
          for i in range(10)
      ]

      merged = retriever._reciprocal_rank_fusion([list_a], top_k=3)

      assert len(merged) == 3

  def test_rrf_higher_rank_gets_higher_score(self):
      """Top-ranked items in input should get higher RRF score."""
      retriever = self._make_retriever()

      list_a = [
          {"chunk_id": "first", "content": "1", "metadata": {}},
          {"chunk_id": "second", "content": "2", "metadata": {}},
          {"chunk_id": "third", "content": "3", "metadata": {}},
      ]

      merged = retriever._reciprocal_rank_fusion([list_a], top_k=3)

      # Scores should be strictly decreasing
      scores = [r["rrf_score"] for r in merged]
      assert scores[0] > scores[1] > scores[2]
