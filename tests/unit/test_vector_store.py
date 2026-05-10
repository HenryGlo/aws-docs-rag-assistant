"""Unit tests for VectorStore."""
import shutil

import pytest

from src.ingestion.chunker import Chunk
from src.retrieval.vector_store import VectorStore


@pytest.fixture
def temp_store(tmp_path):
  """Create a temporary vector store for testing."""
  store = VectorStore(
      persist_dir=tmp_path / "test_chroma",
      collection_name="test_collection",
  )
  yield store
  # Cleanup
  if (tmp_path / "test_chroma").exists():
      shutil.rmtree(tmp_path / "test_chroma")


@pytest.fixture
def sample_chunks():
  """Create sample chunks for testing."""
  return [
      Chunk(
          content="AWS Lambda is a serverless compute service.",
          chunk_id="https://example.com/lambda#chunk-0",
          source_url="https://example.com/lambda",
          title="AWS Lambda",
          chunk_index=0,
          total_chunks=1,
          token_count=10,
          metadata={
              "source": "https://example.com/lambda",
              "title": "AWS Lambda",
          },
      ),
      Chunk(
          content="Amazon S3 provides object storage in the cloud.",
          chunk_id="https://example.com/s3#chunk-0",
          source_url="https://example.com/s3",
          title="Amazon S3",
          chunk_index=0,
          total_chunks=1,
          token_count=10,
          metadata={
              "source": "https://example.com/s3",
              "title": "Amazon S3",
          },
      ),
  ]


@pytest.fixture
def fake_embeddings():
  """Create simple 3-dim embeddings for testing."""
  return [
      [1.0, 0.0, 0.0],  # First chunk
      [0.0, 1.0, 0.0],  # Second chunk
  ]


class TestVectorStore:
  """Tests for VectorStore behavior."""

  def test_add_and_count(self, temp_store, sample_chunks, fake_embeddings):
      """Adding chunks should increase count."""
      assert temp_store.count() == 0
      temp_store.add_chunks(sample_chunks, fake_embeddings)
      assert temp_store.count() == 2

  def test_search_returns_most_similar_first(
      self, temp_store, sample_chunks, fake_embeddings
  ):
      """Searching should return the closest chunk first."""
      temp_store.add_chunks(sample_chunks, fake_embeddings)

      # Query close to the first embedding [1.0, 0.0, 0.0]
      results = temp_store.search([0.9, 0.1, 0.0], top_k=2)

      assert len(results) == 2
      # First result should be the Lambda chunk (closer to query vector)
      assert "Lambda" in results[0]["content"]

  def test_mismatched_lengths_raises(self, temp_store, sample_chunks):
      """Mismatched chunks/embeddings should raise ValueError."""
      with pytest.raises(ValueError):
          temp_store.add_chunks(sample_chunks, [[1.0, 0.0, 0.0]])  # only 1

  def test_reset_clears_store(
      self, temp_store, sample_chunks, fake_embeddings
  ):
      """Reset should empty the store."""
      temp_store.add_chunks(sample_chunks, fake_embeddings)
      assert temp_store.count() == 2

      temp_store.reset()
      assert temp_store.count() == 0
