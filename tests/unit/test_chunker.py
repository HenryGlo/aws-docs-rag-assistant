"""Unit tests for DocumentChunker."""
from src.ingestion.chunker import DocumentChunker
from src.ingestion.loader import Document


class TestDocumentChunker:
  """Tests for DocumentChunker behavior."""

  def test_chunk_document_creates_multiple_chunks(self):
      """Long documents should produce multiple chunks."""
      chunker = DocumentChunker(chunk_size_tokens=100, chunk_overlap_tokens=10)

      long_content = "This is a sample sentence about AWS Lambda. " * 100
      doc = Document(
          content=long_content,
          source_url="https://example.com/lambda",
          title="AWS Lambda",
          metadata={},
      )

      chunks = chunker.chunk_document(doc)

      assert len(chunks) > 1
      assert all(chunk.source_url == "https://example.com/lambda" for chunk in chunks)
      assert all(chunk.title == "AWS Lambda" for chunk in chunks)
      assert chunks[0].chunk_index == 0
      assert chunks[-1].chunk_index == len(chunks) - 1

  def test_chunk_metadata_is_consistent(self):
      """All chunks should share document-level metadata."""
      chunker = DocumentChunker(chunk_size_tokens=100, chunk_overlap_tokens=10)

      doc = Document(
          content="AWS Lambda is great. " * 50,
          source_url="https://example.com/test",
          title="Test Doc",
          metadata={},
      )

      chunks = chunker.chunk_document(doc)

      total = chunks[0].total_chunks
      assert all(chunk.total_chunks == total for chunk in chunks)
      assert all("token_count" in chunk.metadata for chunk in chunks)

  def test_token_counting_is_reasonable(self):
      """Token counting should produce expected ranges."""
      chunker = DocumentChunker()
      text = "AWS Lambda is a serverless compute service."

      token_count = chunker.count_tokens(text)

      # ~9 words, expect 8-12 tokens
      assert 5 < token_count < 20

  def test_chunk_id_is_unique_within_document(self):
      """Each chunk should have a unique chunk_id."""
      chunker = DocumentChunker(chunk_size_tokens=50, chunk_overlap_tokens=5)

      doc = Document(
          content="Sentence about AWS. " * 30,
          source_url="https://example.com/x",
          title="X",
          metadata={},
      )

      chunks = chunker.chunk_document(doc)
      chunk_ids = [c.chunk_id for c in chunks]
      assert len(chunk_ids) == len(set(chunk_ids))
