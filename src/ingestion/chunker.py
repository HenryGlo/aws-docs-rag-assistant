"""Document chunking strategies for RAG.

Why chunking matters: LLMs and embedding models have context limits.
We split documents into semantically coherent pieces that:
- Fit in the embedding model's context window
- Are small enough for precise retrieval
- Are large enough to preserve meaning
- Overlap slightly to avoid breaking concepts at boundaries

Strategy: Recursive character splitting with token-aware sizing.
Splits on natural boundaries (paragraphs, sentences) before falling
back to word-level or character-level splits.
"""
from dataclasses import dataclass
from pathlib import Path

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.loader import Document
from src.logger import logger


@dataclass
class Chunk:
  """A chunk of a document with metadata for retrieval."""

  content: str
  chunk_id: str
  source_url: str
  title: str
  chunk_index: int
  total_chunks: int
  token_count: int
  metadata: dict


class DocumentChunker:
  """Splits documents into chunks suitable for embedding and retrieval.

  Default config:
  - 512 tokens per chunk (fits most embedding models comfortably)
  - 50 token overlap (preserves context across boundaries)
  - Encoder: cl100k_base (used by GPT-4 / Claude tokenization)
  """

  def __init__(
      self,
      chunk_size_tokens: int = 512,
      chunk_overlap_tokens: int = 50,
      encoding_name: str = "cl100k_base",
  ):
      self.chunk_size_tokens = chunk_size_tokens
      self.chunk_overlap_tokens = chunk_overlap_tokens
      self.encoder = tiktoken.get_encoding(encoding_name)

      # Approximate char-per-token ratio for English text (~4 chars/token).
      # We size by chars because RecursiveCharacterTextSplitter measures
      # length in chars, not tokens. This is a heuristic that works well
      # for English; for multilingual content, use a token-based splitter.
      chars_per_token = 4

      self.splitter = RecursiveCharacterTextSplitter(
          chunk_size=chunk_size_tokens * chars_per_token,
          chunk_overlap=chunk_overlap_tokens * chars_per_token,
          # Order matters: tries each separator until chunks fit.
          separators=["\n\n", "\n", ". ", " ", ""],
          length_function=len,
      )

  def count_tokens(self, text: str) -> int:
      """Count tokens for a given text using tiktoken."""
      return len(self.encoder.encode(text))

  def chunk_document(self, document: Document) -> list[Chunk]:
      """Split a single document into chunks with metadata."""
      text_chunks = self.splitter.split_text(document.content)
      chunks = []

      for i, text in enumerate(text_chunks):
          token_count = self.count_tokens(text)
          chunk_id = f"{document.source_url}#chunk-{i}"

          chunks.append(
              Chunk(
                  content=text,
                  chunk_id=chunk_id,
                  source_url=document.source_url,
                  title=document.title,
                  chunk_index=i,
                  total_chunks=len(text_chunks),
                  token_count=token_count,
                  metadata={
                      "source": document.source_url,
                      "title": document.title,
                      "chunk_index": i,
                      "total_chunks": len(text_chunks),
                      "token_count": token_count,
                  },
              )
          )

      logger.info(
          "document_chunked",
          url=document.source_url,
          num_chunks=len(chunks),
          avg_tokens=sum(c.token_count for c in chunks) // max(len(chunks), 1),
      )
      return chunks

  def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
      """Chunk multiple documents."""
      all_chunks = []
      for doc in documents:
          all_chunks.extend(self.chunk_document(doc))

      logger.info(
          "chunking_complete",
          num_documents=len(documents),
          num_chunks=len(all_chunks),
      )
      return all_chunks

  def load_documents_from_disk(self, raw_dir: Path) -> list[Document]:
      """Load previously saved documents from disk.

      Files were saved by AWSDocsLoader.save_documents() in this format:
          URL: <url>
          TITLE: <title>
          ---
          <content>
      """
      documents = []

      for file_path in sorted(raw_dir.glob("*.txt")):
          content = file_path.read_text(encoding="utf-8")
          lines = content.split("\n")

          url_line = lines[0].replace("URL: ", "")
          title_line = lines[1].replace("TITLE: ", "")
          # Skip "---" separator on line index 2, content starts at line 3
          body = "\n".join(lines[3:])

          documents.append(
              Document(
                  content=body,
                  source_url=url_line,
                  title=title_line,
                  metadata={"source": url_line, "title": title_line},
              )
          )

      logger.info("documents_loaded_from_disk", count=len(documents))
      return documents
