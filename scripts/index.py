"""Full indexing pipeline: load -> chunk -> embed -> store.

Run this after `scripts.ingest` to build the vector index.

Usage:
  python -m scripts.index
"""
from src.config import settings
from src.embeddings.embedder import Embedder
from src.ingestion.chunker import DocumentChunker
from src.logger import logger
from src.retrieval.vector_store import VectorStore


def main():
  """Run the complete indexing pipeline."""
  logger.info("indexing_pipeline_started")

  # 1. Load documents from disk (saved by scripts/ingest.py)
  chunker = DocumentChunker()
  documents = chunker.load_documents_from_disk(settings.raw_dir)

  if not documents:
      logger.error("no_documents_found", path=str(settings.raw_dir))
      logger.info("hint", message="Run `python -m scripts.ingest` first")
      return

  # 2. Chunk documents
  chunks = chunker.chunk_documents(documents)

  # 3. Generate embeddings (downloads model on first run)
  embedder = Embedder()
  embeddings = embedder.embed_chunks(chunks)

  # 4. Store in vector DB (reset first for clean re-indexing)
  store = VectorStore()
  if store.count() > 0:
      logger.info("clearing_existing_index", count=store.count())
      store.reset()

  store.add_chunks(chunks, embeddings)

  logger.info(
      "indexing_pipeline_complete",
      documents=len(documents),
      chunks=len(chunks),
      total_in_store=store.count(),
  )


if __name__ == "__main__":
  main()
