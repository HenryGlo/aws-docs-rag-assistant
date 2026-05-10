"""Full indexing pipeline: load -> chunk -> embed -> store.

Run this after `scripts.ingest` to build the vector index.
Also persists chunks to disk so BM25 can be reconstructed at runtime.

Usage:
    python -m scripts.index
"""
import pickle

from src.config import settings
from src.embeddings.embedder import Embedder
from src.ingestion.chunker import DocumentChunker
from src.logger import logger
from src.retrieval.vector_store import VectorStore


def main():
    """Run the complete indexing pipeline."""
    logger.info("indexing_pipeline_started")

    # 1. Load documents from disk
    chunker = DocumentChunker()
    documents = chunker.load_documents_from_disk(settings.raw_dir)

    if not documents:
        logger.error("no_documents_found", path=str(settings.raw_dir))
        logger.info("hint", message="Run `python -m scripts.ingest` first")
        return

    # 2. Chunk documents
    chunks = chunker.chunk_documents(documents)

    # 3. Persist chunks for BM25 reconstruction at query time
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = settings.processed_dir / "chunks.pkl"
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    logger.info("chunks_persisted", path=str(chunks_path), count=len(chunks))

    # 4. Generate embeddings
    embedder = Embedder()
    embeddings = embedder.embed_chunks(chunks)

    # 5. Store in vector DB (reset first for clean re-indexing)
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
