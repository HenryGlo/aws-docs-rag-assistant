"""Embedding generation for chunks.

Why sentence-transformers locally:
- Zero API costs during development and indexing
- No external dependencies for indexing pipeline
- Acceptable quality for English technical documentation

Trade-off considered:
- voyage-3 (Voyage AI): better retrieval quality, paid API ($0.06/1M tokens)
- text-embedding-3-large (OpenAI): paid, slightly worse than voyage for retrieval
- bge-large-en-v1.5: top open source quality, larger model
- all-MiniLM-L6-v2 (chosen): small (80MB), fast, good quality for size

For production at scale or domain-specific needs, benchmark
multiple embedders against retrieval metrics.
"""
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.ingestion.chunker import Chunk
from src.logger import logger


class Embedder:
    """Generates embeddings for text chunks using sentence-transformers."""

    def __init__(self, model_name: str | None = None):
        model_name = model_name or settings.embedding_model
        logger.info("loading_embedding_model", model=model_name)
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(
            "embedding_model_loaded",
            model=model_name,
            dimension=self.dimension,
        )

    def embed_text(self, text: str) -> list[float]:
        """Embed a single piece of text. Used for queries at runtime."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_chunks(
        self, chunks: list[Chunk], batch_size: int = 32
    ) -> list[list[float]]:
        """Embed multiple chunks efficiently in batches.

        Batching speeds up inference by ~5-10x vs one-by-one calls.
        """
        texts = [chunk.content for chunk in chunks]

        logger.info(
            "embedding_chunks",
            num_chunks=len(chunks),
            batch_size=batch_size,
        )

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        return [e.tolist() for e in embeddings]
