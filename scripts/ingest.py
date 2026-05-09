"""Script to ingest AWS documentation into the system.

Usage:
  python -m scripts.ingest

Downloads pages from URLs in src/ingestion/aws_urls.py,
parses HTML to extract clean content, and saves to data/raw/.
"""
from src.config import settings
from src.ingestion.aws_urls import AWS_DOCS_URLS
from src.ingestion.loader import AWSDocsLoader
from src.logger import logger


def main():
  """Run the ingestion pipeline."""
  logger.info("ingestion_started", url_count=len(AWS_DOCS_URLS))

  loader = AWSDocsLoader()
  documents = loader.load_urls(AWS_DOCS_URLS)

  loader.save_documents(documents, settings.raw_dir)

  logger.info(
      "ingestion_complete",
      loaded=len(documents),
      attempted=len(AWS_DOCS_URLS),
  )


if __name__ == "__main__":
  main()
