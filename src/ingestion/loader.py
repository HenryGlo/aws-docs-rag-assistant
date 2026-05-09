"""Document loader for AWS documentation.

Fetches public documentation pages from docs.aws.amazon.com,
extracts the main content, and saves to disk for downstream
processing (chunking, embedding, indexing).
"""
from dataclasses import dataclass
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from src.logger import logger


@dataclass
class Document:
  """Represents a single documentation page."""

  content: str
  source_url: str
  title: str
  metadata: dict


class AWSDocsLoader:
  """Loads documentation pages from AWS docs websites.

  Uses BeautifulSoup to extract clean text from HTML pages,
  filtering out navigation, scripts, and other non-content elements.
  """

  def __init__(self, timeout: int = 30):
      self.timeout = timeout
      self.client = httpx.Client(
          timeout=timeout,
          headers={"User-Agent": "AWS-Docs-RAG/1.0 (Educational Project)"},
          follow_redirects=True,
      )

  def load_url(self, url: str) -> Document | None:
      """Load a single AWS doc page from URL.

      Args:
          url: Full URL to the AWS documentation page.

      Returns:
          Document instance, or None if loading or parsing failed.
      """
      try:
          logger.info("loading_url", url=url)
          response = self.client.get(url)
          response.raise_for_status()

          soup = BeautifulSoup(response.text, "html.parser")

          # AWS docs use <main> or <article> for main content
          main_content = (
              soup.find("main")
              or soup.find("article")
              or soup.find("body")
          )

          if not main_content:
              logger.warning("no_main_content", url=url)
              return None

          # Remove navigation, scripts, styles, asides
          for tag in main_content.find_all(["script", "style", "nav", "aside"]):
              tag.decompose()

          # Extract title
          title_tag = soup.find("h1") or soup.find("title")
          title = title_tag.get_text(strip=True) if title_tag else "Untitled"

          # Get clean text with newlines preserved between blocks
          content = main_content.get_text(separator="\n", strip=True)

          # Filter out very short pages (likely error pages or redirects)
          if len(content) < 200:
              logger.warning(
                  "content_too_short",
                  url=url,
                  length=len(content),
              )
              return None

          logger.info(
              "loaded_url",
              url=url,
              content_length=len(content),
              title=title,
          )

          return Document(
              content=content,
              source_url=url,
              title=title,
              metadata={
                  "source": url,
                  "title": title,
                  "content_length": len(content),
              },
          )

      except httpx.HTTPError as e:
          logger.error("http_error", url=url, error=str(e))
          return None
      except Exception as e:
          logger.error("unexpected_error", url=url, error=str(e))
          return None

  def load_urls(self, urls: list[str]) -> list[Document]:
      """Load multiple URLs and return successfully loaded documents."""
      documents = []
      for url in urls:
          doc = self.load_url(url)
          if doc:
              documents.append(doc)

      logger.info(
          "load_complete",
          total=len(urls),
          successful=len(documents),
          success_rate=f"{len(documents) / len(urls) * 100:.1f}%",
      )
      return documents

  def save_documents(self, documents: list[Document], output_dir: Path) -> None:
      """Save documents to disk for later processing.

      Each document is saved as a .txt file with header metadata:
          URL: <url>
          TITLE: <title>
          ---
          <content>
      """
      output_dir.mkdir(parents=True, exist_ok=True)

      for i, doc in enumerate(documents):
          filename = f"doc_{i:04d}.txt"
          filepath = output_dir / filename
          filepath.write_text(
              f"URL: {doc.source_url}\n"
              f"TITLE: {doc.title}\n"
              f"---\n"
              f"{doc.content}",
              encoding="utf-8",
          )

      logger.info(
          "documents_saved",
          count=len(documents),
          path=str(output_dir),
      )

  def __del__(self):
      """Close HTTP client when loader is garbage collected."""
      if hasattr(self, "client"):
          self.client.close()
