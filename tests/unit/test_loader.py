"""Unit tests for AWSDocsLoader."""
from unittest.mock import Mock, patch

from src.ingestion.loader import AWSDocsLoader, Document


class TestAWSDocsLoader:
  """Tests for AWSDocsLoader behavior."""

  def test_load_url_success(self):
      """Test successful URL loading and parsing."""
      loader = AWSDocsLoader()

      mock_html = """
      <html>
          <head><title>AWS Lambda</title></head>
          <body>
              <main>
                  <h1>What is AWS Lambda</h1>
                  <p>AWS Lambda is a serverless compute service that runs your
                  code in response to events. This is sample content with enough
                  length to pass the minimum content length filter applied by
                  the loader during processing of AWS documentation pages.</p>
              </main>
          </body>
      </html>
      """

      with patch.object(loader.client, "get") as mock_get:
          mock_response = Mock()
          mock_response.text = mock_html
          mock_response.raise_for_status = Mock()
          mock_get.return_value = mock_response

          result = loader.load_url("https://docs.aws.amazon.com/lambda/test.html")

      assert result is not None
      assert isinstance(result, Document)
      assert "AWS Lambda" in result.title
      assert "serverless" in result.content
      assert result.source_url == "https://docs.aws.amazon.com/lambda/test.html"

  def test_load_url_short_content_returns_none(self):
      """Test that pages with too little content are filtered out."""
      loader = AWSDocsLoader()

      mock_html = "<html><body><main>Too short</main></body></html>"

      with patch.object(loader.client, "get") as mock_get:
          mock_response = Mock()
          mock_response.text = mock_html
          mock_response.raise_for_status = Mock()
          mock_get.return_value = mock_response

          result = loader.load_url("https://example.com")

      assert result is None

  def test_load_url_handles_http_error(self):
      """HTTP errors should return None and not raise."""
      loader = AWSDocsLoader()

      with patch.object(loader.client, "get") as mock_get:
          mock_get.side_effect = Exception("Network error")
          result = loader.load_url("https://example.com")

      assert result is None
