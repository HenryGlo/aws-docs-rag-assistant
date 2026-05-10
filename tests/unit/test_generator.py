"""Unit tests for AnswerGenerator (mocked, no API calls)."""
from unittest.mock import MagicMock

from src.generation.generator import AnswerGenerator


class TestAnswerGenerator:
  """Tests for AnswerGenerator behavior."""

  def _sample_chunks(self):
      """Return sample retrieved chunks for testing."""
      return [
          {
              "content": "AWS Lambda is a serverless compute service.",
              "metadata": {
                  "source": "https://docs.aws.amazon.com/lambda/welcome.html",
                  "title": "What is AWS Lambda?",
              },
          },
          {
              "content": "Lambda functions are stateless and event-driven.",
              "metadata": {
                  "source": "https://docs.aws.amazon.com/lambda/foundation.html",
                  "title": "Lambda Foundation",
              },
          },
      ]

  def test_format_context_includes_all_sources(self):
      """Context formatter should include each source with [Source N] header."""
      gen = AnswerGenerator()
      chunks = self._sample_chunks()

      formatted = gen._format_context(chunks)

      assert "[Source 1]" in formatted
      assert "[Source 2]" in formatted
      assert "AWS Lambda" in formatted
      assert "stateless" in formatted

  def test_format_sources_returns_indexed_list(self):
      """Sources should be formatted with index, title, url."""
      gen = AnswerGenerator()
      chunks = self._sample_chunks()

      sources = gen._format_sources(chunks)

      assert len(sources) == 2
      assert sources[0]["index"] == 1
      assert sources[0]["title"] == "What is AWS Lambda?"
      assert sources[0]["url"].startswith("https://")

  def test_generate_returns_no_context_message_when_empty(self):
      """When no chunks are retrieved, return a 'no info' message."""
      gen = AnswerGenerator()
      result = gen.generate("Anything?", retrieved_chunks=[])

      assert "could not find" in result.answer.lower()
      assert result.sources == []
      assert result.input_tokens == 0
      assert result.output_tokens == 0

  def test_generate_calls_ollama_with_retrieved_context(self):
      """Generator should call Ollama with structured prompt + retrieved context."""
      gen = AnswerGenerator()

      # Mock the httpx response from Ollama's /api/chat endpoint
      mock_response = MagicMock()
      mock_response.raise_for_status = MagicMock()
      mock_response.json.return_value = {
          "model": gen.model,
          "message": {"role": "assistant", "content": "Lambda is serverless [1]."},
          "done": True,
          "prompt_eval_count": 100,
          "eval_count": 20,
      }
      gen.client = MagicMock()
      gen.client.post.return_value = mock_response

      result = gen.generate("What is Lambda?", self._sample_chunks())

      assert "Lambda" in result.answer
      assert result.input_tokens == 100
      assert result.output_tokens == 20
      assert len(result.sources) == 2

      # Verify Ollama was called with model, system + user messages, temperature
      call_kwargs = gen.client.post.call_args.kwargs
      payload = call_kwargs["json"]
      assert payload["model"] == gen.model
      assert payload["stream"] is False
      assert payload["options"]["temperature"] == 0.0
      roles = [m["role"] for m in payload["messages"]]
      assert roles == ["system", "user"]
