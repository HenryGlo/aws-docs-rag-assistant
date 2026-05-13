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

  def test_generate_calls_claude_with_retrieved_context(self):
      """Generator should call Claude with structured prompt + retrieved context."""
      gen = AnswerGenerator()

      # Mock the Anthropic client response
      text_block = MagicMock()
      text_block.type = "text"
      text_block.text = "Lambda is serverless [1]."

      usage = MagicMock()
      usage.input_tokens = 100
      usage.output_tokens = 20
      usage.cache_read_input_tokens = 0
      usage.cache_creation_input_tokens = 0

      mock_response = MagicMock()
      mock_response.content = [text_block]
      mock_response.usage = usage

      gen.client = MagicMock()
      gen.client.messages.create.return_value = mock_response

      result = gen.generate("What is Lambda?", self._sample_chunks())

      assert "Lambda" in result.answer
      assert result.input_tokens == 100
      assert result.output_tokens == 20
      assert len(result.sources) == 2

      # Verify Claude was called with model, system + user messages
      call_kwargs = gen.client.messages.create.call_args.kwargs
      assert call_kwargs["model"] == gen.model
      assert call_kwargs["max_tokens"] == gen.max_tokens
      assert isinstance(call_kwargs["system"], list)
      assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
      assert call_kwargs["messages"][0]["role"] == "user"
