"""Answer generation using a local Ollama model with retrieved context.

Design decisions:
- Temperature 0.0: deterministic, factual responses (RAG should not be creative)
- Structured prompt with <context> and <question> tags (most LLMs respond well to this)
- Source citations [N] format: easy to parse and link back to URLs
- System prompt enforces no-hallucination policy
- Token usage tracked per call (Ollama returns prompt_eval_count / eval_count)
"""
from dataclasses import dataclass

import httpx

from src.config import settings
from src.logger import logger


@dataclass
class GeneratedAnswer:
  """A generated answer with cited sources and metadata."""

  answer: str
  sources: list[dict]
  model: str
  input_tokens: int
  output_tokens: int


SYSTEM_PROMPT = """You are a helpful assistant that answers questions about \
AWS services using only the provided documentation context.

Rules:
1. Answer ONLY using information from the provided context.
2. If the context does not contain enough information to answer, say so clearly.
3. Cite your sources using [N] notation, where N matches the source number.
4. Be precise, technical, and concise.
5. Do not invent or hallucinate information not present in the context.
6. If multiple sources contradict, mention the discrepancy.
7. If the question is outside the scope of the provided context, say so.
"""


class AnswerGenerator:
  """Generates answers using a local Ollama model with retrieved context."""

  def __init__(self):
      self.host = settings.ollama_host
      self.model = settings.ollama_model
      self.max_tokens = settings.max_tokens
      self.temperature = settings.temperature
      # Long timeout: local LLMs can be slow on first token, especially
      # when the model is being loaded from disk into RAM.
      self.client = httpx.Client(base_url=self.host, timeout=300)

  def generate(
      self,
      question: str,
      retrieved_chunks: list[dict],
  ) -> GeneratedAnswer:
      """Generate an answer based on the question and retrieved context.

      Args:
          question: The user's natural language question.
          retrieved_chunks: List of chunks from the retriever, each with
              'content' and 'metadata' keys.

      Returns:
          GeneratedAnswer with the answer text, source citations, and
          token usage statistics.
      """
      if not retrieved_chunks:
          return GeneratedAnswer(
              answer=(
                  "I could not find relevant information in the AWS "
                  "documentation to answer your question."
              ),
              sources=[],
              model=self.model,
              input_tokens=0,
              output_tokens=0,
          )

      context_str = self._format_context(retrieved_chunks)
      user_message = self._build_user_message(question, context_str)

      logger.info(
          "calling_ollama",
          model=self.model,
          num_sources=len(retrieved_chunks),
          question_length=len(question),
      )

      response = self.client.post(
          "/api/chat",
          json={
              "model": self.model,
              "messages": [
                  {"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_message},
              ],
              "stream": False,
              "options": {
                  "temperature": self.temperature,
                  # Ollama's name for max output tokens.
                  "num_predict": self.max_tokens,
              },
          },
      )
      response.raise_for_status()
      data = response.json()

      answer_text = data["message"]["content"]
      input_tokens = data.get("prompt_eval_count", 0)
      output_tokens = data.get("eval_count", 0)

      logger.info(
          "ollama_response_received",
          input_tokens=input_tokens,
          output_tokens=output_tokens,
      )

      return GeneratedAnswer(
          answer=answer_text,
          sources=self._format_sources(retrieved_chunks),
          model=self.model,
          input_tokens=input_tokens,
          output_tokens=output_tokens,
      )

  @staticmethod
  def _format_context(retrieved_chunks: list[dict]) -> str:
      """Format retrieved chunks as numbered context for the prompt.

      Each chunk gets a [Source N] header so the LLM can cite it.
      """
      parts = []
      for i, chunk in enumerate(retrieved_chunks, 1):
          metadata = chunk["metadata"]
          parts.append(
              f"[Source {i}] {metadata['title']}\n"
              f"URL: {metadata['source']}\n"
              f"Content: {chunk['content']}\n"
          )
      return "\n---\n".join(parts)

  @staticmethod
  def _build_user_message(question: str, context: str) -> str:
      """Build the user message with structured XML-like tags."""
      return (
          f"Answer the following question using the provided AWS "
          f"documentation context.\n\n"
          f"<context>\n{context}\n</context>\n\n"
          f"<question>{question}</question>\n\n"
          f"Provide a clear, accurate answer with citations [N] referring "
          f"to the sources above."
      )

  @staticmethod
  def _format_sources(retrieved_chunks: list[dict]) -> list[dict]:
      """Format sources for the response payload.

      Returns a list with index, title, and URL for each source.
      Used to render citations in the UI/CLI.
      """
      return [
          {
              "index": i,
              "title": chunk["metadata"]["title"],
              "url": chunk["metadata"]["source"],
          }
          for i, chunk in enumerate(retrieved_chunks, 1)
      ]

  def __del__(self):
      """Close HTTP client when generator is garbage collected."""
      if hasattr(self, "client"):
          self.client.close()
