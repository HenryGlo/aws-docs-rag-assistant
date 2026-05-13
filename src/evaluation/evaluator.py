"""Evaluation pipeline using RAGAs metrics.

Measures the quality of the RAG system on three dimensions:
- Faithfulness: Does the answer stay grounded in the retrieved context?
(Detects hallucinations.)
- Answer Relevancy: Does the answer actually address the question asked?
- Context Precision: Are the retrieved chunks relevant to the question?

RAGAs uses an LLM internally to judge these properties. The judge is
Claude via the Anthropic API; embeddings stay local via
sentence-transformers so the evaluation isn't billed twice.

Reference: https://docs.ragas.io/
"""
import hashlib
import json
import time
from pathlib import Path

from datasets import Dataset
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
  LLMContextPrecisionWithoutReference,
  answer_relevancy,
  faithfulness,
)
from ragas.run_config import RunConfig

from src.config import settings
from src.logger import logger
from src.pipeline import RAGPipeline


class RAGEvaluator:
  """Evaluates RAG pipeline performance using RAGAs metrics."""

  def __init__(self, pipeline: RAGPipeline):
      self.pipeline = pipeline
      # LLM-judge: Claude. temperature=0 for deterministic scoring.
      self.judge_llm = LangchainLLMWrapper(
          ChatAnthropic(
              model=settings.claude_model,
              api_key=settings.anthropic_api_key or None,
              temperature=0.0,
              max_tokens=4096,
          )
      )
      # Embeddings for context_precision etc. Reuses the same local
      # sentence-transformers model the retriever uses.
      self.judge_embeddings = LangchainEmbeddingsWrapper(
          HuggingFaceEmbeddings(model_name=settings.embedding_model)
      )

  def run_pipeline_on_dataset(self, questions: list[str]) -> dict:
      """Run pipeline on questions, collect data needed for RAGAs.

      For each question, runs the full RAG pipeline and records:
      - question (input)
      - answer (generated)
      - contexts (retrieved chunks)
      - latency (wall-clock time)
      """
      records = {
          "question": [],
          "answer": [],
          "contexts": [],
          "latencies": [],
      }

      for i, question in enumerate(questions, 1):
          logger.info(
              "evaluating_question",
              index=i,
              total=len(questions),
              question=question[:80],
          )

          start = time.time()
          response = self.pipeline.query(question)
          latency = time.time() - start

          records["question"].append(question)
          records["answer"].append(response.answer)
          records["contexts"].append([
              chunk["content"] for chunk in response.retrieved_chunks
          ])
          records["latencies"].append(latency)

      return records

  def _load_or_generate_records(
      self,
      questions: list[str],
      cache_path: Path,
      dataset_hash: str,
      use_cache: bool,
  ) -> tuple[dict, list[float]]:
      """Return (records, latencies), loading from cache when valid."""
      if use_cache and cache_path.exists():
          try:
              cached = json.loads(cache_path.read_text())
              if cached.get("dataset_hash") == dataset_hash:
                  logger.info(
                      "records_cache_hit",
                      cache_path=str(cache_path),
                      num_questions=len(cached["records"]["question"]),
                  )
                  return cached["records"], cached["latencies"]
              logger.info("records_cache_stale", cache_path=str(cache_path))
          except (json.JSONDecodeError, KeyError) as e:
              logger.warning("records_cache_unreadable", error=str(e))

      records = self.run_pipeline_on_dataset(questions)
      latencies = records.pop("latencies")

      if use_cache:
          cache_path.write_text(json.dumps({
              "dataset_hash": dataset_hash,
              "records": records,
              "latencies": latencies,
          }))
          logger.info("records_cache_written", cache_path=str(cache_path))

      return records, latencies

  def evaluate(self, qa_dataset_path: Path, use_cache: bool = True) -> dict:
      """Run full evaluation on the dataset and return aggregated metrics.

      If use_cache is True (default), the generation phase results are cached
      to data/eval/.records.cache.json keyed by a hash of the qa dataset, the
      Claude model, and top_k. The cache is reused on subsequent runs so a
      crash during scoring doesn't re-run all queries. Delete the cache
      file or pass use_cache=False to force regeneration.
      """
      qa_bytes = qa_dataset_path.read_bytes()
      qa_data = json.loads(qa_bytes)
      questions = [item["question"] for item in qa_data]

      logger.info("evaluation_started", num_questions=len(questions))

      cache_path = qa_dataset_path.parent / ".records.cache.json"
      dataset_hash = hashlib.sha256(
          qa_bytes
          + settings.claude_model.encode()
          + str(settings.top_k).encode()
      ).hexdigest()

      # 1. Run the pipeline on all questions (or reuse from cache)
      records, latencies = self._load_or_generate_records(
          questions, cache_path, dataset_hash, use_cache
      )

      # 2. Convert to HuggingFace Dataset (RAGAs expects this format)
      ds = Dataset.from_dict(records)

      # 3. Compute RAGAs metrics. Judge LLM is Claude; embeddings are
      # local sentence-transformers.
      logger.info("computing_ragas_metrics")
      # context_precision normally needs a `reference` (ground-truth answer);
      # the without-reference variant judges chunk relevance from the question
      # alone, which matches our qa_dataset.json shape.
      # Bound concurrency: ragas defaults to 16 parallel workers, which
      # bursts past the Anthropic per-minute rate limit on lower tiers.
      run_config = RunConfig(
          max_workers=4,    # Lower concurrency = fewer 429 errors
          max_retries=10,   # More retries when rate-limited
          max_wait=60,      # Longer max wait between retries
          timeout=180,      # Longer timeout per call
      )
      result = evaluate(
          ds,
          metrics=[
              faithfulness,
              answer_relevancy,
              LLMContextPrecisionWithoutReference(),
          ],
          llm=self.judge_llm,
          embeddings=self.judge_embeddings,
          run_config=run_config,
      )

      # 4. Aggregate metrics into a dict
      metrics_df = result.to_pandas()
      result_dict = metrics_df.mean(numeric_only=True).to_dict()

      # 5. Add latency statistics
      sorted_latencies = sorted(latencies)
      result_dict["latency_p50_seconds"] = sorted_latencies[
          len(sorted_latencies) // 2
      ]
      result_dict["latency_p95_seconds"] = sorted_latencies[
          int(len(sorted_latencies) * 0.95)
      ]
      result_dict["latency_avg_seconds"] = sum(latencies) / len(latencies)
      result_dict["num_questions_evaluated"] = len(questions)

      logger.info("evaluation_complete", **result_dict)
      return result_dict
