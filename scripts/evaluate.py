"""Run RAGAs evaluation and save results.

Usage:
  python -m scripts.evaluate              # reuse cached generation if valid
  python -m scripts.evaluate --no-cache   # force regeneration

Output:
  data/eval/results.json — aggregated metrics
  Console — formatted summary
"""
import argparse
import json
from pathlib import Path

from src.evaluation.evaluator import RAGEvaluator
from src.pipeline import RAGPipeline


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--no-cache",
      action="store_true",
      help="Ignore the records cache and regenerate every answer.",
  )
  args = parser.parse_args()
  print("\n" + "=" * 60)
  print("📊 RAG Evaluation with RAGAs")
  print("=" * 60)

  print("\nInitializing pipeline...")
  pipeline = RAGPipeline()

  print("Initializing evaluator...")
  evaluator = RAGEvaluator(pipeline)

  qa_path = Path("data/eval/qa_dataset.json")
  if not qa_path.exists():
      print(f"\n❌ Dataset not found at {qa_path}")
      print("   Create data/eval/qa_dataset.json with evaluation questions.\n")
      return

  print(f"Running evaluation on {qa_path}...")
  print("(This will take 5-10 minutes and costs ~$1-2 in API calls.)\n")

  results = evaluator.evaluate(qa_path, use_cache=not args.no_cache)

  # Save results to JSON for the README
  output_path = Path("data/eval/results.json")
  with open(output_path, "w") as f:
      json.dump(results, f, indent=2)

  # Print formatted summary
  print("\n" + "=" * 60)
  print("📊 Evaluation Results")
  print("=" * 60)
  for metric, value in results.items():
      if isinstance(value, float):
          print(f"  {metric:35s} {value:.4f}")
      else:
          print(f"  {metric:35s} {value}")
  print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
  main()
