"""Interactive Q&A using the full RAG pipeline.

Usage:
  python -m scripts.ask
"""
from src.pipeline import RAGPipeline


def main():
  pipeline = RAGPipeline()

  print("\n" + "=" * 60)
  print("🤖 AWS Docs RAG Assistant")
  print("=" * 60)
  print("Ask questions about AWS services. Type 'quit' to exit.\n")

  while True:
      question = input("\n❓ Question: ").strip()
      if question.lower() in ("quit", "exit", "q", ""):
          break

      try:
          response = pipeline.query(question)

          print("\n💡 Answer:")
          print(response.answer)

          print("\n📚 Sources:")
          for src in response.sources:
              print(f"  [{src['index']}] {src['title']}")
              print(f"      {src['url']}")

          print(
              f"\n📊 Tokens: {response.metrics['input_tokens']} in / "
              f"{response.metrics['output_tokens']} out"
          )
          print(f"   Model: {response.metrics['model']}")

      except Exception as e:
          print(f"\n❌ Error: {e}")


if __name__ == "__main__":
  main()
