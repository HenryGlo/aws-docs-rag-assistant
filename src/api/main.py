"""FastAPI application exposing the RAG pipeline as a REST API.

Design decisions:
- Pipeline is initialized once at startup via lifespan context manager
(avoids reloading models on every request — a common mistake).
- Pydantic models validate request/response payloads.
- Health endpoint reports indexed chunk count for monitoring.
- Errors are logged and returned as proper HTTP status codes.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.logger import logger
from src.pipeline import RAGPipeline


class QueryRequest(BaseModel):
  """Request body for POST /query."""

  question: str = Field(
      ...,
      min_length=3,
      max_length=1000,
      description="The user's natural language question.",
      examples=["What is AWS Lambda?"],
  )
  top_k: int = Field(
      default=5,
      ge=1,
      le=20,
      description="Number of chunks to retrieve.",
  )


class SourceResponse(BaseModel):
  """A single cited source in the response."""

  index: int
  title: str
  url: str


class QueryResponse(BaseModel):
  """Response body for POST /query."""

  question: str
  answer: str
  sources: list[SourceResponse]
  metrics: dict


class HealthResponse(BaseModel):
  """Response body for GET /health."""

  status: str
  indexed_chunks: int


# Pipeline lives in module-level state, initialized at startup.
# Avoids reloading the embedding model and BM25 index on every request.
pipeline_instance: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
  """Initialize the RAG pipeline at startup, log shutdown."""
  global pipeline_instance
  logger.info("api_starting")
  pipeline_instance = RAGPipeline()
  logger.info("api_ready")
  yield
  logger.info("api_shutting_down")


app = FastAPI(
  title="AWS Docs RAG Assistant",
  description=(
      "Production-grade RAG system that answers questions about AWS "
      "services using cited sources from official documentation."
  ),
  version="0.1.0",
  lifespan=lifespan,
)


@app.get("/", tags=["info"])
async def root():
  """API metadata and available endpoints."""
  return {
      "name": "AWS Docs RAG Assistant",
      "version": "0.1.0",
      "description": "RAG over AWS documentation using the Claude API",
      "endpoints": {
          "health": "GET /health",
          "query": "POST /query",
          "docs": "GET /docs (OpenAPI/Swagger UI)",
      },
  }


@app.get("/health", response_model=HealthResponse, tags=["info"])
async def health_check():
  """Health check: returns OK and indexed chunk count.

  Used by container orchestrators (AWS App Runner, ECS, K8s) for
  liveness and readiness probes.
  """
  if pipeline_instance is None:
      raise HTTPException(status_code=503, detail="Pipeline not initialized")
  return HealthResponse(
      status="healthy",
      indexed_chunks=pipeline_instance.vector_store.count(),
  )


@app.post("/query", response_model=QueryResponse, tags=["rag"])
async def query(request: QueryRequest) -> QueryResponse:
  """Run a question through the RAG pipeline.

  Retrieves relevant chunks (hybrid: semantic + BM25 + RRF) and
  generates an answer with Claude, including source citations.
  """
  if pipeline_instance is None:
      raise HTTPException(status_code=503, detail="Pipeline not initialized")

  try:
      response = pipeline_instance.query(
          question=request.question,
          top_k=request.top_k,
      )

      return QueryResponse(
          question=response.question,
          answer=response.answer,
          sources=[SourceResponse(**s) for s in response.sources],
          metrics=response.metrics,
      )
  except Exception as e:
      logger.error("query_failed", error=str(e), exc_info=True)
      raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
