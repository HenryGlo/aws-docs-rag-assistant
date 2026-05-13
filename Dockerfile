# Multi-stage build for smaller, more secure final image.
# Stage 1: build dependencies into a clean directory.
# Stage 2: copy only the installed packages and source code.

FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed for some Python packages (sentence-transformers, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

# Copy dependency declaration and install
COPY pyproject.toml ./
RUN pip install --no-cache-dir --user .


# ---- Final stage ----
FROM python:3.11-slim

WORKDIR /app

# Copy installed Python packages from the builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application source code
COPY src/ ./src/
COPY data/ ./data/

# Expose API port
EXPOSE 8000

# Health check used by container orchestrators
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health', timeout=5); r.raise_for_status()" || exit 1

# Run the API with uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
