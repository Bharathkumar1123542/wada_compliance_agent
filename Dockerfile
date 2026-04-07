# ── Stage 1: Build dependencies ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first (Docker layer cache)
COPY pyproject.toml ./

# Install to a local target directory
RUN pip install --upgrade pip && \
    pip install --prefix=/install . && \
    pip install --prefix=/install uvicorn[standard]

# ── Stage 2: Production image ─────────────────────────────────────────────────
FROM python:3.11-slim AS production

# Security: non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY schemas.py tools.py main.py ./
COPY agents/ ./agents/
COPY api/ ./api/

# Create __init__.py for packages
RUN touch agents/__init__.py api/__init__.py

# Set environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# Drop to non-root
USER appuser

EXPOSE 8080

# Health check (Cloud Run will use /health)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# Entrypoint — use PORT env var (Cloud Run sets this automatically)
CMD ["python", "main.py"]
