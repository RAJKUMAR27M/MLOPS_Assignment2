# --- .dockerignore content ---
# .git
# data/
# mlruns/
# __pycache__
# .venv
# *.pyc
# models/*.pt
# !models/.gitkeep

# Stage 1: Builder
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Stage 2: Final
FROM python:3.11-slim

# Create non-root user
RUN addgroup --system appgroup && adduser --system appuser --ingroup appgroup

# Set working directory
WORKDIR /app

# Copy wheels from builder and install
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache /wheels/*

# Copy application code
COPY app/ app/
COPY src/ src/
COPY models/ models/

# Ensure the model artifact is present at runtime
RUN ls -la /app/models || true

# Set permissions
RUN chown -R appuser:appgroup /app

# Use non-root user
USER appuser

# Expose port
EXPOSE 8000

# Environment variables
ENV PORT=8000
ENV MODEL_PATH=/app/models/cnn_latest.pt

# Healthcheck without requiring curl in slim image
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run uvicorn
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
