# Dockerfile for SABRE Server (AWS App Runner)
FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Copy application code
COPY sabre/ ./sabre/

# Expose port
EXPOSE 8011

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8011/health')"

# Run server
CMD ["uv", "run", "sabre-server"]
