# Dockerfile for SABRE Server (AWS App Runner)
# IMPORTANT: This image contains NO credentials
# All secrets must be passed as environment variables at runtime
FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock README.md ./

# Install dependencies
RUN uv sync --frozen

# Copy application code
COPY sabre/ ./sabre/

# Expose port
EXPOSE 8011

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8011/health')"

# Flexible entrypoint for both server and CLI modes
# Required environment variables (pass at runtime):
#   OPENAI_API_KEY - Your OpenAI API key (required)
#   OPENAI_MODEL   - Model to use (optional, default: gpt-4o)
#   OPENAI_BASE_URL - Custom API endpoint (optional)
#   PORT           - Server port (optional, default: 8011)
#
# Usage:
#   Server mode (default):  docker run sabre:latest
#   CLI mode:              docker run sabre:latest sabre "your task"
ENTRYPOINT ["uv", "run"]
CMD ["sabre-server"]
