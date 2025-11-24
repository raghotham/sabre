# Dockerfile for SABRE Server (AWS App Runner)
# IMPORTANT: This image contains NO credentials
# All secrets must be passed as environment variables at runtime
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for Playwright
RUN apt-get update && apt-get install -y \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock README.md ./

# Install dependencies (without the local package first)
RUN uv sync --frozen --no-install-project

# Copy application code
COPY sabre/ ./sabre/

# Now install the sabre package
RUN uv sync --frozen

# Install Playwright chromium_headless_shell (required for Web helper)
RUN uv run playwright install chromium --only-shell

# Expose port
EXPOSE 8011

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8011/health')"

# Run server
# Required environment variables (pass at runtime):
#   OPENAI_API_KEY - Your OpenAI API key (required)
#   OPENAI_MODEL   - Model to use (optional, default: gpt-4o)
#   OPENAI_BASE_URL - Custom API endpoint (optional)
#   PORT           - Server port (optional, default: 8011)
CMD ["uv", "run", "sabre-server"]
