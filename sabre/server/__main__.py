"""
Entry point for running llmvm2 server.

Usage:
    python -m llmvm2.server
    uv run llmvm2-server

Environment variables:
    LLMVM_PORT: Port to run server on (default: 8011)
    OPENAI_API_KEY: OpenAI API key
    OPENAI_BASE_URL: Custom OpenAI base URL (optional)
    OPENAI_MODEL: Default model (optional, default: gpt-4o)
"""
import logging
import os
import uvicorn


def main():
    """Run the llmvm2 server."""
    # Get port from environment or use default
    port = int(os.getenv("LLMVM_PORT", "8011"))

    # Setup logging to file
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "server.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )

    logging.info(f"Starting llmvm2 server on port {port}")

    # Configure uvicorn logging to match application format
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_config["formatters"]["access"]["fmt"] = '%(asctime)s - %(name)s - %(levelname)s - %(client_addr)s - "%(request_line)s" %(status_code)s'

    # Run server
    uvicorn.run(
        "llmvm2.server.api.server:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        log_config=log_config
    )


if __name__ == "__main__":
    main()
