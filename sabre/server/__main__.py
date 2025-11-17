"""
Entry point for running sabre server.

Usage:
    python -m sabre.server
    uv run sabre-server

Environment variables:
    SABRE_HOME: Override base directory for debugging (default: ~/.local)
    PORT: Port to run server on (default: 8011)
    OPENAI_API_KEY: OpenAI API key
    OPENAI_BASE_URL: Custom OpenAI base URL (optional)
    OPENAI_MODEL: Default model (optional, default: gpt-4o)
"""

import logging
import os
import uvicorn

from sabre.common.paths import get_logs_dir, ensure_dirs, migrate_from_old_structure


def main():
    """Run the sabre server."""
    # Migrate from old structure if needed
    migrate_from_old_structure()

    # Ensure all directories exist
    ensure_dirs()

    # Get port from environment or use default
    port = int(os.getenv("PORT", "8011"))

    # Setup logging to file
    log_dir = get_logs_dir()
    log_file = log_dir / "server.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),  # Also log to console
        ],
    )

    logging.info(f"Starting sabre server on port {port}")

    # Configure uvicorn logging to match application format
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_config["formatters"]["access"]["fmt"] = (
        '%(asctime)s - %(name)s - %(levelname)s - %(client_addr)s - "%(request_line)s" %(status_code)s'
    )

    # Run server with extended timeout for agentic workflows
    # timeout_keep_alive: 300 seconds (5 minutes) for long-running requests
    uvicorn.run(
        "sabre.server.api.server:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        log_config=log_config,
        timeout_keep_alive=300,  # 5 minutes for agentic workflows
    )


if __name__ == "__main__":
    main()
