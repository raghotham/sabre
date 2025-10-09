"""
LLMVM2 Client Entry Point

Usage:
    python -m llmvm2.client
    uv run llmvm2-client
"""
import sys
import asyncio


def main():
    """Main entry point for LLMVM2 client"""
    from sabre.client.client import main as client_main
    sys.exit(asyncio.run(client_main()))


if __name__ == "__main__":
    main()
