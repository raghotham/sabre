"""
SABRE Client Entry Point

Usage:
    python -m sabre.client
    uv run sabre-client
"""

import sys
import asyncio


def main():
    """Main entry point for SABRE client"""
    from sabre.client.client import main as client_main

    sys.exit(asyncio.run(client_main()))


if __name__ == "__main__":
    main()
