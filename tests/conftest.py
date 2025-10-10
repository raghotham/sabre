"""
Pytest configuration for llmvm2 tests.

This file contains shared fixtures and configuration for all tests.
"""

import pytest
import os
import logging

# Setup logging for tests
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


@pytest.fixture(scope="session")
def check_api_key():
    """Check that OpenAI API key is set before running tests.

    Only use this fixture for tests that need the API key (executor tests).
    Client tests don't need it.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use default event loop policy for all async tests."""
    import asyncio

    return asyncio.DefaultEventLoopPolicy()
