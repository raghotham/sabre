"""
Tests for async helper scheduling in llm_call utilities.

Run with: uv run pytest tests/test_llm_call_async.py
"""

import asyncio

import pytest

from sabre.server.helpers.llm_call import run_async_from_sync
from sabre.common.execution_context import set_execution_context, clear_execution_context


@pytest.mark.asyncio
async def test_run_async_from_sync_uses_orchestrator_loop():
    """Ensure run_async_from_sync schedules work on the orchestrator loop when available."""
    orchestrator_loop = asyncio.get_running_loop()

    async def sample_coro():
        # Should execute on the same loop captured in the execution context
        assert asyncio.get_running_loop() is orchestrator_loop
        await asyncio.sleep(0)
        return "ok"

    def invoke_helper():
        set_execution_context(
            event_callback=None,
            tree=None,
            tree_context={},
            conversation_id="conv-test",
            session_id="session-test",
            loop=orchestrator_loop,
        )
        try:
            return run_async_from_sync(sample_coro())
        finally:
            clear_execution_context()

    result = await asyncio.to_thread(invoke_helper)
    assert result == "ok"


@pytest.mark.asyncio
async def test_run_async_from_sync_fallback_loop():
    """Verify run_async_from_sync falls back to its own loop when no context is set."""
    async def sample_coro():
        await asyncio.sleep(0)
        return 42

    def invoke_helper():
        # No execution context provided - should create a temporary loop internally
        return run_async_from_sync(sample_coro())

    result = await asyncio.to_thread(invoke_helper)
    assert result == 42
