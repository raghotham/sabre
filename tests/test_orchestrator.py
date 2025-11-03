"""
Tests for Orchestrator high-level flow using patched internals.

Run with: uv run pytest tests/test_orchestrator.py
"""

import pytest
from unittest.mock import AsyncMock

from sabre.server.orchestrator import Orchestrator, ParsedResponse
from sabre.server.python_runtime import PythonRuntime
from sabre.common import ExecutionTree


@pytest.mark.asyncio
async def test_orchestrator_no_helpers():
    """Orchestrator should return streamed text when no helpers are present."""
    runtime = PythonRuntime()
    executor = AsyncMock()
    tree = ExecutionTree()

    orchestrator = Orchestrator(executor, runtime)
    orchestrator._stream_and_parse_response = AsyncMock(
        return_value=ParsedResponse(
            full_text="This is a plain response",
            helpers=[],
            response_id="resp_123",
        )
    )
    orchestrator._ensure_images_uploaded = AsyncMock(side_effect=lambda x: x)

    result = await orchestrator.run(conversation_id="conv_123", input_text="Hello", tree=tree)

    assert result.success
    assert result.final_response == "This is a plain response"
    orchestrator._stream_and_parse_response.assert_awaited()


@pytest.mark.asyncio
async def test_orchestrator_with_helpers():
    """Orchestrator should execute helpers and continue the loop."""
    runtime = PythonRuntime()
    executor = AsyncMock()
    tree = ExecutionTree()

    orchestrator = Orchestrator(executor, runtime)
    orchestrator._stream_and_parse_response = AsyncMock(
        side_effect=[
            ParsedResponse(
                full_text="Helper incoming",
                helpers=["x = 2 + 2\nresult(x)"],
                response_id="resp_1",
            ),
            ParsedResponse(
                full_text="The result is 4",
                helpers=[],
                response_id="resp_2",
            ),
        ]
    )
    orchestrator._execute_helpers = AsyncMock(return_value=[("Computed result: 4", [])])
    orchestrator._ensure_images_uploaded = AsyncMock(side_effect=lambda x: x)

    result = await orchestrator.run(conversation_id="conv_123", input_text="What is 2 + 2?", tree=tree)

    assert result.success
    assert result.final_response == "The result is 4"
    assert orchestrator._execute_helpers.await_count == 1


@pytest.mark.asyncio
async def test_orchestrator_max_iterations():
    """Orchestrator should stop after hitting the max iteration cap."""
    runtime = PythonRuntime()
    executor = AsyncMock()
    tree = ExecutionTree()

    orchestrator = Orchestrator(executor, runtime, max_iterations=3)
    orchestrator._stream_and_parse_response = AsyncMock(
        return_value=ParsedResponse(
            full_text="Still helper",
            helpers=["x = 1"],
            response_id="resp_iter",
        )
    )
    orchestrator._execute_helpers = AsyncMock(return_value=[("loop", [])])
    orchestrator._ensure_images_uploaded = AsyncMock(side_effect=lambda x: x)

    result = await orchestrator.run(conversation_id="conv_123", input_text="Test", tree=tree)

    assert not result.success
    assert "Max iterations" in result.error
    assert orchestrator._execute_helpers.await_count == 3
