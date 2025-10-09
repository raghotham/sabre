"""
Tests for Orchestrator.

Run with: uv run pytest tests/test_orchestrator.py
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from sabre.server.orchestrator import Orchestrator
from sabre.server.python_runtime import PythonRuntime, ExecutionResult
from sabre.common import Assistant, TextContent, ExecutionTree


def test_parse_helpers():
    """Test helper block parsing."""
    runtime = PythonRuntime()
    executor = MagicMock()
    orchestrator = Orchestrator(executor, runtime)

    text = """
    Some text before
    <helpers>
    x = 1 + 1
    result(x)
    </helpers>
    Text in between
    <helpers>
    y = 2 + 2
    result(y)
    </helpers>
    Text after
    """

    helpers = orchestrator.parse_helpers(text)

    assert len(helpers) == 2
    assert "x = 1 + 1" in helpers[0]
    assert "y = 2 + 2" in helpers[1]


def test_replace_helpers_with_results():
    """Test replacing helpers with results."""
    runtime = PythonRuntime()
    executor = MagicMock()
    orchestrator = Orchestrator(executor, runtime)

    text = """
    Before
    <helpers>code1</helpers>
    Middle
    <helpers>code2</helpers>
    After
    """

    results = ["result1", "result2"]
    replaced = orchestrator.replace_helpers_with_results(text, results)

    assert "<helpers>code1</helpers>" not in replaced
    assert "<helpers>code2</helpers>" not in replaced
    assert "<helpers_result>result1</helpers_result>" in replaced
    assert "<helpers_result>result2</helpers_result>" in replaced


@pytest.mark.asyncio
async def test_orchestrator_no_helpers():
    """Test orchestrator with response containing no helpers."""
    runtime = PythonRuntime()
    executor = AsyncMock()
    tree = ExecutionTree()

    # Mock executor to return response without helpers
    mock_response = Assistant([TextContent("This is a plain response")])
    mock_response.response_id = "resp_123"
    executor.execute = AsyncMock(return_value=mock_response)

    orchestrator = Orchestrator(executor, runtime)

    result = await orchestrator.run(
        conversation_id="conv_123",
        input_text="Hello",
        tree=tree
    )

    assert result.success
    assert result.final_response == "This is a plain response"
    assert executor.execute.call_count == 1


@pytest.mark.asyncio
async def test_orchestrator_with_helpers():
    """Test orchestrator with response containing helpers."""
    runtime = PythonRuntime()
    executor = AsyncMock()
    tree = ExecutionTree()

    # First call: response with helpers
    response1 = Assistant([TextContent("""
    Let me calculate that for you.
    <helpers>
    x = 2 + 2
    result(x)
    </helpers>
    """)])
    response1.response_id = "resp_1"

    # Second call: response without helpers
    response2 = Assistant([TextContent("The result is 4")])
    response2.response_id = "resp_2"

    executor.execute = AsyncMock(side_effect=[response1, response2])

    orchestrator = Orchestrator(executor, runtime)

    result = await orchestrator.run(
        conversation_id="conv_123",
        input_text="What is 2 + 2?",
        tree=tree
    )

    assert result.success
    assert result.final_response == "The result is 4"
    assert executor.execute.call_count == 2


@pytest.mark.asyncio
async def test_orchestrator_max_iterations():
    """Test orchestrator stops at max iterations."""
    runtime = PythonRuntime()
    executor = AsyncMock()
    tree = ExecutionTree()

    # Always return response with helpers
    response = Assistant([TextContent("<helpers>x=1</helpers>")])
    response.response_id = "resp_123"
    executor.execute = AsyncMock(return_value=response)

    orchestrator = Orchestrator(executor, runtime, max_iterations=3)

    result = await orchestrator.run(
        conversation_id="conv_123",
        input_text="Test",
        tree=tree
    )

    assert not result.success
    assert "Max iterations" in result.error
    assert executor.execute.call_count == 3
