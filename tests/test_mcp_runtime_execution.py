"""
Test MCP tools execution through Python runtime.

This test fills the critical gap in MCP test coverage by verifying that
MCP tools return usable Python values when called from LLM-generated code,
not just when called directly via adapter.invoke_tool().

Run with: uv run pytest tests/test_mcp_runtime_execution.py -v
"""

import pytest
from unittest.mock import Mock, AsyncMock
from sabre.server.mcp.models import MCPTool, MCPToolResult, MCPContent
from sabre.server.mcp.helper_adapter import MCPHelperAdapter
from sabre.server.python_runtime import PythonRuntime
from sabre.common.models import TextContent


@pytest.fixture
def mock_manager():
    """Create a mock MCP client manager."""
    manager = Mock()
    manager.name_to_id = {}
    manager.get_all_tools = AsyncMock(return_value={})
    manager.get_client_by_name = Mock()
    return manager


@pytest.fixture
def sample_tools():
    """Create sample MCP tools for testing."""
    return {
        "test_server": [
            MCPTool(
                name="calculate",
                description="Perform arithmetic calculation",
                input_schema={
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["add", "multiply"]},
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["operation", "a", "b"],
                },
                server_name="test_server",
            ),
            MCPTool(
                name="echo",
                description="Echo back a message",
                input_schema={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                    },
                    "required": ["message"],
                },
                server_name="test_server",
            ),
            MCPTool(
                name="get_user",
                description="Get user information",
                input_schema={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "number"},
                    },
                    "required": ["user_id"],
                },
                server_name="test_server",
            ),
        ]
    }


@pytest.mark.asyncio
async def test_mcp_tool_returns_simple_string(mock_manager, sample_tools):
    """Test that MCP tools return Content objects with get_str() method."""
    # Setup mock
    mock_manager.get_all_tools.return_value = sample_tools
    mock_manager.name_to_id = {"test_server": "uuid-123"}

    mock_client = Mock()
    mock_client.call_tool = AsyncMock(
        return_value=MCPToolResult(
            content=[MCPContent(type="text", text="6 multiply 7 = 42")],
            is_error=False,
        )
    )
    mock_manager.get_client_by_name.return_value = mock_client

    # Create adapter and runtime
    adapter = MCPHelperAdapter(mock_manager)
    await adapter.refresh_tools()

    runtime = PythonRuntime(mcp_adapter=adapter)

    # Execute code that calls MCP tool - no explicit result() needed!
    code = """
calc_result = test_server.calculate(operation="multiply", a=6, b=7)
print(f"Type: {type(calc_result).__name__}")
print(f"Has get_str: {hasattr(calc_result, 'get_str')}")
result(calc_result)
"""

    execution_result = runtime.execute(code)

    # CRITICAL ASSERTIONS
    assert execution_result.success, f"Execution failed: {execution_result.error}"

    # Should return Content object with get_str() method
    assert "Type: TextContent" in execution_result.output, \
        f"Expected TextContent type, got output: {execution_result.output}"

    assert "Has get_str: True" in execution_result.output, \
        f"Expected get_str method, got output: {execution_result.output}"

    assert "6 multiply 7 = 42" in execution_result.output, \
        f"Expected result in output: {execution_result.output}"

    # Verify the tool was actually called
    mock_client.call_tool.assert_called_once_with(
        "calculate",
        {"operation": "multiply", "a": 6, "b": 7}
    )

    print("✓ MCP tool returns Content object with get_str()")


@pytest.mark.asyncio
async def test_mcp_tool_result_is_usable_in_python(mock_manager, sample_tools):
    """Test that MCP tool results have get_str() to access text."""
    mock_manager.get_all_tools.return_value = sample_tools
    mock_manager.name_to_id = {"test_server": "uuid-123"}

    mock_client = Mock()
    mock_client.call_tool = AsyncMock(
        return_value=MCPToolResult(
            content=[MCPContent(type="text", text="Hello, World!")],
            is_error=False,
        )
    )
    mock_manager.get_client_by_name.return_value = mock_client

    adapter = MCPHelperAdapter(mock_manager)
    await adapter.refresh_tools()
    runtime = PythonRuntime(mcp_adapter=adapter)

    # Execute code that uses get_str() to access text
    code = """
message_content = test_server.echo(message="Hello, World!")

# Access text via get_str() method (like other SABRE helpers)
message = message_content.get_str()

# Now can use string methods
upper_message = message.upper()
length = len(message)
words = message.split(", ")

print(f"Upper: {upper_message}")
print(f"Length: {length}")
print(f"Words: {words}")

result(f"Processed: {upper_message}, {length} chars, {len(words)} words")
"""

    execution_result = runtime.execute(code)

    assert execution_result.success, f"Execution failed: {execution_result.error}"
    assert "Upper: HELLO, WORLD!" in execution_result.output
    assert "Length: 13" in execution_result.output
    assert "Words: ['Hello', 'World!']" in execution_result.output
    assert "Processed: HELLO, WORLD!, 13 chars, 2 words" in execution_result.output

    print("✓ MCP tool result has get_str() for text access")


@pytest.mark.asyncio
async def test_mcp_tool_with_no_result_call(mock_manager, sample_tools):
    """Test that MCP tool Content objects can be printed directly."""
    mock_manager.get_all_tools.return_value = sample_tools
    mock_manager.name_to_id = {"test_server": "uuid-123"}

    mock_client = Mock()
    mock_client.call_tool = AsyncMock(
        return_value=MCPToolResult(
            content=[MCPContent(type="text", text="Test response")],
            is_error=False,
        )
    )
    mock_manager.get_client_by_name.return_value = mock_client

    adapter = MCPHelperAdapter(mock_manager)
    await adapter.refresh_tools()
    runtime = PythonRuntime(mcp_adapter=adapter)

    # Execute code that prints Content object directly
    code = """
response = test_server.echo(message="test")
print(response.get_str())
"""

    execution_result = runtime.execute(code)

    assert execution_result.success, f"Execution failed: {execution_result.error}"
    assert "Test response" in execution_result.output

    print("✓ MCP tool Content can be printed via get_str()")


@pytest.mark.asyncio
async def test_mcp_tool_multiple_text_content(mock_manager, sample_tools):
    """Test MCP tool that returns multiple text content items."""
    mock_manager.get_all_tools.return_value = sample_tools
    mock_manager.name_to_id = {"test_server": "uuid-123"}

    mock_client = Mock()
    mock_client.call_tool = AsyncMock(
        return_value=MCPToolResult(
            content=[
                MCPContent(type="text", text="Line 1"),
                MCPContent(type="text", text="Line 2"),
                MCPContent(type="text", text="Line 3"),
            ],
            is_error=False,
        )
    )
    mock_manager.get_client_by_name.return_value = mock_client

    adapter = MCPHelperAdapter(mock_manager)
    await adapter.refresh_tools()
    runtime = PythonRuntime(mcp_adapter=adapter)

    code = """
user_info = test_server.get_user(user_id=123)
print(f"Type: {type(user_info).__name__}")
print(f"Content: {user_info.get_str()}")
result(user_info)
"""

    execution_result = runtime.execute(code)

    assert execution_result.success, f"Execution failed: {execution_result.error}"

    # Multiple text contents should be joined with newlines in TextContent
    assert "Type: TextContent" in execution_result.output
    assert "Line 1\nLine 2\nLine 3" in execution_result.output

    print("✓ Multiple text content items joined into TextContent")


@pytest.mark.asyncio
async def test_mcp_tool_empty_response(mock_manager, sample_tools):
    """Test MCP tool that returns empty content."""
    mock_manager.get_all_tools.return_value = sample_tools
    mock_manager.name_to_id = {"test_server": "uuid-123"}

    mock_client = Mock()
    mock_client.call_tool = AsyncMock(
        return_value=MCPToolResult(
            content=[],  # Empty content
            is_error=False,
        )
    )
    mock_manager.get_client_by_name.return_value = mock_client

    adapter = MCPHelperAdapter(mock_manager)
    await adapter.refresh_tools()
    runtime = PythonRuntime(mcp_adapter=adapter)

    code = """
response = test_server.echo(message="test")
print(f"Response: {response}")
print(f"Is None: {response is None}")
result(response if response is not None else "None")
"""

    execution_result = runtime.execute(code)

    assert execution_result.success, f"Execution failed: {execution_result.error}"
    assert "Response: None" in execution_result.output
    assert "Is None: True" in execution_result.output

    print("✓ Empty content returns None")


@pytest.mark.asyncio
async def test_mcp_tool_no_variable_shadowing(mock_manager, sample_tools):
    """Test that MCP tool results don't shadow the result() function."""
    mock_manager.get_all_tools.return_value = sample_tools
    mock_manager.name_to_id = {"test_server": "uuid-123"}

    mock_client = Mock()
    mock_client.call_tool = AsyncMock(
        return_value=MCPToolResult(
            content=[MCPContent(type="text", text="42")],
            is_error=False,
        )
    )
    mock_manager.get_client_by_name.return_value = mock_client

    adapter = MCPHelperAdapter(mock_manager)
    await adapter.refresh_tools()
    runtime = PythonRuntime(mcp_adapter=adapter)

    # This was the original bug - variable named 'result' shadowed result() function
    code = """
# Use a variable name that could shadow result()
result_from_tool = test_server.calculate(operation="add", a=20, b=22)

# Should still be able to call result() function
result(result_from_tool)
result("Additional output")
"""

    execution_result = runtime.execute(code)

    assert execution_result.success, f"Execution failed: {execution_result.error}"
    assert "42" in execution_result.output
    assert "Additional output" in execution_result.output

    print("✓ No variable shadowing of result() function")


@pytest.mark.asyncio
async def test_mcp_tool_in_loop(mock_manager, sample_tools):
    """Test MCP tools can be called multiple times in a loop."""
    mock_manager.get_all_tools.return_value = sample_tools
    mock_manager.name_to_id = {"test_server": "uuid-123"}

    call_count = 0
    def mock_call_tool(tool_name, args):
        nonlocal call_count
        call_count += 1
        return MCPToolResult(
            content=[MCPContent(type="text", text=f"Call {call_count}: {args['message']}")],
            is_error=False,
        )

    mock_client = Mock()
    mock_client.call_tool = AsyncMock(side_effect=mock_call_tool)
    mock_manager.get_client_by_name.return_value = mock_client

    adapter = MCPHelperAdapter(mock_manager)
    await adapter.refresh_tools()
    runtime = PythonRuntime(mcp_adapter=adapter)

    code = """
messages = ["Hello", "World", "Test"]
for msg in messages:
    response = test_server.echo(message=msg)
    print(response)
"""

    execution_result = runtime.execute(code)

    assert execution_result.success, f"Execution failed: {execution_result.error}"
    assert "Call 1: Hello" in execution_result.output
    assert "Call 2: World" in execution_result.output
    assert "Call 3: Test" in execution_result.output
    assert mock_client.call_tool.call_count == 3

    print("✓ MCP tools work correctly in loops")


@pytest.mark.asyncio
async def test_mcp_tool_error_handling(mock_manager, sample_tools):
    """Test that MCP tool errors are handled gracefully."""
    mock_manager.get_all_tools.return_value = sample_tools
    mock_manager.name_to_id = {"test_server": "uuid-123"}

    mock_client = Mock()
    mock_client.call_tool = AsyncMock(
        side_effect=Exception("Connection failed")
    )
    mock_manager.get_client_by_name.return_value = mock_client

    adapter = MCPHelperAdapter(mock_manager)
    await adapter.refresh_tools()
    runtime = PythonRuntime(mcp_adapter=adapter)

    code = """
try:
    response = test_server.echo(message="test")
    result(response)
except Exception as e:
    result(f"Error caught: {type(e).__name__}")
"""

    execution_result = runtime.execute(code)

    assert execution_result.success, f"Execution failed: {execution_result.error}"
    assert "Error caught:" in execution_result.output

    print("✓ MCP tool errors handled gracefully")


@pytest.mark.asyncio
async def test_prepare_result_content_single_text():
    """Test _prepare_result_content with single text content."""
    from sabre.server.mcp.helper_adapter import MCPHelperAdapter

    adapter = MCPHelperAdapter(Mock())

    content = [TextContent(text="Hello, World!")]
    result = adapter._prepare_result_content(content)

    assert isinstance(result, TextContent)
    assert result.get_str() == "Hello, World!"

    print("✓ Single text content returned as TextContent")


@pytest.mark.asyncio
async def test_prepare_result_content_multiple_text():
    """Test _prepare_result_content with multiple text content."""
    from sabre.server.mcp.helper_adapter import MCPHelperAdapter

    adapter = MCPHelperAdapter(Mock())

    content = [
        TextContent(text="Line 1"),
        TextContent(text="Line 2"),
        TextContent(text="Line 3"),
    ]
    result = adapter._prepare_result_content(content)

    assert isinstance(result, TextContent)
    assert result.get_str() == "Line 1\nLine 2\nLine 3"

    print("✓ Multiple text content joined into TextContent")


@pytest.mark.asyncio
async def test_prepare_result_content_empty():
    """Test _prepare_result_content with empty content."""
    from sabre.server.mcp.helper_adapter import MCPHelperAdapter

    adapter = MCPHelperAdapter(Mock())

    content = []
    result = adapter._prepare_result_content(content)

    assert result is None

    print("✓ Empty content returns None")


if __name__ == "__main__":
    # Allow running tests directly
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
