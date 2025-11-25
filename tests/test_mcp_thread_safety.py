"""
Test for MCP SSE transport thread safety issues.

This test reproduces a potential race condition when multiple MCP tools
are called concurrently from Python runtime using SSE transport.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from sabre.server.mcp.helper_adapter import MCPHelperAdapter
from sabre.server.mcp.client_manager import MCPClientManager
from sabre.server.mcp.models import MCPTool, MCPToolResult, MCPContent, MCPServerConfig, MCPTransportType


@pytest.mark.asyncio
async def test_concurrent_mcp_calls_sse_transport():
    """
    Test that concurrent MCP tool calls don't cause thread safety issues.

    This simulates the scenario where:
    1. Python runtime executes code with multiple MCP tool calls
    2. Each call uses Strategy 2 (new thread + new event loop)
    3. Both share the same SSE transport httpx.AsyncClient

    Expected: Should handle safely OR raise clear error
    Actual: May cause race condition or connection pool exhaustion
    """

    # Create mock MCP client manager with SSE transport
    manager = Mock(spec=MCPClientManager)

    # Create mock SSE client (simulating httpx.AsyncClient)
    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(
        return_value=MCPToolResult(
            content=[MCPContent(type="text", text="Result")]
        )
    )

    manager.get_client_by_name.return_value = mock_client
    manager.name_to_id = {"TestServer": "test-id"}

    # Create helper adapter WITHOUT main loop reference (forces Strategy 2/3)
    adapter = MCPHelperAdapter(manager, event_loop=None)

    # Register mock tool
    tool = MCPTool(
        name="test_tool",
        description="Test tool",
        input_schema={"type": "object", "properties": {}},
        server_name="TestServer"
    )
    adapter._tools_cache["TestServer.test_tool"] = tool

    # Get callable
    callable_tool = adapter.get_available_tools()["TestServer.test_tool"]

    # Simulate concurrent calls from Python runtime (sync context)
    # This forces the adapter to use threading
    import concurrent.futures

    def call_tool(i):
        """Simulate synchronous MCP tool call from runtime"""
        try:
            result = callable_tool()
            return f"Success {i}"
        except Exception as e:
            return f"Error {i}: {e}"

    # Launch 10 concurrent calls (simulating parallel tool usage in <helpers>)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(call_tool, i) for i in range(10)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Check for errors
    errors = [r for r in results if r.startswith("Error")]

    if errors:
        print(f"\n❌ Thread safety issue detected! {len(errors)} calls failed:")
        for error in errors:
            print(f"  - {error}")
        pytest.fail(f"Thread safety issues: {len(errors)} failures")
    else:
        print(f"\n✅ All {len(results)} concurrent calls succeeded")


@pytest.mark.asyncio
async def test_event_loop_nesting_detection():
    """
    Test that the adapter correctly detects and handles event loop nesting.

    This simulates calling an MCP tool when already inside an event loop.
    """
    manager = Mock(spec=MCPClientManager)
    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(
        return_value=MCPToolResult(
            content=[MCPContent(type="text", text="Nested result")]
        )
    )

    manager.get_client_by_name.return_value = mock_client
    manager.name_to_id = {"TestServer": "test-id"}

    # Create adapter without main loop
    adapter = MCPHelperAdapter(manager, event_loop=None)

    tool = MCPTool(
        name="nested_tool",
        description="Tool called from event loop",
        input_schema={"type": "object", "properties": {}},
        server_name="TestServer"
    )
    adapter._tools_cache["TestServer.nested_tool"] = tool

    callable_tool = adapter.get_available_tools()["TestServer.nested_tool"]

    # Try calling from within an event loop (should use Strategy 2)
    async def call_from_loop():
        """This runs inside an event loop"""
        # The tool wrapper should detect we're in a loop and spawn a new thread
        return callable_tool()

    # This should NOT raise "asyncio.run() cannot be called from a running event loop"
    try:
        result = await call_from_loop()
        print(f"✅ Successfully handled nested event loop call")
    except RuntimeError as e:
        if "asyncio.run() cannot be called" in str(e):
            pytest.fail(f"❌ Event loop nesting not handled correctly: {e}")
        raise


if __name__ == "__main__":
    # Run tests manually
    print("Testing MCP thread safety...")
    asyncio.run(test_concurrent_mcp_calls_sse_transport())
    asyncio.run(test_event_loop_nesting_detection())
