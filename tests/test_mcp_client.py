"""
Tests for MCP client functionality.

Run with: uv run pytest tests/test_mcp_client.py
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from sabre.server.mcp.models import (
    MCPServerConfig,
    MCPTransportType,
    MCPTool,
    MCPToolResult,
    MCPContent,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPConnectionError,
    MCPTimeoutError,
    MCPProtocolError,
    MCPToolError,
)
from sabre.server.mcp.client import MCPClient


def create_init_response(request_id=1):
    """Helper to create MCP initialization response."""
    return json.dumps({
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": {"name": "test-server", "version": "1.0.0"}
        }
    }) + "\n"


@pytest.fixture
def stdio_config():
    """Create a stdio transport config for testing."""
    return MCPServerConfig(
        name="test-server",
        type=MCPTransportType.STDIO,
        command="echo",  # Simple command for testing
        args=["test"],
        timeout=5,
    )


@pytest.fixture
def mock_process():
    """Create a mock subprocess."""
    process = AsyncMock()
    process.returncode = None
    process.stdin = AsyncMock()
    process.stdout = AsyncMock()
    process.stderr = AsyncMock()
    process.wait = AsyncMock(return_value=0)
    process.terminate = Mock()
    process.kill = Mock()

    # Mock readline to return MCP initialization response by default
    # This simulates a proper MCP server that responds to initialize request
    init_response = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": {"name": "test-server", "version": "1.0.0"}
        }
    }) + "\n"
    process.stdout.readline = AsyncMock(return_value=init_response.encode())

    return process


@pytest.mark.asyncio
async def test_client_initialization(stdio_config):
    """Test MCPClient initialization."""
    client = MCPClient(stdio_config)

    assert client.config == stdio_config
    assert client.connected is False
    assert client.process is None
    assert client.next_id == 1
    print("✓ MCPClient initialized correctly")


@pytest.mark.asyncio
async def test_jsonrpc_request_creation(stdio_config):
    """Test JSON-RPC request message creation."""
    client = MCPClient(stdio_config)

    request = JSONRPCRequest(
        method="tools/list",
        params={"test": "value"},
        id=1,
    )

    request_dict = request.to_dict()
    assert request_dict["jsonrpc"] == "2.0"
    assert request_dict["method"] == "tools/list"
    assert request_dict["params"] == {"test": "value"}
    assert request_dict["id"] == 1
    print(f"✓ JSON-RPC request created: {request_dict}")


@pytest.mark.asyncio
async def test_jsonrpc_response_parsing(stdio_config):
    """Test JSON-RPC response parsing."""
    # Test success response
    success_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"tools": []},
    }

    response = JSONRPCResponse(
        id=success_response["id"],
        result=success_response.get("result"),
        error=success_response.get("error"),
    )

    assert response.is_error is False
    assert response.result == {"tools": []}

    # Test error response
    error_response = {
        "jsonrpc": "2.0",
        "id": 2,
        "error": {"code": -32600, "message": "Invalid Request"},
    }

    response = JSONRPCResponse(
        id=error_response["id"],
        result=error_response.get("result"),
        error=error_response.get("error"),
    )

    assert response.is_error is True
    assert response.error["code"] == -32600

    print("✓ JSON-RPC responses parsed correctly")


@pytest.mark.asyncio
async def test_client_connect_disconnect(stdio_config, mock_process):
    """Test client connection and disconnection lifecycle."""
    client = MCPClient(stdio_config)

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        # Connect
        await client.connect()

        assert client.connected is True
        assert client.process is not None

        # Disconnect
        await client.disconnect()

        assert client.connected is False
        mock_process.terminate.assert_called_once()

    print("✓ Client connect/disconnect lifecycle works")


@pytest.mark.asyncio
async def test_send_request_not_connected(stdio_config):
    """Test that sending request when not connected raises error."""
    client = MCPClient(stdio_config)

    with pytest.raises(MCPConnectionError, match="Not connected"):
        await client._send_request("tools/list")

    print("✓ Sending request when not connected raises MCPConnectionError")


@pytest.mark.asyncio
async def test_list_tools_caching(stdio_config, mock_process):
    """Test that list_tools caches results."""
    client = MCPClient(stdio_config)

    # Mock responses - need to return initialize response first, then tools response
    init_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": {"name": "test-server", "version": "1.0.0"}
        }
    }

    tools_response = {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "tools": [
                {
                    "name": "test_tool",
                    "description": "A test tool",
                    "inputSchema": {"type": "object", "properties": {}},
                }
            ]
        },
    }

    # Return init response first, then tools response
    mock_process.stdout.readline = AsyncMock(
        side_effect=[
            (json.dumps(init_response) + "\n").encode(),
            (json.dumps(tools_response) + "\n").encode(),
        ]
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        await client.connect()

        # First call - should make request
        tools1 = await client.list_tools()
        assert len(tools1) == 1
        assert tools1[0].name == "test_tool"

        # Second call - should use cache
        tools2 = await client.list_tools()
        assert len(tools2) == 1
        assert tools1 is tools2  # Same object from cache

    print("✓ list_tools caching works correctly")


@pytest.mark.asyncio
async def test_call_tool_success(stdio_config, mock_process):
    """Test successful tool invocation."""
    client = MCPClient(stdio_config)

    # Mock responses: init first, then tool response
    tool_response = {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "content": [
                {"type": "text", "text": "Tool result"}
            ]
        },
    }

    mock_process.stdout.readline = AsyncMock(
        side_effect=[
            create_init_response(1).encode(),
            (json.dumps(tool_response) + "\n").encode(),
        ]
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        await client.connect()

        result = await client.call_tool("test_tool", {"arg": "value"})

        assert isinstance(result, MCPToolResult)
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == "Tool result"
        assert result.is_error is False

    print("✓ call_tool success case works")


@pytest.mark.asyncio
async def test_call_tool_error(stdio_config, mock_process):
    """Test tool invocation with error response."""
    client = MCPClient(stdio_config)

    # Mock responses: init first, then error response
    error_response = {
        "jsonrpc": "2.0",
        "id": 2,
        "error": {"code": -32601, "message": "Tool not found"},
    }

    mock_process.stdout.readline = AsyncMock(
        side_effect=[
            create_init_response(1).encode(),
            (json.dumps(error_response) + "\n").encode(),
        ]
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        await client.connect()

        with pytest.raises(MCPToolError, match="Tool not found"):
            await client.call_tool("nonexistent_tool", {})

    print("✓ call_tool error handling works")


@pytest.mark.asyncio
async def test_timeout_handling(stdio_config, mock_process):
    """Test request timeout handling."""
    client = MCPClient(stdio_config)

    # Mock timeout scenario - return init response then timeout
    call_count = 0

    async def timeout_readline():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call - return init response
            return create_init_response(1).encode()
        # Second call - timeout
        await asyncio.sleep(10)  # Longer than timeout
        return b""

    mock_process.stdout.readline = timeout_readline

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        await client.connect()

        with pytest.raises(MCPTimeoutError, match="timed out"):
            await client._send_stdio_request('{"method": "test"}')

    print("✓ Timeout handling works")


@pytest.mark.asyncio
async def test_invalid_json_response(stdio_config, mock_process):
    """Test handling of invalid JSON in response."""
    client = MCPClient(stdio_config)

    # Mock responses: init first, then invalid JSON
    mock_process.stdout.readline = AsyncMock(
        side_effect=[
            create_init_response(1).encode(),
            b"invalid json\n",
        ]
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        await client.connect()

        with pytest.raises(MCPProtocolError, match="Invalid JSON response"):
            await client._send_stdio_request('{"method": "test"}')

    print("✓ Invalid JSON response handling works")


@pytest.mark.asyncio
async def test_empty_response(stdio_config, mock_process):
    """Test handling of empty response."""
    client = MCPClient(stdio_config)

    # Mock responses: init first, then empty response
    mock_process.stdout.readline = AsyncMock(
        side_effect=[
            create_init_response(1).encode(),
            b"\n",
        ]
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        await client.connect()

        with pytest.raises(MCPProtocolError, match="Empty response"):
            await client._send_stdio_request('{"method": "test"}')

    print("✓ Empty response handling works")


@pytest.mark.asyncio
async def test_context_manager(stdio_config, mock_process):
    """Test using MCPClient as async context manager."""
    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        async with MCPClient(stdio_config) as client:
            assert client.connected is True

        # Should be disconnected after exiting context
        mock_process.terminate.assert_called_once()

    print("✓ Async context manager works")


@pytest.mark.asyncio
async def test_list_resources(stdio_config, mock_process):
    """Test listing MCP resources."""
    client = MCPClient(stdio_config)

    # Mock responses: init first, then resources
    resources_response = {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "resources": [
                {
                    "uri": "file:///test.txt",
                    "name": "test.txt",
                    "description": "A test file",
                    "mimeType": "text/plain",
                }
            ]
        },
    }

    mock_process.stdout.readline = AsyncMock(
        side_effect=[
            create_init_response(1).encode(),
            (json.dumps(resources_response) + "\n").encode(),
        ]
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        await client.connect()

        resources = await client.list_resources()
        assert len(resources) == 1
        assert resources[0].uri == "file:///test.txt"
        assert resources[0].name == "test.txt"

    print("✓ list_resources works correctly")


@pytest.mark.asyncio
async def test_read_resource(stdio_config, mock_process):
    """Test reading an MCP resource."""
    client = MCPClient(stdio_config)

    # Mock responses: init first, then resource content
    resource_response = {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "contents": [
                {
                    "uri": "file:///test.txt",
                    "mimeType": "text/plain",
                    "text": "File contents here",
                }
            ]
        },
    }

    mock_process.stdout.readline = AsyncMock(
        side_effect=[
            create_init_response(1).encode(),
            (json.dumps(resource_response) + "\n").encode(),
        ]
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        await client.connect()

        content = await client.read_resource("file:///test.txt")
        assert content.uri == "file:///test.txt"
        assert content.text == "File contents here"

    print("✓ read_resource works correctly")


# SSE Transport Tests


@pytest.fixture
def sse_config():
    """Create a test SSE server config."""
    return MCPServerConfig(
        name="test-sse-server",
        type=MCPTransportType.SSE,
        url="https://example.com/mcp",
        headers={"Authorization": "Bearer test-token"},
        enabled=True,
        timeout=10,
    )


@pytest.fixture
def mock_http_client():
    """Create a mock httpx AsyncClient."""
    client = Mock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.aclose = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_sse_connect(sse_config, mock_http_client):
    """Test connecting to SSE server."""
    # Mock successful connection test (health check) and initialize
    mock_health_response = Mock()
    mock_health_response.status_code = 200
    mock_http_client.get.return_value = mock_health_response

    # Mock initialize request response
    mock_init_response = Mock()
    mock_init_response.status_code = 200
    mock_init_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": {"name": "test-sse-server", "version": "1.0.0"}
        }
    }
    mock_http_client.post.return_value = mock_init_response

    with patch("sabre.server.mcp.client.httpx.AsyncClient", return_value=mock_http_client):
        client = MCPClient(sse_config)
        await client.connect()

        assert client.connected is True
        assert client.http_client is not None
        # Should have made initialize POST request
        assert mock_http_client.post.called

    print("✓ SSE connection works")


@pytest.mark.asyncio
async def test_sse_connect_failure(sse_config):
    """Test SSE connection failure during initialization."""
    # Initialize request fails with 404
    mock_http_client = Mock()
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.text = "Not Found"
    mock_http_client.get = AsyncMock(return_value=mock_response)
    mock_http_client.post = AsyncMock(return_value=mock_response)
    mock_http_client.aclose = AsyncMock()

    with patch("sabre.server.mcp.client.httpx.AsyncClient", return_value=mock_http_client):
        client = MCPClient(sse_config)

        # Connection should fail during initialization
        with pytest.raises(MCPConnectionError, match="HTTP 404"):
            await client.connect()

    print("✓ SSE connection failure handled correctly")


@pytest.mark.asyncio
async def test_sse_disconnect(sse_config, mock_http_client):
    """Test disconnecting from SSE server."""
    mock_health_response = Mock()
    mock_health_response.status_code = 200
    mock_http_client.get.return_value = mock_health_response

    # Mock initialize response
    mock_init_response = Mock()
    mock_init_response.status_code = 200
    mock_init_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": {"name": "test-sse-server", "version": "1.0.0"}
        }
    }
    mock_http_client.post.return_value = mock_init_response

    with patch("sabre.server.mcp.client.httpx.AsyncClient", return_value=mock_http_client):
        client = MCPClient(sse_config)
        await client.connect()

        await client.disconnect()

        assert client.connected is False
        assert client.http_client is None
        mock_http_client.aclose.assert_called_once()

    print("✓ SSE disconnection works")


@pytest.mark.asyncio
async def test_sse_list_tools(sse_config, mock_http_client):
    """Test listing tools via SSE."""
    # Mock connection
    mock_get_response = Mock()
    mock_get_response.status_code = 200
    mock_http_client.get.return_value = mock_get_response

    # Mock initialize response, then tools/list response
    init_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": {"name": "test-sse-server", "version": "1.0.0"}
        }
    }

    tools_response = {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "tools": [
                {
                    "name": "search",
                    "description": "Search the web",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                }
            ]
        },
    }

    # Use side_effect to return different responses for each POST call
    mock_init_response = Mock()
    mock_init_response.status_code = 200
    mock_init_response.json.return_value = init_response

    mock_tools_response = Mock()
    mock_tools_response.status_code = 200
    mock_tools_response.json.return_value = tools_response

    mock_http_client.post = AsyncMock(side_effect=[mock_init_response, mock_tools_response])

    with patch("sabre.server.mcp.client.httpx.AsyncClient", return_value=mock_http_client):
        client = MCPClient(sse_config)
        await client.connect()

        tools = await client.list_tools()

        assert len(tools) == 1
        assert tools[0].name == "search"
        assert tools[0].description == "Search the web"

        # Verify POST was called twice (initialize + tools/list)
        assert mock_http_client.post.call_count == 2

    print("✓ SSE list_tools works")


@pytest.mark.asyncio
async def test_sse_call_tool(sse_config, mock_http_client):
    """Test calling a tool via SSE."""
    # Mock connection
    mock_get_response = Mock()
    mock_get_response.status_code = 200
    mock_http_client.get.return_value = mock_get_response

    # Mock initialize response, then tool call response
    init_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": {"name": "test-sse-server", "version": "1.0.0"}
        }
    }

    tool_response = {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "content": [{"type": "text", "text": "Search results here"}]
        },
    }

    mock_init_response = Mock()
    mock_init_response.status_code = 200
    mock_init_response.json.return_value = init_response

    mock_tool_response = Mock()
    mock_tool_response.status_code = 200
    mock_tool_response.json.return_value = tool_response

    mock_http_client.post = AsyncMock(side_effect=[mock_init_response, mock_tool_response])

    with patch("sabre.server.mcp.client.httpx.AsyncClient", return_value=mock_http_client):
        client = MCPClient(sse_config)
        await client.connect()

        result = await client.call_tool("search", {"query": "test"})

        assert len(result.content) == 1
        assert result.content[0].text == "Search results here"
        assert result.is_error is False

    print("✓ SSE call_tool works")


@pytest.mark.asyncio
async def test_sse_http_error(sse_config, mock_http_client):
    """Test handling HTTP errors in SSE during initialization."""
    # Mock connection
    mock_get_response = Mock()
    mock_get_response.status_code = 200
    mock_http_client.get.return_value = mock_get_response

    # Mock error response for initialize request
    mock_post_response = Mock()
    mock_post_response.status_code = 500
    mock_post_response.text = "Internal Server Error"
    mock_http_client.post.return_value = mock_post_response

    with patch("sabre.server.mcp.client.httpx.AsyncClient", return_value=mock_http_client):
        client = MCPClient(sse_config)

        # Connection should fail during initialization with HTTP 500
        with pytest.raises(MCPConnectionError, match="HTTP 500"):
            await client.connect()

    print("✓ SSE HTTP error handling works")


@pytest.mark.asyncio
async def test_sse_without_httpx():
    """Test SSE connection fails gracefully without httpx."""
    config = MCPServerConfig(
        name="test-sse",
        type=MCPTransportType.SSE,
        url="https://example.com/mcp",
        enabled=True,
    )

    # Mock HTTPX_AVAILABLE as False
    with patch("sabre.server.mcp.client.HTTPX_AVAILABLE", False):
        client = MCPClient(config)

        with pytest.raises(MCPConnectionError, match="httpx is required"):
            await client.connect()

    print("✓ SSE fails gracefully without httpx")
