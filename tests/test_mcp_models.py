"""
Tests for MCP models and data structures.

Run with: uv run pytest tests/test_mcp_models.py
"""

import pytest
from sabre.server.mcp.models import (
    MCPServerConfig,
    MCPTransportType,
    MCPTool,
    MCPToolResult,
    MCPContent,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPError,
    MCPConnectionError,
    MCPToolError,
    MCPServerNotFoundError,
)


def test_mcp_server_config_stdio():
    """Test MCPServerConfig for stdio transport."""
    config = MCPServerConfig(
        name="test-server",
        type=MCPTransportType.STDIO,
        command="npx",
        args=["-y", "@test/server"],
        env={"TEST_VAR": "value"},
        enabled=True,
        timeout=30,
    )

    assert config.name == "test-server"
    assert config.type == MCPTransportType.STDIO
    assert config.command == "npx"
    assert config.args == ["-y", "@test/server"]
    assert config.env == {"TEST_VAR": "value"}
    assert config.enabled is True
    assert config.timeout == 30
    print("✓ MCPServerConfig stdio transport validated")


def test_mcp_server_config_sse():
    """Test MCPServerConfig for SSE transport."""
    config = MCPServerConfig(
        name="remote-server",
        type=MCPTransportType.SSE,
        url="https://example.com/mcp",
        headers={"Authorization": "Bearer token"},
        enabled=True,
    )

    assert config.name == "remote-server"
    assert config.type == MCPTransportType.SSE
    assert config.url == "https://example.com/mcp"
    assert config.headers == {"Authorization": "Bearer token"}
    print("✓ MCPServerConfig SSE transport validated")


def test_mcp_server_config_validation_stdio_missing_command():
    """Test that stdio config requires command."""
    with pytest.raises(ValueError, match="stdio transport requires 'command'"):
        MCPServerConfig(
            name="test",
            type=MCPTransportType.STDIO,
            # Missing command
        )
    print("✓ MCPServerConfig validates missing command for stdio")


def test_mcp_server_config_validation_sse_missing_url():
    """Test that SSE config requires url."""
    with pytest.raises(ValueError, match="SSE transport requires 'url'"):
        MCPServerConfig(
            name="test",
            type=MCPTransportType.SSE,
            # Missing url
        )
    print("✓ MCPServerConfig validates missing url for SSE")


def test_mcp_tool_basic():
    """Test MCPTool basic functionality."""
    tool = MCPTool(
        name="test_tool",
        description="A test tool",
        input_schema={
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
                "param2": {"type": "number"},
            },
            "required": ["param1"],
        },
        server_name="test-server",
    )

    assert tool.name == "test_tool"
    assert tool.description == "A test tool"
    assert tool.server_name == "test-server"
    print("✓ MCPTool basic properties validated")


def test_mcp_tool_signature_generation():
    """Test MCPTool signature generation."""
    tool = MCPTool(
        name="query_database",
        description="Execute SQL query",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL query"},
                "params": {"type": "array", "description": "Query parameters"},
            },
            "required": ["query"],
        },
    )

    signature = tool.get_signature()
    assert "query_database" in signature
    assert "query: str" in signature
    assert "params: list = None" in signature
    print(f"✓ MCPTool signature: {signature}")


def test_mcp_tool_documentation():
    """Test MCPTool documentation generation."""
    tool = MCPTool(
        name="create_file",
        description="Create a new file",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    )

    docs = tool.get_documentation()
    assert "create_file" in docs
    assert "path: str" in docs
    assert "content: str" in docs
    assert "Create a new file" in docs
    print(f"✓ MCPTool documentation:\n{docs}")


def test_mcp_content_text():
    """Test MCPContent for text type."""
    content = MCPContent(type="text", text="Hello, world!")

    assert content.type == "text"
    assert content.text == "Hello, world!"
    print("✓ MCPContent text type validated")


def test_mcp_content_image():
    """Test MCPContent for image type."""
    content = MCPContent(
        type="image",
        data="base64encodeddata",
        mimeType="image/png",
    )

    assert content.type == "image"
    assert content.data == "base64encodeddata"
    assert content.mimeType == "image/png"
    print("✓ MCPContent image type validated")


def test_mcp_tool_result():
    """Test MCPToolResult."""
    result = MCPToolResult(
        content=[
            MCPContent(type="text", text="Result 1"),
            MCPContent(type="text", text="Result 2"),
        ],
        is_error=False,
    )

    assert len(result.content) == 2
    assert result.is_error is False
    text = result.get_text()
    assert "Result 1" in text
    assert "Result 2" in text
    print(f"✓ MCPToolResult validated: {text}")


def test_jsonrpc_request():
    """Test JSON-RPC request serialization."""
    request = JSONRPCRequest(
        method="tools/list",
        params={"arg": "value"},
        id=1,
    )

    request_dict = request.to_dict()
    assert request_dict["jsonrpc"] == "2.0"
    assert request_dict["method"] == "tools/list"
    assert request_dict["params"] == {"arg": "value"}
    assert request_dict["id"] == 1
    print(f"✓ JSONRPCRequest serialized: {request_dict}")


def test_jsonrpc_response_success():
    """Test JSON-RPC response with result."""
    response = JSONRPCResponse(
        id=1,
        result={"data": "success"},
    )

    assert response.is_error is False
    assert response.result == {"data": "success"}
    assert response.error is None
    print("✓ JSONRPCResponse success validated")


def test_jsonrpc_response_error():
    """Test JSON-RPC response with error."""
    response = JSONRPCResponse(
        id=1,
        error={"code": -32600, "message": "Invalid request"},
    )

    assert response.is_error is True
    assert response.error is not None
    assert response.result is None
    print("✓ JSONRPCResponse error validated")


def test_mcp_exceptions():
    """Test MCP exception hierarchy."""
    # Test base exception
    base_error = MCPError("Base error")
    assert isinstance(base_error, Exception)

    # Test connection error
    conn_error = MCPConnectionError("Connection failed")
    assert isinstance(conn_error, MCPError)

    # Test tool error with metadata
    tool_error = MCPToolError("Tool failed", tool_name="test_tool", server_name="test-server")
    assert isinstance(tool_error, MCPError)
    assert tool_error.tool_name == "test_tool"
    assert tool_error.server_name == "test-server"

    # Test server not found error
    server_error = MCPServerNotFoundError("unknown-server")
    assert isinstance(server_error, MCPError)
    assert server_error.server_name == "unknown-server"

    print("✓ MCP exception hierarchy validated")
