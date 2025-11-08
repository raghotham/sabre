"""
Tests for MCP helper adapter.

Run with: uv run pytest tests/test_mcp_helper_adapter.py
"""

import pytest
from unittest.mock import Mock, AsyncMock

from sabre.common.models import TextContent, ImageContent
from sabre.server.mcp.models import (
    MCPTool,
    MCPToolResult,
    MCPContent,
    MCPServerNotFoundError,
)
from sabre.server.mcp.helper_adapter import MCPHelperAdapter
from sabre.server.mcp.client_manager import MCPClientManager


@pytest.fixture
def mock_manager():
    """Create a mock MCPClientManager."""
    manager = Mock(spec=MCPClientManager)
    manager.get_all_tools = AsyncMock()
    manager.has_server = Mock(return_value=True)
    manager.get_client = Mock()
    return manager


@pytest.fixture
def sample_tools():
    """Create sample MCP tools for testing."""
    return {
        "test-server": [
            MCPTool(
                name="query",
                description="Execute a query",
                input_schema={
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string"},
                        "params": {"type": "array"},
                    },
                    "required": ["sql"],
                },
                server_name="test-server",
            ),
            MCPTool(
                name="execute",
                description="Execute a command",
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                    },
                    "required": ["command"],
                },
                server_name="test-server",
            ),
        ],
        "another-server": [
            MCPTool(
                name="create_file",
                description="Create a file",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
                server_name="another-server",
            ),
        ],
    }


@pytest.mark.asyncio
async def test_adapter_initialization(mock_manager):
    """Test MCPHelperAdapter initialization."""
    adapter = MCPHelperAdapter(mock_manager)

    assert adapter.client_manager is mock_manager
    assert len(adapter._tools_cache) == 0
    print("✓ MCPHelperAdapter initialized correctly")


@pytest.mark.asyncio
async def test_refresh_tools(mock_manager, sample_tools):
    """Test refreshing tools from servers."""
    mock_manager.get_all_tools.return_value = sample_tools

    adapter = MCPHelperAdapter(mock_manager)
    await adapter.refresh_tools()

    # Should have 3 tools total (2 from test-server, 1 from another-server)
    assert len(adapter._tools_cache) == 3
    assert "test-server.query" in adapter._tools_cache
    assert "test-server.execute" in adapter._tools_cache
    assert "another-server.create_file" in adapter._tools_cache

    print(f"✓ Refreshed {len(adapter._tools_cache)} tools")


@pytest.mark.asyncio
async def test_get_available_tools(mock_manager, sample_tools):
    """Test getting available tools as callables."""
    mock_manager.get_all_tools.return_value = sample_tools

    adapter = MCPHelperAdapter(mock_manager)
    await adapter.refresh_tools()

    tools = adapter.get_available_tools()

    assert len(tools) == 3
    assert callable(tools["test-server.query"])
    assert callable(tools["test-server.execute"])
    assert callable(tools["another-server.create_file"])

    print(f"✓ Got {len(tools)} callable tools")


@pytest.mark.asyncio
async def test_tool_callable_metadata(mock_manager, sample_tools):
    """Test that tool callables have correct metadata."""
    mock_manager.get_all_tools.return_value = sample_tools

    adapter = MCPHelperAdapter(mock_manager)
    await adapter.refresh_tools()

    tools = adapter.get_available_tools()
    query_tool = tools["test-server.query"]

    assert query_tool.__name__ == "test-server.query"
    assert query_tool.__doc__ == "Execute a query"

    print("✓ Tool callable metadata correct")


@pytest.mark.asyncio
async def test_generate_documentation(mock_manager, sample_tools):
    """Test generating documentation for tools."""
    mock_manager.get_all_tools.return_value = sample_tools

    adapter = MCPHelperAdapter(mock_manager)
    await adapter.refresh_tools()

    docs = adapter.generate_documentation()

    # Should contain markdown sections for each server
    assert "## MCP Tools" in docs
    assert "### test-server Server" in docs
    assert "### another-server Server" in docs

    # Should contain tool signatures
    assert "test-server.query" in docs
    assert "Execute a query" in docs
    assert "sql: str" in docs

    print(f"✓ Generated documentation:\n{docs[:200]}...")


@pytest.mark.asyncio
async def test_transform_text_content():
    """Test transforming MCP text content to SABRE content."""
    adapter = MCPHelperAdapter(Mock())

    mcp_content = MCPContent(type="text", text="Hello, world!")
    sabre_content = adapter._transform_content(mcp_content)

    assert isinstance(sabre_content, TextContent)
    assert sabre_content.text == "Hello, world!"

    print("✓ Text content transformation works")


@pytest.mark.asyncio
async def test_transform_image_content():
    """Test transforming MCP image content to SABRE content."""
    adapter = MCPHelperAdapter(Mock())

    mcp_content = MCPContent(
        type="image",
        data="base64imagedata",
        mimeType="image/png",
    )
    sabre_content = adapter._transform_content(mcp_content)

    assert isinstance(sabre_content, ImageContent)
    assert sabre_content.image_data == "base64imagedata"
    assert sabre_content.mime_type == "image/png"

    print("✓ Image content transformation works")


@pytest.mark.asyncio
async def test_transform_resource_content():
    """Test transforming MCP resource content to SABRE content."""
    adapter = MCPHelperAdapter(Mock())

    mcp_content = MCPContent(
        type="resource",
        uri="file:///test.txt",
        text="Resource text",
    )
    sabre_content = adapter._transform_content(mcp_content)

    assert isinstance(sabre_content, TextContent)
    assert "Resource: file:///test.txt" in sabre_content.text
    assert "Resource text" in sabre_content.text

    print("✓ Resource content transformation works")


@pytest.mark.asyncio
async def test_transform_result(mock_manager):
    """Test transforming full MCP tool result."""
    adapter = MCPHelperAdapter(mock_manager)

    mcp_result = MCPToolResult(
        content=[
            MCPContent(type="text", text="Line 1"),
            MCPContent(type="text", text="Line 2"),
            MCPContent(type="image", data="imagedata", mimeType="image/png"),
        ],
        is_error=False,
    )

    sabre_content = adapter._transform_result(mcp_result)

    assert len(sabre_content) == 3
    assert isinstance(sabre_content[0], TextContent)
    assert isinstance(sabre_content[1], TextContent)
    assert isinstance(sabre_content[2], ImageContent)

    print("✓ Full result transformation works")


@pytest.mark.asyncio
async def test_invoke_tool_server_not_found(mock_manager):
    """Test invoking tool when server not found."""
    mock_manager.has_server.return_value = False

    adapter = MCPHelperAdapter(mock_manager)

    with pytest.raises(MCPServerNotFoundError, match="nonexistent"):
        await adapter.invoke_tool("nonexistent.tool")

    print("✓ invoke_tool raises error for missing server")


@pytest.mark.asyncio
async def test_invoke_tool_invalid_name(mock_manager):
    """Test invoking tool with invalid qualified name."""
    adapter = MCPHelperAdapter(mock_manager)

    with pytest.raises(ValueError, match="qualified with server name"):
        await adapter.invoke_tool("invalid_tool_name")

    print("✓ invoke_tool validates qualified name format")


@pytest.mark.asyncio
async def test_invoke_tool_success(mock_manager, sample_tools):
    """Test successful tool invocation."""
    mock_manager.get_all_tools.return_value = sample_tools

    # Mock client and tool call
    mock_client = Mock()
    mock_client.call_tool = AsyncMock(return_value=MCPToolResult(
        content=[MCPContent(type="text", text="Success!")],
        is_error=False,
    ))
    mock_manager.get_client.return_value = mock_client

    adapter = MCPHelperAdapter(mock_manager)
    await adapter.refresh_tools()

    result = await adapter.invoke_tool("test-server.query", sql="SELECT * FROM users")

    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert result[0].text == "Success!"

    # Verify correct tool was called
    mock_client.call_tool.assert_called_once_with("query", {"sql": "SELECT * FROM users"})

    print("✓ Tool invocation works correctly")


@pytest.mark.asyncio
async def test_get_tool_count(mock_manager, sample_tools):
    """Test getting tool count."""
    mock_manager.get_all_tools.return_value = sample_tools

    adapter = MCPHelperAdapter(mock_manager)
    await adapter.refresh_tools()

    count = adapter.get_tool_count()
    assert count == 3

    print(f"✓ Tool count: {count}")


@pytest.mark.asyncio
async def test_get_server_names(mock_manager, sample_tools):
    """Test getting server names."""
    mock_manager.get_all_tools.return_value = sample_tools

    adapter = MCPHelperAdapter(mock_manager)
    await adapter.refresh_tools()

    servers = adapter.get_server_names()
    assert len(servers) == 2
    assert "test-server" in servers
    assert "another-server" in servers

    print(f"✓ Server names: {servers}")


@pytest.mark.asyncio
async def test_has_tool(mock_manager, sample_tools):
    """Test checking if tool exists."""
    mock_manager.get_all_tools.return_value = sample_tools

    adapter = MCPHelperAdapter(mock_manager)
    await adapter.refresh_tools()

    assert adapter.has_tool("test-server.query") is True
    assert adapter.has_tool("nonexistent.tool") is False

    print("✓ has_tool works correctly")


@pytest.mark.asyncio
async def test_get_tool(mock_manager, sample_tools):
    """Test getting tool definition."""
    mock_manager.get_all_tools.return_value = sample_tools

    adapter = MCPHelperAdapter(mock_manager)
    await adapter.refresh_tools()

    tool = adapter.get_tool("test-server.query")
    assert tool is not None
    assert tool.name == "query"
    assert tool.server_name == "test-server"

    nonexistent = adapter.get_tool("nonexistent.tool")
    assert nonexistent is None

    print("✓ get_tool works correctly")


@pytest.mark.asyncio
async def test_empty_documentation(mock_manager):
    """Test documentation generation with no tools."""
    mock_manager.get_all_tools.return_value = {}

    adapter = MCPHelperAdapter(mock_manager)
    await adapter.refresh_tools()

    docs = adapter.generate_documentation()
    assert docs == ""

    print("✓ Empty documentation handled correctly")
