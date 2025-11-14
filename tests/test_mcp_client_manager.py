"""
Tests for MCP client manager.

Run with: uv run pytest tests/test_mcp_client_manager.py
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from sabre.server.mcp.models import (
    MCPServerConfig,
    MCPTransportType,
    MCPTool,
    MCPServerNotFoundError,
    MCPConnectionError,
)
from sabre.server.mcp.client_manager import MCPClientManager
from sabre.server.mcp.client import MCPClient


@pytest.fixture
def mock_client():
    """Create a mock MCP client."""
    client = Mock(spec=MCPClient)
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.list_tools = AsyncMock(return_value=[
        MCPTool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object"},
            server_name="test-server",
        )
    ])
    client.is_connected = Mock(return_value=True)
    client.tools_cache = [
        MCPTool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object"},
            server_name="test-server",
        )
    ]
    return client


@pytest.fixture
def test_config():
    """Create a test server config."""
    return MCPServerConfig(
        name="test-server",
        type=MCPTransportType.STDIO,
        command="echo",
        args=["test"],
        enabled=True,
    )


@pytest.mark.asyncio
async def test_manager_initialization():
    """Test MCPClientManager initialization."""
    manager = MCPClientManager()

    assert len(manager.clients) == 0
    assert len(manager.configs) == 0
    print("✓ MCPClientManager initialized correctly")


@pytest.mark.asyncio
async def test_connect_single_server(test_config, mock_client):
    """Test connecting to a single MCP server."""
    manager = MCPClientManager()

    with patch("sabre.server.mcp.client_manager.MCPClient", return_value=mock_client):
        connector_id = await manager.connect(test_config)

        # Check UUID-based storage
        assert connector_id in manager.clients
        assert connector_id in manager.configs
        # Check name mappings
        assert "test-server" in manager.name_to_id
        assert manager.name_to_id["test-server"] == connector_id
        mock_client.connect.assert_called_once()
        mock_client.list_tools.assert_called_once()

    print("✓ Single server connection works")


@pytest.mark.asyncio
async def test_connect_disabled_server(test_config, mock_client):
    """Test that disabled servers are skipped."""
    manager = MCPClientManager()
    test_config.enabled = False

    with patch("sabre.server.mcp.client_manager.MCPClient", return_value=mock_client):
        await manager.connect(test_config)

        assert len(manager.clients) == 0
        mock_client.connect.assert_not_called()

    print("✓ Disabled servers are skipped")


@pytest.mark.asyncio
async def test_connect_all_servers(mock_client):
    """Test connecting to multiple servers."""
    manager = MCPClientManager()

    configs = [
        MCPServerConfig(
            name="server1",
            type=MCPTransportType.STDIO,
            command="echo",
            enabled=True,
        ),
        MCPServerConfig(
            name="server2",
            type=MCPTransportType.STDIO,
            command="echo",
            enabled=True,
        ),
    ]

    with patch("sabre.server.mcp.client_manager.MCPClient", return_value=mock_client):
        await manager.connect_all(configs)

        assert len(manager.clients) == 2
        # Check name mappings exist
        assert "server1" in manager.name_to_id
        assert "server2" in manager.name_to_id
        # Verify clients are stored by UUID
        server1_id = manager.name_to_id["server1"]
        server2_id = manager.name_to_id["server2"]
        assert server1_id in manager.clients
        assert server2_id in manager.clients

    print("✓ Multiple server connections work")


@pytest.mark.asyncio
async def test_connect_all_with_failures(mock_client):
    """Test that connect_all continues on failures."""
    manager = MCPClientManager()

    # Create a client that fails to connect
    failing_client = Mock(spec=MCPClient)
    failing_client.connect = AsyncMock(side_effect=MCPConnectionError("Connection failed"))

    configs = [
        MCPServerConfig(name="good-server", type=MCPTransportType.STDIO, command="echo", enabled=True),
        MCPServerConfig(name="bad-server", type=MCPTransportType.STDIO, command="echo", enabled=True),
    ]

    with patch("sabre.server.mcp.client_manager.MCPClient") as mock_client_class:
        # First call returns good client, second returns failing client
        mock_client_class.side_effect = [mock_client, failing_client]

        await manager.connect_all(configs)

        # Should have connected to good server despite bad server failure
        assert len(manager.clients) == 1
        assert "good-server" in manager.name_to_id
        good_server_id = manager.name_to_id["good-server"]
        assert good_server_id in manager.clients
        assert "bad-server" not in manager.name_to_id

    print("✓ connect_all continues on failures")


@pytest.mark.asyncio
async def test_disconnect_server(test_config, mock_client):
    """Test disconnecting from a server."""
    manager = MCPClientManager()

    with patch("sabre.server.mcp.client_manager.MCPClient", return_value=mock_client):
        connector_id = await manager.connect(test_config)
        assert connector_id in manager.clients

        await manager.disconnect(connector_id)

        assert connector_id not in manager.clients
        assert connector_id not in manager.configs
        assert "test-server" not in manager.name_to_id
        mock_client.disconnect.assert_called_once()

    print("✓ Server disconnection works")


@pytest.mark.asyncio
async def test_disconnect_nonexistent_server():
    """Test disconnecting from nonexistent server raises error."""
    manager = MCPClientManager()

    with pytest.raises(MCPServerNotFoundError, match="nonexistent"):
        await manager.disconnect("nonexistent")

    print("✓ Disconnecting nonexistent server raises error")


@pytest.mark.asyncio
async def test_disconnect_all(mock_client):
    """Test disconnecting from all servers."""
    manager = MCPClientManager()

    configs = [
        MCPServerConfig(name="server1", type=MCPTransportType.STDIO, command="echo", enabled=True),
        MCPServerConfig(name="server2", type=MCPTransportType.STDIO, command="echo", enabled=True),
    ]

    with patch("sabre.server.mcp.client_manager.MCPClient", return_value=mock_client):
        await manager.connect_all(configs)
        assert len(manager.clients) == 2

        await manager.disconnect_all()

        assert len(manager.clients) == 0
        assert mock_client.disconnect.call_count == 2

    print("✓ Disconnect all servers works")


@pytest.mark.asyncio
async def test_get_client(test_config, mock_client):
    """Test getting a client by UUID."""
    manager = MCPClientManager()

    with patch("sabre.server.mcp.client_manager.MCPClient", return_value=mock_client):
        connector_id = await manager.connect(test_config)

        # Get client by UUID
        client = manager.get_client(connector_id)
        assert client is mock_client

        # Also test getting client by name
        client_by_name = manager.get_client_by_name("test-server")
        assert client_by_name is mock_client

    print("✓ get_client works correctly")


@pytest.mark.asyncio
async def test_get_nonexistent_client():
    """Test getting nonexistent client raises error."""
    manager = MCPClientManager()

    with pytest.raises(MCPServerNotFoundError, match="nonexistent"):
        manager.get_client("nonexistent")

    print("✓ Getting nonexistent client raises error")


@pytest.mark.asyncio
async def test_has_server(test_config, mock_client):
    """Test checking if server is connected."""
    manager = MCPClientManager()

    # Check by name (using name_to_id mapping)
    assert "test-server" not in manager.name_to_id

    with patch("sabre.server.mcp.client_manager.MCPClient", return_value=mock_client):
        connector_id = await manager.connect(test_config)

        # Check connector registered
        assert manager.has_connector(connector_id) is True
        # Check name mapping exists
        assert "test-server" in manager.name_to_id
        assert "nonexistent" not in manager.name_to_id

    print("✓ has_connector/name_to_id works correctly")


@pytest.mark.asyncio
async def test_list_servers(mock_client):
    """Test listing connected servers."""
    manager = MCPClientManager()

    configs = [
        MCPServerConfig(name="server1", type=MCPTransportType.STDIO, command="echo", enabled=True),
        MCPServerConfig(name="server2", type=MCPTransportType.STDIO, command="echo", enabled=True),
    ]

    with patch("sabre.server.mcp.client_manager.MCPClient", return_value=mock_client):
        await manager.connect_all(configs)

        servers = manager.list_servers()
        assert len(servers) == 2
        assert "server1" in servers
        assert "server2" in servers

    print("✓ list_servers works correctly")


@pytest.mark.asyncio
async def test_get_all_tools(mock_client):
    """Test getting tools from all servers."""
    manager = MCPClientManager()

    # Create different tools for each server
    def create_client(name):
        client = Mock(spec=MCPClient)
        client.connect = AsyncMock()
        client.list_tools = AsyncMock(return_value=[
            MCPTool(
                name=f"{name}_tool",
                description=f"Tool from {name}",
                input_schema={"type": "object"},
                server_name=name,
            )
        ])
        return client

    configs = [
        MCPServerConfig(name="server1", type=MCPTransportType.STDIO, command="echo", enabled=True),
        MCPServerConfig(name="server2", type=MCPTransportType.STDIO, command="echo", enabled=True),
    ]

    with patch("sabre.server.mcp.client_manager.MCPClient") as mock_client_class:
        mock_client_class.side_effect = [create_client("server1"), create_client("server2")]

        await manager.connect_all(configs)

        all_tools = await manager.get_all_tools()

        # get_all_tools returns dict with UUID keys
        assert len(all_tools) == 2

        # Get UUIDs for the servers
        server1_id = manager.name_to_id["server1"]
        server2_id = manager.name_to_id["server2"]

        assert server1_id in all_tools
        assert server2_id in all_tools
        assert len(all_tools[server1_id]) == 1
        assert all_tools[server1_id][0].name == "server1_tool"

    print("✓ get_all_tools works correctly")


@pytest.mark.asyncio
async def test_health_check(test_config, mock_client):
    """Test health checking a server."""
    manager = MCPClientManager()

    with patch("sabre.server.mcp.client_manager.MCPClient", return_value=mock_client):
        connector_id = await manager.connect(test_config)

        # Healthy server - health_check uses UUID keys
        is_healthy = await manager.health_check(connector_id)
        assert is_healthy is True

        # Test with failing list_tools
        mock_client.list_tools = AsyncMock(side_effect=Exception("Health check failed"))
        is_healthy = await manager.health_check(connector_id)
        assert is_healthy is False

    print("✓ health_check works correctly")


@pytest.mark.asyncio
async def test_health_check_all(mock_client):
    """Test health checking all servers."""
    manager = MCPClientManager()

    # Create clients with different health states
    healthy_client = Mock(spec=MCPClient)
    healthy_client.connect = AsyncMock()
    healthy_client.list_tools = AsyncMock(return_value=[])

    # Second client that connects but then becomes unhealthy
    unhealthy_client = Mock(spec=MCPClient)
    unhealthy_client.connect = AsyncMock()
    unhealthy_client.list_tools = AsyncMock(return_value=[])  # Initially healthy for connection

    configs = [
        MCPServerConfig(name="healthy", type=MCPTransportType.STDIO, command="echo", enabled=True),
        MCPServerConfig(name="unhealthy", type=MCPTransportType.STDIO, command="echo", enabled=True),
    ]

    with patch("sabre.server.mcp.client_manager.MCPClient") as mock_client_class:
        mock_client_class.side_effect = [healthy_client, unhealthy_client]

        await manager.connect_all(configs)

        # Now make unhealthy_client fail health checks
        unhealthy_client.list_tools = AsyncMock(side_effect=Exception("Failed"))

        health_status = await manager.health_check_all()

        # health_check_all returns dict with UUID keys
        healthy_id = manager.name_to_id["healthy"]
        unhealthy_id = manager.name_to_id["unhealthy"]

        assert health_status[healthy_id] is True
        assert health_status[unhealthy_id] is False

    print("✓ health_check_all works correctly")


@pytest.mark.asyncio
async def test_get_server_info(test_config, mock_client):
    """Test getting server information."""
    manager = MCPClientManager()

    with patch("sabre.server.mcp.client_manager.MCPClient", return_value=mock_client):
        await manager.connect(test_config)

        info = manager.get_server_info("test-server")

        assert info["name"] == "test-server"
        assert info["type"] == "stdio"
        assert info["enabled"] is True
        assert info["connected"] is True
        assert info["tools_count"] == 1

    print("✓ get_server_info works correctly")


@pytest.mark.asyncio
async def test_reconnect(test_config, mock_client):
    """Test reconnecting to a server."""
    manager = MCPClientManager()

    with patch("sabre.server.mcp.client_manager.MCPClient", return_value=mock_client):
        # Initial connection
        connector_id = await manager.connect(test_config)
        assert connector_id in manager.clients

        # Reconnect using UUID
        await manager.reconnect(connector_id)

        # Should have disconnected and reconnected
        mock_client.disconnect.assert_called()
        assert mock_client.connect.call_count == 2

    print("✓ Reconnect works correctly")


@pytest.mark.asyncio
async def test_context_manager(mock_client):
    """Test using MCPClientManager as context manager."""
    configs = [
        MCPServerConfig(name="server1", type=MCPTransportType.STDIO, command="echo", enabled=True),
    ]

    with patch("sabre.server.mcp.client_manager.MCPClient", return_value=mock_client):
        async with MCPClientManager() as manager:
            await manager.connect_all(configs)
            assert len(manager.clients) == 1

        # Should disconnect all on exit
        mock_client.disconnect.assert_called()

    print("✓ Context manager works correctly")
