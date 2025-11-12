"""
Integration tests for MCPClientManager with ConnectorStore.

Tests UUID support, persistence integration, and connector lifecycle.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from sabre.server.mcp.client_manager import MCPClientManager
from sabre.server.mcp.models import MCPServerConfig, MCPTransportType, MCPServerNotFoundError
from sabre.server.api.connector_store import ConnectorStore


@pytest.fixture
def temp_store():
    """Create a temporary ConnectorStore."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = Path(tmpdir) / "connectors.json"
        yield ConnectorStore(store_path)


@pytest.fixture
def manager_with_store(temp_store):
    """Create MCPClientManager with ConnectorStore."""
    return MCPClientManager(connector_store=temp_store)


@pytest.fixture
def sample_config():
    """Create a sample MCPServerConfig."""
    return MCPServerConfig(
        name="test_server",
        type=MCPTransportType.STDIO,
        command="echo",
        args=["hello"],
        enabled=False,  # Disabled to avoid actual connection attempts
        source="api",
    )


class TestMCPClientManagerInitialization:
    """Test MCPClientManager initialization."""

    def test_init_without_store(self):
        """Test initialization without ConnectorStore."""
        manager = MCPClientManager()

        assert manager.connector_store is None
        assert isinstance(manager.clients, dict)
        assert isinstance(manager.configs, dict)
        assert isinstance(manager.id_to_name, dict)
        assert isinstance(manager.name_to_id, dict)

    def test_init_with_store(self, temp_store):
        """Test initialization with ConnectorStore."""
        manager = MCPClientManager(connector_store=temp_store)

        assert manager.connector_store is temp_store
        assert isinstance(manager.id_to_name, dict)
        assert isinstance(manager.name_to_id, dict)


class TestMCPClientManagerUUIDSupport:
    """Test UUID-based connector management."""

    @pytest.mark.asyncio
    async def test_connect_returns_uuid(self, manager_with_store, sample_config):
        """Test that connect returns connector UUID."""
        # Disabled connector won't try to connect
        connector_id = await manager_with_store.connect(sample_config)

        assert connector_id == sample_config.id
        assert connector_id in manager_with_store.configs

    @pytest.mark.asyncio
    async def test_uuid_mappings_created(self, manager_with_store, sample_config):
        """Test UUID mappings are created on connect."""
        connector_id = await manager_with_store.connect(sample_config)

        # Check id_to_name mapping
        assert connector_id in manager_with_store.id_to_name
        assert manager_with_store.id_to_name[connector_id] == sample_config.name

        # Check name_to_id mapping
        assert sample_config.name in manager_with_store.name_to_id
        assert manager_with_store.name_to_id[sample_config.name] == connector_id

    @pytest.mark.asyncio
    async def test_get_client_by_name(self, manager_with_store, sample_config):
        """Test getting client by name (backward compatibility)."""
        await manager_with_store.connect(sample_config)

        # get_client_by_name should work
        client = manager_with_store.get_client_by_name(sample_config.name)

        # Disabled connector has no client
        assert client is None  # No actual client for disabled connector

    @pytest.mark.asyncio
    async def test_has_connector(self, manager_with_store, sample_config):
        """Test has_connector check by UUID."""
        connector_id = await manager_with_store.connect(sample_config)

        assert manager_with_store.has_connector(connector_id)
        assert not manager_with_store.has_connector("nonexistent-uuid")

    @pytest.mark.asyncio
    async def test_list_connector_ids(self, manager_with_store):
        """Test listing connector UUIDs."""
        configs = [
            MCPServerConfig(
                name=f"server_{i}",
                type=MCPTransportType.STDIO,
                command="echo",
                enabled=False,
                source="api",
            )
            for i in range(3)
        ]

        ids = []
        for config in configs:
            connector_id = await manager_with_store.connect(config)
            ids.append(connector_id)

        connector_ids = manager_with_store.list_connector_ids()

        assert len(connector_ids) == 3
        for conn_id in ids:
            assert conn_id in connector_ids


class TestMCPClientManagerPersistence:
    """Test persistence integration."""

    @pytest.mark.asyncio
    async def test_connect_persists_config(self, manager_with_store, temp_store, sample_config):
        """Test that connect persists config to store."""
        connector_id = await manager_with_store.connect(sample_config)

        # Check config was persisted
        loaded = temp_store.get(connector_id)

        assert loaded is not None
        assert loaded.name == sample_config.name
        assert loaded.id == connector_id

    @pytest.mark.asyncio
    async def test_disconnect_removes_from_store(self, manager_with_store, temp_store, sample_config):
        """Test that disconnect removes config from store."""
        connector_id = await manager_with_store.connect(sample_config)

        # Verify it's in store
        assert temp_store.exists(connector_id)

        # Disconnect
        await manager_with_store.disconnect(connector_id)

        # Should be removed from store
        assert not temp_store.exists(connector_id)

    @pytest.mark.asyncio
    async def test_update_connector_persists(self, manager_with_store, temp_store, sample_config):
        """Test that update_connector persists changes."""
        connector_id = await manager_with_store.connect(sample_config)

        # Update
        updates = {"name": "updated_name", "timeout": 60}
        await manager_with_store.update_connector(connector_id, updates)

        # Check store has updated values
        loaded = temp_store.get(connector_id)

        assert loaded.name == "updated_name"
        assert loaded.timeout == 60

    @pytest.mark.asyncio
    async def test_enable_connector_persists(self, manager_with_store, temp_store, sample_config):
        """Test that enable_connector persists enabled state."""
        connector_id = await manager_with_store.connect(sample_config)

        # Mock the client for enabling
        with patch('sabre.server.mcp.client_manager.MCPClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=[])
            mock_client.is_connected = MagicMock(return_value=True)
            mock_client_class.return_value = mock_client

            # Enable
            await manager_with_store.enable_connector(connector_id)

        # Check store
        loaded = temp_store.get(connector_id)
        assert loaded.enabled is True

    @pytest.mark.asyncio
    async def test_disable_connector_persists(self, manager_with_store, temp_store):
        """Test that disable_connector persists disabled state."""
        config = MCPServerConfig(
            name="test",
            type=MCPTransportType.STDIO,
            command="echo",
            enabled=True,  # Start enabled
            source="api",
        )

        # Mock the client connection
        with patch('sabre.server.mcp.client_manager.MCPClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=[])
            mock_client.is_connected = MagicMock(return_value=True)
            mock_client.disconnect = AsyncMock()
            mock_client_class.return_value = mock_client

            connector_id = await manager_with_store.connect(config)

            # Disable
            await manager_with_store.disable_connector(connector_id)

            # Check store
            loaded = temp_store.get(connector_id)
            assert loaded.enabled is False


class TestMCPClientManagerConnectorInfo:
    """Test connector info methods."""

    @pytest.mark.asyncio
    async def test_get_connector_info(self, manager_with_store, sample_config):
        """Test get_connector_info returns correct data."""
        connector_id = await manager_with_store.connect(sample_config)

        # Mock a client
        mock_client = MagicMock()
        mock_client.is_connected.return_value = False
        mock_client.tools_cache = []

        manager_with_store.clients[connector_id] = mock_client

        info = manager_with_store.get_connector_info(connector_id)

        assert info["id"] == connector_id
        assert info["name"] == sample_config.name
        assert info["type"] == sample_config.type.value
        assert info["enabled"] == sample_config.enabled
        assert info["source"] == sample_config.source
        assert "created_at" in info
        assert "updated_at" in info

    @pytest.mark.asyncio
    async def test_get_connector_info_not_found(self, manager_with_store):
        """Test get_connector_info raises for non-existent connector."""
        with pytest.raises(MCPServerNotFoundError):
            manager_with_store.get_connector_info("nonexistent-id")

    @pytest.mark.asyncio
    async def test_get_all_connector_info(self, manager_with_store):
        """Test get_all_connector_info returns list of info."""
        configs = [
            MCPServerConfig(
                name=f"server_{i}",
                type=MCPTransportType.STDIO,
                command="echo",
                enabled=False,
                source="api",
            )
            for i in range(3)
        ]

        for config in configs:
            connector_id = await manager_with_store.connect(config)

            # Mock client
            mock_client = MagicMock()
            mock_client.is_connected.return_value = False
            mock_client.tools_cache = []
            manager_with_store.clients[connector_id] = mock_client

        all_info = manager_with_store.get_all_connector_info()

        assert len(all_info) == 3
        assert all(isinstance(info, dict) for info in all_info)
        assert all("id" in info for info in all_info)


class TestMCPClientManagerUpdateOperations:
    """Test connector update operations."""

    @pytest.mark.asyncio
    async def test_update_connector_name(self, manager_with_store, sample_config):
        """Test updating connector name updates mappings."""
        connector_id = await manager_with_store.connect(sample_config)

        old_name = sample_config.name
        new_name = "new_name"

        # Update name
        await manager_with_store.update_connector(connector_id, {"name": new_name})

        # Check mappings updated
        assert manager_with_store.id_to_name[connector_id] == new_name
        assert new_name in manager_with_store.name_to_id
        assert manager_with_store.name_to_id[new_name] == connector_id

        # Old name should be removed
        assert old_name not in manager_with_store.name_to_id

    @pytest.mark.asyncio
    async def test_update_connector_partial(self, manager_with_store, sample_config):
        """Test partial connector update."""
        connector_id = await manager_with_store.connect(sample_config)

        # Update only timeout
        await manager_with_store.update_connector(connector_id, {"timeout": 120})

        # Name should be unchanged
        config = manager_with_store.configs[connector_id]
        assert config.name == sample_config.name
        assert config.timeout == 120

    @pytest.mark.asyncio
    async def test_update_connector_not_found(self, manager_with_store):
        """Test updating non-existent connector raises error."""
        with pytest.raises(MCPServerNotFoundError):
            await manager_with_store.update_connector("nonexistent-id", {"timeout": 60})


class TestMCPClientManagerMultipleConnectors:
    """Test managing multiple connectors."""

    @pytest.mark.asyncio
    async def test_multiple_connectors_same_name_different_ids(self, manager_with_store):
        """Test that multiple connectors can have same name (different UUIDs)."""
        # Create two configs with same name
        config1 = MCPServerConfig(
            name="duplicate_name",
            type=MCPTransportType.STDIO,
            command="echo",
            enabled=False,
            source="api",
        )

        config2 = MCPServerConfig(
            name="duplicate_name",
            type=MCPTransportType.STDIO,
            command="echo",
            enabled=False,
            source="api",
        )

        id1 = await manager_with_store.connect(config1)
        id2 = await manager_with_store.connect(config2)

        # Different IDs
        assert id1 != id2

        # Both should be in configs
        assert id1 in manager_with_store.configs
        assert id2 in manager_with_store.configs

        # name_to_id will point to the last one connected
        assert manager_with_store.name_to_id["duplicate_name"] == id2

    @pytest.mark.asyncio
    async def test_mixed_enabled_disabled_connectors(self, manager_with_store):
        """Test mix of enabled and disabled connectors."""
        enabled_config = MCPServerConfig(
            name="enabled",
            type=MCPTransportType.STDIO,
            command="echo",
            enabled=True,
            source="api",
        )

        disabled_config = MCPServerConfig(
            name="disabled",
            type=MCPTransportType.STDIO,
            command="echo",
            enabled=False,
            source="api",
        )

        # Mock client connection for enabled
        with patch('sabre.server.mcp.client_manager.MCPClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=[])
            mock_client.is_connected = MagicMock(return_value=True)
            mock_client_class.return_value = mock_client

            id1 = await manager_with_store.connect(enabled_config)
            id2 = await manager_with_store.connect(disabled_config)

            # Enabled should have client
            assert id1 in manager_with_store.clients

            # Disabled should NOT have client
            assert id2 not in manager_with_store.clients


class TestMCPClientManagerBackwardCompatibility:
    """Test backward compatibility with name-based methods."""

    @pytest.mark.asyncio
    async def test_get_server_info_by_name(self, manager_with_store, sample_config):
        """Test get_server_info still works with name."""
        connector_id = await manager_with_store.connect(sample_config)

        # Mock client
        mock_client = MagicMock()
        mock_client.is_connected.return_value = False
        mock_client.tools_cache = []
        manager_with_store.clients[connector_id] = mock_client

        # Should work with name
        info = manager_with_store.get_server_info(sample_config.name)

        assert info["id"] == connector_id
        assert info["name"] == sample_config.name

    @pytest.mark.asyncio
    async def test_list_servers_returns_names(self, manager_with_store):
        """Test list_servers returns names (backward compatibility)."""
        configs = [
            MCPServerConfig(
                name=f"server_{i}",
                type=MCPTransportType.STDIO,
                command="echo",
                enabled=False,
                source="api",
            )
            for i in range(3)
        ]

        for config in configs:
            await manager_with_store.connect(config)

        names = manager_with_store.list_servers()

        assert len(names) == 3
        assert "server_0" in names
        assert "server_1" in names
        assert "server_2" in names
