"""
Integration tests for connector bootstrap (YAML + persisted merge).

Tests that YAML configs and API-created configs are properly merged on startup.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import yaml

from sabre.server.mcp.config import MCPConfigLoader
from sabre.server.mcp.models import MCPServerConfig, MCPTransportType
from sabre.server.mcp.client_manager import MCPClientManager
from sabre.server.api.connector_store import ConnectorStore


@pytest.fixture
def temp_yaml_config():
    """Create a temporary YAML config file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "mcp.yaml"

        config_data = {
            "mcp_servers": {
                "yaml_server_1": {
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem"],
                    "enabled": True,
                },
                "yaml_server_2": {
                    "type": "sse",
                    "url": "https://api.example.com/mcp",
                    "headers": {"Authorization": "Bearer yaml-token"},
                    "enabled": False,
                },
            }
        }

        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        yield config_path


@pytest.fixture
def temp_store():
    """Create a temporary ConnectorStore."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = Path(tmpdir) / "connectors.json"
        yield ConnectorStore(store_path)


class TestBootstrapYAMLOnly:
    """Test bootstrap with YAML configs only."""

    def test_load_yaml_configs(self, temp_yaml_config):
        """Test loading configs from YAML."""
        configs = MCPConfigLoader.load(temp_yaml_config)

        assert len(configs) == 2

        # Find configs by name
        yaml_server_1 = next(c for c in configs if c.name == "yaml_server_1")
        yaml_server_2 = next(c for c in configs if c.name == "yaml_server_2")

        assert yaml_server_1.type == MCPTransportType.STDIO
        assert yaml_server_1.command == "npx"
        assert yaml_server_1.enabled is True

        assert yaml_server_2.type == MCPTransportType.SSE
        assert yaml_server_2.url == "https://api.example.com/mcp"
        assert yaml_server_2.enabled is False

    def test_yaml_configs_have_unique_ids(self, temp_yaml_config):
        """Test that YAML configs get unique IDs."""
        configs = MCPConfigLoader.load(temp_yaml_config)

        ids = [c.id for c in configs]
        assert len(ids) == len(set(ids))  # All unique

    @pytest.mark.asyncio
    async def test_connect_yaml_only(self, temp_yaml_config, temp_store):
        """Test connecting with only YAML configs."""
        yaml_configs = MCPConfigLoader.load(temp_yaml_config)

        # Mark as YAML source
        for config in yaml_configs:
            config.source = "yaml"

        # Create manager and connect
        manager = MCPClientManager(connector_store=temp_store)

        # Mock client connection
        with patch('sabre.server.mcp.client_manager.MCPClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=[])
            mock_client.is_connected = MagicMock(return_value=True)
            mock_client.tools_cache = []
            mock_client_class.return_value = mock_client

            # Connect all enabled configs
            for config in yaml_configs:
                if config.enabled:
                    await manager.connect(config)

            # Should have connected to enabled configs only
            assert len(manager.clients) == 1  # Only yaml_server_1 is enabled


class TestBootstrapAPIOnly:
    """Test bootstrap with API-persisted configs only."""

    @pytest.mark.asyncio
    async def test_load_persisted_configs(self, temp_store):
        """Test loading configs from ConnectorStore."""
        # Create and save some configs
        api_config_1 = MCPServerConfig(
            name="api_server_1",
            type=MCPTransportType.STDIO,
            command="echo",
            source="api",
        )

        api_config_2 = MCPServerConfig(
            name="api_server_2",
            type=MCPTransportType.STDIO,
            command="echo",
            source="api",
        )

        temp_store.save(api_config_1.id, api_config_1)
        temp_store.save(api_config_2.id, api_config_2)

        # Load all
        loaded = temp_store.load_all()

        assert len(loaded) == 2
        assert all(c.source == "api" for c in loaded.values())

    @pytest.mark.asyncio
    async def test_connect_api_only(self, temp_store):
        """Test connecting with only API-persisted configs."""
        # Create and save config
        api_config = MCPServerConfig(
            name="api_only_test",
            type=MCPTransportType.STDIO,
            command="echo",
            enabled=True,
            source="api",
        )

        temp_store.save(api_config.id, api_config)

        # Load and connect
        persisted_configs = temp_store.load_all()
        manager = MCPClientManager(connector_store=temp_store)

        with patch('sabre.server.mcp.client_manager.MCPClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=[])
            mock_client.is_connected = MagicMock(return_value=True)
            mock_client.tools_cache = []
            mock_client_class.return_value = mock_client

            for config in persisted_configs.values():
                if config.enabled:
                    await manager.connect(config)

            assert len(manager.clients) == 1


class TestBootstrapMerge:
    """Test merging YAML and API configs."""

    @pytest.mark.asyncio
    async def test_merge_yaml_and_api_configs(self, temp_yaml_config, temp_store):
        """Test merging YAML and persisted configs."""
        # Load YAML configs
        yaml_configs = MCPConfigLoader.load(temp_yaml_config)
        for config in yaml_configs:
            config.source = "yaml"

        # Create API config with different ID
        api_config = MCPServerConfig(
            name="api_server",
            type=MCPTransportType.STDIO,
            command="echo",
            source="api",
        )

        temp_store.save(api_config.id, api_config)

        # Merge
        all_configs = {}

        # Add YAML configs
        for config in yaml_configs:
            all_configs[config.id] = config

        # Add persisted configs (should not override since different IDs)
        persisted_configs = temp_store.load_all()
        for connector_id, config in persisted_configs.items():
            all_configs[connector_id] = config

        # Should have all configs
        assert len(all_configs) == 3  # 2 from YAML + 1 from API

        # Check sources
        yaml_count = sum(1 for c in all_configs.values() if c.source == "yaml")
        api_count = sum(1 for c in all_configs.values() if c.source == "api")

        assert yaml_count == 2
        assert api_count == 1

    @pytest.mark.asyncio
    async def test_persisted_overrides_yaml_same_id(self, temp_yaml_config, temp_store):
        """Test that persisted config overrides YAML for same ID."""
        # Load YAML configs
        yaml_configs = MCPConfigLoader.load(temp_yaml_config)
        yaml_config = yaml_configs[0]
        yaml_config.source = "yaml"

        # Simulate: User creates connector via API with SAME ID as YAML
        # (This would happen if YAML config was previously saved via API)
        modified_config = MCPServerConfig(
            id=yaml_config.id,  # Same ID!
            name=yaml_config.name + "_modified",
            type=yaml_config.type,
            command=yaml_config.command,
            source="api",  # Source is API
        )

        temp_store.save(modified_config.id, modified_config)

        # Merge logic: persisted takes precedence
        all_configs = {}

        # Add YAML
        for config in yaml_configs:
            all_configs[config.id] = config

        # Add persisted (overrides YAML for same ID)
        persisted_configs = temp_store.load_all()
        for connector_id, config in persisted_configs.items():
            all_configs[connector_id] = config

        # Should have the modified (API) version
        assert all_configs[modified_config.id].name == modified_config.name
        assert all_configs[modified_config.id].source == "api"


class TestBootstrapEdgeCases:
    """Test edge cases in bootstrap."""

    def test_empty_yaml_config(self):
        """Test loading from empty YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "empty.yaml"
            config_path.write_text("")

            configs = MCPConfigLoader.load(config_path)

            assert configs == []

    def test_yaml_with_no_mcp_servers_section(self):
        """Test YAML without mcp_servers section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "no_section.yaml"

            with open(config_path, 'w') as f:
                yaml.dump({"other_section": {}}, f)

            configs = MCPConfigLoader.load(config_path)

            assert configs == []

    def test_yaml_with_all_disabled(self, temp_store):
        """Test YAML with all servers disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "all_disabled.yaml"

            config_data = {
                "mcp_servers": {
                    "server_1": {
                        "type": "stdio",
                        "command": "echo",
                        "enabled": False,
                    },
                    "server_2": {
                        "type": "stdio",
                        "command": "echo",
                        "enabled": False,
                    },
                }
            }

            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)

            configs = MCPConfigLoader.load(config_path)

            assert len(configs) == 2
            assert all(not c.enabled for c in configs)

    @pytest.mark.asyncio
    async def test_empty_persisted_configs(self, temp_store):
        """Test loading when ConnectorStore is empty."""
        loaded = temp_store.load_all()

        assert loaded == {}


class TestBootstrapPriority:
    """Test bootstrap priority rules."""

    @pytest.mark.asyncio
    async def test_api_config_survives_restart(self, temp_yaml_config, temp_store):
        """Test that API-created config persists across 'restarts'."""
        # First "startup": Load YAML configs
        yaml_configs = MCPConfigLoader.load(temp_yaml_config)
        for config in yaml_configs:
            config.source = "yaml"

        # User creates a new connector via API
        api_config = MCPServerConfig(
            name="user_created",
            type=MCPTransportType.STDIO,
            command="echo",
            source="api",
        )

        temp_store.save(api_config.id, api_config)

        # Second "startup": Merge again
        all_configs_restart = {}

        # Load YAML (same as before)
        yaml_configs_restart = MCPConfigLoader.load(temp_yaml_config)
        for config in yaml_configs_restart:
            config.source = "yaml"
            all_configs_restart[config.id] = config

        # Load persisted (should have user's connector)
        persisted_configs = temp_store.load_all()
        for connector_id, config in persisted_configs.items():
            all_configs_restart[connector_id] = config

        # Should have YAML + user's connector
        assert len(all_configs_restart) == 3

        # User's connector should still be there
        user_connector = next(
            (c for c in all_configs_restart.values() if c.name == "user_created"),
            None
        )

        assert user_connector is not None
        assert user_connector.source == "api"

    @pytest.mark.asyncio
    async def test_user_disables_yaml_connector(self, temp_yaml_config, temp_store):
        """Test user disabling a YAML connector via API."""
        # Load YAML configs
        yaml_configs = MCPConfigLoader.load(temp_yaml_config)
        yaml_enabled_config = next(c for c in yaml_configs if c.enabled)
        yaml_enabled_config.source = "yaml"

        # User disables it via API (persists with same ID)
        disabled_version = MCPServerConfig(
            id=yaml_enabled_config.id,  # Same ID
            name=yaml_enabled_config.name,
            type=yaml_enabled_config.type,
            command=yaml_enabled_config.command,
            enabled=False,  # User disabled it
            source="api",
        )

        temp_store.save(disabled_version.id, disabled_version)

        # On restart: merge
        all_configs = {}

        # YAML
        for config in yaml_configs:
            all_configs[config.id] = config

        # Persisted (overrides)
        persisted_configs = temp_store.load_all()
        for connector_id, config in persisted_configs.items():
            all_configs[connector_id] = config

        # The connector should be disabled now (API version wins)
        connector = all_configs[disabled_version.id]
        assert connector.enabled is False
        assert connector.source == "api"


class TestBootstrapRealWorldScenario:
    """Test real-world bootstrap scenarios."""

    @pytest.mark.asyncio
    async def test_yaml_bootstrap_then_api_usage(self, temp_yaml_config, temp_store):
        """
        Scenario: User starts with YAML config, then uses API to add/modify connectors.
        On restart, both YAML and API changes should be present.
        """
        # Initial startup - YAML only
        yaml_configs = MCPConfigLoader.load(temp_yaml_config)
        for config in yaml_configs:
            config.source = "yaml"

        manager = MCPClientManager(connector_store=temp_store)

        # Mock connections
        with patch('sabre.server.mcp.client_manager.MCPClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=[])
            mock_client.is_connected = MagicMock(return_value=True)
            mock_client.disconnect = AsyncMock()
            mock_client.tools_cache = []
            mock_client_class.return_value = mock_client

            # Connect YAML configs
            for config in yaml_configs:
                if config.enabled:
                    await manager.connect(config)

            yaml_connector_count = len(manager.clients)

            # User creates a new connector via API
            new_api_config = MCPServerConfig(
                name="new_api_connector",
                type=MCPTransportType.STDIO,
                command="echo",
                enabled=True,
                source="api",
            )

            new_id = await manager.connect(new_api_config)

            # Should have YAML + API connectors
            assert len(manager.clients) == yaml_connector_count + 1

            # User modifies a YAML connector via API (disables it)
            yaml_connector_id = next(iter(manager.configs.keys()))
            await manager.disable_connector(yaml_connector_id)

            # Simulate restart: Create new manager, merge configs
            manager2 = MCPClientManager(connector_store=temp_store)

            # Merge
            all_configs = {}

            # YAML (fresh load)
            yaml_configs_restart = MCPConfigLoader.load(temp_yaml_config)
            for config in yaml_configs_restart:
                config.source = "yaml"
                all_configs[config.id] = config

            # Persisted (has user's changes)
            persisted = temp_store.load_all()
            for cid, cfg in persisted.items():
                all_configs[cid] = cfg

            # Should have:
            # - Original YAML connectors (may be overridden if modified)
            # - New API connector
            # - Modified state for disabled connector

            # New API connector should be present
            assert any(c.name == "new_api_connector" for c in all_configs.values())

            # Check a connector was modified
            if yaml_connector_id in persisted:
                assert persisted[yaml_connector_id].enabled is False
