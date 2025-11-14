"""
Integration tests for ConnectorStore.

Tests JSON persistence, file locking, and CRUD operations.
"""

import pytest
import tempfile
from pathlib import Path
import json
from datetime import datetime

from sabre.server.api.connector_store import ConnectorStore
from sabre.server.mcp.models import MCPServerConfig, MCPTransportType


@pytest.fixture
def temp_store():
    """Create a temporary ConnectorStore for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = Path(tmpdir) / "connectors.json"
        store = ConnectorStore(store_path)
        yield store


@pytest.fixture
def sample_config():
    """Create a sample MCPServerConfig for testing."""
    return MCPServerConfig(
        name="test_postgres",
        type=MCPTransportType.STDIO,
        command="npx",
        args=["-y", "@modelcontextprotocol/server-postgres"],
        env={"POSTGRES_URL": "postgresql://localhost/test"},
        enabled=True,
        source="api",
    )


class TestConnectorStoreBasics:
    """Test basic ConnectorStore operations."""

    def test_initialization(self, temp_store):
        """Test store initialization creates empty file."""
        assert temp_store.storage_path.exists()

        with open(temp_store.storage_path) as f:
            data = json.load(f)

        assert data == {}

    def test_save_and_load(self, temp_store, sample_config):
        """Test saving and loading a connector."""
        # Save
        temp_store.save(sample_config.id, sample_config)

        # Load
        loaded = temp_store.load_all()

        assert len(loaded) == 1
        assert sample_config.id in loaded

        loaded_config = loaded[sample_config.id]
        assert loaded_config.name == sample_config.name
        assert loaded_config.type == sample_config.type
        assert loaded_config.command == sample_config.command

    def test_save_updates_timestamp(self, temp_store, sample_config):
        """Test that save updates the updated_at timestamp."""
        # First save
        temp_store.save(sample_config.id, sample_config)
        first_updated = sample_config.updated_at

        # Wait a tiny bit and save again
        import time
        time.sleep(0.01)

        temp_store.save(sample_config.id, sample_config)

        # Load and check timestamp changed
        loaded = temp_store.load_all()
        second_updated = loaded[sample_config.id].updated_at

        assert second_updated > first_updated

    def test_get_existing(self, temp_store, sample_config):
        """Test getting an existing connector."""
        temp_store.save(sample_config.id, sample_config)

        loaded = temp_store.get(sample_config.id)

        assert loaded is not None
        assert loaded.name == sample_config.name
        assert loaded.id == sample_config.id

    def test_get_nonexistent(self, temp_store):
        """Test getting a non-existent connector returns None."""
        loaded = temp_store.get("nonexistent-id")
        assert loaded is None

    def test_exists(self, temp_store, sample_config):
        """Test exists check."""
        assert not temp_store.exists(sample_config.id)

        temp_store.save(sample_config.id, sample_config)

        assert temp_store.exists(sample_config.id)

    def test_delete(self, temp_store, sample_config):
        """Test deleting a connector."""
        temp_store.save(sample_config.id, sample_config)
        assert temp_store.exists(sample_config.id)

        temp_store.delete(sample_config.id)

        assert not temp_store.exists(sample_config.id)
        loaded = temp_store.load_all()
        assert len(loaded) == 0

    def test_delete_nonexistent(self, temp_store):
        """Test deleting a non-existent connector (should not error)."""
        # Should log warning but not raise
        temp_store.delete("nonexistent-id")


class TestConnectorStoreMultiple:
    """Test ConnectorStore with multiple connectors."""

    def test_save_multiple(self, temp_store):
        """Test saving multiple connectors."""
        configs = [
            MCPServerConfig(
                name=f"server_{i}",
                type=MCPTransportType.STDIO,
                command="echo",
                args=[str(i)],
                source="api",
            )
            for i in range(5)
        ]

        # Save all
        for config in configs:
            temp_store.save(config.id, config)

        # Load all
        loaded = temp_store.load_all()

        assert len(loaded) == 5

        # Verify all configs present
        for config in configs:
            assert config.id in loaded
            assert loaded[config.id].name == config.name

    def test_update_one_of_many(self, temp_store):
        """Test updating one connector among many."""
        configs = [
            MCPServerConfig(
                name=f"server_{i}",
                type=MCPTransportType.STDIO,
                command="echo",
                source="api",
            )
            for i in range(3)
        ]

        # Save all
        for config in configs:
            temp_store.save(config.id, config)

        # Update middle one
        configs[1].name = "updated_server"
        temp_store.save(configs[1].id, configs[1])

        # Load and verify
        loaded = temp_store.load_all()
        assert len(loaded) == 3
        assert loaded[configs[1].id].name == "updated_server"
        assert loaded[configs[0].id].name == "server_0"
        assert loaded[configs[2].id].name == "server_2"


class TestConnectorStorePersistence:
    """Test persistence across store instances."""

    def test_persistence_across_instances(self, sample_config):
        """Test data persists across ConnectorStore instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "connectors.json"

            # First instance - save
            store1 = ConnectorStore(store_path)
            store1.save(sample_config.id, sample_config)

            # Second instance - load
            store2 = ConnectorStore(store_path)
            loaded = store2.load_all()

            assert len(loaded) == 1
            assert loaded[sample_config.id].name == sample_config.name


class TestConnectorStoreEdgeCases:
    """Test edge cases and error handling."""

    def test_corrupted_json(self):
        """Test handling of corrupted JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "connectors.json"

            # Write invalid JSON
            with open(store_path, 'w') as f:
                f.write("{invalid json")

            # Initialize store - should handle gracefully
            store = ConnectorStore(store_path)

            # Should return empty dict (and backup corrupted file)
            loaded = store.load_all()
            assert loaded == {}

            # Backup file should exist
            backup_files = list(Path(tmpdir).glob("connectors.backup.*"))
            assert len(backup_files) == 1

    def test_empty_file(self):
        """Test handling of empty file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "connectors.json"

            # Create empty file
            store_path.touch()

            store = ConnectorStore(store_path)
            loaded = store.load_all()

            assert loaded == {}

    def test_special_characters_in_config(self, temp_store):
        """Test handling of special characters in config values."""
        config = MCPServerConfig(
            name="test-server!@#$%",
            type=MCPTransportType.STDIO,
            command="echo",
            args=["hello\nworld", "tab\there"],
            env={"KEY": "value with spaces & special chars!"},
            source="api",
        )

        temp_store.save(config.id, config)
        loaded = temp_store.get(config.id)

        assert loaded.name == config.name
        assert loaded.args == config.args
        assert loaded.env == config.env


class TestConnectorStoreTypes:
    """Test different connector types."""

    def test_stdio_connector(self, temp_store):
        """Test saving stdio connector."""
        config = MCPServerConfig(
            name="stdio_server",
            type=MCPTransportType.STDIO,
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem"],
            env={"PATH": "/allowed/path"},
            source="yaml",
        )

        temp_store.save(config.id, config)
        loaded = temp_store.get(config.id)

        assert loaded.type == MCPTransportType.STDIO
        assert loaded.command == "npx"
        assert loaded.url is None

    def test_sse_connector(self, temp_store):
        """Test saving SSE connector."""
        config = MCPServerConfig(
            name="sse_server",
            type=MCPTransportType.SSE,
            url="https://api.example.com/mcp",
            headers={"Authorization": "Bearer token123"},
            source="api",
        )

        temp_store.save(config.id, config)
        loaded = temp_store.get(config.id)

        assert loaded.type == MCPTransportType.SSE
        assert loaded.url == "https://api.example.com/mcp"
        assert loaded.command is None
        assert loaded.headers["Authorization"] == "Bearer token123"

    def test_mixed_types(self, temp_store):
        """Test saving both stdio and SSE connectors."""
        stdio_config = MCPServerConfig(
            name="stdio",
            type=MCPTransportType.STDIO,
            command="echo",
            source="api",
        )

        sse_config = MCPServerConfig(
            name="sse",
            type=MCPTransportType.SSE,
            url="https://example.com",
            source="api",
        )

        temp_store.save(stdio_config.id, stdio_config)
        temp_store.save(sse_config.id, sse_config)

        loaded = temp_store.load_all()
        assert len(loaded) == 2

        # Verify types preserved
        stdio_loaded = loaded[stdio_config.id]
        sse_loaded = loaded[sse_config.id]

        assert stdio_loaded.type == MCPTransportType.STDIO
        assert sse_loaded.type == MCPTransportType.SSE


class TestConnectorStoreMetadata:
    """Test connector metadata tracking."""

    def test_source_tracking(self, temp_store):
        """Test tracking connector source (yaml vs api)."""
        yaml_config = MCPServerConfig(
            name="yaml_connector",
            type=MCPTransportType.STDIO,
            command="echo",
            source="yaml",
        )

        api_config = MCPServerConfig(
            name="api_connector",
            type=MCPTransportType.STDIO,
            command="echo",
            source="api",
        )

        temp_store.save(yaml_config.id, yaml_config)
        temp_store.save(api_config.id, api_config)

        loaded = temp_store.load_all()

        assert loaded[yaml_config.id].source == "yaml"
        assert loaded[api_config.id].source == "api"

    def test_timestamps_preserved(self, temp_store, sample_config):
        """Test that timestamps are preserved across save/load."""
        original_created = sample_config.created_at
        original_updated = sample_config.updated_at

        temp_store.save(sample_config.id, sample_config)
        loaded = temp_store.get(sample_config.id)

        # created_at should be preserved
        assert loaded.created_at == original_created
        # updated_at will be newer (updated by save)
        assert loaded.updated_at >= original_updated

    def test_enabled_flag_persistence(self, temp_store):
        """Test enabled flag is persisted correctly."""
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

        temp_store.save(enabled_config.id, enabled_config)
        temp_store.save(disabled_config.id, disabled_config)

        loaded = temp_store.load_all()

        assert loaded[enabled_config.id].enabled is True
        assert loaded[disabled_config.id].enabled is False
