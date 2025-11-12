"""
ConnectorStore - Persistent storage for MCP connector configurations.

Provides JSON-based persistence with file locking for concurrent access safety.
"""

from pathlib import Path
import json
import logging
from typing import Optional
from datetime import datetime

try:
    from filelock import FileLock
except ImportError:
    FileLock = None

from sabre.server.mcp.models import MCPServerConfig, MCPTransportType

logger = logging.getLogger(__name__)


class ConnectorStore:
    """
    Manages persistent storage of MCP connector configurations.

    Storage format: JSON file with connector configs keyed by UUID.
    Location: ~/.local/state/sabre/connectors.json

    Thread-safe: Uses file-based locking to prevent concurrent access issues.
    """

    def __init__(self, storage_path: Path):
        """
        Initialize connector store.

        Args:
            storage_path: Path to JSON storage file
        """
        self.storage_path = storage_path
        self.lock_path = storage_path.with_suffix(".lock")

        # Initialize file locking if available
        if FileLock:
            self.lock = FileLock(self.lock_path, timeout=10)
        else:
            logger.warning("filelock not installed, concurrent access may cause issues")
            self.lock = None

        # Ensure parent directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize empty file if not exists
        if not self.storage_path.exists():
            self._write_data({})
            logger.info(f"Initialized empty connector store at {self.storage_path}")

    def save(self, connector_id: str, config: MCPServerConfig) -> None:
        """
        Save or update a connector configuration.

        Args:
            connector_id: UUID of the connector
            config: Connector configuration to save
        """
        if self.lock:
            with self.lock:
                self._save_unlocked(connector_id, config)
        else:
            self._save_unlocked(connector_id, config)

    def _save_unlocked(self, connector_id: str, config: MCPServerConfig) -> None:
        """Internal save without locking."""
        data = self._read_data()

        # Update timestamp
        config.updated_at = datetime.now().isoformat()

        data[connector_id] = self._config_to_dict(config)
        self._write_data(data)
        logger.info(f"Saved connector {connector_id} ({config.name}) to store")

    def load_all(self) -> dict[str, MCPServerConfig]:
        """
        Load all persisted connector configurations.

        Returns:
            Dictionary mapping connector_id to MCPServerConfig
        """
        if self.lock:
            with self.lock:
                return self._load_all_unlocked()
        else:
            return self._load_all_unlocked()

    def _load_all_unlocked(self) -> dict[str, MCPServerConfig]:
        """Internal load all without locking."""
        data = self._read_data()
        configs = {}

        for connector_id, config_dict in data.items():
            try:
                config = self._dict_to_config(config_dict)
                configs[connector_id] = config
            except Exception as e:
                logger.error(f"Failed to load connector {connector_id}: {e}")
                # Continue with other connectors

        logger.info(f"Loaded {len(configs)} connectors from store")
        return configs

    def delete(self, connector_id: str) -> None:
        """
        Delete a connector configuration.

        Args:
            connector_id: UUID of the connector to delete
        """
        if self.lock:
            with self.lock:
                self._delete_unlocked(connector_id)
        else:
            self._delete_unlocked(connector_id)

    def _delete_unlocked(self, connector_id: str) -> None:
        """Internal delete without locking."""
        data = self._read_data()

        if connector_id in data:
            connector_name = data[connector_id].get("name", "unknown")
            del data[connector_id]
            self._write_data(data)
            logger.info(f"Deleted connector {connector_id} ({connector_name}) from store")
        else:
            logger.warning(f"Connector {connector_id} not found in store")

    def get(self, connector_id: str) -> Optional[MCPServerConfig]:
        """
        Get a specific connector configuration.

        Args:
            connector_id: UUID of the connector

        Returns:
            MCPServerConfig if found, None otherwise
        """
        if self.lock:
            with self.lock:
                return self._get_unlocked(connector_id)
        else:
            return self._get_unlocked(connector_id)

    def _get_unlocked(self, connector_id: str) -> Optional[MCPServerConfig]:
        """Internal get without locking."""
        data = self._read_data()

        if connector_id in data:
            try:
                return self._dict_to_config(data[connector_id])
            except Exception as e:
                logger.error(f"Failed to load connector {connector_id}: {e}")
                return None

        return None

    def exists(self, connector_id: str) -> bool:
        """
        Check if a connector exists.

        Args:
            connector_id: UUID of the connector

        Returns:
            True if connector exists, False otherwise
        """
        if self.lock:
            with self.lock:
                data = self._read_data()
                return connector_id in data
        else:
            data = self._read_data()
            return connector_id in data

    def _read_data(self) -> dict:
        """
        Read data from JSON file.

        Returns:
            Dictionary of connector data
        """
        try:
            if not self.storage_path.exists():
                return {}

            with open(self.storage_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Corrupted connector store: {e}")
            # Backup corrupted file
            backup_path = self.storage_path.with_suffix(f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.storage_path.rename(backup_path)
            logger.warning(f"Backed up corrupted file to {backup_path}")
            return {}

    def _write_data(self, data: dict) -> None:
        """
        Write data to JSON file.

        Args:
            data: Dictionary to write
        """
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _config_to_dict(self, config: MCPServerConfig) -> dict:
        """
        Convert MCPServerConfig to dictionary.

        Args:
            config: Configuration to convert

        Returns:
            Dictionary representation
        """
        return {
            "id": config.id,
            "name": config.name,
            "type": config.type.value,
            "command": config.command,
            "args": config.args,
            "env": config.env,
            "url": config.url,
            "headers": config.headers,
            "enabled": config.enabled,
            "timeout": config.timeout,
            "source": config.source,
            "created_at": config.created_at,
            "updated_at": config.updated_at,
        }

    def _dict_to_config(self, data: dict) -> MCPServerConfig:
        """
        Convert dictionary to MCPServerConfig.

        Args:
            data: Dictionary to convert

        Returns:
            MCPServerConfig instance
        """
        return MCPServerConfig(
            id=data["id"],
            name=data["name"],
            type=MCPTransportType(data["type"]),
            command=data.get("command"),
            args=data.get("args", []),
            env=data.get("env", {}),
            url=data.get("url"),
            headers=data.get("headers", {}),
            enabled=data.get("enabled", True),
            timeout=data.get("timeout", 30),
            source=data.get("source", "api"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )
