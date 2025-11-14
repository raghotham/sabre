"""
MCP Client Manager.

Manages connections to multiple MCP servers and provides a registry for tool routing.
"""

import asyncio
import logging
from typing import Optional
from datetime import datetime

from .models import (
    MCPServerConfig,
    MCPTool,
    MCPServerNotFoundError,
    MCPConnectionError,
)
from .client import MCPClient

logger = logging.getLogger(__name__)


class MCPClientManager:
    """
    Manages multiple MCP client connections.

    Responsibilities:
    - Connect/disconnect to MCP servers
    - Maintain server registry (UUID → client, name → UUID mappings)
    - Provide access to connected clients
    - Handle connection lifecycle
    - Persist connector state (if connector_store provided)
    """

    def __init__(self, connector_store=None):
        """
        Initialize MCP client manager.

        Args:
            connector_store: Optional ConnectorStore for persistence
        """
        # Core storage: UUID → MCPClient
        self.clients: dict[str, MCPClient] = {}
        # Core configs: UUID → MCPServerConfig
        self.configs: dict[str, MCPServerConfig] = {}

        # UUID mappings for lookup
        self.id_to_name: dict[str, str] = {}  # UUID → name
        self.name_to_id: dict[str, str] = {}  # name → UUID

        # Persistence layer (optional)
        self.connector_store = connector_store

    async def connect(self, config: MCPServerConfig) -> str:
        """
        Connect to an MCP server and persist config.

        Args:
            config: Server configuration

        Returns:
            connector_id: UUID of the connected connector

        Raises:
            MCPConnectionError: If connection fails
        """
        # Check if already registered (in configs)
        if config.id in self.configs:
            logger.warning(f"Connector already registered: {config.name} (id={config.id})")
            return config.id

        # Store config and mappings first (even if disabled)
        self.configs[config.id] = config
        self.id_to_name[config.id] = config.name
        self.name_to_id[config.name] = config.id

        # Persist if store available
        if self.connector_store:
            self.connector_store.save(config.id, config)

        # If disabled, don't create client
        if not config.enabled:
            logger.info(f"Registered disabled MCP server: {config.name} (id={config.id})")
            return config.id

        # Check if already connected by UUID
        if config.id in self.clients:
            logger.warning(f"Already connected to MCP server: {config.name} (id={config.id})")
            return config.id

        # Check if name already exists (different UUID)
        if config.name in self.name_to_id and self.name_to_id[config.name] != config.id:
            existing_id = self.name_to_id[config.name]
            logger.warning(f"Connector name '{config.name}' already exists with different ID: {existing_id}")
            # Allow it but warn (users might want multiple connectors with same name)

        try:
            # Create and connect client
            client = MCPClient(config)
            await client.connect()

            # Discover tools
            tools = await client.list_tools()
            logger.info(f"Connected to MCP server '{config.name}' (id={config.id}) with {len(tools)} tools: {[t.name for t in tools]}")

            # Store client by UUID
            self.clients[config.id] = client

            return config.id

        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{config.name}': {e}")

            # Clean up on connection failure
            if config.id in self.configs:
                del self.configs[config.id]
            if config.id in self.id_to_name:
                del self.id_to_name[config.id]
            if config.name in self.name_to_id and self.name_to_id[config.name] == config.id:
                del self.name_to_id[config.name]
            if self.connector_store:
                self.connector_store.delete(config.id)

            raise MCPConnectionError(f"Failed to connect to {config.name}: {e}") from e

    async def connect_all(self, configs: list[MCPServerConfig]) -> None:
        """
        Connect to multiple MCP servers.

        Args:
            configs: List of server configurations

        Note:
            Failed connections are logged but don't prevent other connections.
        """
        for config in configs:
            try:
                await self.connect(config)
            except Exception as e:
                logger.error(f"Failed to connect to {config.name}, skipping: {e}")
                # Continue with other servers

    async def disconnect(self, connector_id: str) -> None:
        """
        Disconnect and remove an MCP connector by UUID.

        Args:
            connector_id: UUID of the connector

        Raises:
            MCPServerNotFoundError: If connector not found
        """
        if connector_id not in self.configs:
            raise MCPServerNotFoundError(connector_id)

        config = self.configs[connector_id]
        name = config.name

        # Disconnect client if connected
        if connector_id in self.clients:
            client = self.clients[connector_id]
            await client.disconnect()
            del self.clients[connector_id]

        # Remove config
        del self.configs[connector_id]

        # Update mappings
        if connector_id in self.id_to_name:
            del self.id_to_name[connector_id]
        if name in self.name_to_id and self.name_to_id[name] == connector_id:
            del self.name_to_id[name]

        # Remove from persistence
        if self.connector_store:
            self.connector_store.delete(connector_id)

        logger.info(f"Disconnected from MCP server: {name} (id={connector_id})")

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers"""
        server_names = list(self.clients.keys())

        for name in server_names:
            try:
                await self.disconnect(name)
            except Exception as e:
                logger.error(f"Error disconnecting from {name}: {e}")

    async def reconnect(self, connector_id: str) -> None:
        """
        Reconnect to an MCP server by UUID.

        Args:
            connector_id: UUID of the connector

        Raises:
            MCPServerNotFoundError: If connector not found in configs
        """
        if connector_id not in self.configs:
            raise MCPServerNotFoundError(connector_id)

        config = self.configs[connector_id]

        # Disconnect if connected
        if connector_id in self.clients:
            try:
                await self.disconnect(connector_id)
            except Exception as e:
                logger.warning(f"Error during disconnect before reconnect: {e}")

        # Reconnect
        await self.connect(config)

    def get_client(self, connector_id: str) -> MCPClient:
        """
        Get MCP client for a connector by UUID.

        Args:
            connector_id: UUID of the connector

        Returns:
            MCP client instance

        Raises:
            MCPServerNotFoundError: If connector not found
        """
        if connector_id not in self.clients:
            raise MCPServerNotFoundError(connector_id)

        return self.clients[connector_id]

    def get_client_by_name(self, name: str) -> Optional[MCPClient]:
        """
        Get MCP client by connector name.

        Args:
            name: Connector name

        Returns:
            MCP client instance or None if not found
        """
        connector_id = self.name_to_id.get(name)
        if connector_id:
            return self.clients.get(connector_id)
        return None

    def has_connector(self, connector_id: str) -> bool:
        """
        Check if connector is registered by UUID.

        Args:
            connector_id: UUID of the connector

        Returns:
            True if connector is registered (even if disabled)
        """
        return connector_id in self.configs

    def list_connector_ids(self) -> list[str]:
        """
        Get list of registered connector UUIDs.

        Returns:
            List of connector UUIDs (includes both enabled and disabled)
        """
        return list(self.configs.keys())

    def list_servers(self) -> list[str]:
        """
        Get list of registered server names (for backward compatibility).

        Returns:
            List of server names (includes both enabled and disabled)
        """
        return [self.id_to_name.get(cid, "unknown") for cid in self.configs.keys()]

    async def get_all_tools(self) -> dict[str, list[MCPTool]]:
        """
        Get all tools from all connected servers.

        Returns:
            Dictionary mapping server name to list of tools
        """
        all_tools = {}

        for name, client in self.clients.items():
            try:
                tools = await client.list_tools()
                all_tools[name] = tools
            except Exception as e:
                logger.error(f"Error getting tools from {name}: {e}")
                all_tools[name] = []

        return all_tools

    async def health_check(self, name: str) -> bool:
        """
        Check health of an MCP server.

        Args:
            name: Server name

        Returns:
            True if server is healthy
        """
        if name not in self.clients:
            return False

        client = self.clients[name]

        try:
            # Try to list tools as a health check
            await client.list_tools()
            return True
        except Exception as e:
            logger.error(f"Health check failed for {name}: {e}")
            return False

    async def health_check_all(self) -> dict[str, bool]:
        """
        Check health of all connected servers.

        Returns:
            Dictionary mapping server name to health status
        """
        health_status = {}

        for name in self.clients.keys():
            health_status[name] = await self.health_check(name)

        return health_status

    def get_connector_info(self, connector_id: str) -> dict:
        """
        Get information about a connector by UUID.

        Args:
            connector_id: UUID of the connector

        Returns:
            Dictionary with connector info

        Raises:
            MCPServerNotFoundError: If connector not found
        """
        if connector_id not in self.configs:
            raise MCPServerNotFoundError(connector_id)

        config = self.configs[connector_id]
        client = self.clients.get(connector_id)

        return {
            "id": connector_id,
            "name": config.name,
            "type": config.type.value,
            "enabled": config.enabled,
            "connected": client.is_connected() if client else False,
            "tools_count": len(client.tools_cache) if client and client.tools_cache else 0,
            "source": config.source,
            "created_at": config.created_at,
            "updated_at": config.updated_at,
        }

    def get_server_info(self, name: str) -> dict:
        """
        Get information about a server by name (backward compatibility).

        Args:
            name: Server name

        Returns:
            Dictionary with server info

        Raises:
            MCPServerNotFoundError: If server not found
        """
        connector_id = self.name_to_id.get(name)
        if not connector_id:
            raise MCPServerNotFoundError(name)

        return self.get_connector_info(connector_id)

    def get_all_connector_info(self) -> list[dict]:
        """
        Get information about all connectors (including disabled ones).

        Returns:
            List of connector info dictionaries
        """
        return [self.get_connector_info(cid) for cid in self.configs.keys()]

    def get_all_server_info(self) -> list[dict]:
        """
        Get information about all servers (backward compatibility).

        Returns:
            List of server info dictionaries
        """
        return self.get_all_connector_info()

    async def get_connector_tools(self, connector_id: str) -> list[MCPTool]:
        """
        Get tools for a specific connector by UUID.

        Args:
            connector_id: UUID of the connector

        Returns:
            List of tools from the connector

        Raises:
            MCPServerNotFoundError: If connector not found
        """
        if connector_id not in self.clients:
            raise MCPServerNotFoundError(connector_id)

        client = self.clients[connector_id]
        return await client.list_tools()

    async def update_connector(self, connector_id: str, updates: dict) -> None:
        """
        Update connector configuration and reconnect.

        Args:
            connector_id: UUID of connector
            updates: Partial configuration updates

        Raises:
            MCPServerNotFoundError: If connector not found
        """
        if connector_id not in self.configs:
            raise MCPServerNotFoundError(connector_id)

        config = self.configs[connector_id]

        # Apply updates
        if "name" in updates:
            # Update name mapping
            old_name = config.name
            new_name = updates["name"]
            if old_name in self.name_to_id:
                del self.name_to_id[old_name]
            self.name_to_id[new_name] = connector_id
            self.id_to_name[connector_id] = new_name
            config.name = new_name

        if "command" in updates:
            config.command = updates["command"]
        if "args" in updates:
            config.args = updates["args"]
        if "env" in updates:
            config.env = updates["env"]
        if "url" in updates:
            config.url = updates["url"]
        if "headers" in updates:
            config.headers = updates["headers"]
        if "timeout" in updates:
            config.timeout = updates["timeout"]
        if "enabled" in updates:
            config.enabled = updates["enabled"]

        # Update timestamp
        config.updated_at = datetime.now().isoformat()

        # Persist changes
        if self.connector_store:
            self.connector_store.save(connector_id, config)

        # Reconnect to apply changes
        await self.reconnect(connector_id)

    async def enable_connector(self, connector_id: str) -> None:
        """
        Enable and connect a disabled connector.

        Args:
            connector_id: UUID of connector

        Raises:
            MCPServerNotFoundError: If connector not found
        """
        if connector_id not in self.configs:
            raise MCPServerNotFoundError(connector_id)

        config = self.configs[connector_id]
        config.enabled = True
        config.updated_at = datetime.now().isoformat()

        # Persist changes
        if self.connector_store:
            self.connector_store.save(connector_id, config)

        # Connect if not already connected
        if connector_id not in self.clients:
            try:
                # Create and connect client directly (don't call connect() which checks if already registered)
                client = MCPClient(config)
                await client.connect()

                # Discover tools
                tools = await client.list_tools()
                logger.info(f"Connected to MCP server '{config.name}' (id={config.id}) with {len(tools)} tools: {[t.name for t in tools]}")

                # Store client
                self.clients[config.id] = client
            except Exception as e:
                logger.error(f"Failed to enable connector '{config.name}': {e}")
                # Revert enabled state on failure
                config.enabled = False
                if self.connector_store:
                    self.connector_store.save(connector_id, config)
                raise MCPConnectionError(f"Failed to enable {config.name}: {e}") from e

    async def disable_connector(self, connector_id: str) -> None:
        """
        Disable and disconnect a connector without removing it.

        Args:
            connector_id: UUID of connector

        Raises:
            MCPServerNotFoundError: If connector not found
        """
        if connector_id not in self.configs:
            raise MCPServerNotFoundError(connector_id)

        config = self.configs[connector_id]
        config.enabled = False
        config.updated_at = datetime.now().isoformat()

        # Persist changes
        if self.connector_store:
            self.connector_store.save(connector_id, config)

        # Disconnect client if connected (but keep config)
        if connector_id in self.clients:
            client = self.clients[connector_id]
            await client.disconnect()
            del self.clients[connector_id]
            logger.info(f"Disabled connector: {config.name} (id={connector_id})")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect_all()
