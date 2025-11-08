"""
MCP Client Manager.

Manages connections to multiple MCP servers and provides a registry for tool routing.
"""

import asyncio
import logging
from typing import Optional

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
    - Maintain server registry (name → client)
    - Provide access to connected clients
    - Handle connection lifecycle
    """

    def __init__(self):
        """Initialize MCP client manager"""
        self.clients: dict[str, MCPClient] = {}
        self.configs: dict[str, MCPServerConfig] = {}

    async def connect(self, config: MCPServerConfig) -> None:
        """
        Connect to an MCP server.

        Args:
            config: Server configuration

        Raises:
            MCPConnectionError: If connection fails
        """
        if not config.enabled:
            logger.info(f"Skipping disabled MCP server: {config.name}")
            return

        if config.name in self.clients:
            logger.warning(f"Already connected to MCP server: {config.name}")
            return

        try:
            # Create and connect client
            client = MCPClient(config)
            await client.connect()

            # Discover tools
            tools = await client.list_tools()
            logger.info(f"Connected to MCP server '{config.name}' with {len(tools)} tools: {[t.name for t in tools]}")

            # Store client and config
            self.clients[config.name] = client
            self.configs[config.name] = config

        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{config.name}': {e}")
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

    async def disconnect(self, name: str) -> None:
        """
        Disconnect from an MCP server.

        Args:
            name: Server name

        Raises:
            MCPServerNotFoundError: If server not found
        """
        if name not in self.clients:
            raise MCPServerNotFoundError(name)

        client = self.clients[name]
        await client.disconnect()

        del self.clients[name]
        del self.configs[name]

        logger.info(f"Disconnected from MCP server: {name}")

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers"""
        server_names = list(self.clients.keys())

        for name in server_names:
            try:
                await self.disconnect(name)
            except Exception as e:
                logger.error(f"Error disconnecting from {name}: {e}")

    async def reconnect(self, name: str) -> None:
        """
        Reconnect to an MCP server.

        Args:
            name: Server name

        Raises:
            MCPServerNotFoundError: If server not found in configs
        """
        if name not in self.configs:
            raise MCPServerNotFoundError(name)

        config = self.configs[name]

        # Disconnect if connected
        if name in self.clients:
            try:
                await self.disconnect(name)
            except Exception as e:
                logger.warning(f"Error during disconnect before reconnect: {e}")

        # Reconnect
        await self.connect(config)

    def get_client(self, name: str) -> MCPClient:
        """
        Get MCP client for a server.

        Args:
            name: Server name

        Returns:
            MCP client instance

        Raises:
            MCPServerNotFoundError: If server not found
        """
        if name not in self.clients:
            raise MCPServerNotFoundError(name)

        return self.clients[name]

    def has_server(self, name: str) -> bool:
        """
        Check if server is connected.

        Args:
            name: Server name

        Returns:
            True if server is connected
        """
        return name in self.clients

    def list_servers(self) -> list[str]:
        """
        Get list of connected server names.

        Returns:
            List of server names
        """
        return list(self.clients.keys())

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

    def get_server_info(self, name: str) -> dict:
        """
        Get information about a server.

        Args:
            name: Server name

        Returns:
            Dictionary with server info

        Raises:
            MCPServerNotFoundError: If server not found
        """
        if name not in self.clients:
            raise MCPServerNotFoundError(name)

        config = self.configs[name]
        client = self.clients[name]

        return {
            "name": name,
            "type": config.type.value,
            "enabled": config.enabled,
            "connected": client.is_connected(),
            "tools_count": len(client.tools_cache) if client.tools_cache else 0,
        }

    def get_all_server_info(self) -> list[dict]:
        """
        Get information about all servers.

        Returns:
            List of server info dictionaries
        """
        return [self.get_server_info(name) for name in self.clients.keys()]

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect_all()
