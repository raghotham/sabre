"""
MCP Configuration Loader.

Loads and parses MCP server configurations from mcp.yaml.
"""

import os
import logging
from pathlib import Path
from typing import Optional
import re

try:
    import yaml
except ImportError:
    yaml = None

from sabre.common.paths import SabrePaths
from .models import MCPServerConfig, MCPTransportType

logger = logging.getLogger(__name__)


class MCPConfigLoader:
    """
    Load MCP server configurations from YAML file.

    Configuration file location:
        ~/.config/sabre/mcp.yaml (or $XDG_CONFIG_HOME/sabre/mcp.yaml)
    """

    DEFAULT_CONFIG_FILENAME = "mcp.yaml"

    @staticmethod
    def get_config_path() -> Path:
        """
        Get path to MCP configuration file.

        Returns:
            Path to mcp.yaml
        """
        config_dir = SabrePaths.get_config_home()
        return config_dir / MCPConfigLoader.DEFAULT_CONFIG_FILENAME

    @staticmethod
    def load(config_path: Optional[Path] = None) -> list[MCPServerConfig]:
        """
        Load MCP server configurations from YAML file.

        Args:
            config_path: Optional path to config file. If None, uses default location.

        Returns:
            List of MCP server configurations

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config is invalid
        """
        if yaml is None:
            logger.warning("PyYAML not installed, cannot load MCP config. Install with: pip install pyyaml")
            return []

        # Use default path if not specified
        if config_path is None:
            config_path = MCPConfigLoader.get_config_path()

        if not config_path.exists():
            logger.info(f"MCP config file not found: {config_path}")
            return []

        logger.info(f"Loading MCP config from: {config_path}")

        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)

            if not config_data or "mcp_servers" not in config_data:
                logger.warning("No 'mcp_servers' section in config file")
                return []

            servers_data = config_data["mcp_servers"]

            # Handle case where all servers are commented out (servers_data is None)
            if servers_data is None:
                logger.info("No MCP servers configured (all commented out)")
                return []

            if not isinstance(servers_data, dict):
                raise ValueError("'mcp_servers' must be a dictionary")

            # Parse each server config
            configs = []
            for name, server_data in servers_data.items():
                try:
                    config = MCPConfigLoader._parse_server_config(name, server_data)
                    configs.append(config)
                except Exception as e:
                    logger.error(f"Error parsing config for server '{name}': {e}")
                    # Continue with other servers

            logger.info(f"Loaded {len(configs)} MCP server configs")
            return configs

        except Exception as e:
            logger.error(f"Error loading MCP config from {config_path}: {e}")
            raise

    @staticmethod
    def _parse_server_config(name: str, data: dict) -> MCPServerConfig:
        """
        Parse a single server configuration.

        Args:
            name: Server name
            data: Server configuration data

        Returns:
            MCPServerConfig instance

        Raises:
            ValueError: If config is invalid
        """
        if not isinstance(data, dict):
            raise ValueError(f"Server config must be a dictionary, got {type(data)}")

        # Parse transport type
        transport_type_str = data.get("type", "stdio")
        try:
            transport_type = MCPTransportType(transport_type_str)
        except ValueError:
            raise ValueError(f"Invalid transport type: {transport_type_str}")

        # Expand environment variables in all string values
        command = MCPConfigLoader._expand_env(data.get("command"))
        args = [MCPConfigLoader._expand_env(arg) for arg in data.get("args", [])]
        env = {k: MCPConfigLoader._expand_env(v) for k, v in data.get("env", {}).items()}
        url = MCPConfigLoader._expand_env(data.get("url"))
        headers = {k: MCPConfigLoader._expand_env(v) for k, v in data.get("headers", {}).items()}

        enabled = data.get("enabled", True)
        timeout = data.get("timeout", 30)

        return MCPServerConfig(
            name=name,
            type=transport_type,
            command=command,
            args=args,
            env=env,
            url=url,
            headers=headers,
            enabled=enabled,
            timeout=timeout,
        )

    @staticmethod
    def _expand_env(value: Optional[str]) -> Optional[str]:
        """
        Expand environment variables in string value.

        Supports both ${VAR} and $VAR syntax.

        Args:
            value: String value to expand

        Returns:
            Expanded string, or None if value is None
        """
        if value is None:
            return None

        if not isinstance(value, str):
            return value

        # Replace ${VAR} with environment variable value
        def replace_var(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        # Handle ${VAR} syntax
        expanded = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace_var, value)

        # Handle $VAR syntax
        expanded = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", replace_var, expanded)

        return expanded

    @staticmethod
    def create_example_config(config_path: Optional[Path] = None) -> Path:
        """
        Create an example MCP configuration file.

        Args:
            config_path: Optional path for config file. If None, uses default location.

        Returns:
            Path to created config file
        """
        if config_path is None:
            config_path = MCPConfigLoader.get_config_path()

        # Create config directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)

        example_config = """# SABRE MCP Server Configuration
#
# This file configures external MCP (Model Context Protocol) servers
# that SABRE can connect to for additional tools and capabilities.
#
# See: https://modelcontextprotocol.io/

mcp_servers:
  # Example: Postgres database server
  # Uncomment and configure to enable
  # postgres:
  #   type: stdio
  #   command: npx
  #   args: ["-y", "@modelcontextprotocol/server-postgres"]
  #   env:
  #     POSTGRES_URL: "${POSTGRES_URL}"  # Read from environment
  #   enabled: false

  # Example: GitHub integration
  # github:
  #   type: stdio
  #   command: npx
  #   args: ["-y", "@modelcontextprotocol/server-github"]
  #   env:
  #     GITHUB_TOKEN: "${GITHUB_TOKEN}"
  #   enabled: false

  # Example: Filesystem access (with restricted path)
  # filesystem:
  #   type: stdio
  #   command: npx
  #   args: ["-y", "@modelcontextprotocol/server-filesystem", "/allowed/path"]
  #   enabled: false

  # Example: Custom server (local script)
  # custom:
  #   type: stdio
  #   command: python
  #   args: ["/path/to/your/mcp_server.py"]
  #   env:
  #     API_KEY: "${MY_API_KEY}"
  #   enabled: false
  #   timeout: 60  # Custom timeout in seconds

  # Example: Remote MCP server via SSE (HTTP)
  # remote_api:
  #   type: sse
  #   url: "https://my-mcp-server.com/mcp"
  #   headers:
  #     Authorization: "Bearer ${API_TOKEN}"
  #     X-API-Key: "${API_KEY}"
  #   enabled: false
  #   timeout: 30

# Notes:
# - Use ${VAR} or $VAR to reference environment variables
# - Only enabled servers will be connected at startup
# - stdio: Local subprocess servers (command + args)
# - sse: Remote HTTP servers (url + headers)
"""

        with open(config_path, "w") as f:
            f.write(example_config)

        logger.info(f"Created example MCP config at: {config_path}")
        return config_path

    @staticmethod
    def validate_config(config: MCPServerConfig) -> list[str]:
        """
        Validate an MCP server configuration.

        Args:
            config: Configuration to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required fields based on transport type
        if config.type == MCPTransportType.STDIO:
            if not config.command:
                errors.append(f"stdio transport requires 'command' for server {config.name}")
        elif config.type == MCPTransportType.SSE:
            if not config.url:
                errors.append(f"SSE transport requires 'url' for server {config.name}")

        # Check timeout is positive
        if config.timeout <= 0:
            errors.append(f"timeout must be positive, got {config.timeout}")

        return errors
