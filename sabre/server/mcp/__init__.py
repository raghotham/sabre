"""
MCP (Model Context Protocol) Integration.

This package provides integration with the Model Context Protocol,
allowing SABRE to connect to external MCP servers and use their tools.
"""

from .models import (
    MCPTransportType,
    MCPServerConfig,
    MCPToolParameter,
    MCPTool,
    MCPContent,
    MCPToolResult,
    MCPResource,
    MCPResourceContent,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPError,
    MCPConnectionError,
    MCPTimeoutError,
    MCPProtocolError,
    MCPToolError,
    MCPServerNotFoundError,
    MCPToolNotFoundError,
)
from .client import MCPClient
from .client_manager import MCPClientManager
from .config import MCPConfigLoader
from .helper_adapter import MCPHelperAdapter

__all__ = [
    "MCPTransportType",
    "MCPServerConfig",
    "MCPToolParameter",
    "MCPTool",
    "MCPContent",
    "MCPToolResult",
    "MCPResource",
    "MCPResourceContent",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "MCPError",
    "MCPConnectionError",
    "MCPTimeoutError",
    "MCPProtocolError",
    "MCPToolError",
    "MCPServerNotFoundError",
    "MCPToolNotFoundError",
    "MCPClient",
    "MCPClientManager",
    "MCPConfigLoader",
    "MCPHelperAdapter",
]
