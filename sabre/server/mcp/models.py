"""
MCP data models and types.

This module defines the data structures used for MCP protocol communication.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from enum import Enum


class MCPTransportType(Enum):
    """MCP transport types"""

    STDIO = "stdio"
    SSE = "sse"


@dataclass
class MCPServerConfig:
    """
    Configuration for an MCP server.

    Example (stdio):
        MCPServerConfig(
            name="postgres",
            type=MCPTransportType.STDIO,
            command="npx",
            args=["-y", "@modelcontextprotocol/server-postgres"],
            env={"POSTGRES_URL": "postgresql://localhost/mydb"},
            enabled=True
        )

    Example (SSE):
        MCPServerConfig(
            name="remote_api",
            type=MCPTransportType.SSE,
            url="https://my-mcp-server.com/mcp",
            headers={"Authorization": "Bearer token123"},
            enabled=True
        )
    """

    name: str
    type: MCPTransportType
    command: Optional[str] = None  # For stdio transport
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: Optional[str] = None  # For SSE transport
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    timeout: int = 30  # Timeout in seconds for operations

    def __post_init__(self):
        """Validate configuration"""
        if self.type == MCPTransportType.STDIO:
            if not self.command:
                raise ValueError(f"stdio transport requires 'command' for server {self.name}")
        elif self.type == MCPTransportType.SSE:
            if not self.url:
                raise ValueError(f"SSE transport requires 'url' for server {self.name}")


@dataclass
class MCPToolParameter:
    """
    MCP tool parameter definition.

    Based on JSON Schema for input validation.
    """

    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: Optional[str] = None
    required: bool = False
    default: Any = None
    properties: Optional[dict[str, Any]] = None  # For object type
    items: Optional[dict[str, Any]] = None  # For array type


@dataclass
class MCPTool:
    """
    MCP tool definition.

    Represents a tool exposed by an MCP server.
    """

    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema for tool input
    server_name: str = ""  # Added by client manager during registration

    def get_signature(self) -> str:
        """
        Generate Python-style function signature for tool.

        Example:
            query_database(query: str, params: list = None) -> Any
        """
        params = []

        # Extract parameters from input schema
        properties = self.input_schema.get("properties", {})
        required = set(self.input_schema.get("required", []))

        for param_name, param_schema in properties.items():
            param_type = param_schema.get("type", "Any")
            param_desc = param_schema.get("description", "")

            # Map JSON Schema types to Python type hints
            type_map = {
                "string": "str",
                "number": "float",
                "integer": "int",
                "boolean": "bool",
                "array": "list",
                "object": "dict",
            }
            python_type = type_map.get(param_type, "Any")

            # Add parameter with type hint
            if param_name in required:
                params.append(f"{param_name}: {python_type}")
            else:
                params.append(f"{param_name}: {python_type} = None")

        params_str = ", ".join(params)
        return f"{self.name}({params_str})"

    def get_documentation(self) -> str:
        """
        Generate documentation string for tool.

        Example:
            query_database(query: str, params: list = None)
            Execute SQL query on connected database.
        """
        return f"{self.get_signature()}\n{self.description}"


@dataclass
class MCPContent:
    """
    MCP content type.

    MCP tools can return different content types:
    - text: Plain text
    - image: Base64 encoded image
    - resource: Reference to a resource
    """

    type: Literal["text", "image", "resource"]
    text: Optional[str] = None
    data: Optional[str] = None  # Base64 for images
    mimeType: Optional[str] = None
    uri: Optional[str] = None  # For resources


@dataclass
class MCPToolResult:
    """
    Result from MCP tool invocation.

    Contains list of content items returned by the tool.
    """

    content: list[MCPContent] = field(default_factory=list)
    is_error: bool = False

    def get_text(self) -> str:
        """Extract all text content as a single string"""
        texts = [c.text for c in self.content if c.type == "text" and c.text]
        return "\n".join(texts)


@dataclass
class MCPResource:
    """
    MCP resource definition.

    Resources are data sources that can be read (e.g., files, database schemas).
    """

    uri: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None


@dataclass
class MCPResourceContent:
    """Content returned from reading an MCP resource"""

    uri: str
    mimeType: Optional[str] = None
    text: Optional[str] = None
    blob: Optional[str] = None  # Base64 for binary content


# JSON-RPC Types


@dataclass
class JSONRPCRequest:
    """JSON-RPC 2.0 request"""

    jsonrpc: str = "2.0"
    method: str = ""
    params: Optional[dict[str, Any]] = None
    id: Optional[int | str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {"jsonrpc": self.jsonrpc, "method": self.method}
        if self.params is not None:
            result["params"] = self.params
        if self.id is not None:
            result["id"] = self.id
        return result


@dataclass
class JSONRPCResponse:
    """JSON-RPC 2.0 response"""

    jsonrpc: str = "2.0"
    id: Optional[int | str] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[dict[str, Any]] = None

    @property
    def is_error(self) -> bool:
        """Check if response contains an error"""
        return self.error is not None


# Exceptions


class MCPError(Exception):
    """Base exception for MCP-related errors"""

    pass


class MCPConnectionError(MCPError):
    """Error connecting to MCP server"""

    pass


class MCPTimeoutError(MCPError):
    """MCP operation timed out"""

    pass


class MCPProtocolError(MCPError):
    """MCP protocol violation or invalid message"""

    pass


class MCPToolError(MCPError):
    """Error executing MCP tool"""

    def __init__(self, message: str, tool_name: str = "", server_name: str = ""):
        super().__init__(message)
        self.tool_name = tool_name
        self.server_name = server_name


class MCPServerNotFoundError(MCPError):
    """MCP server not found in registry"""

    def __init__(self, server_name: str):
        super().__init__(f"MCP server not found: {server_name}")
        self.server_name = server_name


class MCPToolNotFoundError(MCPError):
    """MCP tool not found"""

    def __init__(self, tool_name: str, server_name: str = ""):
        super().__init__(f"Tool '{tool_name}' not found on server '{server_name}'")
        self.tool_name = tool_name
        self.server_name = server_name
