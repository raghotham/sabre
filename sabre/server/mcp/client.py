"""
MCP Client implementation.

Handles low-level JSON-RPC communication with MCP servers via stdio or SSE transports.
"""

import asyncio
import json
import logging
from typing import Any, Optional
from asyncio.subprocess import Process

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from .models import (
    MCPServerConfig,
    MCPTool,
    MCPToolResult,
    MCPContent,
    MCPResource,
    MCPResourceContent,
    MCPTransportType,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPConnectionError,
    MCPTimeoutError,
    MCPProtocolError,
    MCPToolError,
)

logger = logging.getLogger(__name__)


class MCPClient:
    """
    MCP Client for communicating with MCP servers.

    Supports stdio and SSE transports.
    """

    def __init__(self, config: MCPServerConfig):
        """
        Initialize MCP client.

        Args:
            config: Server configuration
        """
        self.config = config
        self.process: Optional[Process] = None
        self.http_client: Optional[Any] = None  # httpx.AsyncClient for SSE
        self.next_id = 1
        self.connected = False
        self.tools_cache: Optional[list[MCPTool]] = None
        self._disconnect_lock = asyncio.Lock()  # Prevent race during shutdown

    async def connect(self) -> None:
        """
        Connect to MCP server.

        Raises:
            MCPConnectionError: If connection fails
        """
        if self.connected:
            logger.warning(f"Already connected to {self.config.name}")
            return

        try:
            # Step 1: Establish transport connection (stdio subprocess or HTTP client)
            if self.config.type == MCPTransportType.STDIO:
                await self._connect_stdio()
            elif self.config.type == MCPTransportType.SSE:
                await self._connect_sse()

            # Step 2: Perform MCP protocol initialization handshake
            # This is required by the MCP specification before making any other requests
            await self._initialize_protocol()

            self.connected = True
            logger.info(f"Connected to MCP server: {self.config.name} ({self.config.type.value})")
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.config.name}: {e}")
            raise MCPConnectionError(f"Failed to connect to {self.config.name}: {e}") from e

    async def _connect_stdio(self) -> None:
        """Connect via stdio transport (subprocess)"""
        try:
            import os

            # Build command and arguments
            cmd = [self.config.command] + self.config.args

            logger.debug(f"Starting MCP server process: {' '.join(cmd)}")

            # Merge config env with current environment (config env overrides)
            process_env = {**os.environ, **self.config.env}

            # Spawn subprocess with stdio pipes
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=process_env,
            )

            # Wait a bit for process to start
            await asyncio.sleep(0.5)

            # Check if process is still alive
            if self.process.returncode is not None:
                stderr = await self.process.stderr.read()
                raise MCPConnectionError(f"Process exited immediately: {stderr.decode()}")

        except Exception as e:
            if self.process:
                self.process.kill()
                await self.process.wait()
            raise MCPConnectionError(f"Failed to start stdio process: {e}") from e

    async def _connect_sse(self) -> None:
        """Connect via SSE transport (HTTP)"""
        if not HTTPX_AVAILABLE:
            raise MCPConnectionError("httpx is required for SSE transport. Install with: pip install httpx")

        try:
            # Create HTTP client with custom headers and timeout
            self.http_client = httpx.AsyncClient(
                headers=self.config.headers,
                timeout=httpx.Timeout(self.config.timeout, connect=10.0),
                follow_redirects=True,
            )

            # Test connection with a simple request
            logger.debug(f"Connecting to SSE endpoint: {self.config.url}")

            # We don't actually "connect" with HTTP - connection is per-request
            # Try to validate the endpoint is accessible by attempting a health check
            # or just accept that we'll find out on first real request
            # (Many MCP servers only accept POST to their main endpoint)
            try:
                # Try to access the base URL (remove /mcp path if present) for health check
                from urllib.parse import urlparse
                parsed = urlparse(self.config.url)
                base_url = f"{parsed.scheme}://{parsed.netloc}"

                # Try health endpoint first
                try:
                    health_response = await self.http_client.get(f"{base_url}/health", timeout=2.0)
                    logger.debug(f"Health check response: {health_response.status_code}")
                except:
                    # Health endpoint not available, that's okay
                    logger.debug("Health endpoint not available, will validate on first request")
            except httpx.HTTPError as e:
                # Connection test failed, but we'll let it through
                # The real validation happens on first tools/list call
                logger.debug(f"Connection pre-check failed (will retry on first request): {e}")

        except Exception as e:
            if self.http_client:
                await self.http_client.aclose()
                self.http_client = None
            raise MCPConnectionError(f"Failed to connect via SSE: {e}") from e

    async def _initialize_protocol(self) -> None:
        """
        Perform MCP protocol initialization handshake.

        According to the MCP specification, clients must:
        1. Send 'initialize' request with client info and capabilities
        2. Receive server info and capabilities in response
        3. Send 'notifications/initialized' notification

        This must be done before any other requests (like tools/list).

        Raises:
            MCPProtocolError: If initialization fails
        """
        logger.debug(f"[{self.config.name}] Starting MCP protocol initialization")

        # Step 1 & 2: Send initialize request and receive server capabilities
        init_response = await self._send_request_internal("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "sabre",
                "version": "1.0.0"
            }
        })

        if init_response.is_error:
            error_msg = init_response.error.get("message", "Unknown error") if init_response.error else "Unknown error"
            raise MCPProtocolError(f"MCP initialization failed: {error_msg}")

        logger.debug(f"[{self.config.name}] Received server capabilities: {init_response.result}")

        # Step 3: Send initialized notification (no response expected)
        await self._send_notification("notifications/initialized")

        logger.debug(f"[{self.config.name}] MCP protocol initialization complete")

    async def _send_notification(self, method: str, params: Optional[dict[str, Any]] = None) -> None:
        """
        Send JSON-RPC notification (no response expected).

        Notifications are one-way messages that don't expect a response.

        Args:
            method: JSON-RPC method name
            params: Optional method parameters
        """
        # Build JSON-RPC notification (no 'id' field)
        notification = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params:
            notification["params"] = params

        notification_json = json.dumps(notification)
        logger.debug(f"[{self.config.name}] Sending notification: {notification_json}")

        # Send notification based on transport type
        if self.config.type == MCPTransportType.STDIO:
            await self._write_stdio(notification)
        elif self.config.type == MCPTransportType.SSE:
            # SSE transport typically doesn't use notifications
            # But if needed, we could POST without expecting response
            logger.debug(f"[{self.config.name}] Skipping notification for SSE transport")

    async def _write_stdio(self, data: dict) -> None:
        """
        Write JSON data to stdio without expecting a response.

        Args:
            data: Dictionary to serialize and write
        """
        if not self.process or not self.process.stdin:
            raise MCPConnectionError("Process stdin not available")

        json_data = json.dumps(data) if isinstance(data, dict) else data
        self.process.stdin.write((json_data + "\n").encode())
        await self.process.stdin.drain()

    async def disconnect(self) -> None:
        """Disconnect from MCP server"""
        # Use lock to prevent race conditions during shutdown
        async with self._disconnect_lock:
            if not self.connected:
                return

            try:
                # Mark as disconnected first to prevent new requests
                self.connected = False

                if self.config.type == MCPTransportType.STDIO and self.process:
                    # Terminate process gracefully
                    self.process.terminate()
                    try:
                        await asyncio.wait_for(self.process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.warning(f"Process {self.config.name} did not terminate, killing")
                        self.process.kill()
                        await self.process.wait()

                elif self.config.type == MCPTransportType.SSE and self.http_client:
                    # Close HTTP client
                    await self.http_client.aclose()
                    self.http_client = None

                self.tools_cache = None
                logger.info(f"Disconnected from MCP server: {self.config.name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {self.config.name}: {e}")

    async def _send_request(self, method: str, params: Optional[dict[str, Any]] = None) -> JSONRPCResponse:
        """
        Send JSON-RPC request and wait for response.

        Args:
            method: JSON-RPC method name
            params: Method parameters

        Returns:
            JSON-RPC response

        Raises:
            MCPConnectionError: If not connected or connection lost
            MCPTimeoutError: If request times out
            MCPProtocolError: If response is invalid
        """
        if not self.connected:
            raise MCPConnectionError(f"Not connected to {self.config.name}")

        return await self._send_request_internal(method, params)

    async def _send_request_internal(self, method: str, params: Optional[dict[str, Any]] = None) -> JSONRPCResponse:
        """
        Internal method to send JSON-RPC request without checking connection status.

        This is used during initialization when self.connected is not yet True.

        Args:
            method: JSON-RPC method name
            params: Method parameters

        Returns:
            JSON-RPC response

        Raises:
            MCPTimeoutError: If request times out
            MCPProtocolError: If response is invalid
        """
        # Build JSON-RPC request
        request = JSONRPCRequest(method=method, params=params, id=self.next_id)
        self.next_id += 1

        request_json = json.dumps(request.to_dict())
        logger.debug(f"[{self.config.name}] Sending: {request_json}")

        try:
            if self.config.type == MCPTransportType.STDIO:
                return await self._send_stdio_request(request_json)
            elif self.config.type == MCPTransportType.SSE:
                return await self._send_sse_request(request_json)
        except asyncio.TimeoutError as e:
            raise MCPTimeoutError(f"Request to {self.config.name} timed out") from e
        except Exception as e:
            logger.error(f"Error sending request to {self.config.name}: {e}")
            raise MCPProtocolError(f"Failed to send request: {e}") from e

    async def _send_stdio_request(self, request_json: str) -> JSONRPCResponse:
        """Send request via stdio transport"""
        if not self.process or not self.process.stdin or not self.process.stdout:
            raise MCPConnectionError("Process not available")

        if self.process.returncode is not None:
            raise MCPConnectionError(f"Process terminated with exit code {self.process.returncode}")

        # Extract request ID for validation (Fix Bug #3)
        request_data = json.loads(request_json)
        request_id = request_data.get("id")

        # Write request to stdin
        self.process.stdin.write((request_json + "\n").encode())
        await self.process.stdin.drain()

        # Read response from stdout (wait for timeout)
        try:
            response_json = ""
            max_empty_lines = 10  # Prevent infinite loop on misbehaving servers

            # Skip empty lines from server debug output
            for _ in range(max_empty_lines):
                try:
                    response_line = await asyncio.wait_for(
                        self.process.stdout.readline(),
                        timeout=self.config.timeout
                    )
                except StopAsyncIteration:
                    # Stream ended without data (empty response)
                    break

                response_json = response_line.decode().strip()

                if response_json:
                    break

            # Check if we got a response after skipping empty lines
            if not response_json:
                # Check if process died while waiting
                if self.process.returncode is not None:
                    raise MCPConnectionError(f"Process terminated with exit code {self.process.returncode}")
                raise MCPProtocolError("Empty response from server")

            logger.debug(f"[{self.config.name}] Received: {response_json}")

            # Parse JSON-RPC response
            response_data = json.loads(response_json)
            response = JSONRPCResponse(
                jsonrpc=response_data.get("jsonrpc", "2.0"),
                id=response_data.get("id"),
                result=response_data.get("result"),
                error=response_data.get("error"),
            )

            # Validate response ID matches request ID
            if request_id is not None and response.id != request_id:
                raise MCPProtocolError(
                    f"Response ID mismatch: expected {request_id}, got {response.id}"
                )

            return response

        except asyncio.TimeoutError:
            # Check if process died during timeout
            if self.process.returncode is not None:
                raise MCPConnectionError(f"Process terminated with exit code {self.process.returncode}")
            raise MCPTimeoutError(f"Request timed out after {self.config.timeout}s")
        except json.JSONDecodeError as e:
            raise MCPProtocolError(f"Invalid JSON response: {e}")

    async def _send_sse_request(self, request_json: str) -> JSONRPCResponse:
        """Send request via SSE transport (HTTP POST)"""
        if not self.http_client:
            raise MCPConnectionError("HTTP client not initialized")

        try:
            # Parse JSON to send as proper JSON body
            request_data = json.loads(request_json)

            logger.debug(f"[{self.config.name}] Sending HTTP POST to {self.config.url}")

            # Send POST request with JSON-RPC payload
            # Include Accept headers for Streamable HTTP protocol compatibility
            response = await self.http_client.post(
                self.config.url,
                json=request_data,
                timeout=self.config.timeout,
                headers={
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                },
            )

            # Check HTTP status
            if response.status_code >= 400:
                raise MCPProtocolError(f"HTTP {response.status_code}: {response.text[:200]}")

            # Parse JSON-RPC response
            response_data = response.json()

            logger.debug(f"[{self.config.name}] Received: {json.dumps(response_data)}")

            # Build JSONRPCResponse
            json_rpc_response = JSONRPCResponse(
                jsonrpc=response_data.get("jsonrpc", "2.0"),
                id=response_data.get("id"),
                result=response_data.get("result"),
                error=response_data.get("error"),
            )

            return json_rpc_response

        except httpx.HTTPError as e:
            raise MCPProtocolError(f"HTTP request failed: {e}") from e
        except json.JSONDecodeError as e:
            raise MCPProtocolError(f"Invalid JSON response: {e}") from e
        except asyncio.TimeoutError as e:
            raise MCPTimeoutError(f"Request timed out after {self.config.timeout}s") from e

    async def list_tools(self) -> list[MCPTool]:
        """
        Discover available tools from MCP server.

        Returns:
            List of available tools

        Raises:
            MCPConnectionError: If not connected
            MCPProtocolError: If response is invalid
        """
        # Return cached tools if available
        if self.tools_cache is not None:
            return self.tools_cache

        response = await self._send_request("tools/list")

        if response.is_error:
            error_msg = response.error.get("message", "Unknown error") if response.error else "Unknown error"
            raise MCPProtocolError(f"tools/list failed: {error_msg}")

        if not response.result:
            raise MCPProtocolError("tools/list returned no result")

        # Parse tools from response
        tools_data = response.result.get("tools", [])
        tools = []

        for tool_data in tools_data:
            tool = MCPTool(
                name=tool_data.get("name", ""),
                description=tool_data.get("description", ""),
                input_schema=tool_data.get("inputSchema", {}),
                server_name=self.config.name,
            )
            tools.append(tool)

        logger.info(f"Discovered {len(tools)} tools from {self.config.name}: {[t.name for t in tools]}")

        # Cache tools
        self.tools_cache = tools
        return tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """
        Invoke an MCP tool.

        Args:
            tool_name: Name of tool to invoke
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            MCPToolError: If tool execution fails
        """
        params = {"name": tool_name, "arguments": arguments}

        try:
            response = await self._send_request("tools/call", params)

            if response.is_error:
                error_msg = response.error.get("message", "Unknown error") if response.error else "Unknown error"
                raise MCPToolError(error_msg, tool_name=tool_name, server_name=self.config.name)

            if not response.result:
                raise MCPToolError("Tool returned no result", tool_name=tool_name, server_name=self.config.name)

            # Parse content from result
            content_data = response.result.get("content", [])
            content_list = []

            for item in content_data:
                content = MCPContent(
                    type=item.get("type", "text"),
                    text=item.get("text"),
                    data=item.get("data"),
                    mimeType=item.get("mimeType"),
                    uri=item.get("uri"),
                )
                content_list.append(content)

            result = MCPToolResult(content=content_list, is_error=False)
            logger.debug(f"Tool {tool_name} returned {len(content_list)} content items")

            return result

        except MCPToolError:
            raise
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            raise MCPToolError(f"Tool execution failed: {e}", tool_name=tool_name, server_name=self.config.name) from e

    async def list_resources(self) -> list[MCPResource]:
        """
        List available resources from MCP server.

        Returns:
            List of available resources

        Raises:
            MCPConnectionError: If not connected
            MCPProtocolError: If response is invalid
        """
        response = await self._send_request("resources/list")

        if response.is_error:
            error_msg = response.error.get("message", "Unknown error") if response.error else "Unknown error"
            raise MCPProtocolError(f"resources/list failed: {error_msg}")

        if not response.result:
            return []

        # Parse resources from response
        resources_data = response.result.get("resources", [])
        resources = []

        for resource_data in resources_data:
            resource = MCPResource(
                uri=resource_data.get("uri", ""),
                name=resource_data.get("name", ""),
                description=resource_data.get("description"),
                mimeType=resource_data.get("mimeType"),
            )
            resources.append(resource)

        logger.info(f"Discovered {len(resources)} resources from {self.config.name}")
        return resources

    async def read_resource(self, uri: str) -> MCPResourceContent:
        """
        Read content from an MCP resource.

        Args:
            uri: Resource URI to read

        Returns:
            Resource content

        Raises:
            MCPProtocolError: If read fails
        """
        params = {"uri": uri}
        response = await self._send_request("resources/read", params)

        if response.is_error:
            error_msg = response.error.get("message", "Unknown error") if response.error else "Unknown error"
            raise MCPProtocolError(f"resources/read failed: {error_msg}")

        if not response.result:
            raise MCPProtocolError("resources/read returned no result")

        # Parse resource content
        contents = response.result.get("contents", [])
        if not contents:
            raise MCPProtocolError("No content in resource")

        content_data = contents[0]  # Take first content item
        resource_content = MCPResourceContent(
            uri=content_data.get("uri", uri),
            mimeType=content_data.get("mimeType"),
            text=content_data.get("text"),
            blob=content_data.get("blob"),
        )

        return resource_content

    def is_connected(self) -> bool:
        """Check if client is connected to server"""
        return self.connected

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
