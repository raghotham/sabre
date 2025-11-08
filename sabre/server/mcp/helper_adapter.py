"""
MCP Helper Adapter.

Bridges MCP tools into SABRE's helper system, making them available in the Python runtime.
"""

import asyncio
import logging
from typing import Any, Callable

from sabre.common.models import Content, TextContent, ImageContent
from .models import (
    MCPTool,
    MCPToolResult,
    MCPContent,
    MCPToolError,
    MCPServerNotFoundError,
)
from .client_manager import MCPClientManager

logger = logging.getLogger(__name__)


class MCPHelperAdapter:
    """
    Adapter that exposes MCP tools as SABRE helpers.

    Responsibilities:
    - Convert MCP tools to Python callables
    - Generate documentation for LLM prompts
    - Route tool calls to appropriate MCP server
    - Transform MCP results to SABRE Content
    """

    def __init__(self, client_manager: MCPClientManager, event_loop=None):
        """
        Initialize MCP helper adapter.

        Args:
            client_manager: MCP client manager instance
            event_loop: Optional event loop to use for async operations (for thread-safe execution)
        """
        self.client_manager = client_manager
        self._tools_cache: dict[str, MCPTool] = {}
        self._main_loop = event_loop  # Store reference to main event loop for thread-safe calls

    async def refresh_tools(self) -> None:
        """
        Refresh tools cache from all connected servers.

        This should be called after connecting to servers or when tools might have changed.
        """
        self._tools_cache.clear()

        all_tools = await self.client_manager.get_all_tools()

        for server_name, tools in all_tools.items():
            for tool in tools:
                # Create qualified tool name: ServerName.tool_name
                qualified_name = f"{server_name}.{tool.name}"
                self._tools_cache[qualified_name] = tool
                logger.debug(f"Registered MCP tool: {qualified_name}")

        logger.info(f"Refreshed MCP tools cache: {len(self._tools_cache)} tools available")

    def get_available_tools(self) -> dict[str, Callable]:
        """
        Get all MCP tools as Python callables.

        Returns:
            Dictionary mapping tool name to callable function

        Example:
            {
                "Postgres.query": <callable>,
                "GitHub.create_pr": <callable>,
            }
        """
        tools = {}

        for qualified_name, tool in self._tools_cache.items():
            # Create a callable that wraps the tool invocation
            callable_func = self._create_tool_callable(qualified_name, tool)
            tools[qualified_name] = callable_func

        return tools

    def _create_tool_callable(self, qualified_name: str, tool: MCPTool) -> Callable:
        """
        Create a Python callable for an MCP tool.

        The callable will invoke the tool asynchronously and return results.

        Args:
            qualified_name: Qualified tool name (ServerName.tool_name)
            tool: MCP tool definition

        Returns:
            Callable function
        """

        def tool_wrapper(*args, **kwargs) -> Any:
            """
            Wrapper function that invokes the MCP tool.

            This runs the async tool call synchronously using a robust event loop pattern,
            since it's called from Python runtime's exec() context.

            Accepts both positional and keyword arguments to be flexible with LLM calls.
            """
            try:
                # Convert positional args to keyword args based on tool schema
                if args:
                    # Get required parameters from tool schema
                    schema = tool.input_schema
                    if "properties" in schema:
                        prop_names = list(schema["properties"].keys())
                        # Map positional args to parameter names in order
                        for i, arg in enumerate(args):
                            if i < len(prop_names):
                                param_name = prop_names[i]
                                if param_name not in kwargs:  # Don't override explicit kwargs
                                    kwargs[param_name] = arg

                # Run async invocation synchronously with robust event loop handling

                # If we have a reference to the main loop (from server), use it for thread-safe execution
                if self._main_loop is not None:
                    # We're in a worker thread - submit coroutine to main loop
                    future = asyncio.run_coroutine_threadsafe(
                        self.invoke_tool(qualified_name, **kwargs),
                        self._main_loop
                    )
                    return future.result()

                # Check if we're in a running event loop
                try:
                    loop = asyncio.get_running_loop()
                    # We're in a running loop - need to execute in a new thread
                    import concurrent.futures

                    def run_async_in_thread():
                        # Create new event loop for this thread
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(self.invoke_tool(qualified_name, **kwargs))
                        finally:
                            new_loop.close()

                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(run_async_in_thread)
                        return future.result()
                except RuntimeError:
                    # No running loop - can use asyncio.run() directly
                    return asyncio.run(self.invoke_tool(qualified_name, **kwargs))
            except Exception as e:
                logger.error(f"Error invoking MCP tool {qualified_name}: {e}")
                raise

        # Set function metadata for introspection
        tool_wrapper.__name__ = qualified_name
        tool_wrapper.__doc__ = tool.description

        return tool_wrapper

    async def invoke_tool(self, qualified_name: str, **kwargs) -> list[Content]:
        """
        Invoke an MCP tool and return SABRE Content.

        Args:
            qualified_name: Qualified tool name (ServerName.tool_name)
            **kwargs: Tool arguments

        Returns:
            List of SABRE Content objects

        Raises:
            MCPServerNotFoundError: If server not found
            MCPToolError: If tool invocation fails
        """
        # Parse server name and tool name from qualified name
        if "." not in qualified_name:
            raise ValueError(f"Tool name must be qualified with server name: {qualified_name}")

        server_name, tool_name = qualified_name.split(".", 1)

        # Get client for server
        if not self.client_manager.has_server(server_name):
            raise MCPServerNotFoundError(server_name)

        client = self.client_manager.get_client(server_name)

        # Call tool via MCP client
        logger.debug(f"Invoking MCP tool: {qualified_name} with args: {kwargs}")
        mcp_result = await client.call_tool(tool_name, kwargs)

        # Transform MCP result to SABRE Content
        sabre_content = self._transform_result(mcp_result)

        logger.debug(f"MCP tool {qualified_name} returned {len(sabre_content)} content items")
        return sabre_content

    def _transform_result(self, mcp_result: MCPToolResult) -> list[Content]:
        """
        Transform MCP tool result to SABRE Content.

        Args:
            mcp_result: MCP tool result

        Returns:
            List of SABRE Content objects
        """
        sabre_content = []

        for mcp_content in mcp_result.content:
            content = self._transform_content(mcp_content)
            if content:
                sabre_content.append(content)

        return sabre_content

    def _transform_content(self, mcp_content: MCPContent) -> Content | None:
        """
        Transform a single MCP content item to SABRE Content.

        Args:
            mcp_content: MCP content item

        Returns:
            SABRE Content object, or None if unsupported type
        """
        if mcp_content.type == "text":
            return TextContent(text=mcp_content.text or "")

        elif mcp_content.type == "image":
            # MCP images come as base64 data
            if mcp_content.data:
                return ImageContent(image_data=mcp_content.data, mime_type=mcp_content.mimeType or "image/png")

        elif mcp_content.type == "resource":
            # Resources are represented as text with URI reference
            resource_text = f"Resource: {mcp_content.uri}"
            if mcp_content.text:
                resource_text += f"\n{mcp_content.text}"
            return TextContent(text=resource_text)

        else:
            logger.warning(f"Unsupported MCP content type: {mcp_content.type}")
            return None

    def generate_documentation(self) -> str:
        """
        Generate documentation for all MCP tools.

        This is included in the system prompt to inform the LLM about available tools.

        Returns:
            Markdown-formatted documentation string
        """
        if not self._tools_cache:
            return ""

        # Group tools by server
        servers = {}
        for qualified_name, tool in self._tools_cache.items():
            server_name = tool.server_name
            if server_name not in servers:
                servers[server_name] = []
            servers[server_name].append((qualified_name, tool))

        # Generate markdown documentation
        doc_lines = ["## MCP Tools", ""]
        doc_lines.append("The following tools are available from connected MCP servers:")
        doc_lines.append("")

        for server_name, tools in sorted(servers.items()):
            doc_lines.append(f"### {server_name} Server")
            doc_lines.append("")

            for qualified_name, tool in tools:
                # Generate function signature
                signature = tool.get_signature()
                # Replace tool.name with qualified_name in signature
                signature = signature.replace(f"{tool.name}(", f"{qualified_name}(", 1)

                doc_lines.append(f"**{signature}**")
                doc_lines.append(f"{tool.description}")
                doc_lines.append("")

        return "\n".join(doc_lines)

    def get_tool_count(self) -> int:
        """Get number of available MCP tools"""
        return len(self._tools_cache)

    def get_server_names(self) -> list[str]:
        """Get list of servers that have tools registered"""
        servers = set()
        for tool in self._tools_cache.values():
            servers.add(tool.server_name)
        return sorted(servers)

    def has_tool(self, qualified_name: str) -> bool:
        """Check if a tool is available"""
        return qualified_name in self._tools_cache

    def get_tool(self, qualified_name: str) -> MCPTool | None:
        """Get tool definition by qualified name"""
        return self._tools_cache.get(qualified_name)
