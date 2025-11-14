#!/usr/bin/env python3
"""
Simple HTTP-based MCP server for testing SABRE SSE transport.

This server implements the MCP protocol over HTTP POST requests.
It provides a simple "echo" tool that returns whatever you send it.

Usage:
    python simple_mcp_server.py

The server will start on http://localhost:8080
MCP endpoint: http://localhost:8080/mcp
"""

import json
import asyncio
from typing import Any
from aiohttp import web
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleMCPServer:
    """Simple MCP server with echo and calculate tools."""

    def __init__(self):
        self.tools = [
            {
                "name": "echo",
                "description": "Echo back the input message",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to echo back"
                        }
                    },
                    "required": ["message"]
                }
            },
            {
                "name": "calculate",
                "description": "Perform basic arithmetic",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"],
                            "description": "The arithmetic operation"
                        },
                        "a": {
                            "type": "number",
                            "description": "First number"
                        },
                        "b": {
                            "type": "number",
                            "description": "Second number"
                        }
                    },
                    "required": ["operation", "a", "b"]
                }
            }
        ]

    async def handle_request(self, request: web.Request) -> web.Response:
        """Handle incoming JSON-RPC requests."""
        try:
            # Parse JSON-RPC request
            data = await request.json()
            logger.info(f"Received request: {data}")

            # Extract method and params
            method = data.get("method")
            params = data.get("params", {})
            request_id = data.get("id")

            # Route to handler
            if method == "tools/list":
                result = await self.list_tools()
            elif method == "tools/call":
                result = await self.call_tool(params)
            else:
                return web.json_response({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }, status=404)

            # Return JSON-RPC response
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }

            logger.info(f"Sending response: {response}")
            return web.json_response(response)

        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return web.json_response({
                "jsonrpc": "2.0",
                "id": data.get("id") if "data" in locals() else None,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }, status=500)

    async def list_tools(self) -> dict[str, Any]:
        """Return list of available tools."""
        return {"tools": self.tools}

    async def call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool call."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        logger.info(f"Calling tool: {tool_name} with args: {arguments}")

        if tool_name == "echo":
            message = arguments.get("message", "")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Echo: {message}"
                    }
                ]
            }

        elif tool_name == "calculate":
            operation = arguments.get("operation")
            a = arguments.get("a")
            b = arguments.get("b")

            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                if b == 0:
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": "Error: Division by zero"
                            }
                        ],
                        "isError": True
                    }
                result = a / b
            else:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Unknown operation: {operation}"
                        }
                    ],
                    "isError": True
                }

            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Result: {result}"
                    }
                ]
            }

        else:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Unknown tool: {tool_name}"
                    }
                ],
                "isError": True
            }

    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({"status": "healthy"})


async def create_app() -> web.Application:
    """Create and configure the web application."""
    server = SimpleMCPServer()
    app = web.Application()

    # Routes
    app.router.add_get("/", server.health_check)
    app.router.add_get("/health", server.health_check)
    app.router.add_post("/mcp", server.handle_request)

    return app


if __name__ == "__main__":
    print("=" * 60)
    print("Simple MCP Server (HTTP)")
    print("=" * 60)
    print("\nStarting server on http://localhost:8080")
    print("MCP endpoint: http://localhost:8080/mcp")
    print("\nAvailable tools:")
    print("  - echo(message: str)")
    print("  - calculate(operation: str, a: number, b: number)")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)
    print()

    web.run_app(create_app(), host="0.0.0.0", port=8080)
