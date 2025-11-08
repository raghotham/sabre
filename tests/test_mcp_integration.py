"""
Integration tests for MCP system.

These tests verify the end-to-end flow of MCP integration with SABRE.

Run with: uv run pytest tests/test_mcp_integration.py
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from sabre.server.mcp.models import MCPServerConfig, MCPTransportType
from sabre.server.mcp.config import MCPConfigLoader
from sabre.server.mcp.client_manager import MCPClientManager
from sabre.server.mcp.helper_adapter import MCPHelperAdapter
from sabre.server.python_runtime import PythonRuntime


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def mock_mcp_process():
    """Create a mock MCP server process that responds to requests."""

    class MockMCPProcess:
        def __init__(self):
            self.returncode = None
            self.stdin = AsyncMock()
            self.stdout = AsyncMock()
            self.stderr = AsyncMock()
            self.request_count = 0

        async def readline(self):
            """Simulate MCP server responses."""
            self.request_count += 1

            # First request is usually tools/list
            if self.request_count == 1:
                response = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "tools": [
                            {
                                "name": "echo",
                                "description": "Echo back the input",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"},
                                    },
                                    "required": ["message"],
                                },
                            }
                        ]
                    },
                }
            else:
                # Subsequent requests are tool calls
                response = {
                    "jsonrpc": "2.0",
                    "id": self.request_count,
                    "result": {
                        "content": [{"type": "text", "text": "Echo: test message"}]
                    },
                }

            return (json.dumps(response) + "\n").encode()

        async def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

        def kill(self):
            pass

    process = MockMCPProcess()
    process.stdout.readline = process.readline
    return process


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
@pytest.mark.asyncio
async def test_full_integration_config_to_tool_call(temp_config_file, mock_mcp_process):
    """Test full integration from config loading to tool invocation."""

    # 1. Create config file
    config_content = """
mcp_servers:
  echo_server:
    type: stdio
    command: echo
    args: ["test"]
    enabled: true
    timeout: 5
"""
    temp_config_file.write_text(config_content)

    # 2. Load configuration
    configs = MCPConfigLoader.load(temp_config_file)
    assert len(configs) == 1
    print("✓ Step 1: Configuration loaded")

    # 3. Create client manager and connect
    with patch("asyncio.create_subprocess_exec", return_value=mock_mcp_process):
        manager = MCPClientManager()
        await manager.connect_all(configs)
        assert len(manager.list_servers()) == 1
        print("✓ Step 2: Connected to MCP server")

        # 4. Create helper adapter and refresh tools
        adapter = MCPHelperAdapter(manager)
        await adapter.refresh_tools()
        assert adapter.get_tool_count() == 1
        print("✓ Step 3: Tools discovered and cached")

        # 5. Get available tools as callables
        tools = adapter.get_available_tools()
        assert "echo_server.echo" in tools
        print("✓ Step 4: Tools exposed as callables")

        # 6. Invoke tool
        result = await adapter.invoke_tool("echo_server.echo", message="test message")
        assert len(result) == 1
        assert "Echo: test message" in result[0].text
        print("✓ Step 5: Tool invocation successful")

        # 7. Cleanup
        await manager.disconnect_all()
        print("✓ Step 6: Cleanup complete")


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
@pytest.mark.asyncio
async def test_runtime_integration(temp_config_file, mock_mcp_process):
    """Test MCP integration with Python runtime."""

    # Setup config
    config_content = """
mcp_servers:
  test_server:
    type: stdio
    command: test
    enabled: true
"""
    temp_config_file.write_text(config_content)

    # Load and connect
    configs = MCPConfigLoader.load(temp_config_file)

    with patch("asyncio.create_subprocess_exec", return_value=mock_mcp_process):
        manager = MCPClientManager()
        await manager.connect_all(configs)

        adapter = MCPHelperAdapter(manager)
        await adapter.refresh_tools()

        # Create runtime with MCP adapter
        runtime = PythonRuntime(mcp_adapter=adapter)

        # Verify MCP tools are in namespace
        # Tools are now accessible as server objects (e.g., test_server.echo)
        assert "test_server" in runtime.namespace
        test_server = runtime.namespace["test_server"]
        assert hasattr(test_server, "echo")
        assert callable(test_server.echo)

        print("✓ MCP tools integrated into Python runtime")

        # Cleanup
        await manager.disconnect_all()


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
@pytest.mark.asyncio
async def test_multiple_servers_integration(temp_config_file):
    """Test integration with multiple MCP servers."""

    config_content = """
mcp_servers:
  server1:
    type: stdio
    command: test1
    enabled: true

  server2:
    type: stdio
    command: test2
    enabled: true
"""
    temp_config_file.write_text(config_content)

    configs = MCPConfigLoader.load(temp_config_file)
    assert len(configs) == 2

    # Create separate mock processes for each server
    def create_mock_process():
        class MockMCPProcess:
            def __init__(self):
                self.returncode = None
                self.stdin = AsyncMock()
                self.stdout = AsyncMock()
                self.stderr = AsyncMock()
                self.request_count = 0

            async def readline(self):
                self.request_count += 1
                if self.request_count == 1:
                    response = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "result": {
                            "tools": [
                                {
                                    "name": "echo",
                                    "description": "Echo back the input",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {"message": {"type": "string"}},
                                        "required": ["message"],
                                    },
                                }
                            ]
                        },
                    }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": self.request_count,
                        "result": {"content": [{"type": "text", "text": "Echo: test"}]},
                    }
                return (json.dumps(response) + "\n").encode()

            async def wait(self, timeout=None):
                return 0

            def terminate(self):
                pass

            def kill(self):
                pass

        process = MockMCPProcess()
        process.stdout.readline = process.readline
        return process

    with patch("asyncio.create_subprocess_exec", side_effect=[create_mock_process(), create_mock_process()]):
        manager = MCPClientManager()
        await manager.connect_all(configs)

        assert len(manager.list_servers()) == 2
        assert manager.has_server("server1")
        assert manager.has_server("server2")

        # Get tools from both servers
        adapter = MCPHelperAdapter(manager)
        await adapter.refresh_tools()

        # Should have tools from both servers
        assert adapter.has_tool("server1.echo")
        assert adapter.has_tool("server2.echo")

        print("✓ Multiple servers integration works")

        await manager.disconnect_all()


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
@pytest.mark.asyncio
async def test_error_handling_integration(temp_config_file):
    """Test error handling throughout the integration."""

    config_content = """
mcp_servers:
  failing_server:
    type: stdio
    command: nonexistent_command
    enabled: true
"""
    temp_config_file.write_text(config_content)

    configs = MCPConfigLoader.load(temp_config_file)

    manager = MCPClientManager()

    # Should handle connection failure gracefully
    await manager.connect_all(configs)

    # Server should not be in the list
    assert len(manager.list_servers()) == 0

    print("✓ Error handling works across integration")


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
@pytest.mark.asyncio
async def test_disabled_server_integration(temp_config_file, mock_mcp_process):
    """Test that disabled servers are properly skipped."""

    config_content = """
mcp_servers:
  enabled_server:
    type: stdio
    command: test1
    enabled: true

  disabled_server:
    type: stdio
    command: test2
    enabled: false
"""
    temp_config_file.write_text(config_content)

    configs = MCPConfigLoader.load(temp_config_file)
    assert len(configs) == 2

    with patch("asyncio.create_subprocess_exec", return_value=mock_mcp_process):
        manager = MCPClientManager()
        await manager.connect_all(configs)

        # Only enabled server should be connected
        assert len(manager.list_servers()) == 1
        assert manager.has_server("enabled_server")
        assert not manager.has_server("disabled_server")

        print("✓ Disabled servers properly skipped")

        await manager.disconnect_all()


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
@pytest.mark.asyncio
async def test_documentation_generation_integration(temp_config_file, mock_mcp_process):
    """Test documentation generation from discovered tools."""

    config_content = """
mcp_servers:
  doc_server:
    type: stdio
    command: test
    enabled: true
"""
    temp_config_file.write_text(config_content)

    configs = MCPConfigLoader.load(temp_config_file)

    with patch("asyncio.create_subprocess_exec", return_value=mock_mcp_process):
        manager = MCPClientManager()
        await manager.connect_all(configs)

        adapter = MCPHelperAdapter(manager)
        await adapter.refresh_tools()

        # Generate documentation
        docs = adapter.generate_documentation()

        assert "## MCP Tools" in docs
        assert "### doc_server Server" in docs
        assert "doc_server.echo" in docs
        assert "Echo back the input" in docs
        assert "message: str" in docs

        print(f"✓ Documentation generated:\n{docs[:300]}...")

        await manager.disconnect_all()


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
@pytest.mark.asyncio
async def test_reconnection_integration(temp_config_file, mock_mcp_process):
    """Test server reconnection functionality."""

    config_content = """
mcp_servers:
  reconnect_server:
    type: stdio
    command: test
    enabled: true
"""
    temp_config_file.write_text(config_content)

    configs = MCPConfigLoader.load(temp_config_file)

    with patch("asyncio.create_subprocess_exec", return_value=mock_mcp_process):
        manager = MCPClientManager()
        await manager.connect_all(configs)

        assert manager.has_server("reconnect_server")

        # Reconnect
        await manager.reconnect("reconnect_server")

        # Should still be connected
        assert manager.has_server("reconnect_server")

        print("✓ Server reconnection works")

        await manager.disconnect_all()


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
@pytest.mark.asyncio
async def test_health_check_integration(temp_config_file, mock_mcp_process):
    """Test health checking across the integration."""

    config_content = """
mcp_servers:
  healthy_server:
    type: stdio
    command: test
    enabled: true
"""
    temp_config_file.write_text(config_content)

    configs = MCPConfigLoader.load(temp_config_file)

    with patch("asyncio.create_subprocess_exec", return_value=mock_mcp_process):
        manager = MCPClientManager()
        await manager.connect_all(configs)

        # Check health
        health_status = await manager.health_check_all()

        assert "healthy_server" in health_status
        assert health_status["healthy_server"] is True

        print("✓ Health checking works")

        await manager.disconnect_all()


@pytest.mark.asyncio
async def test_concurrent_tool_calls(mock_mcp_process):
    """Test concurrent tool invocations."""

    with patch("asyncio.create_subprocess_exec", return_value=mock_mcp_process):
        config = MCPServerConfig(
            name="concurrent_server",
            type=MCPTransportType.STDIO,
            command="test",
            enabled=True,
        )

        manager = MCPClientManager()
        await manager.connect(config)

        adapter = MCPHelperAdapter(manager)
        await adapter.refresh_tools()

        # Make concurrent tool calls
        tasks = [
            adapter.invoke_tool("concurrent_server.echo", message=f"msg{i}")
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        for result in results:
            assert len(result) > 0

        print("✓ Concurrent tool calls work")

        await manager.disconnect_all()
