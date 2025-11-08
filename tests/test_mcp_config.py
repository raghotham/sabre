"""
Tests for MCP configuration loading.

Run with: uv run pytest tests/test_mcp_config.py
"""

import pytest
import os
import tempfile
from pathlib import Path

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from sabre.server.mcp.models import MCPServerConfig, MCPTransportType
from sabre.server.mcp.config import MCPConfigLoader


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yield Path(f.name)
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
def test_config_loader_basic(temp_config_file):
    """Test loading a basic MCP config."""
    config_content = """
mcp_servers:
  test_server:
    type: stdio
    command: npx
    args: ["-y", "@test/server"]
    env:
      TEST_VAR: "test_value"
    enabled: true
    timeout: 30
"""
    temp_config_file.write_text(config_content)

    configs = MCPConfigLoader.load(temp_config_file)

    assert len(configs) == 1
    config = configs[0]
    assert config.name == "test_server"
    assert config.type == MCPTransportType.STDIO
    assert config.command == "npx"
    assert config.args == ["-y", "@test/server"]
    assert config.env == {"TEST_VAR": "test_value"}
    assert config.enabled is True
    assert config.timeout == 30
    print("✓ Basic config loaded successfully")


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
def test_config_loader_multiple_servers(temp_config_file):
    """Test loading multiple MCP servers."""
    config_content = """
mcp_servers:
  postgres:
    type: stdio
    command: npx
    args: ["-y", "@mcp/postgres"]
    enabled: true

  github:
    type: stdio
    command: npx
    args: ["-y", "@mcp/github"]
    enabled: false

  remote:
    type: sse
    url: "https://example.com/mcp"
    headers:
      Authorization: "Bearer token"
    enabled: true
"""
    temp_config_file.write_text(config_content)

    configs = MCPConfigLoader.load(temp_config_file)

    assert len(configs) == 3

    # Check postgres
    postgres = next(c for c in configs if c.name == "postgres")
    assert postgres.type == MCPTransportType.STDIO
    assert postgres.enabled is True

    # Check github
    github = next(c for c in configs if c.name == "github")
    assert github.type == MCPTransportType.STDIO
    assert github.enabled is False

    # Check remote
    remote = next(c for c in configs if c.name == "remote")
    assert remote.type == MCPTransportType.SSE
    assert remote.url == "https://example.com/mcp"
    assert remote.enabled is True

    print(f"✓ Loaded {len(configs)} servers successfully")


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
def test_config_loader_env_var_expansion(temp_config_file):
    """Test environment variable expansion."""
    # Set test environment variables
    os.environ["TEST_API_KEY"] = "secret123"
    os.environ["TEST_URL"] = "postgresql://localhost/test"

    config_content = """
mcp_servers:
  test:
    type: stdio
    command: npx
    args: ["-y", "@test/server"]
    env:
      API_KEY: "${TEST_API_KEY}"
      DATABASE_URL: "$TEST_URL"
    enabled: true
"""
    temp_config_file.write_text(config_content)

    try:
        configs = MCPConfigLoader.load(temp_config_file)

        assert len(configs) == 1
        config = configs[0]
        assert config.env["API_KEY"] == "secret123"
        assert config.env["DATABASE_URL"] == "postgresql://localhost/test"
        print("✓ Environment variables expanded successfully")
    finally:
        # Cleanup env vars
        os.environ.pop("TEST_API_KEY", None)
        os.environ.pop("TEST_URL", None)


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
def test_config_loader_empty_servers(temp_config_file):
    """Test loading config with no servers (all commented out)."""
    config_content = """
mcp_servers:
  # All servers commented out
"""
    temp_config_file.write_text(config_content)

    configs = MCPConfigLoader.load(temp_config_file)

    assert len(configs) == 0
    print("✓ Empty servers config handled correctly")


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
def test_config_loader_missing_file():
    """Test loading non-existent config file."""
    non_existent = Path("/tmp/does_not_exist_mcp_config.yaml")

    configs = MCPConfigLoader.load(non_existent)

    assert len(configs) == 0
    print("✓ Missing config file handled gracefully")


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
def test_config_loader_invalid_transport(temp_config_file):
    """Test loading config with invalid transport type."""
    config_content = """
mcp_servers:
  test:
    type: invalid_transport
    command: test
    enabled: true
"""
    temp_config_file.write_text(config_content)

    configs = MCPConfigLoader.load(temp_config_file)

    # Should skip invalid configs
    assert len(configs) == 0
    print("✓ Invalid transport type handled gracefully")


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
def test_config_validation():
    """Test config validation."""
    # Valid stdio config
    valid_stdio = MCPServerConfig(
        name="test",
        type=MCPTransportType.STDIO,
        command="npx",
        args=["-y", "@test/server"],
    )
    errors = MCPConfigLoader.validate_config(valid_stdio)
    assert len(errors) == 0

    # Invalid timeout
    invalid_timeout = MCPServerConfig(
        name="test",
        type=MCPTransportType.STDIO,
        command="npx",
        timeout=-5,
    )
    errors = MCPConfigLoader.validate_config(invalid_timeout)
    assert len(errors) > 0
    assert any("timeout" in err.lower() for err in errors)

    print("✓ Config validation working correctly")


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
def test_create_example_config(tmp_path):
    """Test creating example config file."""
    config_path = tmp_path / "test_mcp.yaml"

    result_path = MCPConfigLoader.create_example_config(config_path)

    assert result_path.exists()
    assert result_path == config_path

    # Verify content
    content = config_path.read_text()
    assert "mcp_servers:" in content
    assert "postgres:" in content
    assert "github:" in content
    assert "modelcontextprotocol.io" in content

    print(f"✓ Example config created at {config_path}")


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
def test_expand_env_vars():
    """Test environment variable expansion utility."""
    os.environ["TEST_VAR"] = "value123"

    try:
        # Test ${VAR} syntax
        expanded = MCPConfigLoader._expand_env("prefix_${TEST_VAR}_suffix")
        assert expanded == "prefix_value123_suffix"

        # Test $VAR syntax
        expanded = MCPConfigLoader._expand_env("prefix_$TEST_VAR")
        assert expanded == "prefix_value123"

        # Test missing variable (should leave as-is)
        expanded = MCPConfigLoader._expand_env("${MISSING_VAR}")
        assert expanded == "${MISSING_VAR}"

        # Test None value
        expanded = MCPConfigLoader._expand_env(None)
        assert expanded is None

        print("✓ Environment variable expansion working correctly")
    finally:
        os.environ.pop("TEST_VAR", None)


def test_config_loader_without_yaml():
    """Test that config loader handles missing PyYAML gracefully."""
    # This test will run even without PyYAML
    if not YAML_AVAILABLE:
        configs = MCPConfigLoader.load()
        assert len(configs) == 0
        print("✓ Config loader handles missing PyYAML gracefully")
    else:
        print("⊘ Skipped (PyYAML is installed)")
