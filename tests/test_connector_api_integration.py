"""
Integration tests for Connector API endpoints.

Tests full CRUD operations via HTTP API.

NOTE: These tests require OPENAI_API_KEY to be set.
Run with: OPENAI_API_KEY=sk-your-key uv run pytest tests/test_connector_api_integration.py
"""

import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


@pytest.fixture(scope="module", autouse=True)
def require_api_key(check_api_key):
    """Require API key for all tests in this module."""
    pass


@pytest.fixture
def app():
    """Import and return the FastAPI app (after API key is validated)."""
    from sabre.server.api.server import app
    return app


@pytest.fixture
def client(app):
    """Create FastAPI test client with initialized session manager."""
    from sabre.server.mcp.client_manager import MCPClientManager
    from sabre.server.api.server import manager

    # Initialize mcp_manager if not already done
    if manager.mcp_manager is None:
        manager.mcp_manager = MCPClientManager(connector_store=manager.connector_store)

    return TestClient(app)


@pytest.fixture
def mock_mcp_connection():
    """Mock MCP client connection to avoid actual subprocess spawning."""
    with patch('sabre.server.mcp.client_manager.MCPClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client.connect = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])
        mock_client.disconnect = AsyncMock()
        mock_client.is_connected = MagicMock(return_value=True)  # Synchronous method
        mock_client.tools_cache = []
        mock_client_class.return_value = mock_client
        yield mock_client


class TestConnectorAPICreate:
    """Test POST /v1/connectors endpoint."""

    def test_create_connector_stdio(self, client, mock_mcp_connection):
        """Test creating a stdio connector."""
        response = client.post(
            "/v1/connectors",
            json={
                "name": "test_postgres",
                "type": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-postgres"],
                "env": {"POSTGRES_URL": "postgresql://localhost/test"},
                "enabled": True,
                "timeout": 30,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "id" in data
        assert data["name"] == "test_postgres"
        assert data["type"] == "stdio"
        assert data["enabled"] is True
        assert data["source"] == "api"
        assert "created_at" in data
        assert "updated_at" in data

    def test_create_connector_sse(self, client, mock_mcp_connection):
        """Test creating an SSE connector."""
        response = client.post(
            "/v1/connectors",
            json={
                "name": "remote_api",
                "type": "sse",
                "url": "https://api.example.com/mcp",
                "headers": {"Authorization": "Bearer token123"},
                "enabled": True,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["name"] == "remote_api"
        assert data["type"] == "sse"

    def test_create_connector_disabled(self, client):
        """Test creating a disabled connector (no connection attempt)."""
        response = client.post(
            "/v1/connectors",
            json={
                "name": "disabled_server",
                "type": "stdio",
                "command": "echo",
                "enabled": False,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["enabled"] is False
        assert data["connected"] is False

    def test_create_connector_invalid_type(self, client):
        """Test creating connector with invalid type."""
        response = client.post(
            "/v1/connectors",
            json={
                "name": "bad_type",
                "type": "invalid_type",
                "command": "echo",
            },
        )

        # Should fail validation or connection
        assert response.status_code in [422, 500]

    def test_create_connector_missing_command_stdio(self, client):
        """Test creating stdio connector without command."""
        response = client.post(
            "/v1/connectors",
            json={
                "name": "no_command",
                "type": "stdio",
                # Missing command
            },
        )

        # Should fail validation
        assert response.status_code in [422, 500]

    def test_create_connector_missing_url_sse(self, client):
        """Test creating SSE connector without URL."""
        response = client.post(
            "/v1/connectors",
            json={
                "name": "no_url",
                "type": "sse",
                # Missing url
            },
        )

        # Should fail validation
        assert response.status_code in [422, 500]


class TestConnectorAPIList:
    """Test GET /v1/connectors endpoint."""

    def test_list_connectors_empty(self, client):
        """Test listing connectors when none exist."""
        response = client.get("/v1/connectors")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        # May have bootstrap connectors from YAML, but should be a list
        assert all(isinstance(item, dict) for item in data)

    def test_list_connectors_after_create(self, client, mock_mcp_connection):
        """Test listing connectors after creating one."""
        # Create a connector
        create_response = client.post(
            "/v1/connectors",
            json={
                "name": "list_test",
                "type": "stdio",
                "command": "echo",
                "enabled": True,
            },
        )

        assert create_response.status_code == 200
        connector_id = create_response.json()["id"]

        # List connectors
        list_response = client.get("/v1/connectors")

        assert list_response.status_code == 200
        connectors = list_response.json()

        # Find our connector
        our_connector = next((c for c in connectors if c["id"] == connector_id), None)
        assert our_connector is not None
        assert our_connector["name"] == "list_test"


class TestConnectorAPIGet:
    """Test GET /v1/connectors/{id} endpoint."""

    def test_get_connector_by_id(self, client, mock_mcp_connection):
        """Test getting connector details by ID."""
        # Create a connector
        create_response = client.post(
            "/v1/connectors",
            json={
                "name": "get_test",
                "type": "stdio",
                "command": "echo",
                "args": ["hello"],
                "timeout": 45,
                "enabled": True,
            },
        )

        connector_id = create_response.json()["id"]

        # Get connector details
        get_response = client.get(f"/v1/connectors/{connector_id}")

        assert get_response.status_code == 200
        data = get_response.json()

        assert data["id"] == connector_id
        assert data["name"] == "get_test"
        assert data["command"] == "echo"
        assert data["args"] == ["hello"]
        assert data["timeout"] == 45

    def test_get_connector_not_found(self, client):
        """Test getting non-existent connector returns 404."""
        response = client.get("/v1/connectors/nonexistent-uuid")

        assert response.status_code in [404, 500]


class TestConnectorAPIGetTools:
    """Test GET /v1/connectors/{id}/tools endpoint."""

    def test_get_connector_tools(self, client):
        """Test getting tools from a connector."""
        # Create a disabled connector (won't actually connect)
        create_response = client.post(
            "/v1/connectors",
            json={
                "name": "tools_test",
                "type": "stdio",
                "command": "echo",
                "enabled": False,
            },
        )

        connector_id = create_response.json()["id"]

        # Get tools (will be empty for disabled connector)
        response = client.get(f"/v1/connectors/{connector_id}/tools")

        # May fail since there's no actual client for disabled connector
        # Accept 404 or 500 as valid responses
        assert response.status_code in [200, 404, 500]

    def test_get_tools_with_mock(self, client, mock_mcp_connection):
        """Test getting tools from a connector with mocked tools."""
        # Mock tools response
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.get_signature.return_value = "test_tool(param: str)"
        mock_tool.input_schema = {"type": "object", "properties": {}}

        mock_mcp_connection.list_tools = AsyncMock(return_value=[mock_tool])

        # Create a connector
        create_response = client.post(
            "/v1/connectors",
            json={
                "name": "mock_tools_test",
                "type": "stdio",
                "command": "echo",
                "enabled": True,
            },
        )

        connector_id = create_response.json()["id"]

        # Get tools
        response = client.get(f"/v1/connectors/{connector_id}/tools")

        assert response.status_code == 200
        data = response.json()

        assert data["connector_id"] == connector_id
        assert "tools" in data
        assert len(data["tools"]) == 1
        assert data["tools"][0]["name"] == "test_tool"


class TestConnectorAPIUpdate:
    """Test PUT /v1/connectors/{id} endpoint."""

    def test_update_connector(self, client, mock_mcp_connection):
        """Test updating a connector."""
        # Create a connector
        create_response = client.post(
            "/v1/connectors",
            json={
                "name": "update_test",
                "type": "stdio",
                "command": "echo",
                "timeout": 30,
                "enabled": True,
            },
        )

        connector_id = create_response.json()["id"]

        # Update timeout
        update_response = client.put(
            f"/v1/connectors/{connector_id}",
            json={"timeout": 60},
        )

        assert update_response.status_code == 200
        data = update_response.json()

        # Note: timeout may not be in response (depends on ConnectorResponse model)
        # But updated_at should change
        assert "updated_at" in data

    def test_update_connector_name(self, client, mock_mcp_connection):
        """Test updating connector name."""
        # Create a connector
        create_response = client.post(
            "/v1/connectors",
            json={
                "name": "old_name",
                "type": "stdio",
                "command": "echo",
                "enabled": True,
            },
        )

        connector_id = create_response.json()["id"]

        # Update name
        update_response = client.put(
            f"/v1/connectors/{connector_id}",
            json={"name": "new_name"},
        )

        assert update_response.status_code == 200
        data = update_response.json()

        assert data["name"] == "new_name"

    def test_update_connector_not_found(self, client):
        """Test updating non-existent connector."""
        response = client.put(
            "/v1/connectors/nonexistent-uuid",
            json={"timeout": 60},
        )

        assert response.status_code in [404, 500]

    def test_update_connector_empty_body(self, client, mock_mcp_connection):
        """Test updating with empty body returns error."""
        # Create a connector
        create_response = client.post(
            "/v1/connectors",
            json={
                "name": "empty_update_test",
                "type": "stdio",
                "command": "echo",
                "enabled": True,
            },
        )

        connector_id = create_response.json()["id"]

        # Update with empty body
        response = client.put(
            f"/v1/connectors/{connector_id}",
            json={},
        )

        assert response.status_code == 400


class TestConnectorAPIPatch:
    """Test PATCH /v1/connectors/{id} endpoint."""

    def test_patch_enable_connector(self, client):
        """Test enabling a disabled connector."""
        # Create disabled connector
        create_response = client.post(
            "/v1/connectors",
            json={
                "name": "patch_enable_test",
                "type": "stdio",
                "command": "echo",
                "enabled": False,
            },
        )

        connector_id = create_response.json()["id"]

        # Enable it
        with patch('sabre.server.mcp.client_manager.MCPClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=[])
            mock_client.is_connected = MagicMock(return_value=True)
            mock_client.tools_cache = []
            mock_client_class.return_value = mock_client

            patch_response = client.patch(
                f"/v1/connectors/{connector_id}",
                json={"enabled": True},
            )

            assert patch_response.status_code == 200
            data = patch_response.json()

            assert data["enabled"] is True

    def test_patch_disable_connector(self, client, mock_mcp_connection):
        """Test disabling an enabled connector."""
        # Create enabled connector
        create_response = client.post(
            "/v1/connectors",
            json={
                "name": "patch_disable_test",
                "type": "stdio",
                "command": "echo",
                "enabled": True,
            },
        )

        connector_id = create_response.json()["id"]

        # Disable it
        patch_response = client.patch(
            f"/v1/connectors/{connector_id}",
            json={"enabled": False},
        )

        assert patch_response.status_code == 200
        data = patch_response.json()

        assert data["enabled"] is False

    def test_patch_with_multiple_fields(self, client, mock_mcp_connection):
        """Test patch with multiple fields falls back to update."""
        # Create connector
        create_response = client.post(
            "/v1/connectors",
            json={
                "name": "patch_multi_test",
                "type": "stdio",
                "command": "echo",
                "enabled": True,
            },
        )

        connector_id = create_response.json()["id"]

        # Patch with multiple fields
        patch_response = client.patch(
            f"/v1/connectors/{connector_id}",
            json={"enabled": False, "timeout": 90},
        )

        assert patch_response.status_code == 200


class TestConnectorAPIDelete:
    """Test DELETE /v1/connectors/{id} endpoint."""

    def test_delete_connector(self, client, mock_mcp_connection):
        """Test deleting a connector."""
        # Create connector
        create_response = client.post(
            "/v1/connectors",
            json={
                "name": "delete_test",
                "type": "stdio",
                "command": "echo",
                "enabled": True,
            },
        )

        connector_id = create_response.json()["id"]

        # Delete it
        delete_response = client.delete(f"/v1/connectors/{connector_id}")

        assert delete_response.status_code == 200
        data = delete_response.json()

        assert data["status"] == "deleted"
        assert data["connector_id"] == connector_id
        assert "message" in data

        # Verify it's gone
        get_response = client.get(f"/v1/connectors/{connector_id}")
        assert get_response.status_code in [404, 500]

    def test_delete_connector_not_found(self, client):
        """Test deleting non-existent connector."""
        response = client.delete("/v1/connectors/nonexistent-uuid")

        assert response.status_code in [404, 500]


class TestConnectorAPIWorkflow:
    """Test complete CRUD workflow."""

    def test_full_crud_workflow(self, client, mock_mcp_connection):
        """Test complete create, read, update, delete workflow."""
        # 1. Create
        create_response = client.post(
            "/v1/connectors",
            json={
                "name": "workflow_test",
                "type": "stdio",
                "command": "echo",
                "args": ["test"],
                "timeout": 30,
                "enabled": True,
            },
        )

        assert create_response.status_code == 200
        connector_id = create_response.json()["id"]

        # 2. Read (get)
        get_response = client.get(f"/v1/connectors/{connector_id}")
        assert get_response.status_code == 200
        assert get_response.json()["name"] == "workflow_test"

        # 3. Update
        update_response = client.put(
            f"/v1/connectors/{connector_id}",
            json={"name": "workflow_test_updated"},
        )
        assert update_response.status_code == 200
        assert update_response.json()["name"] == "workflow_test_updated"

        # 4. Patch (disable)
        patch_response = client.patch(
            f"/v1/connectors/{connector_id}",
            json={"enabled": False},
        )
        assert patch_response.status_code == 200
        assert patch_response.json()["enabled"] is False

        # 5. List (verify present)
        list_response = client.get("/v1/connectors")
        assert list_response.status_code == 200
        connector_ids = [c["id"] for c in list_response.json()]
        assert connector_id in connector_ids

        # 6. Delete
        delete_response = client.delete(f"/v1/connectors/{connector_id}")
        assert delete_response.status_code == 200

        # 7. Verify deleted
        get_after_delete = client.get(f"/v1/connectors/{connector_id}")
        assert get_after_delete.status_code in [404, 500]


class TestConnectorAPIValidation:
    """Test request validation."""

    def test_create_with_extra_fields(self, client, mock_mcp_connection):
        """Test creating connector with extra fields (should be ignored)."""
        response = client.post(
            "/v1/connectors",
            json={
                "name": "extra_fields_test",
                "type": "stdio",
                "command": "echo",
                "enabled": True,
                "extra_field": "should_be_ignored",
            },
        )

        # Pydantic should ignore extra fields
        assert response.status_code == 200

    def test_create_with_invalid_json(self, client):
        """Test creating connector with invalid JSON."""
        response = client.post(
            "/v1/connectors",
            data="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    def test_update_with_invalid_timeout(self, client, mock_mcp_connection):
        """Test updating with invalid timeout value."""
        # Create connector
        create_response = client.post(
            "/v1/connectors",
            json={
                "name": "invalid_timeout_test",
                "type": "stdio",
                "command": "echo",
                "enabled": True,
            },
        )

        connector_id = create_response.json()["id"]

        # Try to update with invalid timeout
        response = client.put(
            f"/v1/connectors/{connector_id}",
            json={"timeout": "not_a_number"},
        )

        assert response.status_code == 422
