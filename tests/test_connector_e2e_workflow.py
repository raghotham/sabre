"""
End-to-end workflow tests for MCP Connector API.

Tests complete user workflows from API calls to persistence to runtime updates.

NOTE: These tests require OPENAI_API_KEY to be set.
Run with: OPENAI_API_KEY=sk-your-key uv run pytest tests/test_connector_e2e_workflow.py
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


@pytest.fixture(scope="module", autouse=True)
def require_api_key(check_api_key):
    """Require API key for all tests in this module."""
    pass


@pytest.fixture
def app():
    """Import and return the FastAPI app (after API key is validated)."""
    from sabre.server.api.server import app, manager
    return app, manager


@pytest.fixture
def client(app):
    """Create FastAPI test client with initialized session manager."""
    from sabre.server.mcp.client_manager import MCPClientManager

    app_instance, manager = app

    # Initialize mcp_manager if not already done
    if manager.mcp_manager is None:
        manager.mcp_manager = MCPClientManager(connector_store=manager.connector_store)

    return TestClient(app_instance)


@pytest.fixture
def mock_mcp_client():
    """Mock MCP client for all tests."""
    with patch('sabre.server.mcp.client_manager.MCPClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client.connect = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])
        mock_client.disconnect = AsyncMock()
        mock_client.is_connected = MagicMock(return_value=True)  # Synchronous method
        mock_client.tools_cache = []
        mock_client_class.return_value = mock_client
        yield mock_client


class TestE2EConnectorLifecycle:
    """Test complete connector lifecycle."""

    def test_create_list_update_delete_workflow(self, client, mock_mcp_client):
        """
        Test complete workflow:
        1. Create connector
        2. List connectors
        3. Get connector details
        4. Update connector
        5. Get tools
        6. Disable connector
        7. Delete connector
        """
        # 1. Create connector
        create_payload = {
            "name": "e2e_postgres",
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-postgres"],
            "env": {"POSTGRES_URL": "postgresql://localhost/test"},
            "enabled": True,
            "timeout": 30,
        }

        create_resp = client.post("/v1/connectors", json=create_payload)
        assert create_resp.status_code == 200

        connector = create_resp.json()
        connector_id = connector["id"]

        assert connector["name"] == "e2e_postgres"
        assert connector["enabled"] is True
        assert connector["source"] == "api"

        # 2. List connectors
        list_resp = client.get("/v1/connectors")
        assert list_resp.status_code == 200

        connectors = list_resp.json()
        our_connector = next((c for c in connectors if c["id"] == connector_id), None)
        assert our_connector is not None

        # 3. Get connector details
        get_resp = client.get(f"/v1/connectors/{connector_id}")
        assert get_resp.status_code == 200

        details = get_resp.json()
        assert details["command"] == "npx"
        assert details["args"] == ["-y", "@modelcontextprotocol/server-postgres"]
        assert details["timeout"] == 30

        # 4. Update connector (change timeout)
        update_resp = client.put(
            f"/v1/connectors/{connector_id}",
            json={"timeout": 60}
        )
        assert update_resp.status_code == 200

        updated = update_resp.json()
        # updated_at should change
        assert updated["updated_at"] != connector["created_at"]

        # 5. Get tools (will be empty with mock)
        tools_resp = client.get(f"/v1/connectors/{connector_id}/tools")
        assert tools_resp.status_code == 200

        tools_data = tools_resp.json()
        assert tools_data["connector_id"] == connector_id
        assert "tools" in tools_data

        # 6. Disable connector
        disable_resp = client.patch(
            f"/v1/connectors/{connector_id}",
            json={"enabled": False}
        )
        assert disable_resp.status_code == 200

        disabled = disable_resp.json()
        assert disabled["enabled"] is False

        # 7. Delete connector
        delete_resp = client.delete(f"/v1/connectors/{connector_id}")
        assert delete_resp.status_code == 200

        delete_data = delete_resp.json()
        assert delete_data["status"] == "deleted"

        # Verify deletion
        get_after_delete = client.get(f"/v1/connectors/{connector_id}")
        assert get_after_delete.status_code in [404, 500]


class TestE2EPersistenceWorkflow:
    """Test persistence across operations."""

    def test_connector_persists_across_api_calls(self, client, mock_mcp_client):
        """Test that connector persists after creation."""
        # Create connector
        create_resp = client.post(
            "/v1/connectors",
            json={
                "name": "persistence_test",
                "type": "stdio",
                "command": "echo",
                "enabled": True,
            }
        )

        connector_id = create_resp.json()["id"]

        # Make multiple GET calls - should all succeed
        for _ in range(3):
            get_resp = client.get(f"/v1/connectors/{connector_id}")
            assert get_resp.status_code == 200
            assert get_resp.json()["name"] == "persistence_test"

        # Update and verify persisted
        client.put(
            f"/v1/connectors/{connector_id}",
            json={"name": "persistence_test_updated"}
        )

        # Get again - should have new name
        get_resp = client.get(f"/v1/connectors/{connector_id}")
        assert get_resp.json()["name"] == "persistence_test_updated"

    def test_updates_reflect_in_list(self, client, mock_mcp_client):
        """Test that updates are reflected in list endpoint."""
        # Create connector
        create_resp = client.post(
            "/v1/connectors",
            json={
                "name": "list_reflection_test",
                "type": "stdio",
                "command": "echo",
                "enabled": True,
            }
        )

        connector_id = create_resp.json()["id"]

        # List - verify present
        list_resp = client.get("/v1/connectors")
        connectors = list_resp.json()
        connector = next(c for c in connectors if c["id"] == connector_id)
        assert connector["enabled"] is True

        # Disable
        client.patch(f"/v1/connectors/{connector_id}", json={"enabled": False})

        # List again - should show disabled
        list_resp = client.get("/v1/connectors")
        connectors = list_resp.json()
        connector = next(c for c in connectors if c["id"] == connector_id)
        assert connector["enabled"] is False


class TestE2EMultipleConnectors:
    """Test workflows with multiple connectors."""

    def test_manage_multiple_connectors_independently(self, client, mock_mcp_client):
        """Test managing multiple connectors independently."""
        # Create 3 connectors
        connector_ids = []

        for i in range(3):
            resp = client.post(
                "/v1/connectors",
                json={
                    "name": f"multi_test_{i}",
                    "type": "stdio",
                    "command": "echo",
                    "args": [str(i)],
                    "enabled": True,
                }
            )
            connector_ids.append(resp.json()["id"])

        # List - should have all 3 (plus any bootstrap connectors)
        list_resp = client.get("/v1/connectors")
        connectors = list_resp.json()
        our_connectors = [c for c in connectors if c["id"] in connector_ids]
        assert len(our_connectors) == 3

        # Update middle one
        client.put(
            f"/v1/connectors/{connector_ids[1]}",
            json={"name": "multi_test_1_updated"}
        )

        # Disable first one
        client.patch(
            f"/v1/connectors/{connector_ids[0]}",
            json={"enabled": False}
        )

        # Delete last one
        client.delete(f"/v1/connectors/{connector_ids[2]}")

        # List and verify states
        list_resp = client.get("/v1/connectors")
        connectors = list_resp.json()

        # First should be disabled
        c0 = next((c for c in connectors if c["id"] == connector_ids[0]), None)
        assert c0 is not None
        assert c0["enabled"] is False

        # Middle should have new name
        c1 = next((c for c in connectors if c["id"] == connector_ids[1]), None)
        assert c1 is not None
        assert c1["name"] == "multi_test_1_updated"

        # Last should be gone
        c2 = next((c for c in connectors if c["id"] == connector_ids[2]), None)
        assert c2 is None

    def test_different_transport_types(self, client, mock_mcp_client):
        """Test managing both stdio and SSE connectors."""
        # Create stdio connector
        stdio_resp = client.post(
            "/v1/connectors",
            json={
                "name": "stdio_connector",
                "type": "stdio",
                "command": "echo",
                "enabled": True,
            }
        )

        stdio_id = stdio_resp.json()["id"]

        # Create SSE connector
        sse_resp = client.post(
            "/v1/connectors",
            json={
                "name": "sse_connector",
                "type": "sse",
                "url": "https://api.example.com/mcp",
                "headers": {"Authorization": "Bearer token"},
                "enabled": True,
            }
        )

        sse_id = sse_resp.json()["id"]

        # Get both and verify types
        stdio_details = client.get(f"/v1/connectors/{stdio_id}").json()
        sse_details = client.get(f"/v1/connectors/{sse_id}").json()

        assert stdio_details["type"] == "stdio"
        assert stdio_details["command"] == "echo"
        assert stdio_details["url"] is None

        assert sse_details["type"] == "sse"
        assert sse_details["url"] == "https://api.example.com/mcp"
        assert sse_details["command"] is None


class TestE2EErrorHandling:
    """Test error handling in workflows."""

    def test_duplicate_operations_idempotent(self, client, mock_mcp_client):
        """Test that duplicate operations are handled gracefully."""
        # Create connector
        create_resp = client.post(
            "/v1/connectors",
            json={
                "name": "idempotent_test",
                "type": "stdio",
                "command": "echo",
                "enabled": True,
            }
        )

        connector_id = create_resp.json()["id"]

        # Enable already-enabled connector (should succeed)
        enable_resp = client.patch(
            f"/v1/connectors/{connector_id}",
            json={"enabled": True}
        )
        assert enable_resp.status_code == 200

        # Disable, then disable again
        client.patch(f"/v1/connectors/{connector_id}", json={"enabled": False})
        disable_again = client.patch(
            f"/v1/connectors/{connector_id}",
            json={"enabled": False}
        )
        assert disable_again.status_code == 200

    def test_update_nonexistent_connector(self, client):
        """Test updating non-existent connector."""
        fake_id = "00000000-0000-0000-0000-000000000000"

        update_resp = client.put(
            f"/v1/connectors/{fake_id}",
            json={"timeout": 60}
        )

        assert update_resp.status_code in [404, 500]

    def test_delete_nonexistent_connector(self, client):
        """Test deleting non-existent connector."""
        fake_id = "00000000-0000-0000-0000-000000000000"

        delete_resp = client.delete(f"/v1/connectors/{fake_id}")

        assert delete_resp.status_code in [404, 500]

    def test_get_tools_from_nonexistent_connector(self, client):
        """Test getting tools from non-existent connector."""
        fake_id = "00000000-0000-0000-0000-000000000000"

        tools_resp = client.get(f"/v1/connectors/{fake_id}/tools")

        assert tools_resp.status_code in [404, 500]


class TestE2EStateConsistency:
    """Test state consistency across operations."""

    def test_enabled_state_consistency(self, client, mock_mcp_client):
        """Test enabled state remains consistent across operations."""
        # Create enabled connector
        create_resp = client.post(
            "/v1/connectors",
            json={
                "name": "state_consistency_test",
                "type": "stdio",
                "command": "echo",
                "enabled": True,
            }
        )

        connector_id = create_resp.json()["id"]

        # Verify enabled in GET
        get_resp = client.get(f"/v1/connectors/{connector_id}")
        assert get_resp.json()["enabled"] is True

        # Verify enabled in LIST
        list_resp = client.get("/v1/connectors")
        connector = next(c for c in list_resp.json() if c["id"] == connector_id)
        assert connector["enabled"] is True

        # Disable
        client.patch(f"/v1/connectors/{connector_id}", json={"enabled": False})

        # Verify disabled everywhere
        get_resp = client.get(f"/v1/connectors/{connector_id}")
        assert get_resp.json()["enabled"] is False

        list_resp = client.get("/v1/connectors")
        connector = next(c for c in list_resp.json() if c["id"] == connector_id)
        assert connector["enabled"] is False

    def test_metadata_consistency(self, client, mock_mcp_client):
        """Test metadata (source, timestamps) remains consistent."""
        # Create connector
        create_resp = client.post(
            "/v1/connectors",
            json={
                "name": "metadata_test",
                "type": "stdio",
                "command": "echo",
                "enabled": True,
            }
        )

        connector = create_resp.json()
        connector_id = connector["id"]
        created_at = connector["created_at"]

        # Source should always be "api"
        assert connector["source"] == "api"

        # Get and verify
        get_resp = client.get(f"/v1/connectors/{connector_id}")
        assert get_resp.json()["source"] == "api"
        assert get_resp.json()["created_at"] == created_at

        # Update (updated_at should change)
        update_resp = client.put(
            f"/v1/connectors/{connector_id}",
            json={"name": "metadata_test_updated"}
        )

        updated_at = update_resp.json()["updated_at"]
        assert updated_at >= created_at  # Should be later

        # created_at should NOT change
        get_resp = client.get(f"/v1/connectors/{connector_id}")
        assert get_resp.json()["created_at"] == created_at


class TestE2EToolDiscovery:
    """Test tool discovery workflow."""

    def test_tools_available_after_creation(self, client):
        """Test that tools become available after connector creation."""
        # Mock tools
        mock_tool = MagicMock()
        mock_tool.name = "test_query"
        mock_tool.description = "Execute a test query"
        mock_tool.get_signature.return_value = "test_query(sql: str)"
        mock_tool.input_schema = {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"]
        }

        with patch('sabre.server.mcp.client_manager.MCPClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=[mock_tool])
            mock_client.is_connected = MagicMock(return_value=True)  # Synchronous method
            mock_client.tools_cache = [mock_tool]
            mock_client_class.return_value = mock_client

            # Create connector
            create_resp = client.post(
                "/v1/connectors",
                json={
                    "name": "tool_discovery_test",
                    "type": "stdio",
                    "command": "echo",
                    "enabled": True,
                }
            )

            connector_id = create_resp.json()["id"]

            # Get tools
            tools_resp = client.get(f"/v1/connectors/{connector_id}/tools")

            assert tools_resp.status_code == 200
            tools_data = tools_resp.json()

            assert len(tools_data["tools"]) == 1
            assert tools_data["tools"][0]["name"] == "test_query"
            assert tools_data["tools"][0]["description"] == "Execute a test query"


class TestE2ECompleteScenario:
    """Test complete realistic scenario."""

    def test_typical_user_workflow(self, client, mock_mcp_client):
        """
        Test typical user workflow:
        1. Start with no connectors
        2. Add a database connector
        3. Use it (simulated by getting tools)
        4. Realize need for another connector
        5. Add file system connector
        6. Both should be available
        7. Decide database timeout too short
        8. Update database connector
        9. Done with file system for now
        10. Disable it
        11. Later delete it
        """
        # 1. Check starting state
        initial_list = client.get("/v1/connectors").json()
        initial_count = len(initial_list)

        # 2. Add database connector
        db_resp = client.post(
            "/v1/connectors",
            json={
                "name": "my_postgres",
                "type": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-postgres"],
                "env": {"POSTGRES_URL": "postgresql://localhost/myapp"},
                "enabled": True,
                "timeout": 30,
            }
        )

        db_id = db_resp.json()["id"]

        # 3. Check tools available
        tools_resp = client.get(f"/v1/connectors/{db_id}/tools")
        assert tools_resp.status_code == 200

        # 4-5. Add file system connector
        fs_resp = client.post(
            "/v1/connectors",
            json={
                "name": "my_filesystem",
                "type": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
                "enabled": True,
            }
        )

        fs_id = fs_resp.json()["id"]

        # 6. Both should be in list
        list_resp = client.get("/v1/connectors")
        connectors = list_resp.json()
        assert len(connectors) == initial_count + 2

        db_in_list = any(c["id"] == db_id for c in connectors)
        fs_in_list = any(c["id"] == fs_id for c in connectors)
        assert db_in_list and fs_in_list

        # 7-8. Update database timeout
        update_resp = client.put(
            f"/v1/connectors/{db_id}",
            json={"timeout": 60}
        )
        assert update_resp.status_code == 200

        # 9-10. Disable filesystem
        disable_resp = client.patch(
            f"/v1/connectors/{fs_id}",
            json={"enabled": False}
        )
        assert disable_resp.status_code == 200

        # Verify disabled
        fs_details = client.get(f"/v1/connectors/{fs_id}").json()
        assert fs_details["enabled"] is False

        # 11. Delete filesystem
        delete_resp = client.delete(f"/v1/connectors/{fs_id}")
        assert delete_resp.status_code == 200

        # Final state: only database connector
        final_list = client.get("/v1/connectors").json()
        final_ids = [c["id"] for c in final_list]

        assert db_id in final_ids
        assert fs_id not in final_ids
