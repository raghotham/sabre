"""Test ATIF export functionality."""

import json
from sabre.server.atif_export import events_to_atif


def test_atif_export_basic():
    """Test basic ATIF export with minimal events."""
    events = [
        {
            "timestamp": "2024-01-01T12:00:00",
            "event_type": "session_start",
            "session_id": "test-session-123",
            "message": "Hello, can you help me?",
        },
        {
            "timestamp": "2024-01-01T12:00:01",
            "event_type": "assistant_message",
            "session_id": "test-session-123",
            "node_id": "node-1",
            "parent_id": None,
            "depth": 0,
            "conversation_id": "conv-123",
            "content": "Of course! I'll help you with that.",
        },
    ]

    atif = events_to_atif(
        events,
        agent_name="sabre",
        agent_version="1.0.0",
        model_name="gpt-4o",
    )

    # Verify structure
    assert atif["schema_version"] == "ATIF-v1.2"
    assert atif["session_id"] == "test-session-123"
    assert atif["agent"]["name"] == "sabre"
    assert atif["agent"]["version"] == "1.0.0"
    assert atif["agent"]["model_name"] == "gpt-4o"

    # Verify steps
    assert len(atif["steps"]) == 2
    assert atif["steps"][0]["source"] == "user"
    assert atif["steps"][0]["message"] == "Hello, can you help me?"
    assert atif["steps"][1]["source"] == "agent"
    assert atif["steps"][1]["message"] == "Of course! I'll help you with that."

    # Verify metrics
    assert atif["final_metrics"]["total_steps"] == 2

    print("✓ Basic ATIF export test passed")
    print(f"Generated ATIF:\n{json.dumps(atif, indent=2)}")


def test_atif_export_with_helpers():
    """Test ATIF export with helper execution."""
    events = [
        {
            "timestamp": "2024-01-01T12:00:00",
            "event_type": "session_start",
            "session_id": "test-session-456",
            "message": "Calculate 2+2",
        },
        {
            "timestamp": "2024-01-01T12:00:01",
            "event_type": "node_output",
            "session_id": "test-session-456",
            "node_id": "helper-node-1",
            "output_type": "helper_code",
            "content": "result = 2 + 2\nprint(result)",
        },
        {
            "timestamp": "2024-01-01T12:00:02",
            "event_type": "node_output",
            "session_id": "test-session-456",
            "node_id": "helper-node-1",
            "output_type": "helper_result",
            "content": "4",
        },
        {
            "timestamp": "2024-01-01T12:00:03",
            "event_type": "assistant_message",
            "session_id": "test-session-456",
            "node_id": "node-2",
            "parent_id": "node-1",
            "depth": 0,
            "conversation_id": "conv-456",
            "content": "The result is 4.",
        },
        {
            "timestamp": "2024-01-01T12:00:04",
            "event_type": "node_complete",
            "session_id": "test-session-456",
            "node_id": "node-2",
            "status": "success",
            "duration_ms": 1000,
            "tokens": {
                "input_tokens": 100,
                "output_tokens": 50,
                "reasoning_tokens": 10,
            },
        },
    ]

    atif = events_to_atif(
        events,
        agent_name="sabre",
        agent_version="1.0.0",
        model_name="gpt-4o",
    )

    # Verify structure
    assert atif["schema_version"] == "ATIF-v1.2"
    assert atif["session_id"] == "test-session-456"

    # Verify steps
    assert len(atif["steps"]) == 3  # user message, tool call, assistant response

    # Check user message
    assert atif["steps"][0]["source"] == "user"
    assert atif["steps"][0]["message"] == "Calculate 2+2"

    # Check tool call
    step = atif["steps"][1]
    assert step["source"] == "agent"
    assert step["message"] == "Executing code"
    assert len(step["tool_calls"]) == 1
    assert step["tool_calls"][0]["function_name"] == "python_runtime"
    assert "2 + 2" in step["tool_calls"][0]["arguments"]["code"]
    assert step["observation"]["results"][0]["content"] == "4"

    # Check final response
    assert atif["steps"][2]["source"] == "agent"
    assert atif["steps"][2]["message"] == "The result is 4."

    # Verify metrics
    assert atif["final_metrics"]["total_prompt_tokens"] == 100
    assert atif["final_metrics"]["total_completion_tokens"] == 50
    assert atif["final_metrics"]["total_cached_tokens"] == 10
    assert atif["final_metrics"]["total_steps"] == 3

    print("✓ Helper execution ATIF export test passed")
    print(f"Generated ATIF:\n{json.dumps(atif, indent=2)}")


def test_atif_export_empty():
    """Test ATIF export with no events."""
    events = []

    atif = events_to_atif(
        events,
        agent_name="sabre",
        agent_version="1.0.0",
    )

    # Verify structure
    assert atif["schema_version"] == "ATIF-v1.2"
    assert atif["agent"]["name"] == "sabre"
    assert len(atif["steps"]) == 0
    assert atif["final_metrics"]["total_steps"] == 0

    print("✓ Empty ATIF export test passed")


if __name__ == "__main__":
    print("Running ATIF export tests...\n")
    test_atif_export_basic()
    print()
    test_atif_export_with_helpers()
    print()
    test_atif_export_empty()
    print("\n✓ All tests passed!")
