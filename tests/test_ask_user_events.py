"""
Tests for AskUserEvent.

Run with: uv run pytest tests/test_ask_user_events.py
"""

import json
from datetime import datetime
from sabre.common.models.events import AskUserEvent, EventType


def test_ask_user_event_creation():
    """Test creating an AskUserEvent."""
    event = AskUserEvent(
        node_id="node-123",
        parent_id="parent-456",
        depth=2,
        path=["root", "parent-456", "node-123"],
        conversation_id="conv-789",
        question_id="q-uuid-1234",
        questions=["What is your name?", "What is your age?"],
        path_summary="User → Response #1 → Helper #1",
    )

    assert event.type == EventType.ASK_USER
    assert event.node_id == "node-123"
    assert event.parent_id == "parent-456"
    assert event.depth == 2
    assert event.conversation_id == "conv-789"
    assert event.data["question_id"] == "q-uuid-1234"
    assert event.data["questions"] == ["What is your name?", "What is your age?"]
    assert event.path_summary == "User → Response #1 → Helper #1"


def test_ask_user_event_to_dict():
    """Test serializing AskUserEvent to dict."""
    event = AskUserEvent(
        node_id="node-123",
        parent_id="parent-456",
        depth=2,
        path=["root", "node-123"],
        conversation_id="conv-789",
        question_id="q-uuid-1234",
        questions=["Question 1?", "Question 2?"],
    )

    event_dict = event.to_dict()

    assert event_dict["type"] == "ask_user"
    assert event_dict["node_id"] == "node-123"
    assert event_dict["parent_id"] == "parent-456"
    assert event_dict["depth"] == 2
    assert event_dict["conversation_id"] == "conv-789"
    assert event_dict["data"]["question_id"] == "q-uuid-1234"
    assert event_dict["data"]["questions"] == ["Question 1?", "Question 2?"]


def test_ask_user_event_to_json():
    """Test serializing AskUserEvent to JSON."""
    event = AskUserEvent(
        node_id="node-123",
        parent_id="parent-456",
        depth=1,
        path=["root"],
        conversation_id="conv-789",
        question_id="q-uuid-1234",
        questions=["What format?"],
    )

    json_str = event.to_json()
    parsed = json.loads(json_str)

    assert parsed["type"] == "ask_user"
    assert parsed["data"]["question_id"] == "q-uuid-1234"
    assert parsed["data"]["questions"] == ["What format?"]


def test_ask_user_event_from_json():
    """Test deserializing AskUserEvent from JSON."""
    json_str = json.dumps({
        "type": "ask_user",
        "node_id": "node-123",
        "parent_id": "parent-456",
        "depth": 1,
        "path": ["root", "node-123"],
        "conversation_id": "conv-789",
        "timestamp": datetime.now().isoformat(),
        "data": {
            "question_id": "q-uuid-1234",
            "questions": ["Question 1?", "Question 2?"]
        },
        "path_summary": "Test path"
    })

    from sabre.common.models.events import Event
    event = Event.from_json(json_str)

    assert isinstance(event, AskUserEvent)
    assert event.type == EventType.ASK_USER
    assert event.data["question_id"] == "q-uuid-1234"
    assert event.data["questions"] == ["Question 1?", "Question 2?"]


def test_ask_user_event_single_question():
    """Test AskUserEvent with single question."""
    event = AskUserEvent(
        node_id="node-1",
        parent_id=None,
        depth=1,
        path=["node-1"],
        conversation_id="conv-1",
        question_id="q-1",
        questions=["Single question?"],
    )

    assert len(event.data["questions"]) == 1
    assert event.data["questions"][0] == "Single question?"


def test_ask_user_event_multiple_questions():
    """Test AskUserEvent with multiple questions."""
    questions = [
        "Question 1?",
        "Question 2?",
        "Question 3?",
        "Question 4?",
        "Question 5?",
    ]

    event = AskUserEvent(
        node_id="node-1",
        parent_id=None,
        depth=1,
        path=["node-1"],
        conversation_id="conv-1",
        question_id="q-1",
        questions=questions,
    )

    assert len(event.data["questions"]) == 5
    assert event.data["questions"] == questions


def test_ask_user_event_timestamp():
    """Test that AskUserEvent has timestamp."""
    before = datetime.now()

    event = AskUserEvent(
        node_id="node-1",
        parent_id=None,
        depth=1,
        path=["node-1"],
        conversation_id="conv-1",
        question_id="q-1",
        questions=["Question?"],
    )

    after = datetime.now()

    # Timestamp should be between before and after
    assert before <= event.timestamp <= after


def test_ask_user_event_question_id_required():
    """Test that question_id is required in event data."""
    event = AskUserEvent(
        node_id="node-1",
        parent_id=None,
        depth=1,
        path=["node-1"],
        conversation_id="conv-1",
        question_id="my-question-id",
        questions=["Question?"],
    )

    assert "question_id" in event.data
    assert event.data["question_id"] == "my-question-id"


def test_ask_user_event_questions_required():
    """Test that questions are required in event data."""
    event = AskUserEvent(
        node_id="node-1",
        parent_id=None,
        depth=1,
        path=["node-1"],
        conversation_id="conv-1",
        question_id="q-1",
        questions=["Q1", "Q2"],
    )

    assert "questions" in event.data
    assert isinstance(event.data["questions"], list)
    assert len(event.data["questions"]) == 2


def test_ask_user_event_serialization_round_trip():
    """Test that event can be serialized and deserialized."""
    original = AskUserEvent(
        node_id="node-123",
        parent_id="parent-456",
        depth=3,
        path=["root", "parent-456", "node-123"],
        conversation_id="conv-789",
        question_id="q-uuid-1234",
        questions=["Q1?", "Q2?", "Q3?"],
        path_summary="Test → Path → Summary",
    )

    # Serialize
    json_str = original.to_json()

    # Deserialize
    from sabre.common.models.events import Event
    deserialized = Event.from_json(json_str)

    # Verify
    assert isinstance(deserialized, AskUserEvent)
    assert deserialized.type == original.type
    assert deserialized.node_id == original.node_id
    assert deserialized.parent_id == original.parent_id
    assert deserialized.depth == original.depth
    assert deserialized.conversation_id == original.conversation_id
    assert deserialized.data["question_id"] == original.data["question_id"]
    assert deserialized.data["questions"] == original.data["questions"]
    assert deserialized.path_summary == original.path_summary
