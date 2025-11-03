"""
Tests for ask_user helper.

Run with: uv run pytest tests/test_ask_user_helper.py
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from sabre.server.helpers.ask_user import AskUser
from sabre.common.execution_context import set_execution_context, clear_execution_context
from sabre.common.models.events import AskUserEvent
from sabre.common import ExecutionTree, ExecutionNodeType


@pytest.fixture
def mock_pending_questions():
    """Mock pending_questions dictionary."""
    return {}


@pytest.fixture
def ask_user_helper(mock_pending_questions):
    """Create ask_user helper with mock pending_questions."""
    return AskUser(lambda: mock_pending_questions)


def _build_tree_context(tree: ExecutionTree, conversation_id: str) -> dict:
    """Build tree context dict for tests (mimics orchestrator behavior)."""
    if not tree.current:
        return {
            "node_id": "",
            "parent_id": None,
            "depth": 0,
            "path": [],
            "path_summary": "",
            "conversation_id": conversation_id,
        }

    return {
        "node_id": tree.current.id,
        "parent_id": tree.current.parent_id,
        "depth": tree.get_depth(),
        "path": [n.id for n in tree.get_path()],
        "path_summary": "Test → Path",
        "conversation_id": conversation_id,
    }


@pytest.fixture
def mock_execution_context():
    """Set up mock execution context for tests."""
    tree = ExecutionTree()
    tree.push(ExecutionNodeType.CLIENT_REQUEST, metadata={"test": "data"})

    event_callback = AsyncMock()
    conversation_id = "test-conv-123"

    set_execution_context(
        event_callback=event_callback,
        tree=tree,
        tree_context=_build_tree_context(tree, conversation_id),
        conversation_id=conversation_id,
        interactive_mode=True,
    )

    yield event_callback, tree

    clear_execution_context()


def test_ask_user_requires_execution_context(ask_user_helper):
    """Test that ask_user requires execution context."""
    clear_execution_context()

    with pytest.raises(RuntimeError, match="ask_user\\(\\) requires execution context"):
        ask_user_helper("What is your name?")


def test_ask_user_requires_interactive_mode(ask_user_helper):
    """Test that ask_user requires interactive mode."""
    tree = ExecutionTree()
    tree.push(ExecutionNodeType.CLIENT_REQUEST, metadata={})

    conversation_id = "test-conv"

    set_execution_context(
        event_callback=AsyncMock(),
        tree=tree,
        tree_context=_build_tree_context(tree, conversation_id),
        conversation_id=conversation_id,
        interactive_mode=False,  # Autonomous mode
    )

    try:
        with pytest.raises(RuntimeError, match="ask_user\\(\\) called in autonomous mode"):
            ask_user_helper("What is your name?")
    finally:
        clear_execution_context()


@pytest.mark.asyncio
async def test_ask_user_single_question(ask_user_helper, mock_pending_questions, mock_execution_context):
    """Test ask_user with single question."""
    event_callback, tree = mock_execution_context

    # Create task to answer the question after a short delay
    async def answer_question():
        await asyncio.sleep(0.1)
        # Find the question_id from the emitted event
        assert event_callback.call_count == 1
        event = event_callback.call_args[0][0]
        assert isinstance(event, AskUserEvent)

        question_id = event.data["question_id"]
        future = mock_pending_questions[question_id]
        future.set_result(["Alice"])

    # Start answering task
    answer_task = asyncio.create_task(answer_question())

    # Call ask_user
    result = await ask_user_helper.execute("What is your name?")

    # Wait for answer task to complete
    await answer_task

    # Verify
    assert result == "Alice"
    assert event_callback.call_count == 1

    # Verify event was emitted correctly
    event = event_callback.call_args[0][0]
    assert isinstance(event, AskUserEvent)
    assert event.data["questions"] == ["What is your name?"]


@pytest.mark.asyncio
async def test_ask_user_multiple_questions(ask_user_helper, mock_pending_questions, mock_execution_context):
    """Test ask_user with multiple questions."""
    event_callback, tree = mock_execution_context

    questions = [
        "What is your name?",
        "What is your age?",
        "What is your favorite color?"
    ]

    # Create task to answer the questions
    async def answer_questions():
        await asyncio.sleep(0.1)
        assert event_callback.call_count == 1
        event = event_callback.call_args[0][0]

        question_id = event.data["question_id"]
        future = mock_pending_questions[question_id]
        future.set_result(["Alice", "30", "Blue"])

    answer_task = asyncio.create_task(answer_questions())

    # Call ask_user with multiple questions
    result = await ask_user_helper.execute(questions)

    await answer_task

    # Verify
    assert result == ["Alice", "30", "Blue"]
    assert event_callback.call_count == 1

    # Verify event
    event = event_callback.call_args[0][0]
    assert event.data["questions"] == questions


@pytest.mark.asyncio
async def test_ask_user_timeout(ask_user_helper, mock_pending_questions, mock_execution_context):
    """Test ask_user timeout behavior."""
    event_callback, tree = mock_execution_context

    # Patch timeout to be very short for testing
    with patch('sabre.server.helpers.ask_user.asyncio.wait_for') as mock_wait_for:
        mock_wait_for.side_effect = asyncio.TimeoutError()

        with pytest.raises(RuntimeError, match="ask_user\\(\\) timed out"):
            await ask_user_helper.execute("What is your name?")


@pytest.mark.asyncio
async def test_ask_user_empty_questions(ask_user_helper, mock_pending_questions, mock_execution_context):
    """Test ask_user with empty question list."""
    with pytest.raises(ValueError, match="ask_user\\(\\) requires at least one question"):
        await ask_user_helper.execute([])


@pytest.mark.asyncio
async def test_ask_user_cleans_up_on_error(ask_user_helper, mock_pending_questions, mock_execution_context):
    """Test that ask_user cleans up pending_questions on error."""
    event_callback, tree = mock_execution_context

    # Create task that raises an error
    async def cause_error():
        await asyncio.sleep(0.1)
        event = event_callback.call_args[0][0]
        question_id = event.data["question_id"]

        # Instead of setting result, raise an error
        future = mock_pending_questions[question_id]
        future.set_exception(Exception("Test error"))

    error_task = asyncio.create_task(cause_error())

    with pytest.raises(RuntimeError, match="ask_user\\(\\) failed"):
        await ask_user_helper.execute("What is your name?")

    await error_task

    # Verify cleanup - pending_questions should be empty
    assert len(mock_pending_questions) == 0


@pytest.mark.asyncio
async def test_ask_user_event_contains_correct_data(ask_user_helper, mock_pending_questions, mock_execution_context):
    """Test that AskUserEvent contains correct data."""
    event_callback, tree = mock_execution_context

    questions = ["Q1?", "Q2?"]

    async def answer_questions():
        await asyncio.sleep(0.1)
        event = event_callback.call_args[0][0]
        question_id = event.data["question_id"]
        future = mock_pending_questions[question_id]
        future.set_result(["A1", "A2"])

    answer_task = asyncio.create_task(answer_questions())

    result = await ask_user_helper.execute(questions)

    await answer_task

    # Verify event
    event = event_callback.call_args[0][0]
    assert isinstance(event, AskUserEvent)
    assert event.conversation_id == "test-conv-123"
    assert event.data["questions"] == questions
    assert "question_id" in event.data
    assert len(event.data["question_id"]) > 0  # UUID should be non-empty


@pytest.mark.asyncio
async def test_ask_user_concurrent_calls(mock_pending_questions, mock_execution_context):
    """Test multiple concurrent ask_user calls."""
    event_callback, tree = mock_execution_context

    helper1 = AskUser(lambda: mock_pending_questions)
    helper2 = AskUser(lambda: mock_pending_questions)

    # Track which questions were asked
    asked_questions = []

    async def answer_all_questions():
        await asyncio.sleep(0.1)

        # Answer all pending questions
        for question_id, future in list(mock_pending_questions.items()):
            if not future.done():
                future.set_result(["answer"])

    answer_task = asyncio.create_task(answer_all_questions())

    # Make concurrent calls
    result1, result2 = await asyncio.gather(
        helper1.execute("Question 1?"),
        helper2.execute("Question 2?")
    )

    await answer_task

    # Verify both got answers
    assert result1 == "answer"
    assert result2 == "answer"

    # Verify both events were emitted
    assert event_callback.call_count == 2


@pytest.mark.asyncio
async def test_ask_user_preserves_answer_order(ask_user_helper, mock_pending_questions, mock_execution_context):
    """Test that answers maintain the same order as questions."""
    event_callback, tree = mock_execution_context

    questions = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    expected_answers = ["A1", "A2", "A3", "A4", "A5"]

    async def answer_questions():
        await asyncio.sleep(0.1)
        event = event_callback.call_args[0][0]
        question_id = event.data["question_id"]
        future = mock_pending_questions[question_id]
        future.set_result(expected_answers)

    answer_task = asyncio.create_task(answer_questions())

    result = await ask_user_helper.execute(questions)

    await answer_task

    assert result == expected_answers
