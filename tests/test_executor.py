"""
Tests for ResponseExecutor.

Run with: uv run pytest
Or: uv run pytest tests/test_executor.py
Or specific test: uv run pytest tests/test_executor.py::test_simple_execution
"""

import pytest
from sabre.common.executors import ResponseExecutor
from sabre.common.models import User, TextContent, ResponseTokenEvent


@pytest.fixture
def executor():
    """Create a ResponseExecutor instance for testing."""
    return ResponseExecutor()


@pytest.fixture
def tree_context():
    """Create dummy tree context for testing."""
    return {
        "node_id": "test-node",
        "parent_id": None,
        "depth": 0,
        "path": ["test-node"],
    }


@pytest.mark.asyncio
async def test_simple_execution(check_api_key, executor):
    """Test simple non-streaming execution."""
    response = await executor.execute_simple(
        user_message="What is 2+2? Answer in one word.",
        system_message="You are a helpful assistant.",
    )

    # Assertions
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    print(f"✓ Simple execution response: {response}")


@pytest.mark.asyncio
async def test_streaming_execution(check_api_key, executor, tree_context):
    """Test streaming execution with event callback."""
    # Track tokens received
    tokens_received = []

    async def on_event(event):
        if isinstance(event, ResponseTokenEvent):
            tokens_received.append(event.data["token"])

    # Create user message
    messages = [User(content=[TextContent("Count from 1 to 5, one number per line.")])]

    # Execute with streaming
    assistant = await executor.execute(
        messages=messages,
        event_callback=on_event,
        tree_context=tree_context,
    )

    # Assertions
    assert assistant is not None
    assert assistant.response_id is not None
    assert len(assistant.get_str()) > 0
    assert len(tokens_received) > 0, "Should have received streaming tokens"

    full_response = assistant.get_str()
    print(f"✓ Streaming execution response: {full_response}")
    print(f"  Received {len(tokens_received)} token events")
    print(f"  Response ID: {assistant.response_id}")


@pytest.mark.asyncio
async def test_response_continuation(check_api_key, executor):
    """Test response_id continuation."""
    # First message
    messages1 = [User(content=[TextContent("Say 'Hello, I am ready to continue.'")])]

    # Execute first call
    assistant1 = await executor.execute(messages1)

    # Assertions for first call
    assert assistant1 is not None
    assert assistant1.response_id is not None
    first_response = assistant1.get_str()
    first_response_id = assistant1.response_id

    print(f"✓ First call response: {first_response}")
    print(f"  Response ID: {first_response_id}")

    # Continue with response_id
    messages2 = messages1 + [assistant1, User(content=[TextContent("Now count to 3.")])]

    assistant2 = await executor.execute(
        messages2,
        previous_response_id=assistant1.response_id,
    )

    # Assertions for continuation
    assert assistant2 is not None
    assert assistant2.response_id is not None
    assert assistant2.response_id != assistant1.response_id, "Should get new response_id"
    second_response = assistant2.get_str()

    print(f"✓ Continuation response: {second_response}")
    print(f"  Response ID: {assistant2.response_id}")


@pytest.mark.asyncio
async def test_token_counting(check_api_key, executor):
    """Test approximate token counting."""
    messages = [User(content=[TextContent("Hello, how are you?")])]

    token_count = await executor.count_tokens(messages)

    # Assertions
    assert token_count > 0
    assert isinstance(token_count, int)
    print(f"✓ Token counting: ~{token_count} tokens for message")


@pytest.mark.asyncio
async def test_custom_model(check_api_key, executor):
    """Test using custom model parameter."""
    response = await executor.execute_simple(
        user_message="Say 'test' and nothing else.",
        model="gpt-4o-mini",  # Use cheaper model for testing
    )

    # Assertions
    assert response is not None
    assert len(response) > 0
    print(f"✓ Custom model (gpt-4o-mini) response: {response}")


# Mark tests that require API key
pytestmark = pytest.mark.asyncio
