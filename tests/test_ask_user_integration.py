"""
Integration tests for ask_user feature.

These tests verify the full flow from runtime execution to event emission.

Run with: uv run pytest tests/test_ask_user_integration.py
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock
from sabre.server.python_runtime import PythonRuntime
from sabre.common.execution_context import set_execution_context, clear_execution_context
from sabre.common import ExecutionTree, ExecutionNodeType
from sabre.common.models.events import AskUserEvent


def _build_tree_context(tree: ExecutionTree, conversation_id: str) -> dict:
    """Build tree context dict for tests."""
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
        "path_summary": "Test → Integration",
        "conversation_id": conversation_id,
    }


@pytest.fixture
def runtime_with_context():
    """Create runtime and context components (without setting context yet)."""
    runtime = PythonRuntime()

    tree = ExecutionTree()
    tree.push(ExecutionNodeType.CLIENT_REQUEST, metadata={"test": "integration"})

    event_callback = AsyncMock()
    conversation_id = "test-integration-conv"

    yield runtime, event_callback, tree, conversation_id


@pytest.mark.asyncio
async def test_runtime_ask_user_single_question(runtime_with_context):
    """Test ask_user from runtime with single question."""
    runtime, event_callback, tree, conversation_id = runtime_with_context

    # Set execution context with event loop
    current_loop = asyncio.get_running_loop()
    set_execution_context(
        event_callback=event_callback,
        tree=tree,
        tree_context=_build_tree_context(tree, conversation_id),
        conversation_id=conversation_id,
        loop=current_loop,
        interactive_mode=True,
    )

    try:
        # Create async task to execute code
        async def execute_with_answer():
            # Start execution task
            exec_task = asyncio.create_task(
                asyncio.to_thread(
                    runtime.execute,
                    """
answer = ask_user("What is your name?")
result(f"Hello, {answer}!")
"""
                )
            )

            # Wait for event to be emitted
            await asyncio.sleep(0.2)

            # Verify event was emitted
            assert event_callback.call_count == 1
            event = event_callback.call_args[0][0]
            assert isinstance(event, AskUserEvent)

            # Answer the question
            question_id = event.data["question_id"]
            future = runtime.pending_questions[question_id]
            future.set_result(["Alice"])

            # Wait for execution to complete
            result = await exec_task
            return result

        result = await execute_with_answer()

        assert result.success
        assert result.results[0] == "Hello, Alice!"
    finally:
        clear_execution_context()


@pytest.mark.asyncio
async def test_runtime_ask_user_multiple_questions(runtime_with_context):
    """Test ask_user from runtime with multiple questions."""
    runtime, event_callback, tree, conversation_id = runtime_with_context

    # Set execution context with event loop
    current_loop = asyncio.get_running_loop()
    set_execution_context(
        event_callback=event_callback,
        tree=tree,
        tree_context=_build_tree_context(tree, conversation_id),
        conversation_id=conversation_id,
        loop=current_loop,
        interactive_mode=True,
    )

    try:
        async def execute_with_answers():
            exec_task = asyncio.create_task(
                asyncio.to_thread(
                    runtime.execute,
                    """
answers = ask_user([
    "What is your name?",
    "What is your age?",
    "What is your city?"
])
name, age, city = answers
result(f"{name} is {age} years old and lives in {city}")
"""
                )
            )

            await asyncio.sleep(0.2)

            event = event_callback.call_args[0][0]
            question_id = event.data["question_id"]
            future = runtime.pending_questions[question_id]
            future.set_result(["Bob", "25", "New York"])

            result = await exec_task
            return result

        result = await execute_with_answers()

        assert result.success
        assert result.results[0] == "Bob is 25 years old and lives in New York"
    finally:
        clear_execution_context()


@pytest.mark.asyncio
async def test_runtime_ask_user_autonomous_mode_error():
    """Test that ask_user raises error in autonomous mode."""
    runtime = PythonRuntime()

    tree = ExecutionTree()
    tree.push(ExecutionNodeType.CLIENT_REQUEST, metadata={})

    conversation_id = "test-conv"

    # Set execution context with event loop (even though autonomous mode)
    current_loop = asyncio.get_running_loop()
    set_execution_context(
        event_callback=AsyncMock(),
        tree=tree,
        tree_context=_build_tree_context(tree, conversation_id),
        conversation_id=conversation_id,
        loop=current_loop,
        interactive_mode=False,  # Autonomous mode
    )

    try:
        # Run in thread to avoid event loop detection error
        result = await asyncio.to_thread(
            runtime.execute,
            """
answer = ask_user("What is your name?")
result(answer)
"""
        )

        assert not result.success
        assert "autonomous mode" in result.error.lower()
    finally:
        clear_execution_context()


@pytest.mark.asyncio
async def test_runtime_ask_user_in_conditional(runtime_with_context):
    """Test ask_user used conditionally in runtime."""
    runtime, event_callback, tree, conversation_id = runtime_with_context

    # Set execution context with event loop
    current_loop = asyncio.get_running_loop()
    set_execution_context(
        event_callback=event_callback,
        tree=tree,
        tree_context=_build_tree_context(tree, conversation_id),
        conversation_id=conversation_id,
        loop=current_loop,
        interactive_mode=True,
    )

    try:
        async def execute_with_conditional():
            exec_task = asyncio.create_task(
                asyncio.to_thread(
                    runtime.execute,
                    """
data_missing = True

if data_missing:
    strategy = ask_user("Missing data found. Strategy: drop or fill?")
    result(f"Using strategy: {strategy}")
else:
    result("No missing data")
"""
                )
            )

            await asyncio.sleep(0.2)

            event = event_callback.call_args[0][0]
            question_id = event.data["question_id"]
            future = runtime.pending_questions[question_id]
            future.set_result(["fill"])

            result = await exec_task
            return result

        result = await execute_with_conditional()

        assert result.success
        assert result.results[0] == "Using strategy: fill"
    finally:
        clear_execution_context()


@pytest.mark.asyncio
async def test_runtime_ask_user_in_loop():
    """Test multiple ask_user calls in a loop."""
    runtime = PythonRuntime()

    tree = ExecutionTree()
    tree.push(ExecutionNodeType.CLIENT_REQUEST, metadata={})

    event_callback = AsyncMock()
    conversation_id = "test-conv"

    # Set execution context with event loop
    current_loop = asyncio.get_running_loop()
    set_execution_context(
        event_callback=event_callback,
        tree=tree,
        tree_context=_build_tree_context(tree, conversation_id),
        conversation_id=conversation_id,
        loop=current_loop,
        interactive_mode=True,
    )

    try:
        async def execute_with_loop():
            exec_task = asyncio.create_task(
                asyncio.to_thread(
                    runtime.execute,
                    """
items = ["apple", "banana", "cherry"]
results = []

for item in items:
    quantity = ask_user(f"How many {item}s?")
    results.append(f"{quantity} {item}s")

result(results)
"""
                )
            )

            # Answer each question as it comes
            for i, answer in enumerate(["5", "3", "10"]):
                await asyncio.sleep(0.2)

                # Get the event
                call_idx = i
                event = event_callback.call_args_list[call_idx][0][0]
                question_id = event.data["question_id"]
                future = runtime.pending_questions[question_id]
                future.set_result([answer])

            result = await exec_task
            return result

        result = await execute_with_loop()

        assert result.success
        assert result.results[0] == ["5 apples", "3 bananas", "10 cherrys"]

    finally:
        clear_execution_context()


@pytest.mark.asyncio
async def test_runtime_ask_user_event_data(runtime_with_context):
    """Test that emitted event contains correct data."""
    runtime, event_callback, tree, conversation_id = runtime_with_context

    # Set execution context with event loop
    current_loop = asyncio.get_running_loop()
    set_execution_context(
        event_callback=event_callback,
        tree=tree,
        tree_context=_build_tree_context(tree, conversation_id),
        conversation_id=conversation_id,
        loop=current_loop,
        interactive_mode=True,
    )

    try:
        questions = ["Question 1?", "Question 2?", "Question 3?"]

        async def execute_and_verify():
            exec_task = asyncio.create_task(
                asyncio.to_thread(
                    runtime.execute,
                    f"""
questions = {questions}
answers = ask_user(questions)
result(answers)
"""
                )
            )

            await asyncio.sleep(0.2)

            # Verify event
            event = event_callback.call_args[0][0]
            assert isinstance(event, AskUserEvent)
            assert event.data["questions"] == questions
            assert event.conversation_id == "test-integration-conv"
            assert "question_id" in event.data

            # Answer
            question_id = event.data["question_id"]
            future = runtime.pending_questions[question_id]
            future.set_result(["A1", "A2", "A3"])

            result = await exec_task
            return result

        result = await execute_and_verify()

        assert result.success
        assert result.results[0] == ["A1", "A2", "A3"]
    finally:
        clear_execution_context()


@pytest.mark.asyncio
async def test_runtime_ask_user_with_computation(runtime_with_context):
    """Test ask_user integrated with computation."""
    runtime, event_callback, tree, conversation_id = runtime_with_context

    # Set execution context with event loop
    current_loop = asyncio.get_running_loop()
    set_execution_context(
        event_callback=event_callback,
        tree=tree,
        tree_context=_build_tree_context(tree, conversation_id),
        conversation_id=conversation_id,
        loop=current_loop,
        interactive_mode=True,
    )

    try:
        async def execute_with_computation():
            exec_task = asyncio.create_task(
                asyncio.to_thread(
                    runtime.execute,
                    """
# Get user preferences
format_choice = ask_user("Export format: CSV or JSON?")

# Perform computation based on answer
if format_choice.lower() == "csv":
    data = "name,age\\nAlice,30\\nBob,25"
else:
    data = '[{"name":"Alice","age":30},{"name":"Bob","age":25}]'

result(f"Exported as {format_choice}: {data}")
"""
                )
            )

            await asyncio.sleep(0.2)

            event = event_callback.call_args[0][0]
            question_id = event.data["question_id"]
            future = runtime.pending_questions[question_id]
            future.set_result(["JSON"])

            result = await exec_task
            return result

        result = await execute_with_computation()

        assert result.success
        assert "JSON" in result.results[0]
        assert "[{" in result.results[0]  # JSON format
    finally:
        clear_execution_context()


@pytest.mark.asyncio
async def test_runtime_ask_user_cleanup_on_success(runtime_with_context):
    """Test that pending_questions is cleaned up after successful execution."""
    runtime, event_callback, tree, conversation_id = runtime_with_context

    # Set execution context with event loop
    current_loop = asyncio.get_running_loop()
    set_execution_context(
        event_callback=event_callback,
        tree=tree,
        tree_context=_build_tree_context(tree, conversation_id),
        conversation_id=conversation_id,
        loop=current_loop,
        interactive_mode=True,
    )

    try:
        async def execute_and_check_cleanup():
            exec_task = asyncio.create_task(
                asyncio.to_thread(
                    runtime.execute,
                    """
answer = ask_user("Question?")
result(answer)
"""
                )
            )

            await asyncio.sleep(0.2)

            # Verify question is pending
            assert len(runtime.pending_questions) == 1

            event = event_callback.call_args[0][0]
            question_id = event.data["question_id"]
            future = runtime.pending_questions[question_id]
            future.set_result(["Answer"])

            result = await exec_task
            return result

        result = await execute_and_check_cleanup()

        assert result.success

        # Verify cleanup
        assert len(runtime.pending_questions) == 0
    finally:
        clear_execution_context()


@pytest.mark.asyncio
async def test_runtime_ask_user_with_error_handling(runtime_with_context):
    """Test ask_user with error handling in user code."""
    runtime, event_callback, tree, conversation_id = runtime_with_context

    # Set execution context with event loop
    current_loop = asyncio.get_running_loop()
    set_execution_context(
        event_callback=event_callback,
        tree=tree,
        tree_context=_build_tree_context(tree, conversation_id),
        conversation_id=conversation_id,
        loop=current_loop,
        interactive_mode=True,
    )

    try:
        async def execute_with_error_handling():
            exec_task = asyncio.create_task(
                asyncio.to_thread(
                    runtime.execute,
                    """
try:
    age = ask_user("What is your age?")
    age_int = int(age)
    result(f"Age: {age_int}")
except ValueError:
    result("Invalid age entered")
"""
                )
            )

            await asyncio.sleep(0.2)

            event = event_callback.call_args[0][0]
            question_id = event.data["question_id"]
            future = runtime.pending_questions[question_id]
            future.set_result(["not a number"])

            result = await exec_task
            return result

        result = await execute_with_error_handling()

        assert result.success
        assert result.results[0] == "Invalid age entered"
    finally:
        clear_execution_context()
