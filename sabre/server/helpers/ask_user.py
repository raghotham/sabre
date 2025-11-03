"""
Ask user for clarification during execution.

Allows code to pause execution and ask the user questions.
Supports both single questions and batch questions.
"""

import asyncio
import logging
import uuid
from typing import Union
from sabre.common.models.events import AskUserEvent

logger = logging.getLogger(__name__)


class AskUser:
    """
    Ask user for input during execution.

    Creates AskUserEvent and waits for response via pending_questions registry.
    """

    def __init__(self, get_pending_questions: callable):
        """
        Initialize AskUser helper.

        Args:
            get_pending_questions: Function to get pending_questions dict from orchestrator
        """
        self.get_pending_questions = get_pending_questions

    def __call__(self, questions: Union[str, list[str]]) -> Union[str, list[str]]:
        """
        Ask user for input (sync wrapper).

        Examples:
            # Single question
            format = ask_user("What format: CSV or JSON?")
            # Returns: "CSV"

            # Multiple questions (batched)
            answers = ask_user([
                "What format: CSV or JSON?",
                "What date range?",
                "Include headers? (yes/no)"
            ])
            # Returns: ["CSV", "2024-01-01 to 2024-12-31", "yes"]

        Args:
            questions: Single question string or list of questions

        Returns:
            Single answer string or list of answers (matches input type)

        Raises:
            RuntimeError: If not in interactive mode or if execution context not available
        """
        from sabre.server.helpers.llm_call import run_async_from_sync

        # Run async code in sync context (exec is synchronous)
        return run_async_from_sync(self.execute(questions))

    async def execute(self, questions: Union[str, list[str]]) -> Union[str, list[str]]:
        """
        Async implementation of ask_user.

        Flow:
        1. Check if we're in interactive mode
        2. Normalize questions to list
        3. Generate unique question_id
        4. Create Future to wait for response
        5. Get execution context (for event callback and tree)
        6. Emit AskUserEvent
        7. Wait for response via Future
        8. Return answer(s) in same format as input

        Args:
            questions: Single question string or list of questions

        Returns:
            Single answer string or list of answers
        """
        # Get execution context
        from sabre.common.execution_context import get_execution_context

        ctx = get_execution_context()
        if not ctx:
            raise RuntimeError("ask_user() requires execution context")

        # Check if we're in interactive mode
        if not ctx.interactive_mode:
            raise RuntimeError(
                "ask_user() called in autonomous mode. "
                "Use --interactive flag to enable user interaction."
            )

        # Normalize to list for internal processing
        is_single_question = isinstance(questions, str)
        questions_list = [questions] if is_single_question else questions

        if not questions_list:
            raise ValueError("ask_user() requires at least one question")

        # Generate unique question_id
        question_id = str(uuid.uuid4())

        logger.info(f"ask_user: question_id={question_id}, questions={len(questions_list)}")

        # Create Future to wait for response
        future = asyncio.Future()

        # Register in pending_questions
        pending_questions = self.get_pending_questions()
        if pending_questions is None:
            raise RuntimeError("Orchestrator pending_questions not available")

        pending_questions[question_id] = future

        # Emit AskUserEvent
        event_callback = ctx.event_callback
        tree_context = ctx.tree_context

        if event_callback and tree_context:
            event = AskUserEvent(
                node_id=tree_context["node_id"],
                parent_id=tree_context["parent_id"],
                depth=tree_context["depth"],
                path=tree_context["path"],
                conversation_id=ctx.conversation_id,
                question_id=question_id,
                questions=questions_list,
                path_summary=tree_context.get("path_summary", ""),
            )
            await event_callback(event)

        logger.info(f"Waiting for user response to question_id={question_id}")

        try:
            # Wait for response (with timeout)
            answers = await asyncio.wait_for(future, timeout=300.0)  # 5 minute timeout

            logger.info(f"Received user response: {answers}")

            # Clean up the pending question
            pending_questions.pop(question_id, None)

            # Return in same format as input
            if is_single_question:
                return answers[0] if answers else ""
            else:
                return answers

        except asyncio.TimeoutError:
            # Clean up
            pending_questions.pop(question_id, None)
            raise RuntimeError("ask_user() timed out waiting for response (5 minutes)")
        except Exception as e:
            # Clean up
            pending_questions.pop(question_id, None)
            raise RuntimeError(f"ask_user() failed: {e}")
