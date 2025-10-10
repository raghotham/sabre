"""
LLM Call helper - delegate tasks to LLM with fresh context.

Allows code to recursively call the LLM with new context.
"""

import asyncio
import logging
from typing import Callable, Coroutine, TypeVar

from sabre.common.utils.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_async_from_sync(coro: Coroutine[None, None, T], timeout: int = 300) -> T:
    """
    Run an async coroutine from synchronous code.

    Handles the case where we're already inside an event loop (uses thread)
    or outside one (creates new loop).

    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds

    Returns:
        Result of the coroutine
    """
    try:
        loop = asyncio.get_running_loop()
        # We're inside an async context - use thread-safe execution
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout)
    except RuntimeError:
        # No running loop - create a new one
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


class LLMCall:
    """
    Call LLM with context (recursively calls orchestrator).

    Creates NEW conversation and recursively calls orchestrator.run().
    """

    def __init__(self, get_orchestrator: Callable, get_openai_client: Callable):
        """
        Initialize LLMCall helper.

        Args:
            get_orchestrator: Function to get orchestrator instance
            get_openai_client: Function to get/create OpenAI client
        """
        self.get_orchestrator = get_orchestrator
        self.get_openai_client = get_openai_client

    def __call__(self, expr_list: list, instructions: str) -> str:
        """
        Call LLM with context (sync wrapper).

        This bridges from synchronous exec() to async orchestrator.

        Examples:
            result = llm_call(["data.csv contents"], "Analyze this data")
            summary = llm_call([article_text], "Summarize in 3 sentences")

        Args:
            expr_list: Context data (e.g., ["data.csv contents", "analysis task"])
            instructions: What to ask the LLM to do

        Returns:
            LLM response as string
        """
        orchestrator = self.get_orchestrator()
        if not orchestrator:
            raise RuntimeError("Orchestrator not set - cannot use llm_call()")

        # Run async code in sync context (exec is synchronous)
        return run_async_from_sync(self.execute(expr_list, instructions))

    async def execute(self, expr_list: list, instructions: str) -> str:
        """
        Async implementation of llm_call.

        Flow:
        1. Create NEW conversation (for this nested call)
        2. Load llm_call.prompt template
        3. Format input with expr_list context and instructions
        4. Create nested execution tree
        5. RECURSIVELY call orchestrator.run() with new conversation
        6. Return the nested LLM's final response

        Args:
            expr_list: List of context items to provide to LLM
            instructions: Task instructions for the LLM

        Returns:
            LLM response text
        """
        # Step 1: Load llm_call.prompt
        prompt = PromptLoader.load(
            "llm_call.prompt",
            template={
                "llm_call_message": instructions,
            },
        )

        # Combine system_message and user_message as instructions
        system_instructions = f"{prompt['system_message']}\n\n{prompt['user_message']}"
        logger.info(f"llm_call: loaded instructions ({len(system_instructions)} chars)")

        # Step 2: Format input with context
        # Handle Content objects (TextContent, ImageContent, etc.)
        from sabre.common.models.messages import Content, ImageContent

        context_parts = []
        for i, expr in enumerate(expr_list):
            if isinstance(expr, ImageContent):
                # Format image as markdown so LLM can see it
                image_markdown = f"![Image {i + 1}](data:{expr.mime_type};base64,{expr.image_data})"
                context_parts.append(f"### Context {i + 1} (Image)\n{image_markdown}")
            elif isinstance(expr, Content):
                # Use get_str() for Content objects
                context_parts.append(f"### Context {i + 1}\n{expr.get_str()}")
            else:
                # Plain data - convert to string
                context_parts.append(f"### Context {i + 1}\n{str(expr)}")

        context_text = "\n\n".join(context_parts)

        # Input is just the context
        input_text = context_text

        # Step 3: Create nested execution tree
        from sabre.common import ExecutionTree, ExecutionNodeType, ExecutionStatus

        nested_tree = ExecutionTree()
        nested_tree.push(
            ExecutionNodeType.NESTED_LLM_CALL,
            metadata={"instructions_preview": instructions[:200], "context_count": len(expr_list)},
        )

        # Step 4: RECURSIVE CALL to orchestrator (will create conversation with instructions)
        try:
            orchestrator = self.get_orchestrator()
            result = await orchestrator.run(
                conversation_id=None,  # Create new conversation
                input_text=input_text,
                tree=nested_tree,
                instructions=system_instructions,  # Pass our loaded instructions
                event_callback=None,  # No client events for nested calls
            )

            nested_tree.pop(ExecutionStatus.COMPLETED)

            if not result.success:
                logger.error(f"Nested LLM call failed: {result.error}")
                return f"ERROR: Nested LLM call failed: {result.error}"

            logger.info(f"llm_call completed: {len(result.final_response)} chars")
            return result.final_response

        except Exception as e:
            nested_tree.pop(ExecutionStatus.ERROR)
            logger.error(f"Exception in nested LLM call: {e}")
            return f"ERROR: {type(e).__name__}: {str(e)}"
