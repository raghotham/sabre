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


def _cleanup_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """
    Clean up event loop resources: browser instances, pending tasks, async generators.

    Args:
        loop: Event loop to clean up
    """
    loop_id = id(loop)

    # 1. Clean up browser instance for this loop (if exists)
    try:
        from sabre.server.helpers.browser import BrowserHelper
        if loop_id in BrowserHelper._instances:
            logger.debug(f"Cleaning up browser for loop {loop_id}")
            browser = BrowserHelper._instances.pop(loop_id)
            loop.run_until_complete(asyncio.wait_for(browser._cleanup(), timeout=10.0))
    except asyncio.TimeoutError:
        logger.warning("Browser cleanup timed out")
    except Exception as e:
        logger.warning(f"Error cleaning up browser: {e}")

    # 2. Cancel all pending tasks
    pending = asyncio.all_tasks(loop)
    if pending:
        logger.debug(f"Cancelling {len(pending)} pending tasks")
        for task in pending:
            task.cancel()
        try:
            loop.run_until_complete(asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True),
                timeout=5.0
            ))
        except asyncio.TimeoutError:
            logger.warning("Task cancellation timed out")

    # 3. Shutdown async generators
    try:
        loop.run_until_complete(asyncio.wait_for(loop.shutdown_asyncgens(), timeout=5.0))
    except asyncio.TimeoutError:
        logger.warning("Async generator shutdown timed out")


def run_async_from_sync(coro: Coroutine[None, None, T], timeout: int = 300) -> T:
    """
    Run an async coroutine from synchronous code.

    Always creates a new event loop to avoid deadlocks when called from
    worker threads (e.g., via asyncio.to_thread). Using run_coroutine_threadsafe
    would cause a deadlock: parent loop waits for thread, thread waits for parent loop.

    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds

    Returns:
        Result of the coroutine
    """
    # Always create a new event loop to avoid deadlock with parent loop
    # This is safe and simple - worker threads should have their own loop
    logger.debug("run_async_from_sync: creating new event loop for coroutine execution")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Wrap with timeout to prevent infinite hangs
        async def run_with_timeout():
            return await asyncio.wait_for(coro, timeout=timeout)

        try:
            result = loop.run_until_complete(run_with_timeout())
        except asyncio.TimeoutError:
            logger.error(f"run_async_from_sync: operation timed out after {timeout}s")
            raise TimeoutError(f"Operation timed out after {timeout} seconds")

        # Clean up event loop resources
        _cleanup_event_loop(loop)

        return result
    finally:
        loop.close()
        # Clear the event loop for this thread
        asyncio.set_event_loop(None)


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
        # Get execution context (set by orchestrator during helper execution)
        from sabre.common.execution_context import get_execution_context

        ctx = get_execution_context()
        if ctx:
            event_callback = ctx.event_callback
            tree = ctx.tree
        else:
            # Fallback for direct calls outside orchestrator
            event_callback = None
            tree = None

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
        image_attachments = []  # Collect images for structured input

        for i, expr in enumerate(expr_list):
            if isinstance(expr, ImageContent):
                # Don't embed base64 in text - collect for structured attachment
                context_parts.append(f"### Context {i + 1} (Image)\n[Image {i + 1}]")
                image_attachments.append(expr)
            elif isinstance(expr, Content):
                # Use get_str() for Content objects
                context_parts.append(f"### Context {i + 1}\n{expr.get_str()}")
            else:
                # Plain data - convert to string
                context_parts.append(f"### Context {i + 1}\n{str(expr)}")

        context_text = "\n\n".join(context_parts)

        # Input is just the context
        input_text = context_text

        # Step 3: Use existing tree or create nested execution tree
        from sabre.common import ExecutionTree, ExecutionNodeType, ExecutionStatus

        # Use tree from context if available, otherwise create new one
        if tree is None:
            tree = ExecutionTree()

        tree.push(
            ExecutionNodeType.NESTED_LLM_CALL,
            metadata={
                "helper": "llm_call",
                "instructions_preview": instructions[:200],
                "context_count": len(expr_list),
            },
        )

        # Step 4: Create NEW orchestrator instance for nested call
        # (avoids interfering with parent orchestrator's state)
        try:
            from sabre.server.orchestrator import Orchestrator
            from sabre.server.python_runtime import PythonRuntime
            from sabre.common.executors.response import ResponseExecutor

            # Create new executor, runtime, and orchestrator for this nested call
            # ResponseExecutor reads OPENAI_API_KEY and other config from env
            executor = ResponseExecutor()
            runtime = PythonRuntime()
            orchestrator = Orchestrator(executor=executor, python_runtime=runtime, max_iterations=10)

            # Connect runtime to orchestrator (for recursive llm_call support)
            runtime.set_orchestrator(orchestrator)

            logger.info("Created new orchestrator instance for nested llm_call")

            # Upload images to Files API before passing to orchestrator
            uploaded_images = []
            if image_attachments:
                logger.info(f"Uploading {len(image_attachments)} images to Files API...")
                for img in image_attachments:
                    # Upload and replace with file_id reference
                    file_id = await orchestrator._upload_image_to_files_api(img)
                    uploaded_images.append(ImageContent(file_id=file_id, mime_type=img.mime_type))
                logger.info(f"Uploaded {len(uploaded_images)} images")

            # RECURSIVE CALL with new orchestrator instance
            # Pass images as structured input to executor
            result = await orchestrator.run(
                conversation_id=None,  # Create new conversation
                input_text=(input_text, uploaded_images) if uploaded_images else input_text,
                tree=tree,
                instructions=system_instructions,  # Pass our loaded instructions
                event_callback=event_callback,  # Pass through event callback for client visibility
            )

            tree.pop(ExecutionStatus.COMPLETED)

            if not result.success:
                logger.error(f"Nested LLM call failed: {result.error}")
                return f"ERROR: Nested LLM call failed: {result.error}"

            logger.info(f"llm_call completed: {len(result.final_response)} chars")
            return result.final_response

        except Exception as e:
            tree.pop(ExecutionStatus.ERROR)
            logger.error(f"Exception in nested LLM call: {e}")
            return f"ERROR: {type(e).__name__}: {str(e)}"
