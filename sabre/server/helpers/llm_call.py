"""
LLM Call helper - delegate tasks to LLM with fresh context.

Allows code to recursively call the LLM with new context.
"""

import asyncio
import logging
from concurrent.futures import Future
from typing import Callable, Coroutine, TypeVar

from sabre.common.utils.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_async_from_sync(coro: Coroutine[None, None, T], timeout: int = 300) -> T:
    """
    Run an async coroutine from synchronous code.

    Prefers scheduling on the orchestrator's existing event loop (captured
    via execution context) so helpers don't spin up ad-hoc loops.

    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds

    Returns:
        Result of the coroutine
    """
    # If we're somehow already on an event loop (shouldn't happen for sync helpers),
    # fail fast to avoid deadlocks.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass  # No running loop - this is expected
    else:
        msg = "run_async_from_sync() cannot be called from within a running event loop"
        logger.error(msg)
        raise RuntimeError(msg)

    from sabre.common.execution_context import get_execution_context

    ctx = get_execution_context()
    target_loop = ctx.loop if ctx else None

    if target_loop and target_loop.is_running():
        thread_future: Future[T] = Future()

        def _schedule():
            async def runner():
                try:
                    result = await asyncio.wait_for(coro, timeout=timeout)
                except Exception as exc:  # noqa: BLE001
                    thread_future.set_exception(exc)
                else:
                    thread_future.set_result(result)

            target_loop.create_task(runner())

        target_loop.call_soon_threadsafe(_schedule)
        return thread_future.result(timeout=timeout + 5)

    # Fallback: no orchestrator loop available (e.g., called in isolation)
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
        return result
    finally:
        asyncio.set_event_loop(None)
        loop.run_until_complete(loop.shutdown_asyncgens())
        pending = asyncio.all_tasks(loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
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
                # Make it clear to the model that an actual image is attached
                context_parts.append(
                    f"### Context {i + 1} (Image Attachment)\n"
                    f"An image has been attached to this message as part of the input. "
                    f"You can view and analyze this image directly - it is available for visual inspection."
                )
                image_attachments.append(expr)
                logger.info(
                    f"  📷 Collected ImageContent {i + 1}: "
                    f"has_file_id={expr.is_file_reference}, "
                    f"has_base64={expr.image_data is not None}, "
                    f"mime_type={expr.mime_type}"
                )
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

            # Get session logger from parent orchestrator if available
            parent_orchestrator = self.get_orchestrator()
            session_logger = parent_orchestrator.session_logger if parent_orchestrator else None

            # Create new executor, runtime, and orchestrator for this nested call
            # ResponseExecutor reads OPENAI_API_KEY and other config from env
            executor = ResponseExecutor()
            runtime = PythonRuntime()
            orchestrator = Orchestrator(
                executor=executor,
                python_runtime=runtime,
                session_logger=session_logger,  # Pass parent's session logger
                max_iterations=10,
            )

            # Connect runtime to orchestrator (for recursive llm_call support)
            runtime.set_orchestrator(orchestrator)

            logger.info("Created new orchestrator instance for nested llm_call")

            # Get session_id from context early for file saving
            session_id = ctx.session_id if ctx else "unknown"

            # Upload images to Files API before passing to orchestrator
            # Also save them to session files directory for persistence
            uploaded_images = []
            if image_attachments:
                logger.info(
                    f"📤 Processing {len(image_attachments)} images (saving to disk + uploading to Files API)..."
                )
                for idx, img in enumerate(image_attachments):
                    # Save to session files directory first
                    # Format: file_llmcall_{timestamp}_{idx}.png
                    import time
                    import base64
                    from sabre.common.paths import get_session_files_dir

                    if session_id and session_id != "unknown":
                        files_dir = get_session_files_dir(session_id)
                        files_dir.mkdir(parents=True, exist_ok=True)

                        # Create filename with timestamp to avoid collisions
                        timestamp = int(time.time() * 1000)  # milliseconds
                        filename = f"file_llmcall_{timestamp}_{idx + 1}.png"

                        # Save image to disk using parent orchestrator's method
                        parent_orchestrator._save_image_to_disk(img, session_id, "llmcall_attachment", filename)
                        file_path = files_dir / filename
                        logger.info(f"  💾 Saved image {idx + 1} to disk: {filename}")

                        # Log the file save to session.jsonl
                        if session_logger:
                            session_logger.log_file_saved(
                                session_id=session_id,
                                filename=filename,
                                file_path=str(file_path),
                                file_type="image",
                                context="llmcall_attachment",
                                metadata={
                                    "mime_type": img.mime_type,
                                    "size_bytes": len(base64.b64decode(img.image_data)) if img.image_data else 0,
                                    "attachment_index": idx + 1,
                                },
                            )

                    # Upload to Files API for token-efficient LLM input
                    logger.debug(f"  📤 Uploading image {idx + 1}/{len(image_attachments)} to Files API...")
                    file_id = await orchestrator._upload_image_to_files_api(img)
                    uploaded_images.append(ImageContent(file_id=file_id, mime_type=img.mime_type))
                    logger.info(f"  ✓ Uploaded image {idx + 1}: file_id={file_id}, mime_type={img.mime_type}")
                logger.info(f"✅ Successfully processed {len(uploaded_images)} images (saved + uploaded)")

            # RECURSIVE CALL with new orchestrator instance
            # Pass images as structured input to executor

            if uploaded_images:
                logger.info(
                    f"🔄 Calling orchestrator.run() with structured input: "
                    f"text ({len(input_text)} chars) + {len(uploaded_images)} image refs"
                )
            else:
                logger.info(f"🔄 Calling orchestrator.run() with text only ({len(input_text)} chars)")

            result = await orchestrator.run(
                conversation_id=None,  # Create new conversation
                input_text=(input_text, uploaded_images) if uploaded_images else input_text,
                tree=tree,
                session_id=session_id,  # Pass session_id from parent context
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
