"""
Orchestrator for continuation-passing style execution.

Architecture:
- Manages continuation loop (response → helpers → response)
- Streams tokens from ResponseExecutor and parses <helpers> in real-time
- Executes helpers (which may recursively call back via llm_call)
- Tracks execution tree state
- Emits events to client at each step

Organization:
1. Main orchestration loop
2. Streaming and parsing
3. Helper execution
4. Tree management
5. Event emission
6. Utility methods
"""

import asyncio
import os
import re
import logging
import time
import httpx
import openai
from typing import Callable, Awaitable, TYPE_CHECKING
from dataclasses import dataclass

from sabre.common.executors.response import ResponseExecutor
from sabre.server.python_runtime import PythonRuntime
from sabre.server.streaming_parser import StreamingHelperParser
from sabre.common import (
    Event,
    EventType,
    ResponseStartEvent,
    ResponseTextEvent,
    HelpersExecutionStartEvent,
    HelpersExecutionEndEvent,
    CompleteEvent,
    ErrorEvent,
    ExecutionTree,
    ExecutionNode,
    ExecutionNodeType,
    ExecutionStatus,
)

if TYPE_CHECKING:
    from sabre.common.models.messages import ImageContent

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationResult:
    """Result of orchestrating a conversation."""

    success: bool
    final_response: str
    conversation_id: str
    response_id: str
    error: str | None = None


@dataclass
class ParsedResponse:
    """Result of streaming and parsing a response."""

    full_text: str
    helpers: list[str]  # Extracted <helpers> blocks
    response_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


class Orchestrator:
    """
    Orchestrates LLM calls with Python helper execution.

    Implements continuation-passing style:
    - LLM emits text + <helpers>code</helpers> blocks
    - Execute helpers in Python runtime
    - Replace <helpers> with <helpers_result>
    - Continue until completion
    """

    def __init__(
        self,
        executor: ResponseExecutor,
        python_runtime: PythonRuntime,
        conversation_logger=None,
        max_iterations: int = 10,
    ):
        """
        Initialize orchestrator.

        Args:
            executor: ResponseExecutor for LLM calls
            python_runtime: Python runtime for executing helpers
            conversation_logger: Optional conversation logger for logging messages
            max_iterations: Max continuation iterations (prevent infinite loops)
        """
        self.executor = executor
        self.runtime = python_runtime
        self.conversation_logger = conversation_logger
        self.max_iterations = max_iterations
        self.system_instructions = None  # Stored after conversation creation
        self._shared_openai_client = None  # Lazy-initialized fallback client

        # CRITICAL: Wire runtime back to orchestrator for recursive llm_call()
        self.runtime.set_orchestrator(self)

    # ============================================================
    # MAIN ORCHESTRATION LOOP
    # ============================================================

    async def run(
        self,
        conversation_id: str | None,
        input_text: str,
        tree: ExecutionTree,
        instructions: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        event_callback: Callable[[Event], Awaitable[None]] | None = None,
    ) -> OrchestrationResult:
        """
        Run orchestration loop.

        Args:
            conversation_id: OpenAI conversation ID (None to create new)
            input_text: User input text
            tree: Execution tree for tracking
            instructions: System instructions for the conversation (required if conversation_id is None)
            model: Model to use (optional)
            max_tokens: Max output tokens
            temperature: Temperature
            event_callback: Event callback for streaming

        Returns:
            OrchestrationResult with final response
        """
        # Create conversation if needed
        if conversation_id is None:
            if not instructions:
                raise ValueError("instructions required when creating new conversation (conversation_id is None)")
            conversation_id = await self._create_conversation_with_instructions(instructions, model)
            logger.info(f"✨ Created new conversation ID: {conversation_id}")
            logger.info("📝 Conversation data stored on OpenAI servers (not locally)")

        iteration = 0
        current_response_id = None
        current_input = input_text
        full_response_text = ""

        logger.info(f"Starting orchestration for conversation {conversation_id}")

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Orchestration iteration {iteration}/{self.max_iterations}")

            # Create node for this LLM response iteration
            response_node = tree.push(
                ExecutionNodeType.RESPONSE_ROUND,
                metadata={
                    "iteration": iteration,
                    "model": model,
                    "conversation_id": conversation_id,
                },
            )

            # Get tree context for events
            tree_context = self._build_tree_context(tree, response_node, conversation_id)

            # Emit response_start event
            if event_callback:
                await event_callback(
                    ResponseStartEvent(
                        **tree_context,
                        model=model or "gpt-4o",
                        round_id=str(iteration),
                        prompt_tokens=0,  # Will be updated by executor
                        previous_response_id=current_response_id,
                    )
                )

            # Stream response and parse helpers in real-time
            parsed = await self._stream_and_parse_response(
                conversation_id=conversation_id,
                input_text=current_input,
                tree_context=tree_context,
                event_callback=event_callback,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Update state
            current_response_id = parsed.response_id
            full_response_text = parsed.full_text

            logger.debug(f"LLM response ({len(full_response_text)} chars): {full_response_text[:200]}...")
            logger.debug(f"Last 200 chars: ...{full_response_text[-200:]}")

            # Check if response is an error
            if full_response_text.startswith("ERROR:"):
                logger.error(f"API error in response: {full_response_text}")
                tree.pop(ExecutionStatus.ERROR)

                # Emit error event
                if event_callback:
                    await event_callback(
                        ErrorEvent(**tree_context, error_message=full_response_text, error_type="api_error")
                    )

                return OrchestrationResult(
                    success=False,
                    final_response=full_response_text,
                    conversation_id=conversation_id,
                    response_id=current_response_id,
                    error=full_response_text,
                )

            # Check if we're done
            if not parsed.helpers:
                # No helpers - orchestration complete
                logger.info("No helpers found, orchestration complete")
                tree.pop(ExecutionStatus.COMPLETED)

                # Just use the final response - it should include any necessary references
                # to results from previous iterations (the LLM saw them in <helpers_result>)
                final_message = full_response_text

                # Emit complete event
                await self._emit_complete_event(tree_context, final_message, event_callback)

                return OrchestrationResult(
                    success=True,
                    final_response=final_message,
                    conversation_id=conversation_id,
                    response_id=current_response_id,
                )

            # Execute helpers (may trigger recursive orchestrator calls)
            # This saves images to disk for client display
            execution_results = await self._execute_helpers(
                helpers=parsed.helpers, tree=tree, parent_tree_context=tree_context, event_callback=event_callback
            )

            # Upload images to Files API (converts base64 to file_id references)
            # This enables token-efficient continuation (~15k→1 token per image)
            # File IDs are appended to result text for LLM visibility
            execution_results = await self._ensure_images_uploaded(execution_results)

            # Build continuation input with results
            # - <helpers_result> contains text with file_id references (not localhost URLs)
            # - image_refs contains file_ids (for token-efficient continuation)

            # DEBUG: Log what we're passing to _replace_helpers_with_results
            logger.info(f"DEBUG: execution_results has {len(execution_results)} items")
            for idx, (output_text, content_items) in enumerate(execution_results):
                logger.info(f"DEBUG: Result {idx}: output_text={output_text[:200] if output_text else 'EMPTY'}")
                logger.info(f"DEBUG: Result {idx}: {len(content_items)} content items")

            response_with_results, image_refs = self._replace_helpers_with_results(
                full_response_text, execution_results
            )

            logger.info(f"DEBUG: response_with_results={response_with_results[:500]}")
            logger.debug(f"Continuing with results: {response_with_results[:200]}...")
            if image_refs:
                logger.info(f"Including {len(image_refs)} image file_id(s) in continuation")

            # Validate continuation input is not empty
            if not response_with_results or not response_with_results.strip():
                logger.error(f"Empty continuation input generated! response_with_results={repr(response_with_results)}")
                logger.error(f"execution_results had {len(execution_results)} items")
                for idx, (out, items) in enumerate(execution_results):
                    logger.error(f"  Result {idx}: output_text={repr(out[:200]) if out else 'EMPTY'}")
                # Use a fallback message instead of empty string
                response_with_results = "[Helper execution completed with empty result]"
                logger.warning(f"Using fallback continuation message: {response_with_results}")

            # Mark response node as completed (had helpers, continuing)
            tree.pop(ExecutionStatus.COMPLETED)

            # Set input for next iteration (text + image file_ids)
            current_input = (response_with_results, image_refs) if image_refs else response_with_results

        # Max iterations reached
        logger.warning(f"Max iterations ({self.max_iterations}) reached")

        # Note: The last response node was already popped with COMPLETED status
        # after helper execution, so no need to pop again

        return OrchestrationResult(
            success=False,
            final_response=full_response_text,
            conversation_id=conversation_id,
            response_id=current_response_id,
            error=f"Max iterations ({self.max_iterations}) reached",
        )

    # ============================================================
    # STREAMING AND PARSING
    # ============================================================

    async def _stream_and_parse_response(
        self,
        conversation_id: str,
        input_text: str | tuple[str, list],
        tree_context: dict,
        event_callback: Callable[[Event], Awaitable[None]] | None,
        **kwargs,
    ) -> ParsedResponse:
        """
        Stream response from executor and parse <helpers> blocks in real-time.

        This method:
        1. Creates a streaming parser
        2. Calls ResponseExecutor with a streaming token handler
        3. Feeds each token to the parser to detect <helpers> tags
        4. Forwards tokens to client via event callback
        5. Returns parsed response with extracted helpers

        Args:
            conversation_id: OpenAI conversation ID
            input_text: Input text OR (text, image_file_ids) tuple
            tree_context: Tree context for events
            event_callback: Callback for streaming events
            **kwargs: Additional parameters for executor (model, max_tokens, temperature)

        Returns:
            ParsedResponse with full text, helpers, and response_id
        """
        # Unpack input if structured (text + image file_ids)
        image_refs = []
        if isinstance(input_text, tuple):
            input_text, image_refs = input_text
            logger.info(f"📥 Unpacked structured input: {len(input_text)} chars text, {len(image_refs)} image refs")
            # Log details about each image ref
            for idx, img in enumerate(image_refs):
                from sabre.common.models.messages import ImageContent

                if isinstance(img, ImageContent):
                    logger.info(
                        f"  Image ref {idx + 1}: "
                        f"has_file_id={img.is_file_reference}, "
                        f"file_id={img.file_id if img.is_file_reference else 'N/A'}, "
                        f"mime_type={img.mime_type}"
                    )
                else:
                    logger.warning(f"  Image ref {idx + 1}: Unexpected type {type(img).__name__}")

        # Create streaming parser (state machine)
        parser = StreamingHelperParser()

        full_text = ""
        response_id = None

        # Define event handler for streaming tokens
        async def streaming_token_handler(event: Event):
            nonlocal full_text, response_id

            if event.type == EventType.RESPONSE_TOKEN:
                token = event.data["token"]
                full_text += token

                # Feed to parser (detects <helpers> tags)
                parser.feed(token)

                # Don't forward individual tokens to client - only show blocks

            elif event.type == EventType.RESPONSE_THINKING_TOKEN:
                # Don't forward thinking tokens either - only show blocks
                pass

            elif event.type == EventType.RESPONSE_RETRY:
                # Forward retry events to client
                if event_callback:
                    await event_callback(event)

        # Log user message to conversation file
        if self.conversation_logger and conversation_id:
            self.conversation_logger.log_message(conversation_id, "user", input_text)

        # Call executor (streams back via streaming_token_handler)
        # NOTE: No image attachments - images only go to client, not back to LLM
        # Pass instructions on every call (they don't persist in Responses API)
        if self.system_instructions:
            logger.info(f"Passing instructions to executor ({len(self.system_instructions)} chars)")
        else:
            logger.warning("NO INSTRUCTIONS - self.system_instructions is None!")

        response = await self.executor.execute(
            conversation_id=conversation_id,
            input_text=input_text,
            image_attachments=image_refs if image_refs else None,
            instructions=self.system_instructions,
            event_callback=streaming_token_handler,
            tree_context=tree_context,
            **kwargs,
        )

        response_id = response.response_id

        # If no tokens were streamed (e.g., API error), fallback to response content
        if not full_text:
            full_text = response.get_str()
            logger.warning(f"No streaming tokens received, using response content instead: {len(full_text)} chars")

        # Finalize parser
        parser.finalize()

        # Get parsed helpers from parser
        helpers = parser.get_helpers()

        logger.info(f"Parsed response: {len(full_text)} chars, {len(helpers)} helpers")
        logger.info(f"Response text: {full_text[:500]}{'...' if len(full_text) > 500 else ''}")

        # Log assistant response to conversation file
        # (image URLs are embedded in the response text)
        if self.conversation_logger and conversation_id:
            self.conversation_logger.log_message(conversation_id, "assistant", full_text)

        # Emit response_text event with preview (not full text to reduce SSE payload)
        # Send first 500 + last 500 chars for preview, full text only in complete event
        if event_callback:
            # Create preview: first 500 + last 500 chars
            if len(full_text) > 1000:
                text_preview = full_text[:500] + "\n\n[...truncated...]\n\n" + full_text[-500:]
            else:
                text_preview = full_text

            await event_callback(
                ResponseTextEvent(
                    **tree_context,
                    text=text_preview,
                    text_length=len(full_text),
                    has_helpers=len(helpers) > 0,
                    helper_count=len(helpers),
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    reasoning_tokens=response.reasoning_tokens,
                )
            )
            # Yield control to event loop so SSE generator can process queue
            await asyncio.sleep(0)

        # NOTE: We DON'T emit helpers_extracted events here
        # They'll be emitted in _execute_helpers() right before each execution
        # to maintain proper event ordering

        return ParsedResponse(
            full_text=full_text,
            helpers=helpers,
            response_id=response_id,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            reasoning_tokens=response.reasoning_tokens,
        )

    # ============================================================
    # HELPER EXECUTION
    # ============================================================

    async def _execute_helpers(
        self,
        helpers: list[str],
        tree: ExecutionTree,
        parent_tree_context: dict,
        event_callback: Callable[[Event], Awaitable[None]] | None,
    ) -> list[tuple[str, list]]:
        """
        Execute helper code blocks.

        Each helper may call llm_call(), which recursively calls
        orchestrator.run() with a NEW conversation.

        Images (matplotlib figures) are handled with dual operation:
        1. Saved to disk at XDG_DATA_HOME/sabre/files/{conversation_id}/ for client serving
        2. Markdown URLs sent to client via events (for display)
        3. ImageContent with base64 kept in content_list (for Files API upload)

        Args:
            helpers: List of code blocks to execute
            tree: Execution tree (modified in place)
            parent_tree_context: Tree context of parent node
            event_callback: Callback for events

        Returns:
            List of (output_text, content_list) tuples for each helper
            - output_text: Formatted output (without localhost URLs)
            - content_list: ImageContent objects (base64, later converted to file_id)
        """
        results = []

        for i, code in enumerate(helpers):
            logger.info(f"Executing helper {i + 1}/{len(helpers)}")

            # Push node for this helper execution
            helper_node = tree.push(
                ExecutionNodeType.HELPERS_EXECUTION, metadata={"block_number": i + 1, "code_preview": code[:200]}
            )

            # Get tree context for this helper (inherit parent's conversation_id)
            parent_conv_id = parent_tree_context.get("conversation_id", "")
            helper_tree_context = self._build_tree_context(tree, helper_node, parent_conv_id)

            # Emit start event
            if event_callback:
                await event_callback(HelpersExecutionStartEvent(**helper_tree_context, code=code, block_number=i + 1))
                # Yield control to event loop so SSE generator can process queue
                await asyncio.sleep(0)

            # Execute code (may trigger recursive orchestrator.run via llm_call)
            # Run in thread pool to avoid blocking event loop (which prevents SSE from streaming)
            # Set execution context for helpers (like coerce) to access
            from sabre.common.execution_context import set_execution_context, clear_execution_context

            current_loop = asyncio.get_running_loop()
            set_execution_context(
                event_callback=event_callback,
                tree=tree,
                tree_context=helper_tree_context,
                conversation_id=helper_tree_context["conversation_id"],
                loop=current_loop,
            )

            try:
                start_time = time.time()
                # Run synchronous exec() in thread pool to avoid blocking event loop
                result = await asyncio.to_thread(self.runtime.execute, code)
                duration_ms = (time.time() - start_time) * 1000
            finally:
                # Clear execution context (optional - contextvars auto-scopes to task)
                clear_execution_context()

            # Process result
            if result.success:
                # Save images to disk and replace with URLs
                from sabre.common.models.messages import ImageContent, TextContent

                saved_content = []
                image_urls = []

                # Calculate user message number by counting existing figure files
                from sabre.common.paths import get_files_dir
                import glob

                files_dir = get_files_dir(helper_tree_context["conversation_id"])
                files_dir.mkdir(parents=True, exist_ok=True)

                # Count existing figure files to get next message number
                existing_figures = glob.glob(str(files_dir / "figure_*.png"))
                # Extract message numbers from filenames (figure_1_1_1.png -> 1)
                message_nums = set()
                for fig_path in existing_figures:
                    # Use basename only to avoid splitting on directory path separators
                    from pathlib import Path

                    filename = Path(fig_path).name
                    parts = filename.split("_")
                    if len(parts) >= 2 and parts[0] == "figure":
                        try:
                            message_nums.add(int(parts[1]))
                        except ValueError:
                            pass
                message_num = (max(message_nums) + 1) if message_nums else 1

                for item in result.content:
                    if isinstance(item, ImageContent) and item.image_data:
                        # Save image to disk and get URL
                        # Use message_num for unique filenames: figure_{message_num}_{helper_num}_{image_num}.png
                        filename = f"figure_{message_num}_{i + 1}_{len(image_urls) + 1}.png"
                        url = self._save_image_to_disk(item, helper_tree_context["conversation_id"], filename)
                        image_urls.append(url)
                        # Keep original base64 ImageContent for Files API upload
                        saved_content.append(item)
                    else:
                        saved_content.append(item)

                # Format result using result prompts (adds context wrapper)
                has_images = len(image_urls) > 0
                formatted_output, prompt_name_used = self._format_result(result.output, has_images)

                # Output text for LLM - include image URLs so LLM can reference them
                output_text_for_llm = formatted_output
                if image_urls:
                    # Append image URLs as markdown so LLM can reference them in responses
                    output_text_for_llm += "\n\n"
                    for idx, url in enumerate(image_urls, 1):
                        output_text_for_llm += f"![Figure {idx}]({url})\n"

                # If result is very large (>10KB), save to file and reference by file_id
                MAX_INLINE_CHARS = 10000  # 10KB threshold
                if len(output_text_for_llm) > MAX_INLINE_CHARS:
                    logger.info(f"Result is large ({len(output_text_for_llm)} chars), saving to file for LLM reference")
                    # Save to file and upload to Files API
                    file_id = await self._save_large_result_to_file(
                        output_text_for_llm, helper_tree_context["conversation_id"], i + 1
                    )
                    # Replace with file reference for LLM (show preview + file_id)
                    output_text_for_llm = (
                        f"[Large result ({len(output_text_for_llm)} chars) uploaded to file_id: {file_id}]\n\n"
                        f"First 500 chars:\n{output_text_for_llm[:500]}\n\n"
                        f"[...truncated...]\n\n"
                        f"Last 500 chars:\n{output_text_for_llm[-500:]}\n\n"
                        f"The full content is available in the attached file."
                    )

                # Store text for LLM (without localhost URLs), keep original content for events
                results.append((output_text_for_llm, saved_content))
                tree.pop(ExecutionStatus.COMPLETED)
                logger.info(
                    f"Helper {i + 1} succeeded: {len(result.output)} chars, {len(result.content)} content items, {len(image_urls)} images saved"
                )

                # Emit execution end event with results for client display
                if event_callback:
                    # Build display content: TextContent with output + ImageContent with URLs (not base64)
                    from sabre.common.models.messages import TextContent, ImageContent

                    # Send preview of formatted output (first 500 + last 500 chars)
                    if len(formatted_output) > 1000:
                        output_preview = formatted_output[:500] + "\n\n[...truncated...]\n\n" + formatted_output[-500:]
                    else:
                        output_preview = formatted_output

                    display_content = [TextContent(output_preview)]

                    # Add image URLs as ImageContent (URL in image_data field, not base64)
                    # Client will detect http:// URLs and render appropriately (wezterm imgcat or show URL)
                    for url in image_urls:
                        display_content.append(ImageContent(image_data=url))

                    await event_callback(
                        HelpersExecutionEndEvent(
                            **helper_tree_context,
                            duration_ms=duration_ms,
                            success=True,
                            result=display_content,
                            block_number=i + 1,
                        )
                    )
                    # Yield control to event loop so SSE generator can process queue
                    await asyncio.sleep(0)
            else:
                # On error, just text (no content)
                error_text = f"ERROR: {result.error}"
                results.append((error_text, []))
                tree.pop(ExecutionStatus.ERROR)
                logger.error(f"Helper {i + 1} failed: {result.error}")

                # Emit execution end event with error
                if event_callback:
                    from sabre.common.models.messages import TextContent

                    await event_callback(
                        HelpersExecutionEndEvent(
                            **helper_tree_context,
                            duration_ms=duration_ms,
                            success=False,
                            result=[TextContent(error_text)],
                            block_number=i + 1,
                        )
                    )

        return results

    def _format_result(self, raw_output: str, has_images: bool) -> tuple[str, str | None]:
        """
        Format helper result using appropriate *_result.prompt.

        Images skip formatting and go directly as URLs.
        Text results are formatted through result prompts for better context.

        Args:
            raw_output: Raw helper output text
            has_images: Whether result contains images

        Returns:
            Tuple of (formatted_output, prompt_name_used)
            - formatted_output: Formatted output text with context wrapper
            - prompt_name_used: Name of prompt used for formatting, or None if skipped
        """
        # Skip formatting for images - they go directly as markdown URLs
        if has_images:
            logger.debug("Skipping result formatting for image output")
            return raw_output, None

        # Skip formatting for very short outputs (likely not useful to wrap)
        if len(raw_output.strip()) < 10:
            return raw_output, None

        # Determine result type based on content
        from sabre.common.utils.prompt_loader import PromptLoader

        stripped = raw_output.strip()

        # Simple heuristic: check if output looks like a list
        if stripped.startswith("[") and stripped.endswith("]"):
            prompt_name = "list_result.prompt"
            template_key = "list_result"
            logger.debug("Using list_result.prompt for formatting")
        else:
            # Default to str_result for all other outputs
            prompt_name = "str_result.prompt"
            template_key = "str_result"
            logger.debug("Using str_result.prompt for formatting")

        try:
            # Load result prompt and template the raw output
            prompt = PromptLoader.load(prompt_name, template={template_key: raw_output})

            # Return just the user_message part
            formatted = prompt["user_message"]
            logger.info(f"Result formatted: {len(raw_output)} → {len(formatted)} chars")
            return formatted, prompt_name

        except Exception as e:
            logger.warning(f"Failed to format result with {prompt_name}: {e}")
            # Fallback to raw output if formatting fails
            return raw_output, None

    # ============================================================
    # TREE MANAGEMENT
    # ============================================================

    def _build_tree_context(self, tree: ExecutionTree, node: ExecutionNode, conversation_id: str = "") -> dict:
        """
        Build tree context dict for events.

        Args:
            tree: Execution tree
            node: Current node
            conversation_id: OpenAI conversation ID

        Returns:
            Dict with node_id, parent_id, depth, path, conversation_id
        """
        # Build human-readable path
        path_nodes = tree.get_path()
        path_summary = []
        for n in path_nodes:
            if n.node_type.value == "client_request":
                msg = n.metadata.get("message", "")
                path_summary.append(f"User: {msg[:30]}..." if len(msg) > 30 else f"User: {msg}")
            elif n.node_type.value == "response_round":
                iteration = n.metadata.get("iteration", "?")
                path_summary.append(f"Response #{iteration}")
            elif n.node_type.value == "helpers_execution":
                block_num = n.metadata.get("block_number", "?")
                path_summary.append(f"Helper #{block_num}")
            elif n.node_type.value == "nested_llm_call":
                helper = n.metadata.get("helper", "llm_call")
                path_summary.append(f"{helper}()")
            else:
                path_summary.append(n.node_type.value)

        path_summary_str = " → ".join(path_summary)
        logger.info(f"Execution path: {path_summary_str}")

        return {
            "node_id": node.id,
            "parent_id": node.parent_id,
            "depth": tree.get_depth(),
            "path": [n.id for n in tree.get_path()],
            "path_summary": path_summary_str,  # Human-readable path for client display
            "conversation_id": conversation_id,
        }

    # ============================================================
    # EVENT EMISSION
    # ============================================================

    async def _emit_complete_event(
        self, tree_context: dict, final_message: str, callback: Callable[[Event], Awaitable[None]] | None
    ):
        """
        Emit completion event.

        Args:
            tree_context: Tree context dict
            final_message: Final response text
            callback: Event callback
        """
        if callback:
            await callback(CompleteEvent(**tree_context, final_message=final_message))

    def _get_openai_client(self):
        """
        Retrieve a shared AsyncOpenAI client configured like the executor.

        Prefers the executor's client (which already respects api key/base URL),
        falling back to a lazily-created instance if necessary.
        """
        client = getattr(self.executor, "client", None)
        if client is not None:
            return client

        if self._shared_openai_client is None:
            from openai import AsyncOpenAI

            skip_ssl = os.getenv("OPENAI_SKIP_SSL_VERIFY", "").lower() in ("true", "1", "yes")

            # Create httpx client with SSL settings
            http_client = None
            if skip_ssl:
                logger.warning("⚠️  SSL certificate verification is DISABLED - use only for testing!")
                http_client = httpx.AsyncClient(verify=False)

            client_kwargs = {"api_key": self.executor.api_key}
            if getattr(self.executor, "base_url", None):
                client_kwargs["base_url"] = self.executor.base_url
            if http_client:
                client_kwargs["http_client"] = http_client

            self._shared_openai_client = AsyncOpenAI(**client_kwargs)

        return self._shared_openai_client

    # ============================================================
    # CONVERSATION MANAGEMENT
    # ============================================================

    def load_default_instructions(self) -> str:
        """
        Load default system instructions for main conversation.

        Returns:
            Combined system_message + user_message from python_continuation_execution_responses.prompt
        """
        from sabre.common.utils.prompt_loader import PromptLoader

        prompt_name = "python_continuation_execution_responses.prompt"

        # Get available functions dynamically from runtime
        available_functions = self.runtime.get_available_functions()
        logger.info(f"Loading default instructions with {len(available_functions)} chars of functions")

        prompt_parts = PromptLoader.load(
            prompt_name,
            template={
                "context_window_tokens": "128000",
                "context_window_words": "96000",
                "context_window_bytes": "512000",
                "scratchpad_token": "scratchpad",
                "functions": available_functions,
                "user_colon_token": "User:",
                "assistant_colon_token": "Assistant:",
            },
        )

        # Combine system_message and user_message
        instructions = f"{prompt_parts['system_message']}\n\n{prompt_parts['user_message']}"
        logger.info(f"Loaded default instructions from '{prompt_name}' ({len(instructions)} chars)")

        return instructions

    async def _create_conversation_with_instructions(self, instructions: str, model: str | None = None) -> str:
        """
        Create a new conversation with given instructions.

        Args:
            instructions: System instructions for the conversation
            model: Model to use (optional)

        Returns:
            Conversation ID
        """
        client = self._get_openai_client()

        # Create conversation
        conversation = await client.conversations.create(metadata={"session_type": "orchestrator_managed"})

        # Store instructions for all future calls
        self.system_instructions = instructions

        logger.info(f"Creating conversation with instructions ({len(instructions)} chars)")
        logger.info(f"Instructions first 3000 chars:\n{instructions[:3000]}")
        logger.info(f"Instructions last 1000 chars:\n...{instructions[-1000:]}")

        # Send initial message with instructions
        # NOTE: Instructions must be sent on EVERY response.create() call
        try:
            await client.responses.create(
                model=model or self.executor.default_model,
                conversation=conversation.id,
                instructions=instructions,  # System instructions
                input="Are you ready?",
                max_output_tokens=100,  # Just need a short acknowledgment
                stream=False,
                truncation="auto",  # Auto-truncate if context exceeds limit
            )
        except openai.RateLimitError as e:
            # Check if this is insufficient quota (not retriable) or rate limit (retriable)
            error_body = getattr(e, "body", {})
            error_code = error_body.get("error", {}).get("code")

            if error_code == "insufficient_quota":
                logger.error(f"Insufficient API quota: {e}")
                raise RuntimeError(
                    "OpenAI API quota exceeded. Please check your billing details at "
                    "https://platform.openai.com/account/billing"
                ) from e
            else:
                # Regular rate limit - re-raise to be handled by caller
                logger.error(f"Rate limit during conversation creation: {e}")
                raise

        logger.info(f"Conversation {conversation.id} created successfully")

        return conversation.id

    # ============================================================
    # FILE MANAGEMENT
    # ============================================================

    def _save_image_to_disk(self, image_content: "ImageContent", conversation_id: str, filename: str) -> str:
        """
        Save image to disk and return URL.

        Args:
            image_content: ImageContent with base64 data
            conversation_id: Conversation ID for directory organization
            filename: Filename to save as (e.g., "graph.png")

        Returns:
            URL path to access the file
        """
        from sabre.common.paths import get_files_dir
        import base64
        import os

        # Create directory: XDG_DATA_HOME/sabre/files/{conversation_id}/
        files_dir = get_files_dir(conversation_id)
        files_dir.mkdir(parents=True, exist_ok=True)

        # Decode and save image
        image_bytes = base64.b64decode(image_content.image_data)
        file_path = files_dir / filename
        file_path.write_bytes(image_bytes)

        # Generate URL (assumes server on localhost:8011 or PORT)
        port = os.getenv("PORT", "8011")
        url = f"http://localhost:{port}/v1/files/{conversation_id}/{filename}"
        logger.info(f"💾 Saved image to: {file_path}")
        logger.debug(f"   Accessible at: {url}")

        return url

    async def _upload_image_to_files_api(self, image_content: "ImageContent") -> str:
        """
        Upload image to OpenAI Files API and return file_id.

        This allows referencing images by ID instead of sending base64 repeatedly,
        saving ~15k tokens per image per message.

        Args:
            image_content: ImageContent with base64 data

        Returns:
            file_id string (e.g., "file-abc123...")

        Raises:
            Exception if upload fails
        """
        import base64
        import io

        client = self._get_openai_client()

        try:
            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_content.image_data)

            # Create file-like object
            file_obj = io.BytesIO(image_bytes)
            file_obj.name = f"image.{image_content.mime_type.split('/')[-1]}"  # e.g., "image.png"

            # Upload to Files API with 1-day expiration
            logger.info(f"Uploading image to OpenAI Files API ({len(image_bytes)} bytes)")
            file_response = await client.files.create(
                file=file_obj,
                purpose="assistants",  # Required purpose for file uploads
                expires_after={
                    "anchor": "created_at",  # Expire relative to creation time
                    "seconds": 86400,  # Delete after 1 day (86400 seconds)
                },
            )

            logger.info(f"Image uploaded successfully: {file_response.id}")
            return file_response.id

        except Exception as e:
            logger.error(f"Failed to upload image to Files API: {e}")
            raise

    async def _save_large_result_to_file(self, content: str, conversation_id: str, helper_num: int) -> str:
        """
        Save large helper result to file and upload to Files API.

        For results >10KB, saves to disk and uploads to OpenAI Files API,
        returning a file_id that can be referenced in the conversation.

        Args:
            content: Large text content to save
            conversation_id: Conversation ID for directory organization
            helper_num: Helper number for unique filename

        Returns:
            file_id string (e.g., "file-abc123...")

        Raises:
            Exception if upload fails
        """
        from sabre.common.paths import get_files_dir
        import io

        # Save to disk first
        files_dir = get_files_dir(conversation_id)
        files_dir.mkdir(parents=True, exist_ok=True)

        filename = f"helper_{helper_num}_result.txt"
        file_path = files_dir / filename
        file_path.write_text(content, encoding="utf-8")
        logger.info(f"💾 Saved large result to: {file_path}")

        # Upload to Files API
        client = self._get_openai_client()

        try:
            # Create file-like object from content
            file_obj = io.BytesIO(content.encode("utf-8"))
            file_obj.name = filename

            logger.info(f"Uploading large result to OpenAI Files API ({len(content)} chars)")
            file_response = await client.files.create(
                file=file_obj,
                purpose="assistants",  # Required purpose for file uploads
                expires_after={
                    "anchor": "created_at",  # Expire relative to creation time
                    "seconds": 86400,  # Delete after 1 day (86400 seconds)
                },
            )

            logger.info(f"Large result uploaded successfully: {file_response.id}")
            return file_response.id

        except Exception as e:
            logger.error(f"Failed to upload large result to Files API: {e}")
            raise

    async def _ensure_images_uploaded(self, results: list[tuple[str, list]]) -> list[tuple[str, list]]:
        """
        Ensure all images in results are uploaded to Files API.

        Converts ImageContent with base64 data to ImageContent with file_id.
        Updates output text to mention file_id for LLM visibility.

        Args:
            results: List of (output_text, content_items) tuples

        Returns:
            Same structure but with file_id references instead of base64
            and updated output_text mentioning the file_ids
        """
        from sabre.common.models.messages import ImageContent

        updated_results = []

        for output_text, content_items in results:
            updated_content = []
            file_id_mentions = []  # Track file_ids to append to output

            for item in content_items:
                if isinstance(item, ImageContent) and not item.is_file_reference:
                    # Upload base64 image and replace with file_id reference
                    # This is CRITICAL - we must upload successfully or fail the operation
                    file_id = await self._upload_image_to_files_api(item)
                    # Create new ImageContent with file_id instead of base64
                    updated_content.append(ImageContent(file_id=file_id, mime_type=item.mime_type))
                    file_id_mentions.append(file_id)
                    logger.info(f"Replaced base64 image with file reference: {file_id}")
                else:
                    updated_content.append(item)

            # Update output text to mention file_ids for LLM visibility
            if file_id_mentions:
                # Append file_id information to output text
                file_id_info = "\n\n" + "\n".join(
                    f"[Image {idx + 1} uploaded to file_id: {file_id}]" for idx, file_id in enumerate(file_id_mentions)
                )
                output_text += file_id_info
                logger.info(f"Added {len(file_id_mentions)} file_id references to output text")

            updated_results.append((output_text, updated_content))

        return updated_results

    # ============================================================
    # UTILITY METHODS
    # ============================================================

    def _replace_helpers_with_results(self, text: str, results: list[tuple[str, list]]) -> tuple[str, list]:
        """
        Replace <helpers> blocks with <helpers_result> blocks and extract images.

        Images are uploaded to Files API (if not already) and returned as file_id references.
        This allows LLM to see images when needed while saving tokens (~15k→1 per image).

        Args:
            text: Original response with <helpers> blocks
            results: List of (output_text, content_items) tuples

        Returns:
            Tuple of (text_with_results, image_refs)
            - text_with_results: Text with <helpers> replaced by <helpers_result>
            - image_refs: List of ImageContent with file_id for continuation
        """
        from sabre.common.models.messages import ImageContent

        pattern = r"<helpers>.*?</helpers>"
        result_index = 0
        all_images = []

        # Max chars per result to avoid context overflow
        MAX_RESULT_CHARS = 20000

        def replacer(match):
            nonlocal result_index
            logger.info(f"DEBUG: replacer called for match {result_index}, matched text: {match.group(0)[:100]}")
            if result_index < len(results):
                output_text, content_items = results[result_index]
                logger.info(
                    f"DEBUG: replacer using result {result_index}: output_text={output_text[:200] if output_text else 'EMPTY'}"
                )

                # Collect image file_ids for continuation
                for item in content_items:
                    if isinstance(item, ImageContent) and item.is_file_reference:
                        all_images.append(item)

                # Truncate very large results
                if len(output_text) > MAX_RESULT_CHARS:
                    output_text = (
                        output_text[:MAX_RESULT_CHARS]
                        + f"\n\n[...truncated {len(output_text) - MAX_RESULT_CHARS} chars]"
                    )
                    logger.warning(
                        f"Truncated helper result {result_index + 1} from {len(results[result_index][0])} to {MAX_RESULT_CHARS} chars"
                    )

                replacement = f"<helpers_result>{output_text}</helpers_result>"
                logger.info(f"DEBUG: replacement={replacement[:200]}")
                result_index += 1
                return replacement
            logger.warning(f"DEBUG: replacer called but result_index {result_index} >= len(results) {len(results)}")
            return match.group(0)

        logger.info(f"DEBUG: _replace_helpers_with_results: text length={len(text)}, results count={len(results)}")
        logger.info(f"DEBUG: _replace_helpers_with_results: text preview={text[:300]}")
        text_with_results = re.sub(pattern, replacer, text, flags=re.DOTALL)
        logger.info(f"DEBUG: _replace_helpers_with_results: after replacement={text_with_results[:500]}")
        return text_with_results, all_images
