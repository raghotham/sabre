"""
Response Executor for OpenAI Responses API.

This is the ONLY executor in sabre. It's clean, simple, and focused.

Key features:
- Streaming support
- response_id continuation
- Event emission for every token
- Clean async API
- Automatic retry with exponential backoff for rate limits
"""

import os
import re
import asyncio
from typing import Callable, Awaitable
from openai import AsyncOpenAI
import openai
import logging

from sabre.common.models import (
    Message,
    Assistant,
    TextContent,
    ResponseTokenEvent,
    ResponseThinkingTokenEvent,
    ResponseRetryEvent,
    Event,
)

logger = logging.getLogger(__name__)


class ResponseExecutor:
    """
    OpenAI Responses API executor.

    Executes LLM calls using OpenAI's stateful Responses API.
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None, default_model: str = "gpt-4o"):
        """
        Initialize executor.

        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            base_url: OpenAI API base URL (if None, reads from OPENAI_BASE_URL env var)
            default_model: Default model to use (if None, reads from OPENAI_MODEL env var)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required (set OPENAI_API_KEY env var)")

        # Get base URL from env if not provided
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

        # Get default model from env if not provided
        self.default_model = os.getenv("OPENAI_MODEL") or default_model

        # Create client with optional base_url
        if self.base_url:
            logger.info(f"Using custom OpenAI base URL: {self.base_url}")
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = AsyncOpenAI(api_key=self.api_key)

        logger.info(f"Initialized ResponseExecutor with model: {self.default_model}")

    def _message_to_openai(self, message: Message) -> dict:
        """
        Convert our Message to OpenAI format.

        Args:
            message: Our Message object

        Returns:
            OpenAI message dict
        """
        # Get text content
        text = message.get_str()

        return {
            "role": message.role,
            "content": text,
        }

    async def count_tokens(self, messages: list[Message]) -> int:
        """
        Count tokens in messages.

        This is approximate - just counts characters / 4.
        For real token counting, we'd use tiktoken.

        Args:
            messages: List of messages

        Returns:
            Approximate token count
        """
        total_chars = sum(len(m.get_str()) for m in messages)
        return total_chars // 4

    async def execute(
        self,
        conversation_id: str,
        input_text: str,
        image_attachments: list | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        instructions: str | None = None,
        event_callback: Callable[[Event], Awaitable[None]] | None = None,
        tree_context: dict | None = None,
        max_retries: int = 5,
    ) -> Assistant:
        """
        Execute LLM call with streaming using Conversations API.

        Args:
            conversation_id: Conversation ID to use
            input_text: Current turn's input text
            image_attachments: Optional list of ImageContent objects to attach
            model: Model to use (defaults to self.default_model)
            max_tokens: Max completion tokens
            temperature: Sampling temperature
            event_callback: Async callback for streaming events
            tree_context: Execution tree context (node_id, parent_id, depth, path)
            max_retries: Maximum number of retry attempts for rate limits (default 5)

        Returns:
            Assistant message with response

        Emits events:
            - ResponseTokenEvent for each token
            - ResponseThinkingTokenEvent for thinking tokens

        Raises:
            openai.APIError: If non-rate-limit error or max retries exceeded
        """
        model = model or self.default_model

        # Build input - either simple text or structured content with images
        if image_attachments:
            # Structured input: text + images
            # Responses API requires wrapping in "message" type
            from sabre.common.models.messages import ImageContent

            message_content = [{"type": "text", "text": input_text}]

            for img in image_attachments:
                if isinstance(img, ImageContent):
                    if img.is_file_reference:
                        # Format as file reference (minimal token usage)
                        message_content.append({"type": "file", "file_id": img.file_id})
                    else:
                        # Format as base64 image_url
                        message_content.append(
                            {"type": "image_url", "image_url": {"url": f"data:{img.mime_type};base64,{img.image_data}"}}
                        )

            # Wrap in message type for Responses API
            api_input = [{"type": "message", "content": message_content}]

            # Count file refs vs base64 for logging
            file_refs = sum(1 for img in image_attachments if isinstance(img, ImageContent) and img.is_file_reference)
            base64_imgs = len(image_attachments) - file_refs
            logger.info(f"Sending structured input: 1 text + {file_refs} file_id refs + {base64_imgs} base64 images")
        else:
            # Simple text input
            api_input = input_text

        # Build request params for Responses API
        # Always use conversation_id - each call adds a new turn
        params = {
            "model": model,
            "conversation": conversation_id,
            "input": api_input,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "truncation": "auto",  # Automatically drop middle items when context exceeded
        }

        # Add instructions if provided (must be sent on every call, not persisted)
        if instructions:
            params["instructions"] = instructions

        # Check conversation state before making the call
        try:
            from openai import AsyncOpenAI

            temp_client = (
                AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
                if self.base_url
                else AsyncOpenAI(api_key=self.api_key)
            )
            conv = await temp_client.conversations.retrieve(conversation_id)
            num_turns = len(conv.turns) if hasattr(conv, "turns") else "unknown"
            logger.info(f"Conversation {conversation_id} has {num_turns} turns before this call")
        except Exception as e:
            logger.warning(f"Could not retrieve conversation state: {e}")

        logger.info(f"Calling Responses API: conversation={conversation_id}, input={input_text[:50]}..., model={model}")
        logger.info(f"Full input being sent: {input_text}")

        # Retry loop for rate limit handling
        for attempt in range(max_retries):
            # Reset state on each retry attempt
            response_text = ""
            thinking_text = ""
            response_id = None
            input_tokens = 0
            output_tokens = 0
            reasoning_tokens = 0

            try:
                # Create streaming response
                stream = await self.client.responses.create(**params)

                # Process stream (Responses API streaming format)
                async for event in stream:
                    # Get response ID
                    if event.type == "response.created":
                        response_id = event.response.id
                        logger.info(f"Got response_id: {response_id}")

                    # Handle text delta events
                    elif event.type == "response.output_text.delta":
                        token = event.delta
                        response_text += token

                        # Emit token event
                        if event_callback and tree_context:
                            await event_callback(
                                ResponseTokenEvent(
                                    node_id=tree_context["node_id"],
                                    parent_id=tree_context.get("parent_id"),
                                    depth=tree_context["depth"],
                                    path=tree_context["path"],
                                    conversation_id=tree_context.get("conversation_id", conversation_id),
                                    token=token,
                                )
                            )

                    # Handle reasoning/thinking tokens
                    elif event.type == "response.output_reasoning.delta":
                        token = event.delta
                        thinking_text += token

                        # Emit thinking token event
                        if event_callback and tree_context:
                            await event_callback(
                                ResponseThinkingTokenEvent(
                                    node_id=tree_context["node_id"],
                                    parent_id=tree_context.get("parent_id"),
                                    depth=tree_context["depth"],
                                    path=tree_context["path"],
                                    conversation_id=tree_context.get("conversation_id", conversation_id),
                                    token=token,
                                )
                            )

                    elif event.type == "response.completed":
                        # Extract usage stats
                        usage = event.response.usage
                        input_tokens = usage.input_tokens
                        output_tokens = usage.output_tokens
                        reasoning_tokens = usage.output_tokens_details.reasoning_tokens

                        logger.info(
                            f"Response done: input={input_tokens}, output={output_tokens}, reasoning={reasoning_tokens}"
                        )

                # Success - break out of retry loop
                break

            except openai.RateLimitError as e:
                # Extract detailed rate limit headers
                h = getattr(e, "headers", {}) or {}
                req_reset = h.get("x-ratelimit-reset-requests")
                tok_reset = h.get("x-ratelimit-reset-tokens")
                retry_after = h.get("retry-after")

                logger.warning(
                    f"Rate limit error: "
                    f"x-ratelimit-reset-requests={req_reset}, "
                    f"x-ratelimit-reset-tokens={tok_reset}, "
                    f"retry-after={retry_after}"
                )

                # Don't retry on last attempt
                if attempt >= max_retries - 1:
                    logger.error(f"Rate limit error after {max_retries} attempts: {e}")
                    # Return empty response instead of raising
                    return Assistant(
                        content=[
                            TextContent(
                                f"ERROR: Rate limit exceeded after {max_retries} retries. Please try again later."
                            )
                        ],
                        response_id=None,
                    )

                # Parse retry time from error message or use headers
                error_msg = str(e)
                wait_time = None

                # Try retry-after header first
                if retry_after:
                    try:
                        wait_time = float(retry_after) + 1.0  # Add buffer
                    except (ValueError, TypeError):
                        pass

                # Fall back to parsing error message
                if wait_time is None:
                    match = re.search(r"try again in ([\d.]+)s", error_msg)
                    if match:
                        wait_time = float(match.group(1)) + 1.0  # Add buffer
                    else:
                        # Exponential backoff if we can't parse retry time
                        wait_time = (2**attempt) * 2  # 2, 4, 8, 16, 32 seconds

                logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{max_retries}), waiting {wait_time:.1f}s before retry"
                )

                # Send retry event to client (even if streaming started)
                if event_callback and tree_context:
                    await event_callback(
                        ResponseRetryEvent(
                            node_id=tree_context["node_id"],
                            parent_id=tree_context.get("parent_id"),
                            depth=tree_context["depth"],
                            path=tree_context["path"],
                            conversation_id=tree_context.get("conversation_id", conversation_id),
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            wait_seconds=wait_time,
                            reason=error_msg,
                        )
                    )

                # Sleep before retrying
                await asyncio.sleep(wait_time)
                continue

            except openai.APIError as e:
                # Non-rate-limit API error - return error message instead of raising
                error_msg = str(e)
                logger.error(f"API error: {e}")
                return Assistant(
                    content=[TextContent(f"ERROR: {error_msg}")],
                    response_id=None,
                )

            except Exception as e:
                # Unexpected error - return error message instead of raising
                logger.error(f"Unexpected error during streaming: {e}", exc_info=True)
                return Assistant(
                    content=[TextContent(f"ERROR: Unexpected error: {str(e)}")],
                    response_id=None,
                )

        # Create Assistant message with token usage
        assistant = Assistant(
            content=[TextContent(response_text)],
            response_id=response_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
        )

        logger.info(
            f"Response complete: {len(response_text)} chars, response_id={response_id}, tokens=({input_tokens} in, {output_tokens} out, {reasoning_tokens} reasoning)"
        )

        return assistant

    async def execute_simple(
        self,
        user_message: str,
        system_message: str | None = None,
        model: str | None = None,
    ) -> str:
        """
        Simple non-streaming execution (for testing).
        Creates a temporary conversation.

        Args:
            user_message: User's message
            system_message: Optional system message (used as instructions)
            model: Model to use

        Returns:
            Response text
        """
        # Create temporary conversation
        conversation = await self.client.conversations.create()

        # Call execute with conversation_id
        assistant = await self.execute(
            conversation_id=conversation.id,
            input_text=user_message,
            model=model,
        )

        return assistant.get_str()
