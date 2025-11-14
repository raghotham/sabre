"""
OpenAI-compatible client wrapper for SABRE.

Provides the same interface as OpenAI's chat.completions.create() but uses SABRE's server.
"""

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Iterator, List, Optional, Union

import httpx
import jsonpickle

from sabre.common.models.events import (
    ResponseTextEvent,
    ErrorEvent,
    CompleteEvent,
)

logger = logging.getLogger(__name__)


class ChatCompletionMessage:
    """OpenAI-compatible chat completion message."""

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self):
        return {"role": self.role, "content": self.content}


class ChatCompletionChoice:
    """OpenAI-compatible chat completion choice."""

    def __init__(self, index: int, message: ChatCompletionMessage, finish_reason: str):
        self.index = index
        self.message = message
        self.finish_reason = finish_reason


class ChatCompletionUsage:
    """OpenAI-compatible token usage."""

    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class ChatCompletion:
    """OpenAI-compatible chat completion response."""

    def __init__(
        self,
        id: str,
        choices: List[ChatCompletionChoice],
        created: int,
        model: str,
        usage: ChatCompletionUsage,
    ):
        self.id = id
        self.choices = choices
        self.created = created
        self.model = model
        self.usage = usage
        self.object = "chat.completion"


class ChatCompletionChunk:
    """OpenAI-compatible streaming chunk."""

    def __init__(self, id: str, choices: List[dict], created: int, model: str):
        self.id = id
        self.choices = choices
        self.created = created
        self.model = model
        self.object = "chat.completion.chunk"


# ============================================================================
# Base classes for shared functionality
# ============================================================================


class BaseCompletions:
    """Base class for chat completions with shared functionality."""

    def __init__(self, client):
        self._client = client

    async def _parse_sse_line(self, line: str) -> Optional[Any]:
        """Parse SSE line and extract event data."""
        if not line or line.startswith(":"):
            return None

        if not line.startswith("data: "):
            return None

        json_str = line[6:]  # Remove "data: " prefix

        if json_str == "[DONE]":
            return None

        # Handle request_id
        if json_str.startswith("__REQUEST_ID__:"):
            return {"_request_id": json_str.split(":", 1)[1]}

        # Decode with jsonpickle
        try:
            return jsonpickle.decode(json_str)
        except Exception as e:
            logger.warning(f"Failed to decode event: {e}")
            return None

    def _validate_messages(self, messages: Optional[List[dict]]) -> str:
        """Validate messages and extract user input."""
        if not messages:
            raise ValueError("messages parameter is required")

        # Extract user message from messages list
        # SABRE expects a single user message, so we'll concatenate all user messages
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if not user_messages:
            raise ValueError("At least one user message is required")

        return "\n".join(msg.get("content", "") for msg in user_messages)

    async def _complete_response(self, user_input: str, model: Optional[str]) -> ChatCompletion:
        """Get complete response (non-streaming)."""
        conversation_id = self._client._conversation_id
        response_text = ""
        input_tokens = 0
        output_tokens = 0
        created = int(time.time())

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self._client.base_url}/v1/message",
                json={
                    "type": "message",
                    "content": user_input,
                    "conversation_id": conversation_id,
                },
                headers={"Accept": "text/event-stream"},
                timeout=httpx.Timeout(None),
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise Exception(f"Server error ({response.status_code}): {error_text.decode()}")

                async for line in response.aiter_lines():
                    event = await self._parse_sse_line(line)
                    if event is None:
                        continue

                    # Extract request_id for future reference
                    if isinstance(event, dict) and "_request_id" in event:
                        continue

                    # Update conversation_id from events
                    if hasattr(event, "conversation_id") and event.conversation_id:
                        self._client._conversation_id = event.conversation_id

                    # Collect response text
                    if isinstance(event, ResponseTextEvent):
                        response_text = event.data.get("text", "")
                        input_tokens = event.data.get("input_tokens", 0)
                        output_tokens = event.data.get("output_tokens", 0)

                    # Handle errors
                    if isinstance(event, ErrorEvent):
                        raise Exception(f"{event.error_type}: {event.error_message}")

                    # Complete event signals end
                    if isinstance(event, CompleteEvent):
                        break

        # Build OpenAI-compatible response
        message = ChatCompletionMessage(role="assistant", content=response_text)
        choice = ChatCompletionChoice(index=0, message=message, finish_reason="stop")
        usage = ChatCompletionUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

        return ChatCompletion(
            id=self._client._conversation_id or "unknown",
            choices=[choice],
            created=created,
            model=model or "sabre",
            usage=usage,
        )

    async def _stream_response(self, user_input: str, model: Optional[str]) -> AsyncIterator[ChatCompletionChunk]:
        """Stream response chunks."""
        conversation_id = self._client._conversation_id
        created = int(time.time())

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self._client.base_url}/v1/message",
                json={
                    "type": "message",
                    "content": user_input,
                    "conversation_id": conversation_id,
                },
                headers={"Accept": "text/event-stream"},
                timeout=httpx.Timeout(None),
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise Exception(f"Server error ({response.status_code}): {error_text.decode()}")

                async for line in response.aiter_lines():
                    event = await self._parse_sse_line(line)
                    if event is None:
                        continue

                    # Extract request_id for future reference
                    if isinstance(event, dict) and "_request_id" in event:
                        continue

                    # Update conversation_id from events
                    if hasattr(event, "conversation_id") and event.conversation_id:
                        self._client._conversation_id = event.conversation_id

                    # Stream response text as chunks
                    if isinstance(event, ResponseTextEvent):
                        text = event.data.get("text", "")
                        chunk = ChatCompletionChunk(
                            id=self._client._conversation_id or "unknown",
                            choices=[
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": text},
                                    "finish_reason": None,
                                }
                            ],
                            created=created,
                            model=model or "sabre",
                        )
                        yield chunk

                    # Handle errors
                    if isinstance(event, ErrorEvent):
                        raise Exception(f"{event.error_type}: {event.error_message}")

                    # Complete event signals end
                    if isinstance(event, CompleteEvent):
                        # Send final chunk with finish_reason
                        chunk = ChatCompletionChunk(
                            id=self._client._conversation_id or "unknown",
                            choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            created=created,
                            model=model or "sabre",
                        )
                        yield chunk
                        break


# ============================================================================
# Synchronous API
# ============================================================================


class ChatCompletions(BaseCompletions):
    """Synchronous chat completions endpoint."""

    def __init__(self, client: "OpenAI"):
        super().__init__(client)

    def create(
        self,
        model: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """
        Create chat completion (synchronous).

        Args:
            model: Model name (ignored - SABRE uses server's configured model)
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream responses
            **kwargs: Additional arguments (ignored)

        Returns:
            ChatCompletion or Iterator[ChatCompletionChunk]
        """
        user_input = self._validate_messages(messages)

        if stream:
            # Return sync iterator that wraps async generator
            async def _async_stream():
                async for chunk in self._stream_response(user_input, model):
                    yield chunk

            # Convert async generator to sync iterator
            loop = asyncio.new_event_loop()
            queue = asyncio.Queue()

            async def _producer():
                try:
                    async for chunk in self._stream_response(user_input, model):
                        await queue.put(chunk)
                finally:
                    await queue.put(None)  # Sentinel

            def _consumer():
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_producer())

            import threading

            thread = threading.Thread(target=_consumer, daemon=True)
            thread.start()

            def _sync_iterator():
                while True:
                    chunk = asyncio.run_coroutine_threadsafe(queue.get(), loop).result()
                    if chunk is None:
                        break
                    yield chunk

            return _sync_iterator()
        else:
            return asyncio.run(self._complete_response(user_input, model))


class Chat:
    """Synchronous chat API wrapper."""

    def __init__(self, client: "OpenAI"):
        self.completions = ChatCompletions(client)


class OpenAI:
    """
    OpenAI-compatible synchronous client for SABRE.

    Usage:
        client = OpenAI(base_url="http://localhost:8011")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )
        print(response.choices[0].message.content)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8011",
        **kwargs,
    ):
        """
        Initialize SABRE synchronous client.

        Args:
            api_key: Ignored (for compatibility with OpenAI SDK)
            base_url: SABRE server URL
            **kwargs: Additional arguments (ignored)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.chat = Chat(self)
        self._conversation_id: Optional[str] = None

    def reset_conversation(self):
        """Reset conversation ID to start a new conversation."""
        self._conversation_id = None


# ============================================================================
# Asynchronous API
# ============================================================================


class AsyncChatCompletions(BaseCompletions):
    """Asynchronous chat completions endpoint."""

    def __init__(self, client: "AsyncOpenAI"):
        super().__init__(client)

    async def create(
        self,
        model: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """
        Create chat completion (asynchronous).

        Args:
            model: Model name (ignored - SABRE uses server's configured model)
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream responses
            **kwargs: Additional arguments (ignored)

        Returns:
            ChatCompletion or AsyncIterator[ChatCompletionChunk]
        """
        user_input = self._validate_messages(messages)

        if stream:
            return self._stream_response(user_input, model)
        else:
            return await self._complete_response(user_input, model)


class AsyncChat:
    """Asynchronous chat API wrapper."""

    def __init__(self, client: "AsyncOpenAI"):
        self.completions = AsyncChatCompletions(client)


class AsyncOpenAI:
    """
    OpenAI-compatible asynchronous client for SABRE.

    Usage:
        client = AsyncOpenAI(base_url="http://localhost:8011")
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )
        print(response.choices[0].message.content)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8011",
        **kwargs,
    ):
        """
        Initialize SABRE asynchronous client.

        Args:
            api_key: Ignored (for compatibility with OpenAI SDK)
            base_url: SABRE server URL
            **kwargs: Additional arguments (ignored)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.chat = AsyncChat(self)
        self._conversation_id: Optional[str] = None

    def reset_conversation(self):
        """Reset conversation ID to start a new conversation."""
        self._conversation_id = None
