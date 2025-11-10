"""
Lightweight SDK for programmatic SABRE access.

Provides simple async API for eval harnesses and integrations without TUI overhead.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Awaitable

import httpx
import jsonpickle

from sabre.common.models.events import (
    Event,
    ResponseTextEvent,
    HelpersExecutionEndEvent,
    CompleteEvent,
    ErrorEvent,
    CancelledEvent,
)

logger = logging.getLogger(__name__)


@dataclass
class SabreResult:
    """Result from a SABRE execution."""

    success: bool
    response: str
    conversation_id: str
    error: str | None = None

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    # Execution details
    helper_executions: list[dict] = None  # List of helper execution details

    def __post_init__(self):
        if self.helper_executions is None:
            self.helper_executions = []


class SabreClient:
    """
    Lightweight async client for SABRE server.

    Usage:
        client = SabreClient()
        result = await client.run("Plot a sine wave")
        print(result.response)
    """

    def __init__(self, base_url: str = "http://localhost:8011", timeout: float = 300.0):
        """
        Initialize SABRE client.

        Args:
            base_url: Server URL (default: http://localhost:8011)
            timeout: Request timeout in seconds (default: 300s)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def run(
        self,
        message: str,
        conversation_id: str | None = None,
        event_callback: Callable[[Event], Awaitable[None]] | None = None,
    ) -> SabreResult:
        """
        Run a message through SABRE and return the result.

        Args:
            message: User message to process
            conversation_id: Optional conversation ID for continuity
            event_callback: Optional async callback for streaming events

        Returns:
            SabreResult with response and metadata

        Raises:
            httpx.HTTPError: If server returns non-200 status
            asyncio.TimeoutError: If request exceeds timeout
        """
        logger.info(f"Running message: {message[:100]}...")

        # Track result state
        final_response = ""
        conversation_id_result = conversation_id
        input_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0
        helper_executions = []
        error_message = None
        success = True

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout)) as client:
            # POST to /message endpoint
            async with client.stream(
                "POST",
                f"{self.base_url}/message",
                json={
                    "type": "message",
                    "content": message,
                    "conversation_id": conversation_id,
                },
                headers={"Accept": "text/event-stream"},
            ) as response:
                # Check for errors
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise httpx.HTTPError(f"Server returned {response.status_code}: {error_text.decode()}")

                # Stream SSE events
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    # Check for completion
                    if line == "data: [DONE]":
                        logger.info("Stream complete")
                        break

                    # Parse event
                    json_str = line[6:]  # Remove "data: " prefix

                    # Skip request ID
                    if json_str.startswith("__REQUEST_ID__:"):
                        continue

                    # Decode event
                    try:
                        event = jsonpickle.decode(json_str)
                    except Exception as e:
                        logger.warning(f"Failed to decode event: {e}")
                        continue

                    # Update conversation ID
                    if hasattr(event, "conversation_id"):
                        conversation_id_result = event.conversation_id

                    # Call user callback if provided
                    if event_callback:
                        await event_callback(event)

                    # Extract data based on event type
                    if isinstance(event, ResponseTextEvent):
                        # Extract final response text
                        final_response = event.data.get("text", "")
                        input_tokens = event.data.get("input_tokens", 0)
                        output_tokens = event.data.get("output_tokens", 0)
                        reasoning_tokens = event.data.get("reasoning_tokens", 0)

                    elif isinstance(event, HelpersExecutionEndEvent):
                        # Track helper executions
                        helper_executions.append(
                            {
                                "block_number": event.data.get("block_number"),
                                "success": event.data.get("success"),
                                "duration_ms": event.data.get("duration_ms"),
                                "result": event.data.get("result"),
                                "code": event.data.get("code"),
                            }
                        )

                    elif isinstance(event, ErrorEvent):
                        # Capture error
                        error_message = event.data.get("error_message", str(event))
                        success = False

                    elif isinstance(event, CancelledEvent):
                        # Handle cancellation
                        error_message = "Execution cancelled"
                        success = False
                        break

                    elif isinstance(event, CompleteEvent):
                        # Execution complete
                        logger.info("Execution complete")

        logger.info(f"Completed: success={success}, response_length={len(final_response)}")

        return SabreResult(
            success=success,
            response=final_response,
            conversation_id=conversation_id_result or "",
            error=error_message,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            helper_executions=helper_executions,
        )

    async def health_check(self) -> bool:
        """
        Check if server is healthy.

        Returns:
            True if server is reachable and healthy
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def clear_conversation(self, conversation_id: str):
        """
        Clear a conversation's state.

        Args:
            conversation_id: Conversation ID to clear
        """
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{self.base_url}/clear",
                json={"conversation_id": conversation_id},
            )
            response.raise_for_status()
            logger.info(f"Cleared conversation {conversation_id}")
