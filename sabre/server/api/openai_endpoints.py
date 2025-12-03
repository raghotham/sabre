"""
OpenAI-compatible API endpoints for SABRE.

These endpoints allow benchmarks and external tools to use SABRE
as a drop-in replacement for OpenAI's API.
"""

import asyncio
import logging
import os
import time
import uuid
from typing import List, Optional, Dict, Any, Union

from fastapi import Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from sabre.common import (
    ExecutionTree,
    ExecutionNodeType,
    ExecutionStatus,
    Event,
    EventType,
)
from sabre.common.models.events import ResponseTextEvent, ErrorEvent

logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================


class ChatMessage(BaseModel):
    """OpenAI chat message."""

    role: str
    content: Union[str, List[Dict[str, Any]]]  # Can be string or array of content parts


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request."""

    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    # Add other OpenAI params as needed, but we'll ignore most of them


class ChatCompletionResponse(BaseModel):
    """OpenAI chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class ChatCompletionChunkResponse(BaseModel):
    """OpenAI chat completion streaming chunk."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]


class ModelObject(BaseModel):
    """OpenAI model object."""

    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsListResponse(BaseModel):
    """OpenAI models list response."""

    object: str = "list"
    data: List[ModelObject]


# ============================================================================
# Endpoint Handlers
# ============================================================================


async def chat_completions_endpoint(request: Request, manager):
    """
    OpenAI-compatible /v1/chat/completions endpoint.

    Args:
        request: FastAPI request
        manager: SessionManager instance from server.py

    Returns:
        JSON response or StreamingResponse
    """
    try:
        body = await request.json()
        req = ChatCompletionRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    # Extract user message from messages array
    # Concatenate all user messages
    user_messages = []
    for msg in req.messages:
        if msg.role == "user":
            # Content can be string or array of content parts
            if isinstance(msg.content, str):
                user_messages.append(msg.content)
            elif isinstance(msg.content, list):
                # Extract text from content parts
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        user_messages.append(part.get("text", ""))

    if not user_messages:
        raise HTTPException(status_code=400, detail="At least one user message is required")

    user_input = "\n".join(user_messages)

    # Get model name from env or use request
    model_name = os.getenv("OPENAI_MODEL", req.model or "gpt-4o")
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    logger.info(
        f"Chat completion request: model={model_name}, stream={req.stream}, "
        f"messages={len(req.messages)}, input_length={len(user_input)}"
    )

    if req.stream:
        # Streaming response
        return StreamingResponse(
            _stream_chat_completion(manager, user_input, completion_id, created, model_name),
            media_type="text/event-stream",
        )
    else:
        # Non-streaming response
        result = await _complete_chat_completion(manager, user_input, completion_id, created, model_name)
        return JSONResponse(content=result)


async def models_endpoint(request: Request):
    """
    OpenAI-compatible /v1/models endpoint.

    Returns list of available models (currently just the one SABRE is using).
    """
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o")

    response = ModelsListResponse(
        data=[
            ModelObject(
                id=model_name,
                created=int(time.time()),
                owned_by="openai",
            )
        ]
    )

    return JSONResponse(content=response.dict())


# ============================================================================
# Helper Functions
# ============================================================================


async def _complete_chat_completion(
    manager, user_input: str, completion_id: str, created: int, model_name: str
) -> dict:
    """
    Process non-streaming chat completion.

    Runs orchestrator.run() and collects final response.
    """
    # Create new session/tree for this request
    tree = ExecutionTree()
    _ = tree.push(ExecutionNodeType.CLIENT_REQUEST, metadata={"message": user_input})

    # Collect response data
    response_text = ""
    input_tokens = 0
    output_tokens = 0
    error_message = None

    # Event callback to collect data
    async def event_callback(event: Event):
        nonlocal response_text, input_tokens, output_tokens, error_message

        if event.type == EventType.RESPONSE_TEXT:
            # Final response text
            if isinstance(event, ResponseTextEvent):
                response_text = event.data.get("text", "")
                input_tokens = event.data.get("input_tokens", 0)
                output_tokens = event.data.get("output_tokens", 0)

        elif event.type == EventType.ERROR:
            if isinstance(event, ErrorEvent):
                error_message = f"{event.error_type}: {event.error_message}"

    # Load instructions for new conversation
    instructions = manager.orchestrator.load_default_instructions()

    # Run orchestration
    try:
        result = await manager.orchestrator.run(
            conversation_id=None,  # Always new conversation for benchmarks
            input_text=user_input,
            tree=tree,
            instructions=instructions,
            event_callback=event_callback,
        )

        tree.pop(ExecutionStatus.COMPLETED)

        if not result.success:
            error_message = result.error or "Unknown error"

    except Exception as e:
        logger.error(f"Error in chat completion: {e}", exc_info=True)
        tree.pop(ExecutionStatus.ERROR)
        error_message = str(e)

    # If error occurred, raise it
    if error_message:
        raise HTTPException(status_code=500, detail=error_message)

    # Build OpenAI-compatible response
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


async def _stream_chat_completion(manager, user_input: str, completion_id: str, created: int, model_name: str):
    """
    Process streaming chat completion.

    Yields SSE chunks in OpenAI format.
    """
    import json

    # Create new session/tree for this request
    tree = ExecutionTree()
    _ = tree.push(ExecutionNodeType.CLIENT_REQUEST, metadata={"message": user_input})

    # Event queue for streaming
    event_queue: asyncio.Queue = asyncio.Queue()
    response_started = False

    # Event callback to collect streaming data
    async def event_callback(event: Event):
        nonlocal response_started
        await event_queue.put(event)

    # Load instructions for new conversation
    instructions = manager.orchestrator.load_default_instructions()

    # Create task for orchestration
    async def run_orchestration():
        try:
            result = await manager.orchestrator.run(
                conversation_id=None,  # Always new conversation for benchmarks
                input_text=user_input,
                tree=tree,
                session_id=completion_id,  # Use completion_id as session_id
                instructions=instructions,
                event_callback=event_callback,
            )
            tree.pop(ExecutionStatus.COMPLETED)
            return result
        except Exception as e:
            logger.error(f"Error in streaming chat completion: {e}", exc_info=True)
            tree.pop(ExecutionStatus.ERROR)
            raise
        finally:
            # Signal completion
            await event_queue.put(None)

    # Start orchestration task
    task = asyncio.create_task(run_orchestration())

    try:
        # Stream events as OpenAI chunks
        while True:
            event = await event_queue.get()

            if event is None:
                # Task completed - send final chunk
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
                break

            # Only stream final response text
            if event.type == EventType.RESPONSE_TEXT and isinstance(event, ResponseTextEvent):
                text = event.data.get("text", "")

                # Send chunk with full text (OpenAI sends incremental deltas,
                # but for simplicity we send the full text once)
                if not response_started:
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    response_started = True

            # Handle errors
            elif event.type == EventType.ERROR and isinstance(event, ErrorEvent):
                error_chunk = {
                    "id": completion_id,
                    "object": "error",
                    "message": f"{event.error_type}: {event.error_message}",
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                break

    except Exception as e:
        logger.error(f"Error streaming chat completion: {e}", exc_info=True)
        error_chunk = {
            "id": completion_id,
            "object": "error",
            "message": str(e),
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    finally:
        # Ensure task is cancelled if client disconnects
        if not task.done():
            task.cancel()
