"""
FastAPI server for sabre.

HTTP SSE-based server that handles chat messages and streams back responses.
"""

import asyncio
import datetime
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import jsonpickle

from sabre.common import (
    ResponseExecutor,
    ExecutionTree,
    ExecutionNodeType,
    ExecutionStatus,
    Event,
    CancelledEvent,
    ErrorEvent,
)
from sabre.common.paths import get_logs_dir, get_files_dir, ensure_dirs
from sabre.server.orchestrator import Orchestrator
from sabre.server.python_runtime import PythonRuntime

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages sessions by conversation_id (HTTP SSE-based)."""

    def __init__(self):
        # Session data per conversation_id
        self.sessions: dict[str, dict] = {}
        # Track running tasks for cancellation (by request_id)
        self.running_tasks: dict[str, asyncio.Task] = {}
        # Create orchestrator with executor and runtime
        # ResponseExecutor reads OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL from env
        executor = ResponseExecutor()
        runtime = PythonRuntime()
        self.orchestrator = Orchestrator(executor, runtime)

    def get_or_create_session(self, conversation_id: str | None) -> dict:
        """Get existing session or create new one."""
        if conversation_id and conversation_id in self.sessions:
            return self.sessions[conversation_id]

        # Create new session
        # Note: conversation_id will be assigned by orchestrator on first message
        session = {"tree": ExecutionTree()}
        if conversation_id:
            self.sessions[conversation_id] = session
        return session

    def save_session(self, conversation_id: str, session: dict):
        """Save session by conversation_id."""
        self.sessions[conversation_id] = session
        logger.info(f"Saved session for conversation {conversation_id}")

    def clear_session(self, conversation_id: str):
        """Clear session by conversation_id."""
        if conversation_id in self.sessions:
            del self.sessions[conversation_id]
            logger.info(f"Cleared session for conversation {conversation_id}")


async def check_playwright_installation():
    """
    Check if Playwright chromium_headless_shell is installed.

    Raises:
        RuntimeError: If Playwright or browser binaries are not installed
    """
    try:
        import json
        from pathlib import Path
        import site

        # Get expected chromium version from playwright package
        chromium_version = None
        for site_dir in site.getsitepackages():
            browsers_json = Path(site_dir) / "playwright" / "driver" / "package" / "browsers.json"
            if browsers_json.exists():
                with open(browsers_json) as f:
                    data = json.load(f)
                    for browser in data["browsers"]:
                        if browser["name"] == "chromium":
                            chromium_version = browser["revision"]
                            break
                break

        if not chromium_version:
            raise RuntimeError("Could not determine required chromium version")

        # Check if chromium_headless_shell is installed
        playwright_cache = Path.home() / "Library" / "Caches" / "ms-playwright"
        headless_dir = playwright_cache / f"chromium_headless_shell-{chromium_version}"

        if not headless_dir.exists():
            raise RuntimeError(
                f"Playwright chromium_headless_shell-{chromium_version} not installed. "
                f"Please run: uvx playwright install chromium --only-shell"
            )

        logger.info(f"Playwright chromium_headless_shell-{chromium_version} found")

    except ImportError as e:
        raise RuntimeError("Playwright not installed. Please run: uv sync") from e


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Ensure directories exist
    ensure_dirs()

    # Get logs directory
    log_dir = get_logs_dir()

    # Configure logging to both file and console
    # Use LOG_LEVEL env var (DEBUG, INFO, WARNING, ERROR) or default to INFO
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "server.log")],
    )
    logger.info("Starting sabre server...")

    # Check Playwright installation
    try:
        await check_playwright_installation()
    except RuntimeError as e:
        logger.error(f"Playwright check failed: {e}")
        raise

    yield
    logger.info("Shutting down sabre server...")


# Create FastAPI app
app = FastAPI(
    title="sabre API",
    description="Simple chat API with execution tree tracking via HTTP SSE",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session manager
manager = SessionManager()


@app.get("/")
async def root():
    """Root endpoint - reserved for future UI."""
    return {"status": "ok", "service": "sabre", "message": "SABRE API - use /v1/ endpoints"}


@app.get("/v1/health")
async def health():
    """Health check with details."""
    return {
        "status": "ok",
        "model": manager.orchestrator.executor.default_model,
        "active_sessions": len(manager.sessions),
        "running_tasks": len(manager.running_tasks),
    }


@app.get("/v1/files/{conversation_id}/{filename}")
async def serve_file(conversation_id: str, filename: str):
    """
    Serve files generated during conversation (e.g., matplotlib images, saved data).

    Security:
    - Only serves files from conversation directories
    - Validates filename is basename only (no path traversal)
    - Returns 404 for non-existent files
    """
    # Security: basename only (prevent path traversal)
    if os.path.basename(filename) != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Files are stored in XDG_DATA_HOME/sabre/files/{conversation_id}/
    files_dir = get_files_dir(conversation_id)
    file_path = files_dir / filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path)


@app.post("/v1/message")
async def message_endpoint(request: Request):
    """
    HTTP SSE endpoint for chat messages.

    Client POSTs: {"type": "message", "content": "user message text", "conversation_id": "..."|null}
    Server streams: SSE events with jsonpickle-encoded Event objects
    """
    data = await request.json()
    user_message = data.get("content", "")
    conversation_id = data.get("conversation_id")

    logger.info(f"Received message request: conversation_id={conversation_id}, message={user_message[:50]}...")

    if not user_message:
        # Return error as SSE stream
        async def error_stream():
            error_event = {"type": "error", "content": "Empty message"}
            yield f"data: {json.dumps(error_event)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(error_stream(), media_type="text/event-stream")

    # Generate request_id for cancellation
    request_id = str(uuid.uuid4())
    logger.info(f"Created request {request_id} for conversation {conversation_id}")

    async def event_generator():
        """Generate SSE events."""
        try:
            # Send request_id as first chunk (for cancellation)
            yield f"data: __REQUEST_ID__:{request_id}\n\n"

            # Get or create session
            session = manager.get_or_create_session(conversation_id)
            tree = session["tree"]

            # Create execution node for this request
            node = tree.push(ExecutionNodeType.CLIENT_REQUEST, metadata={"message": user_message})

            # Event queue for streaming
            event_queue: asyncio.Queue = asyncio.Queue()

            # Create event callback to stream events
            async def event_callback(event: Event):
                """Encode and queue events for streaming."""
                # Log before encoding
                encode_start = time.time()
                timestamp_str = datetime.datetime.now().isoformat()
                node_id = event.node_id[:8] if hasattr(event, "node_id") and event.node_id else "N/A"
                depth = event.depth if hasattr(event, "depth") else "N/A"
                event_type = event.type.value if hasattr(event, "type") else type(event).__name__

                # CRITICAL: Encode event NOW (before queueing) so generator can yield immediately
                encoded = jsonpickle.encode(event)
                encode_time = (time.time() - encode_start) * 1000

                # Queue the encoded string (not the Event object)
                await event_queue.put(encoded)

                logger.info(
                    f"📦 [{timestamp_str}] ENCODED+QUEUED: {event_type} (node={node_id}, depth={depth}, encode={encode_time:.1f}ms)"
                )

            # Load instructions if creating new conversation
            instructions = None
            if conversation_id is None:
                instructions = manager.orchestrator.load_default_instructions()

            # Create task for orchestration (so we can cancel it)
            async def run_orchestration():
                """Run orchestration and signal completion."""
                try:
                    result = await manager.orchestrator.run(
                        conversation_id=conversation_id,  # None for first message
                        input_text=user_message,
                        tree=tree,
                        instructions=instructions,  # Required for new conversations
                        event_callback=event_callback,
                    )

                    # Save conversation_id from orchestration result
                    if result.conversation_id:
                        manager.save_session(result.conversation_id, session)

                    # Mark client request node as completed
                    tree.pop(ExecutionStatus.COMPLETED)

                    return result

                except asyncio.CancelledError:
                    logger.info(f"Orchestration cancelled for request {request_id}")
                    # Send cancelled event
                    tree_context = {
                        "node_id": node.id,
                        "parent_id": node.parent_id,
                        "depth": tree.get_depth(),
                        "path": [n.id for n in tree.get_path()],
                        "conversation_id": conversation_id or "",
                    }
                    await event_callback(CancelledEvent(**tree_context))
                    tree.pop(ExecutionStatus.ERROR)
                    raise

                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
                    tree.pop(ExecutionStatus.ERROR)
                    tree_context = {
                        "node_id": node.id,
                        "parent_id": node.parent_id,
                        "depth": tree.get_depth(),
                        "path": [n.id for n in tree.get_path()],
                        "conversation_id": conversation_id or "",
                    }
                    # Send error event to client
                    await event_callback(ErrorEvent(**tree_context, error_message=str(e), error_type=type(e).__name__))
                    raise
                finally:
                    # Signal completion
                    await event_queue.put(None)

            # Start orchestration task
            task = asyncio.create_task(run_orchestration())
            manager.running_tasks[request_id] = task

            try:
                # Send keepalive every 15 seconds to maintain connection
                keepalive_interval = 15.0
                last_keepalive = asyncio.get_event_loop().time()

                # Stream events from queue
                while True:
                    try:
                        # Wait for encoded event with timeout to send keepalive
                        encoded = await asyncio.wait_for(event_queue.get(), timeout=keepalive_interval)
                    except asyncio.TimeoutError:
                        # Send keepalive comment
                        current_time = asyncio.get_event_loop().time()
                        if current_time - last_keepalive >= keepalive_interval:
                            yield ": keepalive\n\n"
                            last_keepalive = current_time
                        continue

                    if encoded is None:
                        # Task completed
                        break

                    # Event is already encoded - just yield it immediately!
                    # Decode to get event details for logging (quick)
                    try:
                        decoded = jsonpickle.decode(encoded)
                        event_type = decoded.type.value if hasattr(decoded, "type") else type(decoded).__name__
                        node_id = decoded.node_id[:8] if hasattr(decoded, "node_id") and decoded.node_id else "N/A"
                        depth = decoded.depth if hasattr(decoded, "depth") else "N/A"
                    except:
                        event_type = "unknown"
                        node_id = "N/A"
                        depth = "N/A"

                    yield_start = time.time()
                    timestamp_before = datetime.datetime.now().isoformat()
                    logger.info(f"⚡ [{timestamp_before}] About to YIELD: {event_type} (node={node_id}, depth={depth})")

                    yield f"data: {encoded}\n\n"
                    # Send empty comment to force flush (FastAPI/Starlette buffering workaround)
                    yield ": \n\n"

                    yield_time = (time.time() - yield_start) * 1000
                    timestamp_after = datetime.datetime.now().isoformat()
                    logger.info(
                        f"✅ [{timestamp_after}] YIELDED: {event_type} (node={node_id}, depth={depth}, yield={yield_time:.1f}ms)"
                    )

                    last_keepalive = asyncio.get_event_loop().time()

                # Wait for task to complete (to catch exceptions)
                await task

            except asyncio.CancelledError:
                logger.info(f"Stream cancelled for request {request_id}")
                task.cancel()
                raise
            finally:
                # Remove task from tracking
                if request_id in manager.running_tasks:
                    del manager.running_tasks[request_id]

            # Send completion marker
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error in event generator: {e}", exc_info=True)
            # Send error
            error_event = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_event)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Connection": "keep-alive",
        },
    )


@app.post("/v1/cancel/{request_id}")
async def cancel_request(request_id: str):
    """
    Cancel a running request by ID.
    """
    logger.info(f"🛑 /cancel/{request_id} - Received cancellation request")

    if request_id in manager.running_tasks:
        task = manager.running_tasks[request_id]
        task.cancel()
        logger.info(f"✅ Request {request_id} cancelled")
        return {"status": "cancelled", "request_id": request_id}
    else:
        logger.warning(f"⚠️ Request {request_id} not found for cancellation")
        return {"status": "not_found", "request_id": request_id}


@app.post("/v1/clear")
async def clear_conversation(request: Request):
    """Clear conversation by conversation_id."""
    data = await request.json()
    conversation_id = data.get("conversation_id")

    if conversation_id:
        manager.clear_session(conversation_id)
        logger.info(f"Cleared conversation {conversation_id}")
        return {"status": "cleared", "conversation_id": conversation_id}
    else:
        return {"status": "error", "message": "No conversation_id provided"}


if __name__ == "__main__":
    import uvicorn

    # Configure logging
    # Use LOG_LEVEL env var (DEBUG, INFO, WARNING, ERROR) or default to INFO
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Configure uvicorn logging to match application format
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_config["formatters"]["access"]["fmt"] = (
        '%(asctime)s - %(name)s - %(levelname)s - %(client_addr)s - "%(request_line)s" %(status_code)s'
    )

    # Run server
    uvicorn.run(
        "sabre.server.api.server:app", host="0.0.0.0", port=8011, reload=True, log_level="info", log_config=log_config
    )
