"""
FastAPI server for sabre.

HTTP SSE-based server that handles chat messages and streams back responses.
"""

import asyncio
import datetime
import json
import jsonpickle
import logging
import os
import platform
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from sabre.common import (
    ResponseExecutor,
    ExecutionTree,
    ExecutionNodeType,
    ExecutionStatus,
    Event,
    CancelledEvent,
    ErrorEvent,
)
from sabre.common.paths import (
    get_logs_dir,
    get_session_files_dir,
    get_session_log_file,
    get_sessions_base_dir,
    ensure_dirs,
    SabrePaths,
)
from sabre.server.orchestrator import Orchestrator
from sabre.server.python_runtime import PythonRuntime
from sabre.server.api.connector_store import ConnectorStore
from sabre.server.api import openai_endpoints
from sabre.server.session_logger import SessionLogger

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages sessions by conversation_id (HTTP SSE-based)."""

    def __init__(self):
        # Session data per conversation_id
        self.sessions: dict[str, dict] = {}
        # Track running tasks for cancellation (by request_id)
        self.running_tasks: dict[str, asyncio.Task] = {}

        # Initialize connector store for persistence
        connector_store_path = SabrePaths.get_state_home() / "connectors.json"
        self.connector_store = ConnectorStore(connector_store_path)

        # Initialize MCP integration
        self.mcp_manager = None
        self.mcp_adapter = None
        self._init_mcp()

        # Session logger for execution tree visualization
        self.session_logger = SessionLogger()

        # Create orchestrator with executor and runtime
        # ResponseExecutor reads OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL from env
        executor = ResponseExecutor()
        runtime = PythonRuntime(mcp_adapter=self.mcp_adapter)
        self.orchestrator = Orchestrator(
            executor,
            runtime,
            session_logger=self.session_logger,
        )

    def _init_mcp(self):
        """Initialize MCP config loader (actual connection happens in async context)."""
        try:
            from sabre.server.mcp import MCPConfigLoader

            # Load MCP configurations
            self.mcp_configs = MCPConfigLoader.load()

            if not self.mcp_configs:
                logger.info("No MCP servers configured")
                return

            logger.info(f"Loaded {len(self.mcp_configs)} MCP server configurations")

        except ImportError:
            logger.debug("MCP integration not available (PyYAML not installed)")
            self.mcp_configs = []
        except Exception as e:
            logger.warning(f"Failed to load MCP configurations: {e}")
            logger.debug("Continuing without MCP integration")
            self.mcp_configs = []

    async def connect_mcp_servers(self):
        """Connect to MCP servers (must be called in async context)."""
        try:
            from sabre.server.mcp import MCPClientManager, MCPHelperAdapter

            # Create client manager with connector store for persistence
            self.mcp_manager = MCPClientManager(connector_store=self.connector_store)

            # Load persisted connectors from ConnectorStore
            persisted_configs = self.connector_store.load_all()
            logger.info(f"Loaded {len(persisted_configs)} persisted connectors")

            # Merge bootstrap configs from YAML
            all_configs = {}

            # Add bootstrap configs (from YAML) - mark as yaml source
            if hasattr(self, "mcp_configs"):
                for config in self.mcp_configs:
                    config.source = "yaml"
                    all_configs[config.id] = config

            # Override with persisted configs (from API) - already have correct source
            for connector_id, config in persisted_configs.items():
                all_configs[connector_id] = config

            logger.info(
                f"Total connectors to connect: {len(all_configs)} ({len(self.mcp_configs) if hasattr(self, 'mcp_configs') else 0} from YAML, {len(persisted_configs)} persisted)"
            )

            # Register all connectors (both enabled and disabled)
            # Don't auto-connect on bootstrap to avoid startup errors
            for config in all_configs.values():
                # Temporarily mark as disabled for registration
                original_enabled = config.enabled
                config.enabled = False

                try:
                    # Register the connector (stores config, doesn't connect)
                    await self.mcp_manager.connect(config)

                    # Restore original enabled state
                    config.enabled = original_enabled
                    self.mcp_manager.configs[config.id].enabled = original_enabled

                    # If it was enabled, try to connect now
                    if original_enabled:
                        try:
                            # Enable will trigger connection
                            await self.mcp_manager.enable_connector(config.id)
                        except Exception as e:
                            logger.warning(f"Failed to connect to {config.name} on startup (will retry on demand): {e}")
                            # Keep connector registered but disabled

                except Exception as e:
                    logger.error(f"Failed to register connector {config.name}: {e}")
                    # Continue with other connectors

            # Create helper adapter and refresh tools (pass event loop for thread-safe execution)
            self.mcp_adapter = MCPHelperAdapter(self.mcp_manager, event_loop=asyncio.get_running_loop())
            await self.mcp_adapter.refresh_tools()

            logger.info(
                f"MCP integration initialized with {self.mcp_adapter.get_tool_count()} tools from {len(self.mcp_manager.list_servers())} servers"
            )

            # Update runtime with MCP adapter
            if self.orchestrator and self.orchestrator.runtime:
                self.orchestrator.runtime.mcp_adapter = self.mcp_adapter
                self.orchestrator.runtime.reset()  # Refresh namespace with MCP tools

        except Exception as e:
            logger.error(f"Failed to connect to MCP servers: {e}")
            logger.debug("Continuing without MCP integration")

    async def disconnect_mcp_servers(self):
        """Disconnect from MCP servers."""
        if self.mcp_manager:
            try:
                await self.mcp_manager.disconnect_all()
                logger.info("Disconnected from MCP servers")
            except Exception as e:
                logger.error(f"Error disconnecting from MCP servers: {e}")

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
        # Try platform-specific paths
        system = platform.system()
        if system == "Darwin":  # macOS
            playwright_cache = Path.home() / "Library" / "Caches" / "ms-playwright"
        elif system == "Linux":
            playwright_cache = Path.home() / ".cache" / "ms-playwright"
        elif system == "Windows":
            playwright_cache = Path.home() / "AppData" / "Local" / "ms-playwright"
        else:
            playwright_cache = Path.home() / ".cache" / "ms-playwright"  # Default to Linux path

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

    # Connect to MCP servers
    try:
        await manager.connect_mcp_servers()
    except Exception as e:
        logger.warning(f"MCP initialization failed: {e}")
        logger.info("Server starting without MCP integration")

    yield

    # Disconnect from MCP servers on shutdown
    logger.info("Shutting down sabre server...")
    await manager.disconnect_mcp_servers()


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


@app.get("/v1/health")
async def health():
    """Health check with details."""
    return {
        "status": "ok",
        "model": manager.orchestrator.executor.default_model,
        "active_sessions": len(manager.sessions),
        "running_tasks": len(manager.running_tasks),
    }


@app.get("/v1/sessions/{session_id}/files/{filename}")
async def serve_session_file(session_id: str, filename: str):
    """
    Serve files generated during session (e.g., matplotlib images, saved data).

    Files are organized by session ID, with all files for a session stored in:
    ~/.local/state/sabre/logs/sessions/{session_id}/files/

    Security:
    - Only serves files from session directories
    - Validates filename is basename only (no path traversal)
    - Returns 404 for non-existent files
    """
    # Security: basename only (prevent path traversal)
    if os.path.basename(filename) != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Files are stored in ~/.local/state/sabre/logs/sessions/{session_id}/files/
    files_dir = get_session_files_dir(session_id)
    file_path = files_dir / filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path)


@app.get("/v1/sessions")
async def list_sessions():
    """
    List all available sessions.

    Returns:
        List of session summaries with metadata
    """
    sessions = manager.session_logger.list_sessions(limit=100)
    return {"sessions": sessions}


@app.get("/v1/sessions/{session_id}")
async def get_session_data(session_id: str):
    """
    Get session log data (session.jsonl).

    Returns JSONL file as plain text that can be parsed line-by-line.

    Security:
    - Only serves session.jsonl from session directories
    - Returns 404 for non-existent sessions
    """
    session_file = get_session_log_file(session_id)

    if not session_file.exists() or not session_file.is_file():
        raise HTTPException(status_code=404, detail="Session not found")

    return FileResponse(session_file, media_type="application/x-ndjson")


@app.get("/v1/sessions/{session_id}/atif")
async def get_session_atif(session_id: str, model: str = None):
    """
    Get session in ATIF (Agent Trajectory Interchange Format) v1.2.

    ATIF is a standardized format for representing agent execution traces,
    used by Harbor, Terminal-Bench, and other benchmarking frameworks.

    Query Parameters:
        model: Optional model name to include in agent metadata (default: from env OPENAI_MODEL)

    Returns:
        JSON response with ATIF-formatted trajectory

    Security:
        - Only serves sessions from session directories
        - Returns 404 for non-existent sessions
    """
    from sabre.server.atif_export import events_to_atif
    import os

    # Load session events
    events = manager.session_logger.get_session(session_id)

    if not events:
        raise HTTPException(status_code=404, detail="Session not found or has no events")

    # Get model name from query param, env, or default
    model_name = model or os.environ.get("OPENAI_MODEL", "gpt-4o")

    # Convert to ATIF
    atif = events_to_atif(
        events,
        agent_name="sabre",
        agent_version="latest",  # TODO: Get from package version
        model_name=model_name,
    )

    return atif


@app.get("/v1/sessions/{session_id}/files")
async def list_session_files(session_id: str):
    """
    List all files in a session's files directory.

    Returns:
        JSON array of filenames with metadata
    """

    files_dir = get_sessions_base_dir() / session_id / "files"

    if not files_dir.exists():
        return {"files": []}

    files = []
    for file_path in files_dir.iterdir():
        if file_path.is_file():
            # Determine file type
            suffix = file_path.suffix.lower()
            file_type = "image" if suffix in [".png", ".jpg", ".jpeg", ".gif", ".webp"] else "other"

            files.append(
                {
                    "filename": file_path.name,
                    "type": file_type,
                    "size": file_path.stat().st_size,
                }
            )

    return {"files": files}


@app.get("/v1/sessions/{session_id}/files/{filename}")
async def get_session_file(session_id: str, filename: str):
    """
    Get a file from a session's files directory (e.g., screenshots, images).

    Security:
    - Only serves files from session's files directory
    - Returns 404 for non-existent files or directory traversal attempts
    """

    # Sanitize filename to prevent directory traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Construct file path
    files_dir = get_sessions_base_dir() / session_id / "files"
    file_path = files_dir / filename

    # Verify file exists and is within the files directory
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    if not str(file_path).startswith(str(files_dir)):
        raise HTTPException(status_code=403, detail="Access denied")

    # Determine media type based on extension
    if filename.endswith(".png"):
        media_type = "image/png"
    elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
        media_type = "image/jpeg"
    else:
        media_type = "application/octet-stream"

    return FileResponse(file_path, media_type=media_type)


@app.get("/")
async def serve_session_viewer():
    """
    Serve the session viewer HTML page at root path.
    """
    from pathlib import Path

    # Find session_viewer.html in the sabre/ui directory
    viewer_path = Path(__file__).parent.parent.parent / "ui" / "session_viewer.html"

    if not viewer_path.exists():
        raise HTTPException(status_code=404, detail="Session viewer not found")

    return FileResponse(viewer_path, media_type="text/html")


@app.post("/v1/message")
async def message_endpoint(request: Request):
    """
    HTTP SSE endpoint for chat messages.

    Client POSTs: {"type": "message", "content": "user message text", "session_id": "..."|null, "attachments": "..."|null}
    Server streams: SSE events with jsonpickle-encoded Event objects
    """
    data = await request.json()
    user_message = data.get("content", "")
    conversation_id = data.get("conversation_id")
    session_id = data.get("session_id")
    attachments_json = data.get("attachments")

    # Deserialize attachments if present
    attachments = []
    if attachments_json:
        try:
            attachments = jsonpickle.decode(attachments_json)
            logger.info(f"Received {len(attachments)} attachments")
        except Exception as e:
            error_msg = str(e)  # Capture the error message for async function
            logger.error(f"Failed to deserialize attachments: {e}")

            # Return error as SSE stream
            async def error_stream():
                from sabre.common.models.events import ErrorEvent

                error_event = ErrorEvent(
                    error_type="deserialization_error", message=f"Failed to deserialize attachments: {error_msg}"
                )
                yield f"data: {jsonpickle.encode(error_event)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(error_stream(), media_type="text/event-stream")

    # Generate session ID if not provided (new session)
    if not session_id:
        session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:8]
        logger.info(f"Generated new session ID: {session_id}")
        # Log session start
        manager.session_logger.log_session_start(session_id, user_message)

    logger.info(
        f"Received message request: session_id={session_id}, conversation_id={conversation_id}, message={user_message[:50]}..."
    )

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
                        attachments=attachments if attachments else None,  # Pass attachments
                        tree=tree,
                        instructions=instructions,  # Required for new conversations
                        event_callback=event_callback,
                        session_id=session_id,  # Pass session ID for logging
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


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint.

    Allows external benchmarks (datascibench, gaia, tau2, etc.) to use SABRE
    as a drop-in replacement for OpenAI's API.

    Timeout: 300 seconds (5 minutes) to allow for agentic workflows.
    """
    return await openai_endpoints.chat_completions_endpoint(request, manager)


@app.get("/v1/models")
async def models(request: Request):
    """
    OpenAI-compatible models listing endpoint.

    Returns the model SABRE is currently using.
    """
    return await openai_endpoints.models_endpoint(request)


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


# ============================================================================
# V1 API - Connector Management (MCP Connectors CRUD)
# ============================================================================

from sabre.server.api.models import (
    ConnectorCreateRequest,
    ConnectorUpdateRequest,
    ConnectorResponse,
    ConnectorDetailResponse,
    ToolResponse,
    ConnectorToolsResponse,
)
from sabre.server.mcp.models import MCPServerConfig, MCPTransportType


@app.post("/v1/connectors", response_model=ConnectorResponse)
async def create_connector(request: ConnectorCreateRequest):
    """
    Create and connect to a new MCP server.

    The connector will be persisted and automatically reconnected on server restart.
    """
    try:
        # Convert request to MCPServerConfig
        config = MCPServerConfig(
            name=request.name,
            type=MCPTransportType(request.type),
            command=request.command,
            args=request.args,
            env=request.env,
            url=request.url,
            headers=request.headers,
            enabled=request.enabled,
            timeout=request.timeout,
            source="api",  # Mark as API-created
        )

        # Connect to server
        connector_id = await manager.mcp_manager.connect(config)

        # Refresh runtime tools
        if manager.orchestrator and manager.orchestrator.runtime:
            manager.orchestrator.runtime.mcp_adapter = manager.mcp_adapter
            manager.orchestrator.runtime.reset()

        # Get connector info for response
        info = manager.mcp_manager.get_connector_info(connector_id)

        return ConnectorResponse(**info)

    except Exception as e:
        logger.error(f"Failed to create connector: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/connectors", response_model=list[ConnectorResponse])
async def list_connectors():
    """
    List all configured MCP connectors with their status.
    """
    try:
        if not manager.mcp_manager:
            return []

        connectors_info = manager.mcp_manager.get_all_connector_info()
        return [ConnectorResponse(**info) for info in connectors_info]
    except Exception as e:
        logger.error(f"Failed to list connectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/connectors/{connector_id}", response_model=ConnectorDetailResponse)
async def get_connector(connector_id: str):
    """
    Get detailed information about a specific connector.
    """
    try:
        if not manager.mcp_manager:
            raise HTTPException(status_code=404, detail="MCP manager not initialized")

        info = manager.mcp_manager.get_connector_info(connector_id)
        config = manager.mcp_manager.configs.get(connector_id)

        if not config:
            raise HTTPException(status_code=404, detail="Connector not found")

        return ConnectorDetailResponse(
            **info,
            command=config.command,
            args=config.args,
            url=config.url,
            timeout=config.timeout,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get connector {connector_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/connectors/{connector_id}/tools", response_model=ConnectorToolsResponse)
async def get_connector_tools(connector_id: str):
    """
    List all tools available from a specific connector.
    """
    try:
        if not manager.mcp_manager:
            raise HTTPException(status_code=404, detail="MCP manager not initialized")

        # Get connector info
        info = manager.mcp_manager.get_connector_info(connector_id)

        # Get tools
        tools = await manager.mcp_manager.get_connector_tools(connector_id)

        # Convert to response format
        tool_responses = [
            ToolResponse(
                name=tool.name,
                description=tool.description,
                signature=tool.get_signature(),
                server_name=info["name"],
                input_schema=tool.input_schema,
            )
            for tool in tools
        ]

        return ConnectorToolsResponse(
            connector_id=connector_id,
            connector_name=info["name"],
            tools=tool_responses,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tools for connector {connector_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/v1/connectors/{connector_id}", response_model=ConnectorResponse)
async def update_connector(connector_id: str, request: ConnectorUpdateRequest):
    """
    Update connector configuration and reconnect.

    Only provided fields will be updated (partial update).
    The connector will be disconnected and reconnected with new config.
    """
    try:
        if not manager.mcp_manager:
            raise HTTPException(status_code=404, detail="MCP manager not initialized")

        # Build updates dict from request (only non-None fields)
        updates = {k: v for k, v in request.model_dump().items() if v is not None}

        if not updates:
            raise HTTPException(status_code=400, detail="No updates provided")

        # Update connector
        await manager.mcp_manager.update_connector(connector_id, updates)

        # Refresh runtime tools
        if manager.orchestrator and manager.orchestrator.runtime:
            manager.orchestrator.runtime.mcp_adapter = manager.mcp_adapter
            manager.orchestrator.runtime.reset()

        # Get updated info
        info = manager.mcp_manager.get_connector_info(connector_id)
        return ConnectorResponse(**info)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update connector {connector_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/v1/connectors/{connector_id}", response_model=ConnectorResponse)
async def patch_connector(connector_id: str, request: ConnectorUpdateRequest):
    """
    Partial update to connector (e.g., enable/disable without reconnecting).

    Use this for lightweight updates like toggling enabled state.
    Use PUT for configuration changes that require reconnection.
    """
    try:
        if not manager.mcp_manager:
            raise HTTPException(status_code=404, detail="MCP manager not initialized")

        # Build updates dict from request (only non-None fields)
        updates = {k: v for k, v in request.model_dump().items() if v is not None}

        if not updates:
            raise HTTPException(status_code=400, detail="No updates provided")

        # Special handling for enable/disable
        if "enabled" in updates and len(updates) == 1:
            if updates["enabled"]:
                await manager.mcp_manager.enable_connector(connector_id)
            else:
                await manager.mcp_manager.disable_connector(connector_id)

            # Refresh runtime tools
            if manager.orchestrator and manager.orchestrator.runtime:
                manager.orchestrator.runtime.mcp_adapter = manager.mcp_adapter
                manager.orchestrator.runtime.reset()
        else:
            # Fall back to full update
            await manager.mcp_manager.update_connector(connector_id, updates)

        # Get updated info
        info = manager.mcp_manager.get_connector_info(connector_id)
        return ConnectorResponse(**info)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to patch connector {connector_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/connectors/{connector_id}")
async def delete_connector(connector_id: str):
    """
    Disconnect and remove a connector.

    The connector will be removed from persistent storage.
    """
    try:
        if not manager.mcp_manager:
            raise HTTPException(status_code=404, detail="MCP manager not initialized")

        # Get name before deleting
        info = manager.mcp_manager.get_connector_info(connector_id)
        connector_name = info["name"]

        # Disconnect and delete
        await manager.mcp_manager.disconnect(connector_id)

        # Refresh runtime tools
        if manager.orchestrator and manager.orchestrator.runtime:
            manager.orchestrator.runtime.mcp_adapter = manager.mcp_adapter
            manager.orchestrator.runtime.reset()

        return {
            "status": "deleted",
            "connector_id": connector_id,
            "connector_name": connector_name,
            "message": "Connector disconnected and removed",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete connector {connector_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
