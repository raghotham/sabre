"""
FastAPI server for sabre.

Simple WebSocket-based server that handles chat messages and streams back responses.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from sabre.common import (
    ResponseExecutor,
    ExecutionTree,
    ExecutionNodeType,
    ExecutionStatus,
    Event,
    CancelledEvent,
    ErrorEvent,
)
from sabre.server.orchestrator import Orchestrator
from sabre.server.python_runtime import PythonRuntime

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        # Simple session data per client
        self.sessions: dict[str, dict] = {}
        # Track running tasks for cancellation
        self.running_tasks: dict[str, asyncio.Task] = {}
        # Create orchestrator with executor and runtime
        # ResponseExecutor reads OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL from env
        executor = ResponseExecutor()
        runtime = PythonRuntime()
        self.orchestrator = Orchestrator(executor, runtime)

    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept connection and create session."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        # Session has: conversation_id (created on first message), tree
        self.sessions[client_id] = {"conversation_id": None, "tree": ExecutionTree()}
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        """Remove connection and session."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.sessions:
            del self.sessions[client_id]
        logger.info(f"Client {client_id} disconnected")

    async def send_event(self, client_id: str, event: Event):
        """Send event to client."""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            timestamp_str = event.timestamp.isoformat() if hasattr(event, "timestamp") else "N/A"
            # Log with node_id and depth for tracking
            node_id = event.node_id[:8] if hasattr(event, "node_id") and event.node_id else "N/A"
            depth = event.depth if hasattr(event, "depth") else "N/A"
            logger.info(
                f"→ SEND event: type={event.type.value}, node={node_id}, depth={depth}, timestamp={timestamp_str}"
            )
            try:
                # Use send_text with explicit JSON to ensure proper flushing
                # send_json may buffer in some FastAPI/Starlette versions
                # Use jsonable_encoder to handle datetime and other non-serializable types
                await websocket.send_text(json.dumps(jsonable_encoder(event.to_dict())))
                logger.debug(f"✓ Event sent successfully: {event.type.value}")
            except Exception as e:
                logger.error(f"✗ Failed to send event {event.type.value}: {e}")

    async def send_message(self, client_id: str, message: dict):
        """Send JSON message to client."""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            # Use send_text with explicit JSON to ensure proper flushing
            await websocket.send_text(json.dumps(jsonable_encoder(message)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    from sabre.common.paths import get_logs_dir, ensure_dirs

    # Ensure directories exist
    ensure_dirs()

    # Get logs directory
    log_dir = get_logs_dir()

    # Configure logging to both file and console
    # Use LOG_LEVEL env var (DEBUG, INFO, WARNING, ERROR) or default to INFO
    import os

    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "server.log")],
    )
    logger.info("Starting sabre server...")
    yield
    logger.info("Shutting down sabre server...")


# Create FastAPI app
app = FastAPI(
    title="sabre API",
    description="Simple chat API with execution tree tracking",
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

# Connection manager
manager = ConnectionManager()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "sabre"}


@app.get("/health")
async def health():
    """Health check with details."""
    return {
        "status": "ok",
        "active_connections": len(manager.active_connections),
        "active_sessions": len(manager.sessions),
    }


@app.get("/files/{conversation_id}/{filename}")
async def serve_file(conversation_id: str, filename: str):
    """Serve files generated during conversation (e.g., matplotlib images)."""
    from sabre.common.paths import get_files_dir

    # Files are stored in XDG_DATA_HOME/sabre/files/{conversation_id}/
    files_dir = get_files_dir(conversation_id)
    file_path = files_dir / filename

    if not file_path.exists():
        return {"detail": "Not Found"}, 404

    return FileResponse(file_path)


@app.websocket("/message")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for chat messages.

    Client sends: {"type": "message", "content": "user message text"}
    Server sends: {"type": "token", "content": "token text"}
                  {"type": "complete", "content": "full response"}
                  {"type": "error", "content": "error message"}
    """
    # Generate client ID (in production, use proper auth)
    import uuid

    client_id = str(uuid.uuid4())

    await manager.connect(websocket, client_id)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            logger.info(f"Received WebSocket message: {data}")

            if data.get("type") == "message":
                user_message = data.get("content", "")
                logger.info(f"Processing user message: {user_message}")

                if not user_message:
                    await manager.send_message(client_id, {"type": "error", "content": "Empty message"})
                    continue

                # Create event callback to stream tokens and orchestration events
                async def event_callback(event: Event):
                    # Send raw event to client (all communication via events)
                    await manager.send_event(client_id, event)

                # Process message
                try:
                    session = manager.sessions[client_id]
                    tree = session["tree"]

                    # Create execution node for this request
                    node = tree.push(ExecutionNodeType.CLIENT_REQUEST, metadata={"message": user_message})

                    # Create task for orchestration (so we can cancel it)
                    # Load instructions if creating new conversation
                    instructions = None
                    if session["conversation_id"] is None:
                        instructions = manager.orchestrator.load_default_instructions()

                    task = asyncio.create_task(
                        manager.orchestrator.run(
                            conversation_id=session["conversation_id"],  # None for first message
                            input_text=user_message,
                            tree=tree,
                            instructions=instructions,  # Required for new conversations
                            event_callback=event_callback,
                        )
                    )
                    # Track the task for cancellation
                    manager.running_tasks[client_id] = task

                    # Wait for task to complete
                    try:
                        result = await task

                        # Save conversation_id from orchestration result
                        if result.conversation_id:
                            session["conversation_id"] = result.conversation_id

                    except asyncio.CancelledError:
                        logger.info(f"Orchestration cancelled for client {client_id}")
                        # Send cancelled event
                        tree_context = {
                            "node_id": node.id,
                            "parent_id": node.parent_id,
                            "depth": tree.get_depth(),
                            "path": [n.id for n in tree.get_path()],
                            "conversation_id": session["conversation_id"],
                        }
                        await event_callback(CancelledEvent(**tree_context))
                        tree.pop(ExecutionStatus.ERROR)
                        continue
                    finally:
                        # Remove task from tracking
                        if client_id in manager.running_tasks:
                            del manager.running_tasks[client_id]

                    # Mark client request node as completed
                    tree.pop(ExecutionStatus.COMPLETED)

                    # Note: Complete event is sent by orchestrator via event_callback

                except asyncio.CancelledError:
                    # Task was cancelled - already handled above
                    logger.info(f"Task cancelled for client {client_id}")
                    raise

                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)

                    # Build tree context for error event
                    try:
                        if "tree" in locals() and "node" in locals():
                            tree.pop(ExecutionStatus.ERROR)
                            tree_context = {
                                "node_id": node.id,
                                "parent_id": node.parent_id,
                                "depth": tree.get_depth(),
                                "path": [n.id for n in tree.get_path()],
                                "conversation_id": session.get("conversation_id", ""),
                            }
                            # Send error event to client
                            await event_callback(
                                ErrorEvent(**tree_context, error_message=str(e), error_type=type(e).__name__)
                            )
                        else:
                            # Fallback: send simple error message
                            await manager.send_message(client_id, {"type": "error", "content": str(e)})
                    except Exception as error_handling_error:
                        logger.error(f"Error while handling error: {error_handling_error}")
                        # Last resort: try to send simple message
                        try:
                            await manager.send_message(client_id, {"type": "error", "content": f"Error: {str(e)}"})
                        except:
                            pass  # Give up if we can't send

            elif data.get("type") == "ping":
                await manager.send_message(client_id, {"type": "pong"})

            elif data.get("type") == "clear_conversation":
                # Clear conversation by resetting conversation_id
                logger.info(f"Clearing conversation for client {client_id}")
                session = manager.sessions[client_id]
                session["conversation_id"] = None
                await manager.send_message(client_id, {"type": "conversation_cleared"})

            elif data.get("type") == "cancel":
                # Cancel running orchestration
                logger.info(f"Cancellation requested by client {client_id}")
                if client_id in manager.running_tasks:
                    task = manager.running_tasks[client_id]
                    task.cancel()
                    logger.info(f"Cancelled task for client {client_id}")
                else:
                    logger.warning(f"No running task found for client {client_id}")

    except WebSocketDisconnect as e:
        logger.info(f"WebSocket disconnected: {e}")
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        manager.disconnect(client_id)


if __name__ == "__main__":
    import uvicorn

    # Configure logging
    # Use LOG_LEVEL env var (DEBUG, INFO, WARNING, ERROR) or default to INFO
    import os

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
