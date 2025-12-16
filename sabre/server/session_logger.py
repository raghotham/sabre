"""Session logging for execution tree visualization.

Logs complete execution trees with parent-child relationships for UI rendering.
Each session gets one JSONL file with all events from the entire tree.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from sabre.common.paths import get_sessions_base_dir, get_session_log_file, get_session_dir
from sabre.common.models.events import Event

logger = logging.getLogger(__name__)


class SessionLogger:
    """Logs execution trees by session for visualization."""

    def __init__(self):
        """Initialize session logger."""
        self.sessions_dir = get_sessions_base_dir()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def log_session_start(self, session_id: str, message: str):
        """
        Log the start of a session.

        Args:
            session_id: Session ID
            message: User's initial message
        """
        # Ensure session directory exists
        session_dir = get_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        # Create files subdirectory for session
        files_dir = session_dir / "files"
        files_dir.mkdir(parents=True, exist_ok=True)

        session_file = get_session_log_file(session_id)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "session_start",
            "session_id": session_id,
            "message": message,
        }

        with open(session_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_user_message(
        self,
        session_id: str,
        node_id: str,
        parent_id: Optional[str],
        depth: int,
        message: str,
    ):
        """
        Log a user message.

        Args:
            session_id: Session ID
            node_id: Execution node ID
            parent_id: Parent node ID (None for root)
            depth: Tree depth
            message: User message content
        """
        # Ensure session directory exists
        session_dir = get_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        session_file = get_session_log_file(session_id)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "user_message",
            "session_id": session_id,
            "node_id": node_id,
            "parent_id": parent_id,
            "depth": depth,
            "content": message,
        }

        with open(session_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_assistant_message(
        self,
        session_id: str,
        node_id: str,
        parent_id: Optional[str],
        depth: int,
        conversation_id: str,
        message: str,
    ):
        """
        Log an assistant message.

        Args:
            session_id: Session ID
            node_id: Execution node ID
            parent_id: Parent node ID
            depth: Tree depth
            conversation_id: OpenAI conversation ID
            message: Assistant message content
        """
        session_file = get_session_log_file(session_id)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "assistant_message",
            "session_id": session_id,
            "node_id": node_id,
            "parent_id": parent_id,
            "depth": depth,
            "conversation_id": conversation_id,
            "content": message,
        }

        with open(session_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_node_start(
        self,
        session_id: str,
        node_id: str,
        parent_id: Optional[str],
        depth: int,
        node_type: str,
        conversation_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Log the start of an execution node.

        Args:
            session_id: Session ID
            node_id: Execution node ID
            parent_id: Parent node ID
            depth: Tree depth
            node_type: Type of node (e.g., "response", "helper", "llm_call")
            conversation_id: OpenAI conversation ID (if applicable)
            metadata: Additional metadata
            system_prompt: System prompt/instructions used (if applicable)
        """
        session_file = get_session_log_file(session_id)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "node_start",
            "session_id": session_id,
            "node_id": node_id,
            "parent_id": parent_id,
            "depth": depth,
            "node_type": node_type,
            "conversation_id": conversation_id,
            "metadata": metadata or {},
            "system_prompt": system_prompt,
        }

        with open(session_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_node_output(
        self,
        session_id: str,
        node_id: str,
        output_type: str,
        content: str,
    ):
        """
        Log output from a node (e.g., helper result, token stream).

        Args:
            session_id: Session ID
            node_id: Execution node ID
            output_type: Type of output (e.g., "token", "helper_result", "error")
            content: Output content
        """
        session_file = get_session_log_file(session_id)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "node_output",
            "session_id": session_id,
            "node_id": node_id,
            "output_type": output_type,
            "content": content,
        }

        with open(session_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_node_complete(
        self,
        session_id: str,
        node_id: str,
        status: str,
        duration_ms: Optional[float] = None,
        tokens: Optional[dict] = None,
    ):
        """
        Log completion of a node.

        Args:
            session_id: Session ID
            node_id: Execution node ID
            status: Completion status ("success", "error", "cancelled")
            duration_ms: Duration in milliseconds
            tokens: Token usage stats (input, output, reasoning)
        """
        session_file = get_session_log_file(session_id)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "node_complete",
            "session_id": session_id,
            "node_id": node_id,
            "status": status,
            "duration_ms": duration_ms,
            "tokens": tokens or {},
        }

        with open(session_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_event(self, session_id: str, event: Event):
        """
        Log a generic event (for backward compatibility and flexibility).

        Args:
            session_id: Session ID
            event: Event object
        """
        session_file = get_session_log_file(session_id)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event.type.value if hasattr(event, "type") else type(event).__name__,
            "session_id": session_id,
            "node_id": getattr(event, "node_id", None),
            "parent_id": getattr(event, "parent_id", None),
            "depth": getattr(event, "depth", None),
            "conversation_id": getattr(event, "conversation_id", None),
            "data": getattr(event, "data", {}),
        }

        with open(session_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_file_saved(
        self,
        session_id: str,
        filename: str,
        file_path: str,
        file_type: str = "image",
        context: str = "",
        metadata: dict | None = None,
    ):
        """
        Log a file save event (e.g., screenshot, generated image, etc.).

        Args:
            session_id: Session ID
            filename: Filename (basename only)
            file_path: Full path to the file
            file_type: Type of file (image, pdf, csv, etc.)
            context: Context where file was saved (e.g., "llmcall_attachment", "helper_result")
            metadata: Additional metadata about the file
        """
        session_file = get_session_log_file(session_id)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "file_saved",
            "session_id": session_id,
            "filename": filename,
            "file_path": str(file_path),
            "file_type": file_type,
            "context": context,
            "metadata": metadata or {},
        }

        with open(session_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def get_session(self, session_id: str) -> list[dict]:
        """
        Load a session from disk.

        Args:
            session_id: Session ID

        Returns:
            List of event dicts
        """
        session_file = get_session_log_file(session_id)

        if not session_file.exists():
            return []

        events = []
        with open(session_file) as f:
            for line in f:
                events.append(json.loads(line))

        return events

    def list_sessions(self, limit: int = 100) -> list[dict]:
        """
        List recent sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session summaries
        """
        sessions = []

        # Get all session directories, sorted by modification time (newest first)
        if not self.sessions_dir.exists():
            return []

        session_dirs = [d for d in self.sessions_dir.iterdir() if d.is_dir()]

        for session_dir in sorted(
            session_dirs,
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            if len(sessions) >= limit:
                break

            session_id = session_dir.name
            session_file = session_dir / "session.jsonl"

            if not session_file.exists():
                continue

            # Read all entries
            with open(session_file) as f:
                lines = f.readlines()

            if not lines:
                continue

            first_entry = json.loads(lines[0])
            last_entry = json.loads(lines[-1])

            # Extract first user message for preview
            first_message = None
            for line in lines:
                entry = json.loads(line)
                if entry.get("event_type") in ["session_start", "user_message"]:
                    first_message = entry.get("message") or entry.get("content")
                    if first_message:
                        first_message = first_message[:100]
                        break

            sessions.append(
                {
                    "session_id": session_id,
                    "start_time": first_entry["timestamp"],
                    "end_time": last_entry["timestamp"],
                    "event_count": len(lines),
                    "preview": first_message,
                }
            )

        return sessions
