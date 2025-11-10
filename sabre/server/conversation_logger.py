"""Conversation logging for review and analysis."""

import json
import logging
from datetime import datetime

from sabre.common.paths import get_logs_dir
from sabre.common.models.events import Event

logger = logging.getLogger(__name__)


class ConversationLogger:
    """Logs conversations and events for later review."""

    def __init__(self):
        """Initialize conversation logger."""
        # Store conversations in state directory
        self.conversations_dir = get_logs_dir() / "conversations"
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

    def log_event(self, conversation_id: str, event: Event):
        """
        Log an event to the conversation file.

        Args:
            conversation_id: Conversation ID
            event: Event to log
        """
        if not conversation_id:
            return

        conv_file = self.conversations_dir / f"{conversation_id}.jsonl"

        # Convert event to dict
        event_data = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event.type.value if hasattr(event, "type") else type(event).__name__,
            "node_id": getattr(event, "node_id", None),
            "depth": getattr(event, "depth", None),
            "data": getattr(event, "data", {}),
        }

        # Add event-specific fields
        if hasattr(event, "error_message"):
            event_data["error_message"] = event.error_message
        if hasattr(event, "error_type"):
            event_data["error_type"] = event.error_type

        # Write to file (append mode)
        with open(conv_file, "a") as f:
            f.write(json.dumps(event_data, default=str) + "\n")

    def get_conversation(self, conversation_id: str) -> list[dict]:
        """
        Load a conversation from disk.

        Args:
            conversation_id: Conversation ID

        Returns:
            List of event dicts
        """
        conv_file = self.conversations_dir / f"{conversation_id}.jsonl"

        if not conv_file.exists():
            return []

        events = []
        with open(conv_file) as f:
            for line in f:
                events.append(json.loads(line))

        return events

    def list_conversations(self, limit: int = 100) -> list[dict]:
        """
        List recent conversations.

        Args:
            limit: Maximum number of conversations to return

        Returns:
            List of conversation summaries
        """
        conversations = []

        # Get all conversation files
        for conv_file in sorted(
            self.conversations_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            if len(conversations) >= limit:
                break

            conversation_id = conv_file.stem

            # Read first and last events
            with open(conv_file) as f:
                lines = f.readlines()

            if not lines:
                continue

            first_event = json.loads(lines[0])
            last_event = json.loads(lines[-1])

            conversations.append(
                {
                    "conversation_id": conversation_id,
                    "start_time": first_event["timestamp"],
                    "end_time": last_event["timestamp"],
                    "event_count": len(lines),
                }
            )

        return conversations
