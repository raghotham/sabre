"""
Events for client-server communication.

All execution operations emit events that are streamed to the client.
Every event includes execution tree context (node_id, parent_id, depth, path).
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional
import json


class EventType(Enum):
    """Types of events"""

    # Response events
    RESPONSE_START = "response_start"
    RESPONSE_TOKEN = "response_token"
    RESPONSE_THINKING_TOKEN = "response_thinking_token"
    RESPONSE_TEXT = "response_text"
    RESPONSE_RETRY = "response_retry"

    # Helpers events
    HELPERS_EXECUTION_START = "helpers_execution_start"
    HELPERS_EXECUTION_END = "helpers_execution_end"

    # Nested LLM call events (for coerce, etc.)
    NESTED_CALL_START = "nested_call_start"
    NESTED_CALL_END = "nested_call_end"

    # Session events
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class Event:
    """
    Base event class.

    All events include execution tree context:
    - node_id: Current execution node
    - parent_id: Parent execution node
    - depth: Nesting depth
    - path: Full path from root to current
    - path_summary: Human-readable execution path (e.g., "User: hello → Response #1 → Helper #1")
    - conversation_id: OpenAI conversation ID for this context
    - timestamp: When event occurred
    - data: Event-specific data
    """

    type: EventType
    node_id: str
    parent_id: Optional[str]
    depth: int
    path: list[str]
    conversation_id: str
    timestamp: datetime
    data: dict = field(default_factory=dict)
    path_summary: str = field(default="")  # Human-readable execution path

    def __init__(self, type, node_id, parent_id, depth, path, conversation_id, timestamp, data=None, path_summary=""):
        """Initialize event, accepting path_summary from tree context."""
        self.type = type
        self.node_id = node_id
        self.parent_id = parent_id
        self.depth = depth
        self.path = path
        self.conversation_id = conversation_id
        self.timestamp = timestamp
        self.data = data if data is not None else {}
        self.path_summary = path_summary

    def to_dict(self) -> dict:
        """Convert event to dict, handling enums and Content objects"""

        def convert_value(obj):
            """Recursively convert values to JSON-serializable types"""
            from sabre.common.models.messages import Content

            if isinstance(obj, Content):
                # Serialize Content objects
                return {"type": obj.type.value, "data": obj.data}
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_value(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_value(v) for v in obj]
            else:
                return obj

        d = asdict(self)
        # Convert all values recursively
        return {k: convert_value(v) for k, v in d.items()}

    def to_json(self) -> str:
        """Convert event to JSON"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "Event":
        """Parse event from JSON"""
        d = json.loads(json_str)
        d["type"] = EventType(d["type"])
        d["timestamp"] = datetime.fromisoformat(d["timestamp"])

        # Determine specific event class based on type
        event_classes = {
            EventType.RESPONSE_START: ResponseStartEvent,
            EventType.RESPONSE_TOKEN: ResponseTokenEvent,
            EventType.RESPONSE_THINKING_TOKEN: ResponseThinkingTokenEvent,
            EventType.RESPONSE_TEXT: ResponseTextEvent,
            EventType.RESPONSE_RETRY: ResponseRetryEvent,
            EventType.HELPERS_EXECUTION_START: HelpersExecutionStartEvent,
            EventType.HELPERS_EXECUTION_END: HelpersExecutionEndEvent,
            EventType.NESTED_CALL_START: NestedCallStartEvent,
            EventType.NESTED_CALL_END: NestedCallEndEvent,
            EventType.COMPLETE: CompleteEvent,
            EventType.ERROR: ErrorEvent,
            EventType.CANCELLED: CancelledEvent,
        }

        event_class = event_classes.get(d["type"], Event)
        return event_class(**d)


@dataclass
class ResponseStartEvent(Event):
    """LLM response round starting"""

    def __init__(
        self,
        node_id: str,
        parent_id: Optional[str],
        depth: int,
        path: list[str],
        conversation_id: str,
        model: str,
        round_id: str,
        prompt_tokens: int,
        previous_response_id: Optional[str] = None,
        path_summary: str = "",
    ):
        super().__init__(
            type=EventType.RESPONSE_START,
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            path=path,
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            path_summary=path_summary,
            data={
                "model": model,
                "round_id": round_id,
                "prompt_tokens": prompt_tokens,
                "previous_response_id": previous_response_id,
            },
        )


@dataclass
class ResponseTokenEvent(Event):
    """LLM response token streamed"""

    def __init__(
        self,
        node_id: str,
        parent_id: Optional[str],
        depth: int,
        path: list[str],
        conversation_id: str,
        token: str,
        path_summary: str = "",
    ):
        super().__init__(
            type=EventType.RESPONSE_TOKEN,
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            path=path,
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            path_summary=path_summary,
            data={"token": token},
        )


@dataclass
class ResponseThinkingTokenEvent(Event):
    """LLM thinking token streamed"""

    def __init__(
        self,
        node_id: str,
        parent_id: Optional[str],
        depth: int,
        path: list[str],
        conversation_id: str,
        token: str,
        path_summary: str = "",
    ):
        super().__init__(
            type=EventType.RESPONSE_THINKING_TOKEN,
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            path=path,
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            path_summary=path_summary,
            data={"token": token},
        )


@dataclass
class ResponseTextEvent(Event):
    """Full response text after streaming completes, before helper extraction"""

    def __init__(
        self,
        node_id: str,
        parent_id: Optional[str],
        depth: int,
        path: list[str],
        conversation_id: str,
        text: str,
        text_length: int,
        has_helpers: bool,
        helper_count: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        reasoning_tokens: int = 0,
        path_summary: str = "",
    ):
        super().__init__(
            type=EventType.RESPONSE_TEXT,
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            path=path,
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            path_summary=path_summary,
            data={
                "text": text,
                "text_length": text_length,
                "has_helpers": has_helpers,
                "helper_count": helper_count,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
            },
        )


@dataclass
class ResponseRetryEvent(Event):
    """Rate limit hit, retrying after delay"""

    def __init__(
        self,
        node_id: str,
        parent_id: Optional[str],
        depth: int,
        path: list[str],
        conversation_id: str,
        attempt: int,
        max_retries: int,
        wait_seconds: float,
        reason: str,
        path_summary: str = "",
    ):
        super().__init__(
            type=EventType.RESPONSE_RETRY,
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            path=path,
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            path_summary=path_summary,
            data={
                "attempt": attempt,
                "max_retries": max_retries,
                "wait_seconds": wait_seconds,
                "reason": reason,
            },
        )


@dataclass
class HelpersExecutionStartEvent(Event):
    """Starting to execute <helpers> block"""

    def __init__(
        self,
        node_id: str,
        parent_id: Optional[str],
        depth: int,
        path: list[str],
        conversation_id: str,
        code: str,
        block_number: int = 1,
        helper_name: str = "unknown",
        path_summary: str = "",
    ):
        super().__init__(
            type=EventType.HELPERS_EXECUTION_START,
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            path=path,
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            path_summary=path_summary,
            data={"code": code, "block_number": block_number, "helper_name": helper_name},
        )


@dataclass
class HelpersExecutionEndEvent(Event):
    """Helper execution finished with results"""

    def __init__(
        self,
        node_id: str,
        parent_id: Optional[str],
        depth: int,
        path: list[str],
        conversation_id: str,
        duration_ms: float,
        success: bool,
        result: list,  # list[Content] from sabre.common.models.messages
        block_number: int = 1,
        helper_name: str = "unknown",
        path_summary: str = "",
    ):
        super().__init__(
            type=EventType.HELPERS_EXECUTION_END,
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            path=path,
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            path_summary=path_summary,
            data={
                "duration_ms": duration_ms,
                "success": success,
                "result": result,  # Includes text output and image URLs
                "block_number": block_number,
                "helper_name": helper_name,
            },
        )


@dataclass
class NestedCallStartEvent(Event):
    """Starting nested LLM call (e.g., from coerce)"""

    def __init__(
        self,
        node_id: str,
        parent_id: Optional[str],
        depth: int,
        path: list[str],
        conversation_id: str,
        caller: str,
        instruction: str,
        path_summary: str = "",
    ):
        super().__init__(
            type=EventType.NESTED_CALL_START,
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            path=path,
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            path_summary=path_summary,
            data={
                "caller": caller,
                "instruction": instruction,
            },
        )


@dataclass
class NestedCallEndEvent(Event):
    """Finished nested LLM call"""

    def __init__(
        self,
        node_id: str,
        parent_id: Optional[str],
        depth: int,
        path: list[str],
        conversation_id: str,
        result: str,
        duration_ms: float,
        success: bool = True,
        path_summary: str = "",
    ):
        super().__init__(
            type=EventType.NESTED_CALL_END,
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            path=path,
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            path_summary=path_summary,
            data={
                "result": result,
                "duration_ms": duration_ms,
                "success": success,
            },
        )


@dataclass
class CompleteEvent(Event):
    """Request completed successfully"""

    def __init__(
        self,
        node_id: str,
        parent_id: Optional[str],
        depth: int,
        path: list[str],
        conversation_id: str,
        final_message: str,
        session_id: str = "",
        workspace_dir: str = "",
        path_summary: str = "",
    ):
        super().__init__(
            type=EventType.COMPLETE,
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            path=path,
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            path_summary=path_summary,
            data={
                "final_message": final_message,
                "session_id": session_id,
                "workspace_dir": workspace_dir,
            },
        )


@dataclass
class ErrorEvent(Event):
    """Error occurred"""

    def __init__(
        self,
        node_id: str,
        parent_id: Optional[str],
        depth: int,
        path: list[str],
        conversation_id: str,
        error_message: str,
        error_type: str,
        path_summary: str = "",
    ):
        super().__init__(
            type=EventType.ERROR,
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            path=path,
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            path_summary=path_summary,
            data={
                "error_message": error_message,
                "error_type": error_type,
            },
        )


@dataclass
class CancelledEvent(Event):
    """Execution cancelled by user"""

    def __init__(
        self,
        node_id: str,
        parent_id: Optional[str],
        depth: int,
        path: list[str],
        conversation_id: str,
        message: str = "Execution cancelled by user",
        path_summary: str = "",
    ):
        super().__init__(
            type=EventType.CANCELLED,
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            path=path,
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            path_summary=path_summary,
            data={
                "message": message,
            },
        )
