"""
Execution Tree for tracking the call hierarchy.

The ExecutionTree explicitly tracks every operation:
- CLIENT_REQUEST: User sends a message
- RESPONSE_ROUND: LLM inference round
- HELPERS_EXECUTION: Running <helpers> code
- NESTED_LLM_CALL: llm_call() from within helper

Every node has explicit parent-child relationships, making the
entire call tree traceable.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class ExecutionNodeType(Enum):
    """Types of execution nodes"""

    CLIENT_REQUEST = "client_request"  # User message
    RESPONSE_ROUND = "response_round"  # LLM inference
    HELPERS_EXECUTION = "helpers_execution"  # Executing <helpers>
    NESTED_LLM_CALL = "nested_llm_call"  # llm_call() from helper
    HELPER_FUNCTION = "helper_function"  # Individual helper call


class ExecutionStatus(Enum):
    """Status of execution node"""

    PENDING = "pending"  # Not started
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Finished successfully
    ERROR = "error"  # Failed with error
    CANCELLED = "cancelled"  # User cancelled


@dataclass
class ExecutionNode:
    """
    A node in the execution tree.

    Represents one operation (LLM call, helper execution, etc.)
    with explicit parent-child relationships.
    """

    id: str
    parent_id: Optional[str]
    node_type: ExecutionNodeType
    status: ExecutionStatus
    start_time: datetime
    end_time: Optional[datetime]
    metadata: dict = field(default_factory=dict)
    children: list["ExecutionNode"] = field(default_factory=list)

    def duration_ms(self) -> float:
        """Get duration in milliseconds"""
        if not self.end_time:
            return 0.0
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    def is_complete(self) -> bool:
        """Check if node is complete"""
        return self.status in (ExecutionStatus.COMPLETED, ExecutionStatus.ERROR, ExecutionStatus.CANCELLED)


class ExecutionTree:
    """
    Manages the execution tree.

    Provides context-manager-style push/pop for entering/exiting nodes.
    """

    def __init__(self):
        self.root: Optional[ExecutionNode] = None
        self.current: Optional[ExecutionNode] = None
        self.nodes: dict[str, ExecutionNode] = {}

    def create_node(
        self, node_type: ExecutionNodeType, metadata: dict | None = None, parent_id: str | None = None
    ) -> ExecutionNode:
        """
        Create a new execution node.

        If parent_id is not specified, uses current node as parent.
        """
        node = ExecutionNode(
            id=str(uuid.uuid4()),
            parent_id=parent_id or (self.current.id if self.current else None),
            node_type=node_type,
            status=ExecutionStatus.PENDING,
            start_time=datetime.now(),
            end_time=None,
            metadata=metadata or {},
        )
        self.nodes[node.id] = node
        return node

    def push(self, node_type: ExecutionNodeType, metadata: dict | None = None) -> ExecutionNode:
        """
        Create and enter a new execution context.

        The new node becomes the current node and is added as a child
        of the previous current node.
        """
        node = self.create_node(node_type, metadata)

        # Add as child of current node
        if self.current:
            self.current.children.append(node)
        else:
            # This is the root node
            self.root = node

        # Make this the current node
        self.current = node
        node.status = ExecutionStatus.RUNNING
        node.start_time = datetime.now()

        return node

    def pop(self, status: ExecutionStatus = ExecutionStatus.COMPLETED) -> Optional[ExecutionNode]:
        """
        Exit the current execution context.

        Marks the current node as complete and returns to parent.
        """
        if not self.current:
            raise RuntimeError("No current node to pop")

        # Mark node as complete
        self.current.status = status
        self.current.end_time = datetime.now()

        # Return to parent
        parent_node = None
        if self.current.parent_id:
            parent_node = self.nodes.get(self.current.parent_id)
            self.current = parent_node
        else:
            # Popped root node
            self.current = None

        return parent_node

    def get_path(self, node_id: str | None = None) -> list[ExecutionNode]:
        """
        Get path from root to specified node.

        If node_id is None, uses current node.
        """
        if node_id is None:
            if not self.current:
                return []
            node_id = self.current.id

        path = []
        node = self.nodes.get(node_id)

        while node:
            path.insert(0, node)
            node = self.nodes.get(node.parent_id) if node.parent_id else None

        return path

    def get_depth(self, node_id: str | None = None) -> int:
        """Get nesting depth of node (0 for root)"""
        return len(self.get_path(node_id)) - 1

    def get_round_id(self) -> str:
        """
        Get hierarchical round ID like '1', '1.2', '1.2.3'.

        Counts RESPONSE_ROUND nodes in the path.
        """
        path = self.get_path()
        round_numbers = []

        for node in path:
            if node.node_type == ExecutionNodeType.RESPONSE_ROUND:
                round_numbers.append(1)
            elif round_numbers and node.node_type in (
                ExecutionNodeType.HELPERS_EXECUTION,
                ExecutionNodeType.NESTED_LLM_CALL,
            ):
                # Increment the last round number
                round_numbers[-1] += 1

        if not round_numbers:
            return "1"

        return ".".join(str(n) for n in round_numbers)

    def to_dict(self) -> dict:
        """Serialize tree to dictionary"""

        def node_to_dict(node: ExecutionNode) -> dict:
            return {
                "id": node.id,
                "parent_id": node.parent_id,
                "type": node.node_type.value,
                "status": node.status.value,
                "start_time": node.start_time.isoformat(),
                "end_time": node.end_time.isoformat() if node.end_time else None,
                "duration_ms": node.duration_ms(),
                "metadata": node.metadata,
                "children": [node_to_dict(c) for c in node.children],
            }

        if not self.root:
            return {}

        return node_to_dict(self.root)

    def visualize(self, node: ExecutionNode | None = None, indent: int = 0) -> str:
        """
        Create text visualization of tree.

        Example output:
        ┌─ [CLIENT_REQUEST] User message
        │  ├─ [RESPONSE_ROUND:1] gpt-4o (500ms)
        │  │  └─ [HELPERS_EXECUTION] (100ms)
        │  │     └─ [NESTED_LLM_CALL] llm_call(...)
        │  │        └─ [RESPONSE_ROUND:1.1] gpt-4o (300ms)
        │  └─ [RESPONSE_ROUND:2] gpt-4o (200ms)
        """
        if node is None:
            if not self.root:
                return "(empty tree)"
            node = self.root

        lines = []
        prefix = "│  " * indent

        # Node info
        type_name = node.node_type.value.upper()
        status_icon = {
            ExecutionStatus.PENDING: "⏸",
            ExecutionStatus.RUNNING: "▶",
            ExecutionStatus.COMPLETED: "✓",
            ExecutionStatus.ERROR: "✗",
            ExecutionStatus.CANCELLED: "⊗",
        }.get(node.status, "?")

        duration = f" ({node.duration_ms():.0f}ms)" if node.is_complete() else ""
        metadata_str = ""
        if "round" in node.metadata:
            metadata_str = f":{node.metadata['round']}"
        elif "model" in node.metadata:
            metadata_str = f" {node.metadata['model']}"

        line = f"{prefix}{'┌─' if indent == 0 else '├─'} [{type_name}{metadata_str}] {status_icon}{duration}"
        lines.append(line)

        # Children
        for i, child in enumerate(node.children):
            child_lines = self.visualize(child, indent + 1)
            lines.append(child_lines)

        return "\n".join(lines)
