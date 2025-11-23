"""
Execution Context - Thread-safe context passing for helper execution.

Uses Python's contextvars to pass execution context from orchestrator to helpers
in a clean, type-safe, async-aware way.
"""

import asyncio
from contextvars import ContextVar
from typing import Callable, Awaitable, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from sabre.common import Event, ExecutionTree


@dataclass
class ExecutionContext:
    """
    Execution context passed from orchestrator to helpers.

    Contains all the context needed for helpers to emit events and track execution.
    """

    event_callback: Optional[Callable[["Event"], Awaitable[None]]]
    tree: Optional["ExecutionTree"]
    tree_context: dict
    conversation_id: str
    session_id: str
    loop: Optional[asyncio.AbstractEventLoop] = None


# Context variable (async-safe, thread-aware)
_execution_context_var: ContextVar[Optional[ExecutionContext]] = ContextVar("execution_context", default=None)


def get_execution_context() -> Optional[ExecutionContext]:
    """
    Get current execution context.

    Returns None if not set (e.g., helper called outside orchestrator).

    Returns:
        ExecutionContext or None
    """
    return _execution_context_var.get()


def set_execution_context(
    event_callback: Optional[Callable[["Event"], Awaitable[None]]],
    tree: Optional["ExecutionTree"],
    tree_context: dict,
    conversation_id: str,
    session_id: str,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> None:
    """
    Set execution context for current async task.

    Args:
        event_callback: Callback to emit events to client
        tree: Execution tree for tracking
        tree_context: Tree context metadata
        conversation_id: OpenAI conversation ID
        session_id: Session ID for logging (required)
        loop: Event loop that should run async helper coroutines
    """
    ctx = ExecutionContext(
        event_callback=event_callback,
        tree=tree,
        tree_context=tree_context,
        conversation_id=conversation_id,
        session_id=session_id,
        loop=loop,
    )
    _execution_context_var.set(ctx)


def clear_execution_context() -> None:
    """
    Clear execution context.

    Note: This is optional - contextvars automatically scope to the async task,
    so context is automatically cleaned up when the task exits.
    """
    _execution_context_var.set(None)
