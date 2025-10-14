"""
sabre_call helper - Recursive execution.

Provides SABRE's core recursive execution capability - delegates tasks
to a new orchestration session with ALL helpers enabled.

This is SABRE's equivalent of LLMVM's llmvm_call/delegate_task.
"""

import logging
from typing import Any, Callable

from sabre.common.models.messages import Content, TextContent

logger = logging.getLogger(__name__)


class SabreCall:
    """
    Recursive SABRE orchestration helper.

    Enables the LLM to delegate subtasks with full helper access,
    implementing SABRE's recursive execution model.
    """

    def __init__(self, get_orchestrator: Callable, get_tree: Callable, get_event_callback: Callable):
        """
        Initialize with orchestrator/tree access.

        Args:
            get_orchestrator: Lambda that returns current orchestrator
            get_tree: Lambda that returns current execution tree
            get_event_callback: Lambda that returns event callback
        """
        self.get_orchestrator = get_orchestrator
        self.get_tree = get_tree
        self.get_event_callback = get_event_callback

    async def __call__(
        self,
        task_description: str,
        expr_list: list[Any],
        include_original_task: bool = True,
    ) -> list[Content]:
        """
        Delegate a task to a new SABRE orchestration session.

        Creates a new conversation with ALL helpers enabled and executes
        the task recursively. This is SABRE's core recursive execution mechanism.

        Args:
            task_description: Natural language description of the subtask
            expr_list: Context data to pass to the task (can include Content objects)
            include_original_task: Whether to include high-level task context

        Returns:
            list[Content] with the task results (can include TextContent, ImageContent, etc.)

        Example:
            # Parallelize tasks
            tasks = [
                sabre_call("Analyze CNN headlines", cnn_results),
                sabre_call("Analyze BBC headlines", bbc_results)
            ]
            cnn_analysis, bbc_analysis = await asyncio.gather(*tasks)
        """
        logger.info(f"sabre_call: Delegating task: {task_description}")
        logger.debug(f"sabre_call: expr_list has {len(expr_list)} items")

        # Get orchestrator reference
        orchestrator = self.get_orchestrator()
        if not orchestrator:
            raise RuntimeError("sabre_call() requires orchestrator reference (not set)")

        # Get execution tree and event callback for context
        tree = self.get_tree()
        event_callback = self.get_event_callback()

        # Build context from expr_list - convert to Content objects
        context_content = []
        for item in expr_list:
            if isinstance(item, Content):
                context_content.append(item)
            elif isinstance(item, str):
                context_content.append(TextContent(item))
            else:
                # Convert other types to string
                context_content.append(TextContent(str(item)))

        # Build user message combining context + task
        # Format: "[Context data]\n\nTask: {task_description}"
        context_text = "\n\n".join(c.get_str() for c in context_content)
        user_message = f"{context_text}\n\nTask: {task_description}"

        logger.info(f"sabre_call: Calling orchestrator with {len(user_message)} chars")

        # Push nested call node to execution tree
        if tree:
            from sabre.common import ExecutionNodeType
            tree.push(
                ExecutionNodeType.NESTED_LLM_CALL,
                metadata={
                    "helper": "sabre_call",
                    "task": task_description[:100],
                }
            )

        try:
            # Create new conversation with default instructions
            # This gives the recursive call full access to helpers
            instructions = orchestrator.load_default_instructions()

            # Run orchestration in new conversation (recursive!)
            result = await orchestrator.run(
                conversation_id=None,  # New conversation
                input_text=user_message,
                tree=tree,
                instructions=instructions,
                event_callback=event_callback,
            )

            if tree:
                from sabre.common import ExecutionStatus
                tree.pop(ExecutionStatus.COMPLETED)

            if not result.success:
                logger.error(f"sabre_call failed: {result.error}")
                return [TextContent(f"ERROR: {result.error}")]

            # Parse final_response to extract structured content (text + any image URLs)
            # The final_response may contain markdown image URLs like ![alt](url)
            # For now, return as TextContent - could be enhanced to parse images
            logger.info(f"sabre_call succeeded: {len(result.final_response)} chars")

            # Return as list[Content] for flexibility
            # This allows subtasks to return images, files, etc. in the future
            return [TextContent(result.final_response)]

        except Exception as e:
            logger.error(f"sabre_call exception: {e}", exc_info=True)
            if tree:
                from sabre.common import ExecutionStatus
                tree.pop(ExecutionStatus.ERROR)
            raise RuntimeError(f"sabre_call failed: {e}")
