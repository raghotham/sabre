"""
LLM Bind helper - bind data to function arguments using LLM.

Allows code to extract structured data and bind it to function calls.
"""

import logging
import time
from typing import Any, Callable

from sabre.common.utils.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)


class LLMBind:
    """
    Bind data from expr to function arguments using LLM.
    """

    def __init__(self, get_orchestrator: Callable, get_openai_client: Callable, get_namespace: Callable):
        """
        Initialize LLMBind helper.

        Args:
            get_orchestrator: Function to get orchestrator instance
            get_openai_client: Function to get/create OpenAI client
            get_namespace: Function to get Python namespace for executing bound calls
        """
        self.get_orchestrator = get_orchestrator
        self.get_openai_client = get_openai_client
        self.get_namespace = get_namespace

    def __call__(self, expr: Any, func_str: str) -> Any:
        """
        Bind data from expr to function arguments (sync wrapper).

        Example:
            expr = "The CEO of AMD is Lisa Su"
            func_str = "get_person_info(first_name, last_name, company)"
            # Returns: get_person_info("Lisa", "Su", "AMD")

        Args:
            expr: Data to extract values from
            func_str: Function signature to bind to

        Returns:
            Result of executing the bound function
        """
        logger.debug(f"llm_bind({str(expr)[:50]}, {func_str})")

        from sabre.server.helpers.llm_call import run_async_from_sync

        return run_async_from_sync(self.execute(expr, func_str))

    async def execute(self, expr: Any, func_str: str, max_retries: int = 5) -> Any:
        """
        Async implementation of llm_bind with retry logic.

        Args:
            expr: Data to bind from
            func_str: Function signature to bind to
            max_retries: Maximum number of retry attempts

        Returns:
            Result of executing bound function, or None on failure
        """
        # Get execution context (set by orchestrator during helper execution)
        from sabre.common.execution_context import get_execution_context

        ctx = get_execution_context()
        if ctx:
            event_callback = ctx.event_callback
            tree = ctx.tree
            parent_tree_context = ctx.tree_context
        else:
            # Fallback for direct calls outside orchestrator
            event_callback = None
            tree = None
            parent_tree_context = {}

        # Load prompt
        prompt = PromptLoader.load(
            "llm_bind_global.prompt",
            template={
                "function_definition": func_str,
            },
        )

        # Combine system_message and user_message as instructions
        system_instructions = f"{prompt['system_message']}\n\n{prompt['user_message']}"

        # Get or create OpenAI client (respects env vars)
        client = self.get_openai_client()

        conversation = await client.conversations.create(metadata={"type": "llm_bind", "function": func_str[:100]})

        # Send initial message with instructions
        await client.responses.create(
            model="gpt-4o",
            conversation=conversation.id,
            instructions=system_instructions,  # Pass as instructions
            input="Are you ready?",
            max_output_tokens=100,
            stream=False,
            truncation="auto",
        )

        # Initial input with data
        input_text = f"### Data to bind from\n{str(expr)}"

        from sabre.common import ExecutionNodeType, ExecutionStatus, NestedCallStartEvent, NestedCallEndEvent

        # Retry loop
        for attempt in range(max_retries):
            logger.info(f"llm_bind attempt {attempt + 1}/{max_retries}")

            # Push tree node if tree is available
            if tree:
                node = tree.push(
                    ExecutionNodeType.NESTED_LLM_CALL, metadata={"helper": "llm_bind", "attempt": attempt + 1}
                )
                # Build tree context from orchestrator's _build_tree_context logic
                orchestrator = self.get_orchestrator()
                tree_context = orchestrator._build_tree_context(
                    tree, node, parent_tree_context.get("conversation_id", "")
                )
            else:
                # Fallback if tree not available (shouldn't happen in normal execution)
                tree_context = {
                    "node_id": "llm_bind",
                    "parent_id": None,
                    "depth": 0,
                    "path": [],
                    "conversation_id": parent_tree_context.get("conversation_id", ""),
                    "path_summary": "llm_bind()",
                }

            # Emit start event
            if event_callback:
                await event_callback(
                    NestedCallStartEvent(
                        **tree_context,
                        caller="llm_bind",
                        instruction=f"Bind '{str(expr)[:50]}' to {func_str[:50]}",
                    )
                )

            start_time = time.time()

            try:
                # Call LLM to bind arguments (pass instructions on every call)
                response = await client.responses.create(
                    model="gpt-4o",
                    conversation=conversation.id,
                    instructions=system_instructions,  # Must send on every call
                    input=input_text,
                    max_output_tokens=1000,
                    stream=False,
                    truncation="auto",
                )

                # Extract function call from response
                result_text = response.content.text.strip()

                # Look for function call in response (e.g., "get_person_info('Lisa', 'Su', 'AMD')")
                # Extract just the function call if embedded in text
                import re

                func_name = func_str.split("(")[0]
                match = re.search(rf"{func_name}\([^)]*\)", result_text)
                if match:
                    bound_call = match.group(0)
                else:
                    bound_call = result_text.strip()

                logger.info(f"llm_bind extracted call: {bound_call}")

                # Execute the bound function call
                namespace = self.get_namespace()
                try:
                    result = eval(bound_call, namespace)
                    duration_ms = (time.time() - start_time) * 1000

                    if tree:
                        tree.pop(ExecutionStatus.COMPLETED)

                    # Emit end event
                    if event_callback:
                        await event_callback(
                            NestedCallEndEvent(
                                **tree_context,
                                result=str(result)[:100],  # Truncate for event
                                duration_ms=duration_ms,
                                success=True,
                            )
                        )

                    return result
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000

                    logger.warning(f"Failed to execute bound call '{bound_call}': {e}")

                    if tree:
                        tree.pop(ExecutionStatus.ERROR)

                    # Emit end event with failure
                    if event_callback:
                        await event_callback(
                            NestedCallEndEvent(
                                **tree_context,
                                result=f"Execution error: {str(e)[:50]}",
                                duration_ms=duration_ms,
                                success=False,
                            )
                        )

                    # Retry on execution failure
                    continue

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                if tree:
                    tree.pop(ExecutionStatus.ERROR)

                logger.warning(f"llm_bind attempt {attempt + 1} failed: {e}")

                # Emit end event with failure
                if event_callback:
                    await event_callback(
                        NestedCallEndEvent(
                            **tree_context,
                            result=f"Error: {str(e)[:100]}",
                            duration_ms=duration_ms,
                            success=False,
                        )
                    )

                if attempt >= max_retries - 1:
                    logger.error(f"llm_bind failed after {max_retries} attempts")
                    return None  # Return None on failure
                # Continue to next retry

        return None  # Fallback to None
