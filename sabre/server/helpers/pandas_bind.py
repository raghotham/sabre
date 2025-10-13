"""
Pandas Bind helper - convert data to pandas DataFrame using LLM.

Allows code to convert arbitrary data into structured DataFrames.
"""

import logging
import time
from typing import Any, Callable

import pandas as pd

from sabre.common.utils.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)


class PandasBind:
    """
    Bind expression to Pandas DataFrame.
    """

    def __init__(self, get_orchestrator: Callable, get_openai_client: Callable):
        """
        Initialize PandasBind helper.

        Args:
            get_orchestrator: Function to get orchestrator instance
            get_openai_client: Function to get/create OpenAI client
        """
        self.get_orchestrator = get_orchestrator
        self.get_openai_client = get_openai_client

    def __call__(self, expr: Any) -> pd.DataFrame:
        """
        Bind expression to DataFrame (sync wrapper).

        Examples:
            df = pandas_bind("data.csv")
            df = pandas_bind(html_table)
            df = pandas_bind(json_data)

        Args:
            expr: Expression (URL, data, etc.)

        Returns:
            DataFrame
        """
        logger.debug(f"pandas_bind({str(expr)[:50]})")

        # Handle CSV files/URLs directly
        if isinstance(expr, str):
            if expr.endswith(".csv") or "csv" in expr.lower():
                return pd.read_csv(expr)

        from sabre.server.helpers.llm_call import run_async_from_sync

        return run_async_from_sync(self.execute(expr))

    async def execute(self, expr: Any, max_retries: int = 3) -> pd.DataFrame:
        """
        Async implementation of pandas_bind using LLM.

        Args:
            expr: Expression to convert to DataFrame
            max_retries: Maximum retry attempts

        Returns:
            DataFrame
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

        prompt = PromptLoader.load("pandas_bind.prompt", template={})

        # Combine system_message and user_message as instructions
        system_instructions = f"{prompt['system_message']}\n\n{prompt['user_message']}"

        # Get or create OpenAI client (respects env vars)
        client = self.get_openai_client()

        conversation = await client.conversations.create(metadata={"type": "pandas_bind"})

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

        # Format input with data
        input_text = f"### Data\n{str(expr)}"

        from sabre.common import ExecutionNodeType, ExecutionStatus, NestedCallStartEvent, NestedCallEndEvent

        # Retry loop
        for attempt in range(max_retries):
            logger.info(f"pandas_bind attempt {attempt + 1}/{max_retries}")

            # Push tree node if tree is available
            if tree:
                node = tree.push(
                    ExecutionNodeType.NESTED_LLM_CALL, metadata={"helper": "pandas_bind", "attempt": attempt + 1}
                )
                # Build tree context from orchestrator's _build_tree_context logic
                orchestrator = self.get_orchestrator()
                tree_context = orchestrator._build_tree_context(
                    tree, node, parent_tree_context.get("conversation_id", "")
                )
            else:
                # Fallback if tree not available (shouldn't happen in normal execution)
                tree_context = {
                    "node_id": "pandas_bind",
                    "parent_id": None,
                    "depth": 0,
                    "path": [],
                    "conversation_id": parent_tree_context.get("conversation_id", ""),
                    "path_summary": "pandas_bind()",
                }

            # Emit start event
            if event_callback:
                await event_callback(
                    NestedCallStartEvent(
                        **tree_context,
                        caller="pandas_bind",
                        instruction=f"Convert to DataFrame: {str(expr)[:50]}",
                    )
                )

            start_time = time.time()

            try:
                # Call LLM to convert to DataFrame structure (pass instructions on every call)
                response = await client.responses.create(
                    model="gpt-4o",
                    conversation=conversation.id,
                    instructions=system_instructions,  # Must send on every call
                    input=input_text,
                    max_output_tokens=4000,
                    stream=False,
                    truncation="auto",
                )

                # Extract result from response
                result_text = response.content.text.strip()

                # Try to parse as Python dict/list and convert to DataFrame
                import ast

                try:
                    data = ast.literal_eval(result_text)
                    df = pd.DataFrame(data)
                    duration_ms = (time.time() - start_time) * 1000

                    if tree:
                        tree.pop(ExecutionStatus.COMPLETED)

                    # Emit end event
                    if event_callback:
                        await event_callback(
                            NestedCallEndEvent(
                                **tree_context,
                                result=f"DataFrame {df.shape[0]}x{df.shape[1]}",
                                duration_ms=duration_ms,
                                success=True,
                            )
                        )

                    return df

                except (ValueError, SyntaxError) as e:
                    logger.warning(f"Failed to parse LLM response as data: {e}")

                    # Try to construct DataFrame from text directly
                    try:
                        # Maybe it's JSON
                        import json

                        data = json.loads(result_text)
                        df = pd.DataFrame(data)
                        duration_ms = (time.time() - start_time) * 1000

                        if tree:
                            tree.pop(ExecutionStatus.COMPLETED)

                        # Emit end event
                        if event_callback:
                            await event_callback(
                                NestedCallEndEvent(
                                    **tree_context,
                                    result=f"DataFrame {df.shape[0]}x{df.shape[1]}",
                                    duration_ms=duration_ms,
                                    success=True,
                                )
                            )

                        return df

                    except Exception as json_err:
                        duration_ms = (time.time() - start_time) * 1000

                        logger.error(f"Could not convert to DataFrame: {result_text[:200]}")

                        if tree:
                            tree.pop(ExecutionStatus.ERROR)

                        # Emit end event with failure
                        if event_callback:
                            await event_callback(
                                NestedCallEndEvent(
                                    **tree_context,
                                    result=f"Parse error: {str(json_err)[:50]}",
                                    duration_ms=duration_ms,
                                    success=False,
                                )
                            )

                        # Retry on parse failure
                        if attempt < max_retries - 1:
                            continue
                        else:
                            return pd.DataFrame()  # Return empty DataFrame after all retries

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                if tree:
                    tree.pop(ExecutionStatus.ERROR)

                logger.error(f"pandas_bind attempt {attempt + 1} failed: {e}")

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
                    logger.error(f"pandas_bind failed after {max_retries} attempts")
                    return pd.DataFrame()  # Return empty DataFrame on failure
                # Continue to next retry

        return pd.DataFrame()  # Fallback to empty DataFrame
