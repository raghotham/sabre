"""
Pandas Bind helper - convert data to pandas DataFrame using LLM.

Allows code to convert arbitrary data into structured DataFrames.
"""
import asyncio
import logging
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
            if expr.endswith('.csv') or 'csv' in expr.lower():
                return pd.read_csv(expr)

        from sabre.server.helpers.llm_call import run_async_from_sync
        return run_async_from_sync(self.execute(expr))

    async def execute(self, expr: Any) -> pd.DataFrame:
        """
        Async implementation of pandas_bind using LLM.

        Args:
            expr: Expression to convert to DataFrame

        Returns:
            DataFrame
        """
        prompt = PromptLoader.load('pandas_bind.prompt', template={})

        # Combine system_message and user_message as instructions
        system_instructions = f"{prompt['system_message']}\n\n{prompt['user_message']}"

        # Get or create OpenAI client (respects env vars)
        client = self.get_openai_client()

        conversation = await client.conversations.create(
            metadata={"type": "pandas_bind"}
        )

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

        from sabre.common import ExecutionTree, ExecutionNodeType, ExecutionStatus
        tree = ExecutionTree()
        node = tree.push(ExecutionNodeType.NESTED_LLM_CALL, metadata={"helper": "pandas_bind"})

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
            tree.pop(ExecutionStatus.COMPLETED)

            # Try to parse as Python dict/list and convert to DataFrame
            import ast
            try:
                data = ast.literal_eval(result_text)
                return pd.DataFrame(data)
            except (ValueError, SyntaxError) as e:
                logger.warning(f"Failed to parse LLM response as data: {e}")
                # Try to construct DataFrame from text directly
                try:
                    # Maybe it's JSON
                    import json
                    data = json.loads(result_text)
                    return pd.DataFrame(data)
                except:
                    logger.error(f"Could not convert to DataFrame: {result_text[:200]}")
                    return pd.DataFrame()  # Return empty DataFrame

        except Exception as e:
            tree.pop(ExecutionStatus.ERROR)
            logger.error(f"pandas_bind failed: {e}")
            return pd.DataFrame()  # Return empty DataFrame on failure
