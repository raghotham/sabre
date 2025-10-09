"""
LLM List Bind helper - extract lists from data using LLM.

Allows code to extract structured lists from unstructured data.
"""
import asyncio
import logging
import re
from typing import Any, Callable

from sabre.common.utils.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)


class LLMListBind:
    """
    Bind expression to a list using LLM.
    """

    def __init__(self, get_orchestrator: Callable, get_openai_client: Callable):
        """
        Initialize LLMListBind helper.

        Args:
            get_orchestrator: Function to get orchestrator instance
            get_openai_client: Function to get/create OpenAI client
        """
        self.get_orchestrator = get_orchestrator
        self.get_openai_client = get_openai_client

    def __call__(self, expr: Any, llm_instruction: str, count: int = 999999) -> list:
        """
        Bind expression to a list using LLM (sync wrapper).

        Examples:
            prices = llm_list_bind(html_content, "extract all prices")
            emails = llm_list_bind(text, "find email addresses", count=10)

        Args:
            expr: Expression containing data
            llm_instruction: Instructions for what to extract
            count: Max items to extract

        Returns:
            List of extracted items
        """
        logger.debug(f"llm_list_bind({str(expr)[:50]}, {llm_instruction})")

        from sabre.server.helpers.llm_call import run_async_from_sync
        return run_async_from_sync(self.execute(expr, llm_instruction, count))

    async def execute(self, expr: Any, llm_instruction: str, count: int, max_retries: int = 3) -> list:
        """
        Async implementation of llm_list_bind with retry logic.

        Args:
            expr: Expression containing data
            llm_instruction: Instructions for extraction
            count: Max items to extract
            max_retries: Maximum retry attempts

        Returns:
            List of extracted items
        """
        # Load prompt
        prompt = PromptLoader.load(
            'llm_list_bind.prompt',
            template={
                'goal': llm_instruction.replace('"', ''),
                'context': str(expr),
                'type': 'str',  # Default to string type
            }
        )

        # Combine system_message and user_message as instructions
        system_instructions = f"{prompt['system_message']}\n\n{prompt['user_message']}"

        # Get or create OpenAI client (respects env vars)
        client = self.get_openai_client()

        conversation = await client.conversations.create(
            metadata={"type": "llm_list_bind"}
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

        # Initial input with data
        input_text = f"### Data\n{str(expr)}"

        from sabre.common import ExecutionTree, ExecutionNodeType, ExecutionStatus

        # Retry loop
        for attempt in range(max_retries):
            logger.info(f"llm_list_bind attempt {attempt + 1}/{max_retries}")

            tree = ExecutionTree()
            node = tree.push(
                ExecutionNodeType.NESTED_LLM_CALL,
                metadata={"helper": "llm_list_bind", "attempt": attempt + 1}
            )

            try:
                # Call LLM to extract list (pass instructions on every call)
                response = await client.responses.create(
                    model="gpt-4o",
                    conversation=conversation.id,
                    instructions=system_instructions,  # Must send on every call
                    input=input_text,
                    max_output_tokens=2000,
                    stream=False,
                    truncation="auto",
                )

                # Extract result from response
                result_text = response.content.text.strip()
                tree.pop(ExecutionStatus.COMPLETED)

                # Try to parse as Python list
                import ast
                try:
                    result = ast.literal_eval(result_text)
                    if isinstance(result, list):
                        # Apply count limit
                        return result[:count]
                    else:
                        logger.warning(f"LLM returned non-list: {type(result)}")
                        # Retry if not a list
                        continue
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"Failed to parse LLM response as list: {e}")
                    # Retry on parse failure
                    continue

            except Exception as e:
                tree.pop(ExecutionStatus.ERROR)
                logger.warning(f"llm_list_bind attempt {attempt + 1} failed: {e}")
                if attempt >= max_retries - 1:
                    logger.error(f"llm_list_bind failed after {max_retries} attempts")
                    return []  # Return empty list on failure
                # Continue to next retry

        return []  # Fallback to empty list
