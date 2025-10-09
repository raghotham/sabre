"""
Coerce helper - type coercion using LLM.

Allows code to convert expressions to specific types using LLM understanding.
"""
import asyncio
import logging
from typing import Any, Union, Type, Callable

from sabre.common.utils.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)


class Coerce:
    """
    Coerce expression to specified type using LLM.
    """

    def __init__(self, get_orchestrator: Callable, get_openai_client: Callable):
        """
        Initialize Coerce helper.

        Args:
            get_orchestrator: Function to get orchestrator instance
            get_openai_client: Function to get/create OpenAI client
        """
        self.get_orchestrator = get_orchestrator
        self.get_openai_client = get_openai_client

    def __call__(self, expr: Any, type_name: Union[str, Type]) -> Any:
        """
        Coerce expression to specified type (sync wrapper).

        Examples:
            number = coerce("forty-two", int)
            date = coerce("next friday", "datetime")
            price = coerce("$29.99", float)

        Args:
            expr: Expression to coerce
            type_name: Target type (string or Type)

        Returns:
            Coerced value
        """
        if isinstance(type_name, type):
            type_name = type_name.__name__

        logger.debug(f"coerce({str(expr)[:50]}, {type_name})")

        # Run async coercion
        from sabre.server.helpers.llm_call import run_async_from_sync
        return run_async_from_sync(self.execute(expr, type_name))

    async def execute(self, expr: Any, type_name: str, max_retries: int = 3) -> Any:
        """
        Async implementation of coerce with retry logic.

        Args:
            expr: Value to coerce
            type_name: Target type name
            max_retries: Maximum retry attempts

        Returns:
            Coerced value or original expression on failure
        """
        # Load prompt
        prompt = PromptLoader.load(
            'coerce.prompt',
            template={
                'string': str(expr),
                'type': type_name,
            }
        )

        # Combine system_message and user_message as instructions
        system_instructions = f"{prompt['system_message']}\n\n{prompt['user_message']}"

        # Get or create OpenAI client (respects env vars)
        client = self.get_openai_client()

        conversation = await client.conversations.create(
            metadata={"type": "coerce", "target_type": type_name}
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
        input_text = f"### Value to coerce\n{str(expr)}"

        from sabre.common import ExecutionTree, ExecutionNodeType, ExecutionStatus

        # Retry loop
        for attempt in range(max_retries):
            logger.info(f"coerce attempt {attempt + 1}/{max_retries}")

            tree = ExecutionTree()
            node = tree.push(
                ExecutionNodeType.NESTED_LLM_CALL,
                metadata={"helper": "coerce", "attempt": attempt + 1}
            )

            try:
                # Call LLM to perform coercion (pass instructions on every call)
                response = await client.responses.create(
                    model="gpt-4o",
                    conversation=conversation.id,
                    instructions=system_instructions,  # Must send on every call
                    input=input_text,
                    max_output_tokens=500,
                    stream=False,
                    truncation="auto",
                )

                # Extract result from response
                result_text = response.content.text.strip()
                tree.pop(ExecutionStatus.COMPLETED)

                # Try to parse as Python literal
                import ast
                try:
                    return ast.literal_eval(result_text)
                except (ValueError, SyntaxError):
                    # Return as string if can't parse
                    return result_text

            except Exception as e:
                tree.pop(ExecutionStatus.ERROR)
                logger.warning(f"coerce attempt {attempt + 1} failed: {e}")
                if attempt >= max_retries - 1:
                    logger.error(f"coerce failed after {max_retries} attempts")
                    return expr  # Return original on failure
                # Continue to next retry

        return expr  # Fallback to original value
