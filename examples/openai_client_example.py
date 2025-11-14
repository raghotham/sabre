"""
Example usage of SABRE's OpenAI-compatible client.

This demonstrates how to use the OpenAI-compatible wrapper to interact with SABRE.
"""

import asyncio
from sabre.openai_client import OpenAI


def basic_example():
    """Basic non-streaming example."""
    print("=== Basic Example (Non-streaming) ===\n")

    # Create client (similar to OpenAI SDK)
    client = OpenAI(base_url="http://localhost:8011")

    # Send a message
    response = client.chat.completions.create(
        model="gpt-4",  # Model name is ignored - SABRE uses server's configured model
        messages=[{"role": "user", "content": "What is 2+2?"}],
    )

    # Access response (OpenAI-compatible interface)
    print(f"Response: {response.choices[0].message.content}")
    print(f"\nTokens: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")


async def streaming_example():
    """Streaming example."""
    print("\n\n=== Streaming Example ===\n")

    client = OpenAI(base_url="http://localhost:8011")

    # Reset conversation for fresh start
    client.reset_conversation()

    # Stream response chunks
    stream = await client.chat.completions.create_async(
        model="gpt-4",
        messages=[{"role": "user", "content": "Count to 5"}],
        stream=True,
    )

    print("Response: ", end="", flush=True)
    async for chunk in stream:
        # chunk.choices[0] is a dict, not an object
        if chunk.choices[0]["delta"].get("content"):
            content = chunk.choices[0]["delta"]["content"]
            print(content, end="", flush=True)
    print()


async def conversation_example():
    """Multi-turn conversation example."""
    print("\n\n=== Conversation Example ===\n")

    client = OpenAI(base_url="http://localhost:8011")

    # Reset conversation
    client.reset_conversation()

    # First message
    response1 = await client.chat.completions.create_async(
        messages=[{"role": "user", "content": "My name is Alice"}],
    )
    print(f"Assistant: {response1.choices[0].message.content}")

    # Second message (same conversation)
    response2 = await client.chat.completions.create_async(
        messages=[{"role": "user", "content": "What's my name?"}],
    )
    print(f"Assistant: {response2.choices[0].message.content}")


if __name__ == "__main__":
    # Run basic example
    basic_example()

    # Run async examples
    asyncio.run(streaming_example())
    asyncio.run(conversation_example())
