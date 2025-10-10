#!/usr/bin/env python3
"""Test script to verify the instructions fix."""

import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import AsyncOpenAI


async def test_conversation_create():
    """Test that conversation.create() works without instructions parameter."""
    client = AsyncOpenAI()

    # This should work (no instructions parameter)
    conversation = await client.conversations.create(metadata={"test": "value"})

    print(f"✓ Conversation created successfully: {conversation.id}")

    # Test with responses API and instructions
    response = await client.responses.create(
        model="gpt-4o",
        conversation=conversation.id,
        input="Say hello in one sentence",
        instructions="You are a helpful assistant.",
        max_output_tokens=100,
        stream=False,
    )

    # Extract response text
    response_text = response.output[0].content[0].text if response.output else ""
    print(f"✓ Response created with instructions: {response_text}")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_conversation_create())
