"""
Test script for OpenAI-compatible endpoints.

Tests both /v1/models and /v1/chat/completions endpoints.
"""

import asyncio
import httpx


async def test_models_endpoint():
    """Test the /v1/models endpoint."""
    print("=" * 60)
    print("Testing /v1/models endpoint")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8011/v1/models")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print()


async def test_chat_completions_non_streaming():
    """Test the /v1/chat/completions endpoint (non-streaming)."""
    print("=" * 60)
    print("Testing /v1/chat/completions (non-streaming)")
    print("=" * 60)

    request_body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What is 2+2? Just give the number."}],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        response = await client.post("http://localhost:8011/v1/chat/completions", json=request_body)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"ID: {data['id']}")
            print(f"Model: {data['model']}")
            print(f"Response: {data['choices'][0]['message']['content']}")
            print(f"Usage: {data['usage']}")
        else:
            print(f"Error: {response.text}")
        print()


async def test_chat_completions_streaming():
    """Test the /v1/chat/completions endpoint (streaming)."""
    print("=" * 60)
    print("Testing /v1/chat/completions (streaming)")
    print("=" * 60)

    request_body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Count from 1 to 3."}],
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        async with client.stream("POST", "http://localhost:8011/v1/chat/completions", json=request_body) as response:
            print(f"Status: {response.status_code}")

            if response.status_code == 200:
                print("Chunks:")
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str == "[DONE]":
                            print("Stream complete")
                            break
                        print(f"  {data_str}")
            else:
                error_text = await response.aread()
                print(f"Error: {error_text.decode()}")
        print()


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("OpenAI-Compatible Endpoints Test Suite")
    print("=" * 60 + "\n")

    try:
        await test_models_endpoint()
        await test_chat_completions_non_streaming()
        await test_chat_completions_streaming()

        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except httpx.ConnectError:
        print("\n❌ ERROR: Could not connect to server at http://localhost:8011")
        print("Make sure the SABRE server is running: uv run sabre-server")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
