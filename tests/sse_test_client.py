"""
Simple SSE test client to verify event reception.

Connects to test server and prints received events.

NOTE: This is an integration test that requires a running test server.
It is not a unit test and should not be run as part of the regular test suite.
"""

import asyncio
import httpx
import json
import time
import pytest


@pytest.mark.skip(reason="Requires running test server at localhost:8012 - manual integration test only")
@pytest.mark.asyncio
async def test_sse_client():
    """Connect to SSE test server and print events.

    NOTE: This test requires a running test server at localhost:8012.
    This is a manual integration test, not part of the automated test suite.
    """
    url = "http://localhost:8012/test-sse"

    print(f"[CLIENT] Connecting to {url}")

    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url, timeout=httpx.Timeout(None)) as response:
            if response.status_code != 200:
                print(f"[CLIENT] Error: Status {response.status_code}")
                return

            print("[CLIENT] Connected, waiting for events...")

            # Read SSE stream line by line
            async for line in response.aiter_lines():
                if not line:
                    continue

                # Skip comments
                if line.startswith(":"):
                    continue

                # Parse SSE data
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix

                    if data == "[DONE]":
                        print(f"[CLIENT] Received DONE at {time.time()}")
                        break

                    # Parse JSON
                    try:
                        event_data = json.loads(data)
                        print(f"[CLIENT] Received event {event_data['event_number']} at {time.time()}")
                    except json.JSONDecodeError as e:
                        print(f"[CLIENT] Failed to parse: {data} - {e}")

    print("[CLIENT] Stream ended")


if __name__ == "__main__":
    print("⚠️  This is a manual integration test.")
    print("    Make sure the test server is running at http://localhost:8012")
    print()
    asyncio.run(test_sse_client())
