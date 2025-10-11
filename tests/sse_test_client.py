"""
Simple SSE test client to verify event reception.

Connects to test server and prints received events.
"""

import asyncio
import httpx
import json
import time


async def test_sse_client():
    """Connect to SSE test server and print events."""
    url = "http://localhost:8012/test-sse"

    print(f"[CLIENT] Connecting to {url}")

    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url, timeout=httpx.Timeout(None)) as response:
            if response.status_code != 200:
                print(f"[CLIENT] Error: Status {response.status_code}")
                return

            print(f"[CLIENT] Connected, waiting for events...")

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
    asyncio.run(test_sse_client())
