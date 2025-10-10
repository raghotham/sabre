#!/usr/bin/env python3
"""Test to verify system prompt is sent and model uses <helpers> blocks."""

import asyncio
import json
import websockets


async def test():
    """Test WebSocket connection and message sending."""
    uri = "ws://localhost:8011/message"

    async with websockets.connect(uri) as websocket:
        print("Connected to server")

        # Send a message that should trigger <helpers> usage
        message = {"type": "message", "content": "Use bash to list files in the current directory"}

        print(f"\nSending: {message}")
        await websocket.send(json.dumps(message))

        # Receive and print all responses
        print("\nReceiving responses:")
        try:
            while True:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                data = json.loads(response)

                # Print different event types differently
                if data.get("type") == "token":
                    print(data["content"], end="", flush=True)
                elif data.get("type") == "helper_extracted":
                    print(f"\n\n[HELPER EXTRACTED #{data.get('block_count')}]:")
                    print(data.get("code", ""))
                elif data.get("type") == "helper_start":
                    print("\n[HELPER EXECUTING...]")
                elif data.get("type") == "helper_result":
                    print(f"[HELPER RESULT ({data.get('duration_ms', 0):.0f}ms)]:")
                    print(data.get("result", ""))
                elif data.get("type") == "final_response":
                    print(f"\n\n[FINAL RESPONSE]:\n{data.get('content', '')}")
                elif data.get("type") == "error":
                    print(f"\n[ERROR]: {data.get('content', '')}")
                else:
                    # Print raw event for debugging
                    print(f"\n[EVENT {data.get('type')}]: {json.dumps(data, indent=2)}")

        except asyncio.TimeoutError:
            print("\n\nTimeout waiting for response")
        except websockets.exceptions.ConnectionClosed:
            print("\n\nConnection closed")


if __name__ == "__main__":
    asyncio.run(test())
