"""Test the simple pane client WebSocket connection."""

import asyncio
import json
import pytest
import websockets


@pytest.mark.asyncio
async def test_websocket_connection():
    """Test basic WebSocket connection and message sending."""
    uri = "ws://localhost:8011/message"

    async with websockets.connect(uri) as websocket:
        # Send message
        await websocket.send(json.dumps({"type": "message", "content": "Say hello in one sentence"}))

        # Receive response
        response_text = ""
        async for message in websocket:
            data = json.loads(message)

            if data.get("type") == "token":
                response_text += data.get("content", "")

            elif data.get("type") == "complete":
                final_response = data.get("content", "")
                assert final_response is not None
                assert len(final_response) > 0
                break

        assert len(response_text) > 0


if __name__ == "__main__":
    asyncio.run(test_websocket_connection())
