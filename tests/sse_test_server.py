"""
Simple SSE test server to verify event streaming works.

Sends a sequence of test events to verify SSE delivery.
"""

import asyncio
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn


app = FastAPI()


@app.get("/test-sse")
async def test_sse():
    """
    Test SSE endpoint that sends a sequence of events.
    """

    async def event_generator():
        """Generate test SSE events."""
        # Send 10 events with delays
        for i in range(10):
            # Create event data
            event_data = f'{{"event_number": {i}, "timestamp": {time.time()}}}'

            # Yield SSE event
            yield f"data: {event_data}\n\n"

            # CRITICAL: Force flush with empty comment
            yield ": \n\n"

            print(f"[SERVER] Sent event {i} at {time.time()}")

            # Wait 1 second between events
            await asyncio.sleep(1)

        # Send completion marker
        yield "data: [DONE]\n\n"
        print(f"[SERVER] Sent DONE at {time.time()}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    print("Starting SSE test server on http://localhost:8012")
    uvicorn.run(app, host="0.0.0.0", port=8012, log_level="info")
