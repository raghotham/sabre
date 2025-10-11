#!/bin/bash

# Test SSE server/client communication

echo "Starting SSE test..."
echo ""

# Start server in background
echo "Starting test server on port 8012..."
uv run python tests/sse_test_server.py &
SERVER_PID=$!

# Wait for server to start
sleep 2

# Run client
echo ""
echo "Starting test client..."
uv run python tests/sse_test_client.py

# Kill server
echo ""
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null

echo "Test complete"
