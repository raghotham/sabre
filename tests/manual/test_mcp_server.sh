#!/bin/bash
#
# Test script for the simple MCP server
#
# This script tests the MCP server endpoints to verify it's working correctly
# before connecting SABRE.
#

set -e

echo "============================================================"
echo "Testing Simple MCP Server"
echo "============================================================"
echo ""

# Check if server is running
echo "1. Testing health check..."
HEALTH=$(curl -s http://localhost:8080/health)
if echo "$HEALTH" | grep -q "healthy"; then
    echo "   ✓ Health check passed"
else
    echo "   ✗ Health check failed"
    echo "   Make sure the server is running: python simple_mcp_server.py"
    exit 1
fi
echo ""

# Test tools/list
echo "2. Testing tools/list..."
TOOLS=$(curl -s -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/list",
    "id": 1
  }')

if echo "$TOOLS" | grep -q "echo"; then
    echo "   ✓ Tools list includes 'echo'"
else
    echo "   ✗ Tools list missing 'echo'"
    exit 1
fi

if echo "$TOOLS" | grep -q "calculate"; then
    echo "   ✓ Tools list includes 'calculate'"
else
    echo "   ✗ Tools list missing 'calculate'"
    exit 1
fi
echo ""

# Test echo tool
echo "3. Testing echo tool..."
ECHO=$(curl -s -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "id": 2,
    "params": {
      "name": "echo",
      "arguments": {
        "message": "Hello from test!"
      }
    }
  }')

if echo "$ECHO" | grep -q "Echo: Hello from test!"; then
    echo "   ✓ Echo tool works"
else
    echo "   ✗ Echo tool failed"
    echo "   Response: $ECHO"
    exit 1
fi
echo ""

# Test calculate tool (addition)
echo "4. Testing calculate tool (addition)..."
CALC=$(curl -s -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "id": 3,
    "params": {
      "name": "calculate",
      "arguments": {
        "operation": "add",
        "a": 5,
        "b": 3
      }
    }
  }')

if echo "$CALC" | grep -q "Result: 8"; then
    echo "   ✓ Calculate tool works (5 + 3 = 8)"
else
    echo "   ✗ Calculate tool failed"
    echo "   Response: $CALC"
    exit 1
fi
echo ""

# Test calculate tool (division)
echo "5. Testing calculate tool (division)..."
CALC=$(curl -s -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "id": 4,
    "params": {
      "name": "calculate",
      "arguments": {
        "operation": "divide",
        "a": 10,
        "b": 2
      }
    }
  }')

if echo "$CALC" | grep -q "Result: 5"; then
    echo "   ✓ Division works (10 / 2 = 5)"
else
    echo "   ✗ Division failed"
    echo "   Response: $CALC"
    exit 1
fi
echo ""

echo "============================================================"
echo "All tests passed! ✓"
echo "============================================================"
echo ""
echo "The MCP server is working correctly."
echo "You can now configure SABRE to connect to it."
echo ""
