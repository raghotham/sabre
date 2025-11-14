# Testing SABRE + Remote MCP Server (SSE Transport)

This directory contains everything you need to test SABRE's SSE (Server-Sent Events) transport integration with a remote MCP server.

## Quick Start

### Option 1: Automated Setup (Recommended)

```bash
cd tests/manual

# Run setup script
chmod +x setup_test.sh
./setup_test.sh

# Terminal 1: Start MCP server
python simple_mcp_server.py

# Terminal 2: Verify server works (optional)
chmod +x test_mcp_server.sh
./test_mcp_server.sh

# Terminal 3: Start SABRE
cd ../..
uv run sabre
```

### Option 2: Manual Setup

#### 1. Install Dependencies

```bash
uv pip install aiohttp
```

#### 2. Configure SABRE

```bash
# Create config directory
mkdir -p ~/.config/sabre

# Copy test config
cp tests/manual/mcp_test.yaml ~/.config/sabre/mcp.yaml
```

#### 3. Start MCP Server

```bash
cd tests/manual
python simple_mcp_server.py
```

You should see:
```
============================================================
Simple MCP Server (HTTP)
============================================================

Starting server on http://localhost:8080
MCP endpoint: http://localhost:8080/mcp

Available tools:
  - echo(message: str)
  - calculate(operation: str, a: number, b: number)

Press Ctrl+C to stop
============================================================
```

#### 4. Verify Server (Optional)

In a new terminal:

```bash
cd tests/manual
chmod +x test_mcp_server.sh
./test_mcp_server.sh
```

Expected output:
```
============================================================
Testing Simple MCP Server
============================================================

1. Testing health check...
   ✓ Health check passed

2. Testing tools/list...
   ✓ Tools list includes 'echo'
   ✓ Tools list includes 'calculate'

3. Testing echo tool...
   ✓ Echo tool works

4. Testing calculate tool (addition)...
   ✓ Calculate tool works (5 + 3 = 8)

5. Testing calculate tool (division)...
   ✓ Division works (10 / 2 = 5)

============================================================
All tests passed! ✓
============================================================
```

#### 5. Start SABRE

In a new terminal:

```bash
cd /Users/hnayak/Documents/workspace/sabre
uv run sabre
```

Watch for these log messages:
```
INFO - Loading MCP configuration from: ~/.config/sabre/mcp.yaml
INFO - Found 1 MCP server(s) in config
INFO - Connecting to MCP server: remote_test (sse)
INFO - Connected to MCP server: remote_test (sse)
INFO - Discovered 2 tools from remote_test: ['echo', 'calculate']
INFO - MCP tools integrated into Python runtime
```

## Testing in SABRE

Once SABRE is running, try these prompts:

### Test 1: Echo Tool

**Prompt:**
```
Use the echo tool to say "Hello from SABRE!"
```

**Expected behavior:**
- LLM generates `<helpers>` block calling `remote_test.echo()`
- Output shows: `Echo: Hello from SABRE!`
- MCP server logs show the request

### Test 2: Calculate Tool

**Prompt:**
```
Use the calculator to add 42 and 58
```

**Expected behavior:**
- LLM calls `remote_test.calculate(operation="add", a=42, b=58)`
- Output shows: `Result: 100`

### Test 3: Multiple Operations

**Prompt:**
```
Calculate these for me:
- 15 * 3
- 100 / 4
- 25 - 10
```

**Expected behavior:**
- LLM makes multiple tool calls
- Each operation returns correct result

### Test 4: Discovery

**Prompt:**
```
What MCP tools are available?
```

**Expected behavior:**
- LLM lists tools from remote_test server
- Shows echo and calculate with descriptions

## What's Being Tested

This test verifies:

1. **SSE Transport:** SABRE connects to remote HTTP-based MCP server
2. **Tool Discovery:** SABRE discovers tools via `tools/list` JSON-RPC call
3. **Tool Invocation:** SABRE invokes tools via `tools/call` JSON-RPC call
4. **Result Handling:** MCP responses transformed to SABRE Content model
5. **Error Handling:** Graceful handling of server errors
6. **Integration:** End-to-end flow from LLM → SABRE → MCP Server → back

## Files in This Directory

- **simple_mcp_server.py** - HTTP-based MCP server with echo and calculate tools
- **test_mcp_server.sh** - Script to verify MCP server is working
- **mcp_test.yaml** - SABRE MCP configuration for testing
- **setup_test.sh** - Automated setup script
- **README.md** - This file

## Architecture

```
┌──────────────┐         WebSocket        ┌──────────────┐
│   SABRE      │◀──────────────────────────│  Terminal    │
│   Server     │                           │  Client      │
└──────┬───────┘                           └──────────────┘
       │
       │ SSE Transport (HTTP POST)
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  Simple MCP Server (localhost:8080)                      │
│                                                           │
│  Endpoints:                                               │
│  - GET  /health          → Health check                  │
│  - POST /mcp             → JSON-RPC endpoint             │
│                                                           │
│  Tools:                                                   │
│  - echo(message)         → Returns echoed message        │
│  - calculate(op, a, b)   → Returns calculation result    │
└──────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Server won't start

**Error:** `Address already in use`

**Fix:**
```bash
# Find process using port 8080
lsof -i :8080

# Kill it
kill -9 <PID>
```

### SABRE can't connect

**Error:** `MCPConnectionError: Failed to reach SSE endpoint`

**Fix:**
1. Verify server is running: `curl http://localhost:8080/health`
2. Check SABRE config: `cat ~/.config/sabre/mcp.yaml`
3. Check server logs for errors

### Tools not discovered

**Error:** `Discovered 0 tools from remote_test`

**Fix:**
1. Test tools/list manually:
```bash
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":1}'
```
2. Check server logs
3. Restart both server and SABRE

### httpx not installed

**Error:** `httpx is required for SSE transport`

**Fix:**
```bash
uv pip install httpx
```

## Advanced Testing

### With Authentication

1. Edit `simple_mcp_server.py`, add to `handle_request()`:

```python
# Check authorization
auth_header = request.headers.get("Authorization", "")
if auth_header != "Bearer test-secret-token":
    return web.json_response({
        "jsonrpc": "2.0",
        "error": {"code": -32000, "message": "Unauthorized"}
    }, status=401)
```

2. Update `~/.config/sabre/mcp.yaml`:

```yaml
mcp_servers:
  remote_test:
    type: sse
    url: "http://localhost:8080/mcp"
    headers:
      Authorization: "Bearer test-secret-token"
    enabled: true
```

3. Restart both server and SABRE

### Deploy Remotely

1. Deploy `simple_mcp_server.py` to a cloud VM
2. Update URL in `mcp.yaml`:
```yaml
url: "https://your-server.com/mcp"
```
3. Add authentication headers
4. Ensure HTTPS for production use

## Success Criteria

✅ MCP server starts on port 8080
✅ `test_mcp_server.sh` passes all tests
✅ SABRE connects and discovers 2 tools
✅ Can invoke `remote_test.echo()` from SABRE
✅ Can invoke `remote_test.calculate()` from SABRE
✅ Results displayed correctly in SABRE
✅ Server logs show JSON-RPC requests

## Next Steps

After successful testing:

1. Build custom MCP tools for your use case
2. Deploy to production with authentication
3. Monitor tool usage and performance
4. Add more sophisticated tools
5. Integrate with external APIs

## Support

For issues:
1. Check server logs
2. Check SABRE logs at `~/.local/state/sabre/logs/server.log`
3. Review test output from `test_mcp_server.sh`
4. Verify configuration in `~/.config/sabre/mcp.yaml`

Happy testing! 🎉
