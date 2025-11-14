# SABRE - Model Context Protocol (MCP) Integration Plan

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution: MCP Integration Layer](#solution-mcp-integration-layer)
  - [Key Insight](#key-insight)
- [MCP Protocol Overview](#mcp-protocol-overview)
  - [Architecture](#architecture)
  - [Protocol Flow](#protocol-flow)
  - [MCP Message Types](#mcp-message-types)
- [Integration Architecture](#integration-architecture)
  - [Components](#components)
- [Key Design Decisions](#key-design-decisions)
  - [1. Coexistence Strategy: Built-in + MCP](#1-coexistence-strategy-built-in--mcp)
  - [2. Connection Lifecycle: Lazy + Persistent](#2-connection-lifecycle-lazy--persistent)
  - [3. Tool Discovery: Dynamic + Cached](#3-tool-discovery-dynamic--cached)
  - [4. Error Handling: Graceful Degradation](#4-error-handling-graceful-degradation)
  - [5. Security: Allowlist Configuration](#5-security-allowlist-configuration)
  - [6. Transport Support: stdio First, SSE Later](#6-transport-support-stdio-first-sse-later)
  - [7. Resources: Phase 3](#7-resources-phase-3)
- [Implementation Roadmap](#implementation-roadmap)
  - [Phase 1: Core MCP Infrastructure](#phase-1-core-mcp-infrastructure-week-1-2)
  - [Phase 2: Runtime Integration](#phase-2-runtime-integration-week-3)
  - [Phase 3: Configuration & CLI](#phase-3-configuration--cli-week-4)
  - [Phase 4: SSE Transport](#phase-4-sse-transport-week-5)
  - [Phase 5: Resources & Prompts](#phase-5-resources--prompts-week-6)
- [Integration with Persona System](#integration-with-persona-system)
- [Benefits](#benefits)
- [Challenges & Risks](#challenges--risks)
- [Testing Strategy](#testing-strategy)
- [Success Metrics](#success-metrics)
- [Open Questions](#open-questions)
- [Future Enhancements](#future-enhancements)
- [References](#references)
- [Appendix: Example MCP Server Integration](#appendix-example-mcp-server-integration)
- [Appendix: Custom MCP Server Example](#appendix-custom-mcp-server-example)

## Problem Statement

**Current State:**
- SABRE has a fixed set of built-in helpers (Bash, Search, Web, FS, etc.)
- Adding new tool integrations requires modifying SABRE's codebase
- Each helper is tightly coupled to SABRE's Python runtime
- No standardized way to extend SABRE with third-party tools
- Tool capabilities are limited to what's implemented in `sabre/server/helpers/`

**What We Want:**
- Plug-and-play integration with external tools via MCP protocol
- Add new capabilities without modifying SABRE's core code
- Leverage the growing ecosystem of MCP servers (databases, APIs, IDEs, etc.)
- Standardized tool discovery and invocation
- Coexistence of built-in helpers and MCP tools

## Solution: MCP Integration Layer

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is an open protocol that standardizes how AI systems connect to external tools and data sources. By integrating MCP into SABRE, we enable:

1. **Dynamic tool discovery** - Connect to any MCP server at runtime
2. **Extensibility** - Add capabilities without code changes
3. **Ecosystem access** - Leverage existing MCP servers (Postgres, Slack, GitHub, etc.)
4. **Standardization** - Use common protocol for all external tools

### Key Insight

MCP servers expose tools via a standard JSON-RPC protocol. SABRE can:
- Connect to multiple MCP servers simultaneously
- Discover available tools dynamically
- Invoke tools using standard protocol
- Present tools to the LLM alongside built-in helpers
- Route tool calls to appropriate MCP server

The LLM decides when to use MCP tools vs built-in helpers - they coexist naturally.

## MCP Protocol Overview

### Architecture

```
┌─────────────────────────────────────────────────┐
│  SABRE Orchestrator                             │
│  ┌───────────────────────────────────────────┐  │
│  │  Python Runtime                           │  │
│  │  ┌─────────────────┬─────────────────┐    │  │
│  │  │ Built-in Helpers│  MCP Client     │    │  │
│  │  │  - Bash         │   - Discovery   │    │  │
│  │  │  - Search       │   - Invocation  │    │  │
│  │  │  - Web          │   - Routing     │    │  │
│  │  │  - FS           │                 │    │  │
│  │  └─────────────────┴─────────────────┘    │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┬───────────┐
        │                       │           │
        ▼                       ▼           ▼
┌───────────────┐    ┌──────────────┐  ┌──────────┐
│ MCP Server 1  │    │ MCP Server 2 │  │ MCP ...  │
│  (Postgres)   │    │   (GitHub)   │  │          │
│               │    │              │  │          │
│ Tools:        │    │ Tools:       │  │          │
│  - query()    │    │  - list_prs()│  │          │
│  - execute()  │    │  - create_pr │  │          │
└───────────────┘    └──────────────┘  └──────────┘
```

### Protocol Flow

1. **Connection**: SABRE connects to MCP server via stdio or WebSocket
2. **Discovery**: Server lists available tools with schemas
3. **Presentation**: Tools exposed to LLM in `<helpers>` namespace
4. **Invocation**: LLM calls tool, SABRE routes to correct server
5. **Result**: Response returned in `<helpers_result>` tag

### MCP Message Types

**Tools Discovery:**
```json
// Request
{"jsonrpc": "2.0", "method": "tools/list", "id": 1}

// Response
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {
        "name": "query_database",
        "description": "Execute SQL query on connected database",
        "inputSchema": {
          "type": "object",
          "properties": {
            "query": {"type": "string"},
            "params": {"type": "array"}
          },
          "required": ["query"]
        }
      }
    ]
  }
}
```

**Tool Invocation:**
```json
// Request
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "id": 2,
  "params": {
    "name": "query_database",
    "arguments": {
      "query": "SELECT * FROM users LIMIT 10"
    }
  }
}

// Response
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Found 10 users:\n1. Alice\n2. Bob\n..."
      }
    ]
  }
}
```


## Simple Control Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Request                                │
│                  "Query the database for top users"                 │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      1. Orchestrator.run()                          │
│  - Creates execution tree                                           │
│  - Calls ResponseExecutor with system prompt (includes MCP tools)   │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              2. OpenAI Responses API (streaming)                    │
│  LLM generates response with <helpers> block:                       │
│                                                                     │
│  <helpers>                                                          │
│  result = Postgres.query("SELECT * FROM users LIMIT 10")            │
│  print(result)                                                      │
│  </helpers>                                                         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│               3. Streaming Parser (token-by-token)                  │
│  - Detects <helpers> tags                                           │
│  - Extracts Python code block                                       │
│  - Emits HelpersExtractedEvent                                      │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  4. Python Runtime.execute()                        │
│  - Executes code in namespace with helpers                          │
│  - Encounters: Postgres.query(...)                                  │
│  - Postgres is a callable from MCP namespace                        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  5. MCPHelperAdapter.invoke_tool()                  │
│  - Parses tool call: server="postgres", tool="query"                │
│  - Gets MCPClient for "postgres" server                             │
│  - Transforms arguments to MCP schema                               │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   6. MCPClient.call_tool()                          │
│  - Builds JSON-RPC request:                                         │
│    {                                                                │
│      "jsonrpc": "2.0",                                              │
│      "method": "tools/call",                                        │
│      "params": {                                                    │
│        "name": "query",                                             │
│        "arguments": {"sql": "SELECT..."}                            │
│      }                                                              │
│    }                                                                │
│  - Sends via stdio/SSE transport                                    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              7. MCP Server Process (external)                       │
│  - Receives JSON-RPC request                                        │
│  - Executes SQL query against Postgres database                     │
│  - Returns result:                                                  │
│    {                                                                │
│      "jsonrpc": "2.0",                                              │
│      "result": {                                                    │
│        "content": [                                                 │
│          {"type": "text", "text": "user1\nuser2\n..."}              │
│        ]                                                            │
│      }                                                              │
│    }                                                                │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 8. MCPClient receives response                      │
│  - Parses JSON-RPC response                                         │
│  - Extracts content array                                           │
│  - Returns to adapter                                               │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│            9. MCPHelperAdapter transforms result                    │
│  - Converts MCP content → SABRE Content model                       │
│  - Returns list[Content] to runtime                                 │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              10. Python Runtime completes execution                 │
│  - Postgres.query() returns result                                  │
│  - print(result) executes                                           │
│  - Captures stdout: "user1\nuser2\n..."                             │
│  - Returns ExecutionResult to orchestrator                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│          11. Orchestrator builds continuation message               │
│  - Creates <helpers_result> tag with output                         │
│  - Calls ResponseExecutor again (continuation)                      │
│                                                                     │
│  Previous conversation + new message:                               │
│  <helpers_result>                                                   │
│  user1                                                              │
│  user2                                                              │
│  ...                                                                │
│  </helpers_result>                                                  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│         12. OpenAI Responses API (continuation stream)              │
│  LLM sees result and generates final response:                      │
│                                                                     │
│  "I found the top 10 users in the database:                         │
│   1. user1                                                          │
│   2. user2                                                          │
│   ..."                                                              │
│                                                                     │
│  </complete>                                                        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│               13. Orchestrator detects </complete>                  │
│  - No more <helpers> blocks                                         │
│  - Emits CompleteEvent                                              │
│  - Returns to client                                                │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   User sees final response                          │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Interaction Diagram

```
┌──────────┐     ┌─────────────┐     ┌──────────────┐     ┌──────────┐
│  Client  │────▶│ Orchestrator│────▶│   Response   │────▶│ OpenAI   │
│          │     │             │     │   Executor   │     │   API    │
└──────────┘     └──────┬──────┘     └──────────────┘     └──────────┘
                        │
                        │ delegates to
                        ▼
                 ┌──────────────┐
                 │   Python     │
                 │   Runtime    │
                 └──────┬───────┘
                        │
                        │ MCP tool call detected
                        ▼
                 ┌──────────────────┐
                 │  MCP Helper      │
                 │  Adapter         │
                 └──────┬───────────┘
                        │
                        │ routes to server
                        ▼
                 ┌──────────────────┐
                 │  MCP Client      │
                 │  Manager         │
                 └──────┬───────────┘
                        │
                        │ gets client for server
                        ▼
                 ┌──────────────────┐
                 │   MCP Client     │
                 │   (Postgres)     │
                 └──────┬───────────┘
                        │
                        │ JSON-RPC over stdio
                        ▼
            ┌────────────────────────────┐
            │   MCP Server Process       │
            │   (npx @mcp/postgres)      │
            │                            │
            │  ┌──────────────────────┐  │
            │  │  Postgres Database   │  │
            │  └──────────────────────┘  │
            └────────────────────────────┘
```

## Key Data Flows

### 1. MCP Server Connection (Startup)

```
Orchestrator startup
    │
    ├─▶ MCPClientManager.connect("postgres", config)
    │       │
    │       ├─▶ Spawn subprocess: npx @mcp/postgres
    │       │       │
    │       │       └─▶ MCP Server starts, listens on stdio
    │       │
    │       ├─▶ MCPClient.list_tools()
    │       │       │
    │       │       └─▶ JSON-RPC: {"method": "tools/list"}
    │       │               │
    │       │               └─▶ Returns: [{name: "query", schema: {...}}]
    │       │
    │       └─▶ Store in registry: servers["postgres"] = client
    │
    └─▶ MCPHelperAdapter.get_available_tools()
            │
            └─▶ Returns: {"Postgres.query": <callable>, ...}
                    │
                    └─▶ Injected into Python Runtime namespace
```

### 2. MCP Tool Invocation (Runtime)

```
Python code execution: Postgres.query("SELECT...")
    │
    ├─▶ Callable from namespace (bound to MCPHelperAdapter)
    │
    ├─▶ MCPHelperAdapter.invoke_tool(server="postgres", tool="query", ...)
    │       │
    │       ├─▶ Get client from manager: manager.get_client("postgres")
    │       │
    │       ├─▶ MCPClient.call_tool("query", arguments={...})
    │       │       │
    │       │       ├─▶ Build JSON-RPC request
    │       │       │
    │       │       ├─▶ Send via stdio (write to subprocess stdin)
    │       │       │
    │       │       ├─▶ Read from subprocess stdout
    │       │       │
    │       │       └─▶ Parse JSON-RPC response
    │       │
    │       └─▶ Transform MCP content → SABRE Content
    │
    └─▶ Return result to Python code
```

### 3. Error Handling Flow

```
MCP tool execution error
    │
    ├─▶ MCPClient.call_tool() raises exception
    │       │
    │       ├─▶ Server timeout → MCPTimeoutError
    │       ├─▶ Invalid JSON → MCPProtocolError
    │       ├─▶ Tool error → MCPToolError
    │       └─▶ Server crash → MCPConnectionError
    │
    ├─▶ MCPHelperAdapter catches exception
    │       │
    │       ├─▶ Log error with context
    │       │
    │       └─▶ Re-raise or return error content
    │
    ├─▶ Python Runtime catches exception
    │       │
    │       └─▶ Returns ExecutionResult(success=False, error="...")
    │
    └─▶ Orchestrator includes error in <helpers_result>
            │
            └─▶ LLM sees error and can retry or explain to user
```

## Sequence Diagram (Simplified)

```
User    Client  Orchestrator  Runtime  MCPAdapter  MCPClient  MCPServer
 │        │          │           │         │          │          │
 │─msg───▶│          │           │         │          │          │
 │        │─request─▶│           │         │          │          │
 │        │          │─execute──▶│         │          │          │
 │        │          │           │─tool────▶│         │          │
 │        │          │           │          │─route──▶│          │
 │        │          │           │          │         │─RPC─────▶│
 │        │          │           │          │         │          │─exec─┐
 │        │          │           │          │         │          │      │
 │        │          │           │          │         │          │◀─────┘
 │        │          │           │          │         │◀─result──│
 │        │          │           │          │◀────────│          │
 │        │          │           │◀─────────│         │          │
 │        │          │◀──────────│         │          │          │
 │        │          │─continue─▶│         │          │          │
 │        │          │◀──────────│         │          │          │
 │        │◀─stream──│           │         │          │          │
 │◀───────│          │           │         │          │          │
```

## Integration Architecture

### Components

#### 1. MCP Client Manager (`sabre/server/mcp/client_manager.py`)

Manages connections to multiple MCP servers:

**Responsibilities:**
- Start/stop MCP server processes (stdio transport)
- Maintain WebSocket connections (SSE transport)
- Connection lifecycle (reconnection, health checks)
- Server registry (name → connection mapping)

**API:**
```python
class MCPClientManager:
    async def connect(self, name: str, config: MCPServerConfig) -> None:
        """Connect to MCP server."""

    async def disconnect(self, name: str) -> None:
        """Disconnect from server."""

    async def list_servers(self) -> list[str]:
        """Get list of connected servers."""

    def get_client(self, name: str) -> MCPClient:
        """Get client for server."""
```

#### 2. MCP Client (`sabre/server/mcp/client.py`)

Low-level JSON-RPC client for MCP protocol:

**Responsibilities:**
- Send/receive JSON-RPC messages
- Handle protocol errors and retries
- Parse MCP response schemas
- Support both stdio and SSE transports

**API:**
```python
class MCPClient:
    async def list_tools(self) -> list[Tool]:
        """Discover available tools."""

    async def call_tool(self, name: str, arguments: dict) -> ToolResult:
        """Invoke tool with arguments."""

    async def list_resources(self) -> list[Resource]:
        """List available resources (future)."""

    async def read_resource(self, uri: str) -> ResourceContent:
        """Read resource content (future)."""
```

#### 3. MCP Helper Adapter (`sabre/server/mcp/helper_adapter.py`)

Bridges MCP tools into SABRE's helper system:

**Responsibilities:**
- Convert MCP tool schemas to Python callable signatures
- Generate helper documentation for LLM prompts
- Route tool calls to correct MCP server
- Transform results into SABRE's Content model

**API:**
```python
class MCPHelperAdapter:
    def __init__(self, client_manager: MCPClientManager):
        self.client_manager = client_manager

    def get_available_tools(self) -> dict[str, Callable]:
        """Get all MCP tools as Python callables."""

    def generate_documentation(self) -> str:
        """Generate helper docs for LLM prompt."""

    async def invoke_tool(
        self,
        server_name: str,
        tool_name: str,
        **kwargs
    ) -> list[Content]:
        """Invoke MCP tool and return results."""
```

#### 4. MCP Configuration (`sabre/config/mcp.yaml`)

User configuration for MCP servers:

```yaml
mcp_servers:
  postgres:
    type: stdio
    command: npx
    args: ["-y", "@modelcontextprotocol/server-postgres"]
    env:
      POSTGRES_URL: "postgresql://localhost/mydb"
    enabled: true

  github:
    type: stdio
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_TOKEN: "${GITHUB_TOKEN}"  # Read from environment
    enabled: true

  filesystem:
    type: stdio
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/allowed/path"]
    enabled: false  # Can enable when needed

  # Future: SSE transport
  remote_api:
    type: sse
    url: "https://my-mcp-server.com/mcp"
    headers:
      Authorization: "Bearer ${API_TOKEN}"
    enabled: false
```

**Config Loader:**
```python
@dataclass
class MCPServerConfig:
    name: str
    type: Literal["stdio", "sse"]
    command: str | None = None  # stdio only
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None  # sse only
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True

class MCPConfigLoader:
    @staticmethod
    def load(config_path: Path | None = None) -> list[MCPServerConfig]:
        """Load MCP server configs from YAML."""
```

#### 5. Runtime Integration (`sabre/server/python_runtime.py`)

Inject MCP tools into runtime namespace:

**Changes to `PythonRuntime.reset()`:**
```python
def reset(self):
    """Reset runtime namespace with built-in + MCP helpers."""
    # ... existing built-in helpers ...

    # Add MCP tools if manager available
    if self.mcp_manager:
        mcp_adapter = MCPHelperAdapter(self.mcp_manager)
        mcp_tools = mcp_adapter.get_available_tools()

        # Add tools to namespace
        # Option 1: Flat namespace (GitHub.create_pr)
        for tool_name, tool_func in mcp_tools.items():
            self.namespace[tool_name] = tool_func

        # Option 2: Namespaced (MCP.GitHub.create_pr)
        self.namespace['MCP'] = self._create_mcp_namespace(mcp_tools)
```

#### 6. Prompt Integration (`sabre/server/prompts/`)

Document MCP tools in system prompt:

**Auto-generated section:**
```
## MCP Tools

The following tools are available from connected MCP servers:

### GitHub Server

**GitHub.create_pull_request(title: str, body: str, head: str, base: str)**
Create a new pull request in the repository.

**GitHub.list_pull_requests(state: str = "open")**
List pull requests. State can be "open", "closed", or "all".

### Postgres Server

**Postgres.query(sql: str, params: list = None)**
Execute SQL query and return results.

**Postgres.execute(sql: str, params: list = None)**
Execute SQL statement (INSERT, UPDATE, DELETE).
```

## Key Design Decisions

### 1. Connection Lifecycle: Lazy + Persistent

**Decision**: Connect to MCP servers at SABRE startup, maintain persistent connections.

**Rationale:**
- Avoid connection overhead on every request
- Discover tools once at startup
- Reconnect automatically on failure
- Graceful degradation if server unavailable

**Alternative Considered**: Connect on-demand per conversation
- Pro: Lower resource usage
- Con: Slower first request, complex connection pooling
- **Rejected**: Startup connection is simpler and faster in practice

### 2. Tool Discovery: Dynamic + Cached

**Decision**: Discover tools at connection time, cache in memory.

**Rationale:**
- Tools don't change during server lifetime
- Fast lookup during execution
- Can refresh on reconnection

**Future Enhancement**: Watch for tool schema changes (via notifications)

### 3. Security: Allowlist Configuration

**Decision**: Only connect to explicitly configured servers in `mcp.yaml`.

**Rationale:**
- Prevent arbitrary code execution
- User must opt-in to each server
- Environment variable expansion for secrets (not stored in config)

**Security Considerations:**
- MCP servers run as child processes (stdio) - same trust boundary as SABRE
- SSE servers are remote - validate SSL, use tokens
- Don't expose file paths or env vars to LLM unless needed

### 4. Transport Support: stdio First, SSE Later

**Decision**: Implement stdio transport in Phase 1, SSE transport in Phase 2.

**Rationale:**
- stdio is simpler (subprocess management)
- Most MCP servers use stdio (npx packages)
- SSE adds complexity (HTTP, reconnection, auth)

### 5. Resources: Phase 3

**Decision**: Implement tools first, resources later.

**Rationale:**
- Tools (function calls) are higher priority
- Resources (data sources) require different UX
- Can add resources incrementally after tools proven

**Resources will enable:**
- Reading files/data without explicit tool calls
- Context injection (e.g., "use this database schema")
- Prompt augmentation with external knowledge

## Implementation Roadmap

### Phase 1: Core MCP Infrastructure

**Goal**: Connect to MCP servers, discover tools, basic invocation

**Tasks:**
1. Create `MCPClient` class (JSON-RPC over stdio)
2. Create `MCPClientManager` (multi-server management)
3. Implement stdio transport (subprocess spawning)
4. Add tool discovery (`tools/list`)
5. Add tool invocation (`tools/call`)
6. Create `MCPConfigLoader` (parse `mcp.yaml`)
7. Add basic error handling and logging

**Testing:**
- Unit tests with mock MCP server
- Integration test with real MCP server (filesystem or echo)
- Error handling tests (server crash, invalid JSON, etc.)

**Deliverables:**
- `sabre/server/mcp/client.py`
- `sabre/server/mcp/client_manager.py`
- `sabre/config/mcp_config.py`
- `tests/test_mcp_client.py`

### Phase 2: Runtime Integration

**Goal**: Expose MCP tools in Python runtime, invoke from `<helpers>` blocks

**Tasks:**
1. Create `MCPHelperAdapter` to bridge MCP → SABRE helpers
2. Integrate adapter into `PythonRuntime`
3. Generate helper documentation from MCP tool schemas
4. Add MCP tool section to system prompt
5. Test end-to-end: LLM calls MCP tool via `<helpers>` block

**Testing:**
- Test tool invocation from runtime namespace
- Test error handling (tool not found, invalid args)
- Test result transformation (MCP content → SABRE Content)
- Integration test with real conversation

**Deliverables:**
- `sabre/server/mcp/helper_adapter.py`
- Updated `sabre/server/python_runtime.py`
- Updated `sabre/server/prompts/` templates
- `tests/test_mcp_integration.py`

### Phase 3: Configuration & CLI

**Goal**: User-facing configuration and management

**Tasks:**
1. Define `mcp.yaml` schema
2. Create default config at `~/.config/sabre/mcp.yaml`
3. Add CLI commands:
   - `sabre mcp list` - Show connected servers
   - `sabre mcp add <name> <command>` - Add server to config
   - `sabre mcp enable/disable <name>` - Toggle server
4. Add `/mcp` slash command in client (show status)
5. Document MCP integration in CLAUDE.md

**Testing:**
- Test config loading and validation
- Test CLI commands
- Test environment variable expansion
- Test config hot-reload (optional)

**Deliverables:**
- `sabre/config/mcp.yaml` (example config)
- Updated `sabre/cli.py` (mcp subcommands)
- `/mcp` slash command
- Updated `CLAUDE.md` with MCP docs

### Phase 4: SSE Transport

**Goal**: Support remote MCP servers via Server-Sent Events

**Tasks:**
1. Implement SSE transport in `MCPClient`
2. Add WebSocket connection management
3. Handle authentication (headers, tokens)
4. Add reconnection logic
5. Test with remote MCP server

**Testing:**
- Test SSE connection and streaming
- Test authentication
- Test reconnection on disconnect
- Test concurrent requests

**Deliverables:**
- Updated `sabre/server/mcp/client.py` (SSE support)
- `tests/test_mcp_sse.py`

### Phase 5: Resources & Prompts

**Goal**: Support MCP resources (context injection)

**Tasks:**
1. Implement resource discovery (`resources/list`)
2. Implement resource reading (`resources/read`)
3. Add resource templates (`prompts/get`)
4. Inject resources into conversation context (optional)
5. Add helper to read resources from `<helpers>` blocks

**Testing:**
- Test resource listing and reading
- Test resource content transformation
- Test prompt templates
- Integration test with resource-aware conversation

**Deliverables:**
- Updated `sabre/server/mcp/client.py` (resources)
- Resource helpers in runtime
- `tests/test_mcp_resources.py`

## Integration with Persona System

MCP tools can be featured in persona configurations:

```yaml
personas:
  data-engineer:
    name: "Data Engineering Specialist"
    identity: |
      You are an expert data engineer working with databases and data pipelines.

    examples: |
      ### Example: Query database and visualize results

      <helpers>
      # Query Postgres via MCP
      results = Postgres.query("SELECT date, revenue FROM sales ORDER BY date")

      # Create DataFrame
      import pandas as pd
      df = pd.DataFrame(results)

      # Visualize
      import matplotlib.pyplot as plt
      df.plot(x='date', y='revenue')
      plt.title("Revenue Over Time")

      result("See visualization above")
      </helpers>

    featured_helpers:
      - "Postgres.query"
      - "Postgres.execute"
      - "pandas_bind"
      - "matplotlib"
```

**Interaction:**
- Persona config references MCP tools by name
- If MCP server unavailable, warn in logs
- Persona examples show how to use MCP tools
- Same prompt generation pipeline

## Challenges & Risks

### 🔴 Process Management Complexity
- stdio transport requires subprocess management
- Need to handle crashes, hangs, zombies
- Stdout/stderr multiplexing

**Mitigation:**
- Use battle-tested async subprocess libraries
- Set timeouts on all operations
- Monitor process health, restart on failure
- Log all subprocess lifecycle events

### 🔴 Error Handling Across Boundaries
- MCP server errors must propagate to LLM
- Network failures, timeouts, invalid responses
- Debugging is harder (cross-process)

**Mitigation:**
- Structured error messages in `<helpers_result>`
- Detailed logging at all layers
- Health checks and status monitoring
- Graceful degradation (disable failing servers)

### 🔴 Performance Overhead
- Inter-process communication (stdio) has latency
- Tool discovery at startup adds initialization time
- JSON serialization overhead

**Mitigation:**
- Cache tool schemas in memory
- Use connection pooling for SSE transport
- Async I/O to prevent blocking
- Benchmark and optimize hot paths

### 🔴 Security Concerns
- MCP servers run as subprocesses (same trust boundary)
- SSE servers are remote (network trust boundary)
- Tools may access sensitive data

**Mitigation:**
- Explicit allowlist configuration
- Environment variable expansion (not plaintext secrets)
- SSL validation for SSE transport
- Document security best practices
- Audit logs for tool invocations

### 🔴 Schema Compatibility
- MCP spec may evolve
- Different servers may interpret spec differently
- Need to handle version skew

**Mitigation:**
- Test with multiple MCP servers
- Validate schemas at connection time
- Log warnings for unsupported features
- Graceful degradation for unknown fields

### 🔴 LLM Confusion
- More tools = longer prompts = higher cost
- LLM may choose wrong tool
- Tool names may conflict with built-in helpers

**Mitigation:**
- Use persona system to feature relevant tools only
- Clear naming conventions (namespace MCP tools)
- Good tool descriptions in schemas
- Monitor tool usage patterns

## Testing Strategy

### Unit Tests
- `MCPClient`: JSON-RPC message formatting and parsing
- `MCPClientManager`: Connection lifecycle
- `MCPHelperAdapter`: Schema conversion, tool routing
- `MCPConfigLoader`: YAML parsing, validation

### Integration Tests
- Connect to real MCP server (filesystem server in test mode)
- Discover tools, invoke tools, handle errors
- Test stdio subprocess management
- Test SSE connection and streaming

### End-to-End Tests
- Full conversation with MCP tool usage
- LLM calls MCP tool via `<helpers>` block
- Result transformation and display
- Error recovery (server restart, reconnect)

### Performance Tests
- Tool invocation latency
- Concurrent tool calls
- Large response handling
- Connection pool saturation (SSE)

## Future Enhancements

### Multi-Turn Tool Calls
- Some MCP tools may require follow-up (e.g., auth flow)
- Support stateful tool interactions
- Handle progress updates and streaming

### Tool Composition
- Chain multiple MCP tools together
- Create higher-level helpers that use MCP tools internally
- Example: `ETL.pipeline(source, transform, dest)` uses multiple MCP tools

### Resource Caching
- Cache frequently accessed resources (e.g., database schemas)
- Invalidate on change notifications
- Reduce latency for repeated access

### MCP Server Health Dashboard
- Web UI showing connected servers
- Tool usage statistics
- Error rates and logs
- Restart/reconnect controls

### SABRE as MCP Server
- Expose SABRE's capabilities as MCP server
- Other AI tools can use SABRE's helpers
- Enables composition (SABRE using SABRE)

## References

- [Model Context Protocol Spec](https://spec.modelcontextprotocol.io/)
- [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Official MCP Servers](https://github.com/modelcontextprotocol/servers)

## Appendix: Example MCP Server Integration

### Using Postgres MCP Server

**1. Install MCP server:**
```bash
npm install -g @modelcontextprotocol/server-postgres
```

**2. Configure in `~/.config/sabre/mcp.yaml`:**
```yaml
mcp_servers:
  postgres:
    type: stdio
    command: npx
    args: ["-y", "@modelcontextprotocol/server-postgres"]
    env:
      POSTGRES_URL: "postgresql://localhost/mydb"
    enabled: true
```

**3. Start SABRE:**
```bash
uv run sabre
# Logs: "Connected to MCP server: postgres (2 tools available)"
```

**4. Use in conversation:**
```
User: Show me the top 10 users by revenue

<helpers>
# Query via MCP Postgres tool
results = Postgres.query("""
    SELECT user_id, name, SUM(amount) as total_revenue
    FROM orders
    JOIN users ON orders.user_id = users.id
    GROUP BY user_id, name
    ORDER BY total_revenue DESC
    LIMIT 10
""")

# Display as table
import pandas as pd
df = pd.DataFrame(results)
print(df.to_markdown())

result("Top 10 users by revenue shown above")
</helpers>
```

**5. SABRE orchestrator:**
- Parses `<helpers>` block
- Sees `Postgres.query()` call
- Routes to postgres MCP server via `MCPHelperAdapter`
- MCP server executes SQL query
- Returns results as MCP content
- Adapter transforms to SABRE Content
- Injected into `<helpers_result>`
- LLM continues with results

## Appendix: Custom MCP Server Example

Users can build custom MCP servers for proprietary systems:

**custom_mcp_server.py:**
```python
#!/usr/bin/env python3
from mcp.server import Server, Tool
from mcp.types import TextContent
import asyncio

app = Server("my-custom-server")

@app.tool()
async def get_user_data(user_id: str) -> list[TextContent]:
    """Fetch user data from internal API."""
    # Call your internal API
    data = await fetch_from_api(user_id)
    return [TextContent(type="text", text=str(data))]

@app.tool()
async def send_notification(user_id: str, message: str) -> list[TextContent]:
    """Send notification to user."""
    # Call notification service
    await send_to_service(user_id, message)
    return [TextContent(type="text", text="Notification sent")]

if __name__ == "__main__":
    asyncio.run(app.run())
```

**Add to `mcp.yaml`:**
```yaml
mcp_servers:
  custom:
    type: stdio
    command: python
    args: ["/path/to/custom_mcp_server.py"]
    env:
      API_KEY: "${MY_API_KEY}"
    enabled: true
```

**Use in SABRE:**
```python
<helpers>
# Use custom MCP tools
user_data = Custom.get_user_data("user123")
Custom.send_notification("user123", "Your report is ready")
result("Processed user data and sent notification")
</helpers>
```
