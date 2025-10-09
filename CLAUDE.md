# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Always use `uv run` for running Python commands.**

### Running SABRE
```bash
# Start server and client (recommended)
uv run sabre

# Or run components separately
uv run sabre-server    # Server only
uv run sabre-client    # Client only

# Alternative commands
uv run python -m sabre.server
uv run python -m sabre.client
```

### Dependencies
```bash
# Install all dependencies
uv sync

# Install with test dependencies
uv sync --dev
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_executor.py

# Run specific test
uv run pytest tests/test_executor.py::test_simple_execution

# Run with coverage
uv run pytest --cov=sabre --cov-report=html

# Run with output visible (see prints)
uv run pytest -s
```

### Environment Variables
```bash
# Required
OPENAI_API_KEY="sk-..."

# Optional
OPENAI_BASE_URL="https://api.openai.com/v1"  # For custom endpoints
OPENAI_MODEL="gpt-4o"                        # Override default model
PORT="8011"                                   # Server port (default: 8011)

# XDG directories (for testing)
SABRE_HOME="/tmp/sabre-test"  # Override all XDG paths
```

## High-Level Architecture

### Core Concept

SABRE is a **persona-driven AI agent** using OpenAI's Responses API with a **continuation-passing execution model**:

1. LLM generates response with optional `<helpers>` code blocks
2. Python runtime executes helper code (may call `llm_call()` recursively)
3. Results injected as `<helpers_result>` tags
4. Continue until completion or max iterations

### Key Components

#### 1. Orchestrator (`sabre/server/orchestrator.py`)

The orchestration engine that manages the continuation loop:

- **Main loop**: Streams LLM responses → parses `<helpers>` → executes → continues
- **Streaming parser**: Real-time detection of `<helpers>` blocks using state machine
- **Tree tracking**: Maintains execution tree for debugging and event routing
- **Image handling**: Dual operation for matplotlib figures:
  - Saves to disk at `XDG_DATA_HOME/sabre/files/{conversation_id}/`
  - Uploads to OpenAI Files API for token-efficient continuation (~15k tokens → 1 token)
  - Markdown URLs in `<helpers_result>` for LLM to reference

**Critical**: `orchestrator.run()` can be called recursively from `llm_call()` helper, creating nested execution trees.

#### 2. Python Runtime (`sabre/server/python_runtime.py`)

Isolated Python execution environment for `<helpers>` blocks:

- **Namespace**: Provides helpers (Bash, Search, Web, llm_call, etc.)
- **Execution**: Synchronous `exec()` with stdout capture
- **Matplotlib handling**: Auto-captures figures after execution, converts to base64 PNG
- **Recursive LLM calls**: Runtime has reference to orchestrator for `llm_call()` → `orchestrator.run()`

**Key helpers available**:
- `llm_call(expr_list, instructions)`: Delegate sub-tasks to LLM (uses new conversation)
- `llm_bind(data, "function_sig")`: Extract structured data using LLM
- `llm_list_bind(data, instructions, count)`: Extract list of items
- `pandas_bind(data)`: Create intelligent DataFrame with `.ask()` method
- `Bash.execute(cmd)`: Run bash commands
- `Search.search(query)`: DuckDuckGo search
- `Web.get(url)`: Fetch web content
- `download(urls)`: Download files
- `result(value)`: Collect results for output

#### 3. Response Executor (`sabre/common/executors/response.py`)

Wrapper for OpenAI Responses API:

- **Streaming**: Token-by-token streaming with event callbacks
- **Continuation**: Uses `response_id` for context management
- **Instructions**: System instructions must be passed on EVERY call (not persisted)
- **Error handling**: Automatic retry with exponential backoff for rate limits
- **Token counting**: Tracks input/output/reasoning tokens

#### 4. Client (`sabre/client/client.py`)

Terminal UI using prompt_toolkit:

- **Event-driven**: Receives streaming events from server via WebSocket
- **Rich rendering**: Syntax highlighting for code, markdown for text
- **Slash commands**: `/help`, `/clear`, `/exit`, etc. (see `sabre/client/slash_commands/`)
- **Image display**: Renders matplotlib figures inline using terminal graphics protocols

### Directory Structure

```
sabre/
├── server/
│   ├── orchestrator.py          # Main continuation loop
│   ├── python_runtime.py         # Code execution environment
│   ├── streaming_parser.py       # <helpers> tag parser
│   ├── helpers/                  # Helper implementations
│   │   ├── llm_call.py          # Recursive LLM delegation
│   │   ├── llm_bind.py          # Structured data extraction
│   │   ├── bash.py              # Bash execution
│   │   ├── search.py            # Web search
│   │   └── web.py               # HTTP fetching
│   ├── prompts/                  # System prompts
│   │   └── python_continuation_execution_responses.prompt
│   └── api/                      # WebSocket server
│       └── server.py
├── client/
│   ├── client.py                 # Terminal UI
│   └── slash_commands/           # Client commands
├── common/
│   ├── executors/
│   │   └── response.py           # OpenAI Responses API wrapper
│   ├── models/                   # Data models (events, messages, trees)
│   ├── utils/
│   │   └── prompt_loader.py      # Prompt template system
│   └── paths.py                  # XDG directory management
└── cli.py                        # Main entry point

llmvm/                            # Legacy LLMVM (reference only)
tests/                            # Test suite
plans/                            # Architecture docs
```

### Execution Flow

**Example: User asks to plot data**

1. **Client** sends message via WebSocket → **Server**
2. **Server** creates ExecutionTree, calls `orchestrator.run()`
3. **Orchestrator** calls `executor.execute()` (OpenAI Responses API)
4. **Streaming tokens**: LLM generates response with `<helpers>` block containing matplotlib code
5. **Parser** detects `<helpers>` tags, extracts code
6. **Runtime** executes code: `plt.plot([1,2,3])` → captures figure as base64 PNG
7. **Orchestrator** saves image to disk, uploads to Files API, builds `<helpers_result>` with markdown URL
8. **Continuation**: Orchestrator calls `executor.execute()` again with result
9. **LLM** sees result (with image via file_id), generates final response without `<helpers>`
10. **Complete**: Final response streamed to client

### Image Handling Architecture

SABRE has a sophisticated dual-path approach for matplotlib figures:

**Problem**: Images consume massive tokens (~15k per image per message in base64)

**Solution**: Three-stage handling
1. **Execution**: Runtime captures matplotlib figures as base64 ImageContent
2. **Disk storage**: Orchestrator saves to `XDG_DATA_HOME/sabre/files/{conversation_id}/figure_*.png`
3. **Files API upload**: Converts base64 → `file_id` for token-efficient continuation
4. **LLM reference**: Markdown URLs in `<helpers_result>` for LLM responses

**Result**: Images persist across continuations at 1 token (file_id) instead of 15k tokens (base64)

### Persona System (Planned)

See `plans/PERSONA_PLAN.md` for detailed architecture. The persona system will enable:

- **Helper filtering**: Only relevant tools for domain (e.g., data-scientist gets database helpers only)
- **Domain workflows**: Persona-specific best practices
- **Token efficiency**: Fewer helpers in prompt = fewer tokens
- **Focused expertise**: Clearer guidance for specific domains

## XDG Directory Compliance

SABRE follows XDG Base Directory Specification:

- **Data**: `~/.local/share/sabre/` - Generated files, conversation data
- **Config**: `~/.config/sabre/` - User configuration (future personas.yaml)
- **State**: `~/.local/state/sabre/logs/` - Logs, PID files
- **Cache**: `~/.cache/sabre/` - (reserved for future use)

Override for testing: `export SABRE_HOME=/tmp/sabre-test`

## Development Patterns

### Adding a New Helper

1. Create helper class in `sabre/server/helpers/your_helper.py`
2. Add to runtime namespace in `sabre/server/python_runtime.py:reset()`
3. Document in system prompt at `sabre/server/prompts/python_continuation_execution_responses.prompt`

### Modifying System Prompts

Prompts use template system with `{{variable}}` substitution:

```python
PromptLoader.load(
    'python_continuation_execution_responses.prompt',
    template={'functions': helper_signatures, 'context_window_tokens': '128000'}
)
```

Returns dict with `system_message` and `user_message` keys.

### Event System

All orchestrator operations emit events via callback:
- `ResponseStartEvent` - LLM call starting
- `ResponseTokenEvent` - Streaming token
- `ResponseTextEvent` - Full response received
- `HelpersExtractedEvent` - Code block found
- `HelpersStartEvent` - Execution starting
- `HelpersEndEvent` - Execution complete (includes result)
- `CompleteEvent` - Orchestration finished
- `ErrorEvent` - Error occurred

Client subscribes to events via WebSocket for real-time UI updates.

### Testing Strategy

- **Unit tests**: Individual components (executor, parser, helpers)
- **Integration tests**: Orchestrator with mock LLM
- **E2E tests**: Full client-server flow (requires API key)

Tests require `OPENAI_API_KEY` - automatically skipped if not found.

## Important Quirks

### 1. Instructions Must Be Passed Every Time

OpenAI Responses API does NOT persist instructions across calls. Always pass:

```python
await executor.execute(
    conversation_id=conv_id,
    input_text=message,
    instructions=system_instructions,  # Required on every call!
)
```

### 2. Recursive Orchestrator Calls

`llm_call()` creates a NEW conversation and recursively calls `orchestrator.run()`:

```python
# In runtime namespace
llm_call(['data'], 'Analyze this')
  → orchestrator.run(conversation_id=NEW, input_text=...)
    → nested execution tree
```

This enables powerful decomposition but requires careful tree management.

### 3. Helper Code Runs Synchronously

Runtime uses synchronous `exec()`, but helpers can use `async`:

```python
# In helper
async def llm_call(...):
    result = await orchestrator.run(...)  # Async call
    return result

# Runtime wraps async helpers
if asyncio.iscoroutinefunction(helper):
    result = asyncio.run(helper(...))
```

### 4. Image Base64 Token Consumption

**Never** pass base64 image data back to LLM in text - use Files API:
- Runtime captures matplotlib as ImageContent with base64
- Orchestrator uploads to Files API → file_id
- Only file_id goes in continuation (1 token vs 15k tokens)

### 5. Legacy LLMVM Directory

`llmvm/` contains the original LLMVM implementation for reference only. Active development is in `sabre/`.

## Common Tasks

### Debugging Orchestrator Flow

Enable debug logging:
```python
import logging
logging.getLogger('sabre.server.orchestrator').setLevel(logging.DEBUG)
```

Check logs at: `~/.local/state/sabre/logs/server.log`

### Running Single Test with Output

```bash
uv run pytest tests/test_executor.py::test_simple_execution -s -v
```

### Testing with Custom Model

```bash
export OPENAI_MODEL="gpt-4o-mini"
uv run sabre
```

### Stopping Stuck Server

```bash
uv run sabre --stop
# Or manually: pkill -f sabre.server
```
