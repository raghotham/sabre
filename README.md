# SABRE

**Smart Agent Based on Recursive Execution**

Persona-driven AI agent with:
- Recursive execution engine with error correction
- OpenAI Conversations API and Responses API for state management
- MCP (Model Context Protocol) integration for external tools
- Rewrite of [llmvm](https://github.com/9600dev/llmvm) using Conversations and Responses APIs

## Setup

```bash
git clone https://github.com/raghotham/sabre
cd sabre
uv sync
uvx playwright install chromium --only-shell
```

## Environment Variables

### Required

- **`OPENAI_API_KEY`** - Your OpenAI API key (required for LLM calls)

### Optional

- `OPENAI_BASE_URL` - Custom OpenAI API endpoint (default: OpenAI's official API)
- `OPENAI_MODEL` - Default model to use (default: `gpt-4o`)
- `PORT` - Server port (default: `8011`)
- `SABRE_HOME` - Override all SABRE directories to use a single base directory
- `LOG_LEVEL` - Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`)
- `SABRE_THEME` - Force theme: `light` or `dark` (auto-detected by default)

## Commands

```bash
# Open up Terminal UI
OPENAI_API_KEY=$(cat ~/.openai/key) uv run sabre

# Run as script
OPENAI_API_KEY=$(cat ~/.openai/key) uv run sabre "message"

# Stop the server
uv run sabre --stop

# Clean up all SABRE data (removes logs, files, cache)
uv run sabre --clean
uv run sabre --clean --force  # Skip confirmation
```

## Data Storage

SABRE stores data in XDG-compliant directories:

- **Data**: `~/.local/share/sabre/` - Generated files, conversation data
- **State**: `~/.local/state/sabre/logs/` - Logs, PID files
- **Config**: `~/.config/sabre/` - User configuration (future)
- **Cache**: `~/.cache/sabre/` - Cache (future)

Use `uv run sabre --clean` to remove all stored data.

## Running Benchmarks

See [benchmarks/README.md](benchmarks/README.md) for complete benchmark documentation.
