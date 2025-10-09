# SABRE

**Self-Amplifying Backtracking Recursive Execution**

Persona-driven AI agent with:
- Recursive execution engine with error correction
- Persona-based helper filtering for focused expertise
- OpenAI Responses API for state management
- WebSocket streaming server
- Minimal terminal client with rich rendering

## Quick Start

```bash
# Install dependencies
uv sync

# Run SABRE server
uv run python -m sabre.server

# Or use the CLI
uv run sabre
```

## Commands

```bash
# Server
uv run python -m sabre.server
uv run sabre-server

# Client
uv run python -m sabre.client
uv run sabre-client

# CLI (starts both server and client)
uv run sabre
```

## Architecture

- **`sabre/`** - Main package (persona-driven AI agent)
- **`llmvm/`** - Legacy LLMVM implementation (for reference)
- **`tests/`** - Test suite
- **`plans/`** - Architecture and planning documents

See `plans/PERSONA_PLAN.md` for detailed architecture.
