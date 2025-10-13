# SABRE

**Smart Agent Based on Recursive Execution**

Persona-driven AI agent with:
- Recursive execution engine with error correction
- Persona-based helper filtering for focused expertise
- OpenAI Responses API for state management
- WebSocket streaming server
- Minimal terminal client with rich rendering

## Quick Start

```bash
# Open up Terminal UI
uv run sabre

# Run as script
uv run sabre "message"
```

## Commands

```bash
uv run sabre "message
```

## Architecture

- **`sabre/`** - Main package (persona-driven AI agent)
- **`tests/`** - Test suite
- **`plans/`** - Architecture and planning documents

See `plans/PERSONA_PLAN.md` for detailed architecture.
