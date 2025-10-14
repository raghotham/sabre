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

## Environment Variables

### Required

- **`OPENAI_API_KEY`** - Your OpenAI API key (required for LLM calls)

### Optional - API Configuration

- **`OPENAI_BASE_URL`** - Custom OpenAI API endpoint (default: OpenAI's official API)
- **`OPENAI_MODEL`** - Default model to use (default: `gpt-4o`)
- **`PORT`** - Server port (default: `8011`)

### Optional - Paths (XDG Base Directory Compliance)

- **`SABRE_HOME`** - Override all SABRE directories to use a single base directory
- **`XDG_DATA_HOME`** - Base directory for data files (default: `~/.local/share`)
- **`XDG_CONFIG_HOME`** - Base directory for config files (default: `~/.config`)
- **`XDG_STATE_HOME`** - Base directory for state files like logs (default: `~/.local/state`)
- **`XDG_CACHE_HOME`** - Base directory for cache files (default: `~/.cache`)

### Optional - Configuration

- **`SABRE_CONFIG`** - Path to custom config file (default: `~/.config/sabre/config.yaml`)
- **`LOG_LEVEL`** - Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`)

### Optional - UI/Theme

- **`SABRE_THEME`** - Force theme: `light` or `dark` (auto-detected by default)
- **`COLORFGBG`** - Terminal background color hint (auto-detected)
- **`ITERM_PROFILE`** - iTerm2 profile name (auto-detected)
- **`TERM_PROGRAM`** - Terminal program identifier (auto-detected)

### Directory Structure

When using default XDG paths:

```
~/.local/share/sabre/     # Data directory (XDG_DATA_HOME/sabre)
  ├── files/              # Generated files (images, data)
  │   └── {conv_id}/      # Per-conversation files
  └── memory/             # Conversation memory/context

~/.config/sabre/          # Config directory (XDG_CONFIG_HOME/sabre)
  └── config.yaml         # Configuration file

~/.local/state/sabre/     # State directory (XDG_STATE_HOME/sabre)
  └── logs/               # Application logs
      └── server.log

~/.cache/sabre/           # Cache directory (XDG_CACHE_HOME/sabre)
```

### Example Configuration

```bash
# Minimal setup
export OPENAI_API_KEY="sk-..."

# Custom configuration
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"
export PORT="8080"
export LOG_LEVEL="DEBUG"
export SABRE_THEME="dark"

# Custom base directory (overrides all XDG paths)
export SABRE_HOME="~/my-sabre"
```

## Commands

```bash
uv run sabre "message"
```

## Architecture

- **`sabre/`** - Main package (persona-driven AI agent)
- **`tests/`** - Test suite
- **`plans/`** - Architecture and planning documents

See `plans/PERSONA_PLAN.md` for detailed architecture.
