# SABRE

**Smart Agent Based on Recursive Execution**

Persona-driven AI agent with:
- Recursive execution engine with error correction
- OpenAI Conversastions API and Responses API for state management

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

```
# Open up Terminal UI
OPENAI_API_KEY=$(cat ~/.openai/key) uv run sabre

# Run as script
OPENAI_API_KEY=$(cat ~/.openai/key) uv run sabre "message"
```

```bash
uv run sabre "message"
```
