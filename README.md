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

SABRE includes benchmark infrastructure for evaluating performance against standard LLM benchmarks like DataSciBench.

### Setup

```bash
# Create HuggingFace token file (optional but recommended)
echo "your_hf_token_here" > ~/.hf/key

# Run the setup script (requires Python 3.11)
python benchmarks/DataSciBench/setup_benchmark.py
```

The setup script automatically:
- Clones DataSciBench repository
- Creates Python 3.11 virtual environment
- Installs MetaGPT and all dependencies
- Downloads ground truth data from HuggingFace (if `~/.hf/key` exists)

### Run Benchmarks

The `run_benchmark.py` wrapper:
- Runs benchmark tasks with compatibility patches
- Evaluates outputs against ground truth
- Stores results in `benchmarks/DataSciBench/results/`
- Falls back to downloading ground truth if missing (using `~/.hf/key`)

```bash
# Start SABRE server (required for SABRE benchmarks)
uv run sabre-server

# Run baseline benchmark (gpt-4o-mini via OpenAI)
OPENAI_API_KEY=`cat ~/.openai/key` python benchmarks/DataSciBench/run_benchmark.py \
  --task_id csv_excel_0 \
  --config benchmarks/DataSciBench/configs/config_gpt4o_baseline.yaml \
  --data_type csv \
  --max_runs 1

# Run SABRE benchmark (gpt-4o via localhost:8011)
OPENAI_API_KEY=`cat ~/.openai/key` python benchmarks/DataSciBench/run_benchmark.py \
  --task_id csv_excel_0 \
  --config benchmarks/DataSciBench/configs/config_sabre.yaml \
  --data_type csv \
  --max_runs 1
```

Results are stored in `benchmarks/DataSciBench/results/{config_name}/{task_id}/` with evaluation scores, execution logs, and output files.
