# Harbor Benchmarks

SABRE integration with the [Harbor benchmark framework](https://github.com/laude-institute/harbor).

Harbor provides containerized task execution with standardized verification and scoring across multiple datasets like Terminal-Bench.

## Prerequisites

```bash
# Check prerequisites (SABRE server, Docker, Harbor CLI)
./benchmarks/harbor/setup_benchmark.py
```

The setup script verifies:
- SABRE server is running and accessible
- Docker is installed and running
- Harbor CLI is available (`uvx harbor`)

## Quick Start

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=$(cat ~/.openai/key)

# Run hello-world example (quick test)
uv run benchmarks/harbor/run_benchmark.py --dataset hello-world@head

# Run specific task from a dataset
uv run benchmarks/harbor/run_benchmark.py --dataset terminal-bench@2.0 --task chess-best-move
```

The `run_benchmark.py` wrapper simplifies running Harbor benchmarks by handling:
- Prerequisites checking (Docker, API key)
- Directory setup (jobs, results)
- Results archiving to `benchmarks/harbor/results/`

## Running Benchmarks

### Using the Wrapper Script (Recommended)

```bash
# Run all tasks in a dataset
uv run benchmarks/harbor/run_benchmark.py --dataset terminal-bench@2.0

# Run specific task
uv run benchmarks/harbor/run_benchmark.py --dataset terminal-bench@2.0 --task chess-best-move

# Enable debug logging
uv run benchmarks/harbor/run_benchmark.py --dataset hello-world@head --debug

# List tasks in a dataset
uv run benchmarks/harbor/run_benchmark.py --dataset terminal-bench@2.0 --list-tasks
```

### Testing with Different Models

You can test SABRE with different models or API endpoints by setting environment variables:

```bash
# Use a different OpenAI model
export OPENAI_MODEL=gpt-4o-mini
uv run benchmarks/harbor/run_benchmark.py --dataset hello-world@head

# Use a custom API endpoint (e.g., local model server)
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_MODEL=local-model
uv run benchmarks/harbor/run_benchmark.py --dataset hello-world@head

# Test with Claude via OpenAI-compatible endpoint
export OPENAI_BASE_URL=https://api.anthropic.com/v1
export OPENAI_MODEL=claude-3-5-sonnet-20241022
uv run benchmarks/harbor/run_benchmark.py --dataset hello-world@head
```

These environment variables are automatically passed to SABRE running inside the Harbor container.

### Using Harbor CLI Directly

```bash
# List available datasets
uvx harbor list

# Run Terminal-Bench
uvx harbor run -d terminal-bench@head \
  --agent-import-path benchmarks.harbor.container:SabreAgent \
  --env docker \
  --ek OPENAI_API_KEY=$OPENAI_API_KEY

# Run with multiple trials
uvx harbor run -d hello-world@head \
  --agent-import-path benchmarks.harbor.container:SabreAgent \
  --env docker \
  --ek OPENAI_API_KEY=$OPENAI_API_KEY \
  --num-trials 5
```

## How It Works

1. **Container Build**: Harbor builds a Docker container from the task's Dockerfile
2. **SABRE Installation**: `container/install.sh` installs SABRE and dependencies (including Playwright)
3. **Task Execution**: `container/agent.py` (SabreAgent) runs SABRE in command mode with the task instruction
4. **Verification**: Harbor's verifier checks the results and assigns a reward score

### Architecture

```
benchmarks/harbor/
├── run_benchmark.py       # Simplified wrapper for running benchmarks
├── setup_benchmark.py     # Prerequisites checker (uv script)
├── container/
│   ├── agent.py          # SabreAgent (extends Harbor's BaseInstalledAgent)
│   ├── install.sh        # SABRE installation script for containers
│   └── __init__.py       # Package exports
├── results/              # Permanent results (copied by run_benchmark.py)
│   └── <dataset>_<version>/
│       └── <timestamp>_result.json
└── tmp/harbor/           # Temporary job outputs (Harbor default)
    └── <timestamp>/
        └── <task_name>/
            ├── result.json        # Trial results and metrics
            ├── agent/             # SABRE logs and execution traces
            │   ├── setup/         # Installation logs
            │   ├── command-0/     # SABRE execution output
            │   └── sabre_sessions/ # SABRE session data with ATIF
            └── verifier/          # Test results and reward scores
```

## Results

**Using run_benchmark.py**: Results are automatically copied to `benchmarks/harbor/results/<dataset>_<version>/` for permanent storage.

**Using Harbor CLI directly**: Results are stored in `tmp/harbor/<timestamp>/`.

Each trial includes:

- **`result.json`** - Trial metadata, agent info, verifier results, timing
- **Agent logs** - SABRE execution traces, ATIF output
- **Verifier output** - Test results, reward scores, error messages

Example result.json:
```json
{
  "task_name": "hello-world",
  "agent_info": {
    "name": "sabre",
    "version": "latest",
    "model_info": {
      "name": "gpt-4o",
      "provider": "openai"
    }
  },
  "verifier_result": {
    "rewards": {
      "reward": 1.0
    }
  }
}
```

## Implementation Details

### SabreAgent (container/agent.py)

- Extends `Harbor.agents.installed.base.BaseInstalledAgent`
- Copies SABRE source code to container during setup
- Runs SABRE in command mode: `uv run sabre -m "instruction"`
- Prefixes instructions to guide SABRE's behavior:
  - Use `Bash.execute()` for file operations (not file helpers)
  - Current working directory is `/app`

### Installation Script (container/install.sh)

Installs in the container:
1. System dependencies (curl, git)
2. uv (Python package manager)
3. SABRE (via `uv sync` from copied source)
4. Playwright (for Web helper): `uv run playwright install chromium --only-shell --with-deps`

### Platform Compatibility

SABRE detects the container platform (Linux) and uses appropriate paths:
- Playwright cache: `~/.cache/ms-playwright` (Linux) vs `~/Library/Caches/ms-playwright` (macOS)

## Adding New Datasets

Harbor benchmarks work with any Harbor dataset out-of-the-box. No code changes needed!

```bash
# List available datasets
uvx harbor list

# Run any dataset
uvx harbor run -d <dataset-name>@head \
  --agent-import-path container:SabreAgent \
  --env docker \
  --ek OPENAI_API_KEY=$OPENAI_API_KEY
```

## Troubleshooting

### Prerequisites Check Failed

```bash
# Verify SABRE server
curl http://localhost:8011/v1/health

# Verify Docker
docker info

# Verify Harbor
uvx harbor --help
```

### Container Build Errors

Check Docker logs for the most recent container:
```bash
docker ps -a --format "{{.ID}} {{.CreatedAt}}" | head -1
docker logs <container-id>
```

### SABRE Server Crashes in Container

Check SABRE server logs in the trial results:
```bash
cat jobs/<timestamp>/<task-name>/agent/sabre_source/sabre/logs/server.log
```

Common issues:
- Missing Playwright installation
- Missing OPENAI_API_KEY environment variable
- Platform-specific path issues

## Known Limitations

1. **Token tracking**: Currently `populate_context_post_run()` sets token counts to 0. Future: Parse ATIF trace for actual token usage.

2. **File operations**: SABRE's file helpers (like `write_file()`) save to server-managed directories. The instruction prefix guides SABRE to use `Bash.execute()` instead for proper file placement.

3. **Setup time**: Installing SABRE and Playwright takes ~60-70 seconds per trial. Consider using Harbor's caching features for multiple trials.
