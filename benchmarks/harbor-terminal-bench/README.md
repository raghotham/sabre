# Harbor Terminal-Bench 2.0 Benchmark for SABRE

This directory contains the SABRE agent implementation for evaluation on Terminal-Bench 2.0 via the Harbor framework.

## Overview

Terminal-Bench 2.0 is a benchmark for evaluating AI agents on real-world terminal tasks, from compiling code to training models and setting up servers. Harbor is the execution framework that runs agents in isolated Docker containers.

SABRE is integrated as an `AbstractInstalledAgent`, which means:
1. Harbor installs SABRE inside the container using `install-sabre.sh.j2`
2. SABRE runs in script mode: `uv run sabre "task description"`
3. Results are captured and converted to ATIF trajectory format

## Architecture

```
Harbor Host                     Container
+--------------+               +-------------------------+
| Harbor       |               | 1. setup.sh installs    |
| Orchestrator |--executes---->|    SABRE (via uv)       |
|              |               | 2. "sabre <task>"       |
|              |               |    runs inside          |
|              |               | 3. SABRE has direct     |
|              |               |    shell/file access    |
+--------------+               +-------------------------+
```

## Directory Structure

```
harbor-terminal-bench/
├── README.md                    # This file
├── setup_benchmark.py           # Setup and prerequisite checker
├── run_benchmark.sh             # Main benchmark runner
├── sabre_harbor/                # SABRE Harbor agent package
│   ├── __init__.py              # Exports SabreAgent
│   ├── agent.py                 # AbstractInstalledAgent implementation
│   └── install-sabre.sh.j2      # Container installation script (Jinja2)
└── results/                     # Benchmark results storage
```

## Prerequisites

### 1. Install Harbor

```bash
# Using pip
pip install harbor-bench

# Or using uv
uv pip install harbor-bench

# Verify installation
harbor --version
```

### 2. Install Docker

Harbor runs agents in Docker containers. Install Docker from:
https://docs.docker.com/get-docker/

```bash
# Verify Docker is running
docker info
```

### 3. Set Environment Variables

```bash
# Required: OpenAI API key
export OPENAI_API_KEY="sk-..."

# Optional: Custom API endpoint (for Azure, etc.)
export OPENAI_BASE_URL="https://api.openai.com/v1"

# Optional: Default model
export OPENAI_MODEL="gpt-4o"
```

### 4. Verify Setup

```bash
# Run the setup script to check all prerequisites
python setup_benchmark.py

# Or just check prerequisites without testing agent
python setup_benchmark.py --check

# Test agent import and validation
python setup_benchmark.py --test-agent
```

## Running Benchmarks

### Quick Start

```bash
# Run debug task (hello-world dataset - single simple task)
./run_benchmark.sh --model gpt-4o --dataset hello-world@head

# Run full Terminal-Bench 2.0 (89 tasks)
./run_benchmark.sh --model gpt-4o

# Run specific task within Terminal-Bench
./run_benchmark.sh --model gpt-4o --task-name fix-broken-env

# Run with multiple concurrent tasks
./run_benchmark.sh --model gpt-4o --n-concurrent 4
```

### Command Options

```
./run_benchmark.sh [options]

Options:
  --model MODEL           Model to use (default: gpt-4o)
  --dataset DATASET       Dataset to run (default: terminal-bench@2.0)
  --task-name TASK        Run specific task by name (within the dataset)
  --n-concurrent N        Number of concurrent tasks (default: 4)
  --jobs-dir DIR          Output directory for results
  --env ENV               Environment type: docker or local (default: docker)
  --version VERSION       SABRE version/branch to install (default: main)
  --help                  Show help message
```

### Available Datasets

```bash
# List available datasets
harbor datasets list --verbose
```

Key datasets:
- `hello-world@head` - Single simple task for debugging
- `terminal-bench@2.0` - 89 terminal environment tasks (default)
- `swebench-verified@1.0` - 500 SWE-bench tasks

### Running via Harbor CLI Directly

```bash
# Debug task (hello-world)
harbor run -d hello-world@head \
  --agent-import-path sabre_harbor:SabreAgent \
  --env docker

# Full Terminal-Bench 2.0
harbor run -d terminal-bench@2.0 \
  --agent-import-path sabre_harbor:SabreAgent \
  --jobs-dir ./results/run1 \
  --n-concurrent 4 \
  --env docker

# Specific task within a dataset
harbor run -d terminal-bench@2.0 \
  --agent-import-path sabre_harbor:SabreAgent \
  --task-name fix-broken-env \
  --env docker
```

## Agent Implementation Details

### SabreAgent Class

The `SabreAgent` class (`sabre_harbor/agent.py`) implements Harbor's `BaseInstalledAgent` interface:

| Method/Property | Purpose |
|-----------------|---------|
| `name()` | Returns `"sabre"` |
| `version()` | Returns SABRE version or `"latest"` |
| `_install_agent_template_path` | Path to `install-sabre.sh.j2` |
| `create_run_agent_commands(instruction)` | Generates `uv run sabre "instruction"` command |
| `populate_context_post_run(context)` | Parses output to ATIF trajectory format |

### Installation Script

The `install-sabre.sh.j2` template:
1. Installs system dependencies (curl, git)
2. Installs `uv` (Python package manager)
3. Clones SABRE from GitHub
4. Runs `uv sync` to install dependencies
5. Verifies installation

### Environment Variables Passed to SABRE

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes | API key for LLM calls |
| `OPENAI_MODEL` | No | Model to use (default: gpt-4o) |
| `OPENAI_BASE_URL` | No | Custom API endpoint |
| `LOG_LEVEL` | No | Logging level (default: WARNING) |

## Results Format

Results are stored in the jobs directory with the following structure:

```
results/
└── sabre-gpt-4o-20241210-120000/
    ├── task-1/
    │   ├── trajectory.json    # ATIF format trajectory
    │   ├── stdout.txt         # Agent stdout
    │   └── stderr.txt         # Agent stderr
    ├── task-2/
    │   └── ...
    └── summary.json           # Overall results
```

## Troubleshooting

### Harbor CLI Not Found

```bash
# Install harbor
pip install harbor-bench

# Or add to PATH if installed via uv
export PATH="$HOME/.local/bin:$PATH"
```

### Docker Permission Denied

```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
# Log out and back in

# Or use sudo (not recommended for production)
sudo ./run_benchmark.sh
```

### Agent Import Error

```bash
# Ensure sabre_harbor is in PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Test import
python -c "from sabre_harbor import SabreAgent; print(SabreAgent.name())"
```

### Container Installation Fails

Check the installation logs in the jobs directory. Common issues:
- Network connectivity (can't clone SABRE repo)
- Git not available in container
- uv installation fails

### SABRE Execution Fails

1. Check OPENAI_API_KEY is set correctly
2. Verify model name is valid
3. Check task timeout (default: 1 hour)

## Development

### Testing Locally

```bash
# Test agent import
python setup_benchmark.py --test-agent

# Run with verbose output
LOG_LEVEL=DEBUG ./run_benchmark.sh --task-id hello-world
```

### Modifying the Agent

1. Edit `sabre_harbor/agent.py` for agent behavior
2. Edit `sabre_harbor/install-sabre.sh.j2` for installation changes
3. Test with `python setup_benchmark.py --test-agent`

## References

- [Terminal-Bench](https://www.tbench.ai/) - Benchmark documentation
- [Harbor Framework](https://github.com/laude-institute/harbor) - Execution framework
- [SABRE](https://github.com/raghotham/sabre) - SABRE agent
- [Harbor Docs](https://harborframework.com/docs) - Harbor documentation
