# SABRE Benchmarks

This directory contains benchmark implementations for evaluating SABRE's performance against standard LLM agent benchmarks.

## Available Benchmarks

- **[Harbor](harbor/README.md)** (Recommended) - Containerized benchmark framework supporting Terminal-Bench and other datasets
- **[k8s-ai-bench](k8s-ai-bench/README.md)** - Kubernetes task benchmark using MCP kubectl-ai tools
- **[DataSciBench](DataSciBench/)** - Data science benchmark suite (standalone)

## Benchmark Structure

Standalone benchmarks follow this standardized structure:

```
benchmarks/
└── <BenchmarkName>/
    ├── setup_<benchmark>.py    # Setup script (clone repos, install deps, download data)
    ├── run_<benchmark>.py      # Benchmark runner (execute tasks, evaluate, store results)
    ├── configs/                # Configuration files (baseline, SABRE, etc.)
    │   ├── config_baseline.yaml
    │   └── config_sabre.yaml
    ├── results/                # Benchmark results (auto-generated)
    │   └── <config_name>/
    │       └── <task_id>/
    └── README.md               # Benchmark-specific documentation
```

## Adding a New Benchmark

To add a new standalone benchmark, create a folder in `benchmarks/` with the following required files:

### 1. `setup_<benchmark>.py`

Automated setup script that:
- Clones the benchmark repository (if external)
- Creates virtual environment with required Python version
- Installs dependencies
- Downloads ground truth data or test datasets
- Validates setup (e.g., checks for required API keys)

**Requirements:**
- Must be written in Python (no shell scripts)
- Should be idempotent (safe to run multiple times)
- Must fail with clear error messages if dependencies are missing
- Should use `subprocess.run()` for external commands

**Example:**
```python
#!/usr/bin/env python3
"""
Setup script for <BenchmarkName>

This script:
- Clones <BenchmarkName> repository
- Creates Python X.Y virtual environment
- Installs dependencies
- Downloads ground truth data
"""
import subprocess
import sys
from pathlib import Path
import shutil


def main():
    # Check Python version
    # Clone repository
    # Create virtual environment
    # Install dependencies
    # Download data
    pass


if __name__ == "__main__":
    main()
```

### 2. `run_<benchmark>.py`

Benchmark execution script that:
- Applies any necessary compatibility patches
- Executes benchmark tasks
- Evaluates results against ground truth
- Stores results in organized directory structure

**Requirements:**
- Must support command-line arguments for task selection
- Should handle both baseline and SABRE configurations
- Must generate reproducible results
- Should store results in `results/<config_name>/<task_id>/`

**Example:**
```python
#!/usr/bin/env python3
"""
Benchmark runner for <BenchmarkName>
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_runs", type=int, default=1)
    args = parser.parse_args()

    # Run benchmark
    # Evaluate results
    # Store results
    pass


if __name__ == "__main__":
    main()
```

### 3. `configs/`

Configuration files for different evaluation scenarios:
- `config_baseline.yaml` - Standard LLM API configuration (e.g., OpenAI)
- `config_sabre.yaml` - SABRE configuration (localhost endpoint)
- Additional configs as needed

### 4. `README.md`

Benchmark-specific documentation:
- Overview of the benchmark
- Setup instructions
- Running instructions
- Expected results format
- Known issues or limitations

## Examples

### DataSciBench

See `benchmarks/DataSciBench/` for a complete standalone benchmark implementation:

```bash
# Setup
python benchmarks/DataSciBench/setup_benchmark.py

# Run baseline
OPENAI_API_KEY=$(cat ~/.openai/key) python benchmarks/DataSciBench/run_benchmark.py \
  --task_id csv_excel_0 \
  --config benchmarks/DataSciBench/configs/config_gpt4o_baseline.yaml \
  --data_type csv \
  --max_runs 1

# Run SABRE
uv run sabre-server  # Start SABRE server first
OPENAI_API_KEY=$(cat ~/.openai/key) python benchmarks/DataSciBench/run_benchmark.py \
  --task_id csv_excel_0 \
  --config benchmarks/DataSciBench/configs/config_sabre.yaml \
  --data_type csv \
  --max_runs 1
```

## Best Practices

1. **Python Over Shell**: Always use Python scripts instead of shell scripts for portability
2. **Strict Error Handling**: Fail fast with clear error messages rather than graceful degradation
3. **Idempotent Setup**: Setup scripts should detect existing installations and skip unnecessary steps
4. **Virtual Environments**: Use isolated virtual environments for benchmark dependencies
5. **Result Organization**: Store results in `results/<config_name>/<task_id>/` for easy comparison
6. **Documentation**: Include clear setup and usage instructions in benchmark README

## Testing

When testing new benchmarks:

1. Run setup script from a clean state
2. Verify all dependencies are installed correctly
3. Run a single task first to validate configuration
4. Compare baseline vs SABRE results
5. Document any compatibility issues or patches needed
