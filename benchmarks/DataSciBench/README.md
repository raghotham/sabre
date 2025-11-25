# DataSciBench Benchmark Setup

This directory contains configurations for running DataSciBench benchmarks to compare direct LLM performance vs SABRE's agentic approach.

## Setup Complete

✅ Python 3.11 venv with MetaGPT + DataSciBench installed
✅ Wrapper script with openai/httpx compatibility patch
✅ Environment variable expansion for API keys
✅ Configs created for baseline and SABRE endpoints
✅ **No modifications to DataSciBench code required**

## Architecture

All compatibility fixes are isolated in our codebase:

```
sabre/
├── benchmarks/DataSciBench/
│   ├── run_benchmark.py          # Wrapper with all compatibility fixes
│   └── configs/
│       ├── config_gpt4o_baseline.yaml
│       └── config_sabre.yaml
└── tmp/DataSciBench/              # Clean DataSciBench checkout (no modifications)
```

The wrapper script (`run_benchmark.py`) handles:
- openai/httpx `proxies` parameter compatibility patch
- Environment variable expansion in config files (`${OPENAI_API_KEY}`)
- No changes needed to DataSciBench source code

## Running Benchmarks

### Prerequisites

Ensure `OPENAI_API_KEY` is set in your environment:
```bash
export OPENAI_API_KEY="sk-..."
```

### Run Baseline (gpt-4o-mini direct)

```bash
cd /Users/raghu/dev/sabre/tmp/DataSciBench

/Users/raghu/dev/sabre/benchmarks/DataSciBench/run_benchmark.py \
    --task_id csv_excel_0 \
    --config /Users/raghu/dev/sabre/benchmarks/DataSciBench/configs/config_gpt4o_baseline.yaml \
    --data_type csv \
    --max_runs 1
```

Or using the venv Python directly:
```bash
cd /Users/raghu/dev/sabre/tmp/DataSciBench

.venv/bin/python /Users/raghu/dev/sabre/benchmarks/DataSciBench/run_benchmark.py \
    --task_id csv_excel_0 \
    --config /Users/raghu/dev/sabre/benchmarks/DataSciBench/configs/config_gpt4o_baseline.yaml \
    --data_type csv \
    --max_runs 1
```

### Run with SABRE Endpoint

First, ensure SABRE server is running:
```bash
cd /Users/raghu/dev/sabre
uv run sabre-server
```

Then run the benchmark:
```bash
cd /Users/raghu/dev/sabre/tmp/DataSciBench

/Users/raghu/dev/sabre/benchmarks/DataSciBench/run_benchmark.py \
    --task_id csv_excel_0 \
    --config /Users/raghu/dev/sabre/benchmarks/DataSciBench/configs/config_sabre.yaml \
    --data_type csv \
    --max_runs 1
```

## Configuration Files

### `configs/config_gpt4o_baseline.yaml`
Runs DataSciBench directly against OpenAI's gpt-4o-mini to establish baseline performance.

```yaml
llm:
  api_type: "openai"
  base_url: "https://api.openai.com/v1"
  api_key: "${OPENAI_API_KEY}"  # Expanded by wrapper script
  model: "gpt-4o-mini"
  timeout: 600

repair_llm_output: true
```

### `configs/config_sabre.yaml`
Runs DataSciBench through SABRE's OpenAI-compatible endpoint (`http://localhost:8011/v1`) to test agentic workflow performance.

```yaml
llm:
  api_type: "openai"
  base_url: "http://localhost:8011/v1"
  api_key: "dummy"  # SABRE doesn't require real API key
  model: "gpt-4o"
  timeout: 600

repair_llm_output: true
```

## Results

Results are saved to:
- `/Users/raghu/dev/sabre/tmp/DataSciBench/data/{task_id}/{model_name}_{run_id}/`
  - `logs.txt` - Execution logs
  - `sys_logs.txt` - System logs
  - `{model_name}_outputs.jsonl` - Structured output with metrics

Compare results using:
```bash
cd /Users/raghu/dev/sabre/tmp/DataSciBench
.venv/bin/python -m experiments.evaluate --task_id csv_excel_0
```

## Implementation Details

### Wrapper Script (`run_benchmark.py`)

The wrapper script provides a clean interface without modifying DataSciBench code:

1. **openai/httpx compatibility patch**
   - Patches `AsyncHttpxClientWrapper` at runtime
   - Accepts both `proxy` and `proxies` parameters
   - Fixes incompatibility between openai 1.38.0 and httpx 0.28.1

2. **Environment variable expansion**
   - Reads config YAML files
   - Expands `${VAR_NAME}` patterns to actual values
   - Creates temporary config file with expanded values
   - Cleans up temp file after execution

3. **Transparent execution**
   - Changes to DataSciBench directory
   - Imports and runs DataSciBench's main code
   - Passes through all command-line arguments
   - No modifications to DataSciBench source required

### Why This Approach?

- **Maintainable**: All custom code in our repository (`sabre/benchmarks/`)
- **Non-invasive**: DataSciBench checkout remains pristine
- **Upstream-friendly**: No changes that need to be upstreamed
- **Secure**: API keys stay in environment variables, not committed to git
- **Reusable**: Same wrapper works for both baseline and SABRE configs

## Troubleshooting

### Proxy Error
If you see `AsyncClient.__init__() got an unexpected keyword argument 'proxies'`:
- Ensure you're using `run_benchmark.py` wrapper (applies compatibility patch)
- Do NOT run `python -m experiments.run_examples` directly

### API Key Not Expanded
If you see `"${OPENAI_API_KEY}"` in logs:
- Ensure you're using `run_benchmark.py` wrapper (expands env vars)
- Verify `OPENAI_API_KEY` is set: `echo $OPENAI_API_KEY`

### Import Errors
If you see `ModuleNotFoundError`:
- Ensure you're running from DataSciBench directory: `cd /Users/raghu/dev/sabre/tmp/DataSciBench`
- Verify venv is activated or use `.venv/bin/python` directly
