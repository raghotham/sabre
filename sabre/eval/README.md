# SABRE Evaluation Framework

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Evaluation Harness (databench.py)                     │
│  - Load tasks from benchmark                           │
│  - Run through SABRE                                   │
│  - Collect results                                     │
│  - Calculate metrics                                   │
└─────────────────┬───────────────────────────────────────┘
                  │
                  │ Uses SDK
                  ▼
┌─────────────────────────────────────────────────────────┐
│  SABRE SDK (sdk.py)                                     │
│  - Lightweight async client                            │
│  - POST to /message endpoint                           │
│  - Consume SSE stream                                  │
│  - Return SabreResult                                  │
└─────────────────┬───────────────────────────────────────┘
                  │
                  │ HTTP SSE
                  ▼
┌─────────────────────────────────────────────────────────┐
│  SABRE Server (server.py)                              │
│  - HTTP SSE endpoint                                   │
│  - Session management                                  │
│  - Orchestrator execution                              │
└─────────────────┬───────────────────────────────────────┘
                  │
                  │ Orchestrates
                  ▼
┌─────────────────────────────────────────────────────────┐
│  Orchestrator + Python Runtime                         │
│  - Continuation loop                                   │
│  - Helper execution                                    │
│  - LLM calls                                          │
└─────────────────────────────────────────────────────────┘
```

## Components

### 1. SDK (`sabre/sdk.py`)

**Purpose**: Lightweight programmatic interface to SABRE server

**Key Classes**:
- `SabreClient`: Async client for running messages
- `SabreResult`: Structured result with response, tokens, helper executions

**Usage**:
```python
from sabre.sdk import SabreClient

client = SabreClient(base_url="http://localhost:8011")
result = await client.run("Plot a sine wave")

print(result.response)  # Final assistant response
print(result.success)  # True/False
print(result.input_tokens)  # Token usage
print(result.helper_executions)  # List of helper runs
```

**Why not call orchestrator directly?**
- ✅ Keeps eval harness decoupled from server internals
- ✅ Works with any SABRE server (local or remote)
- ✅ Can be used by other eval frameworks (CIBench, DABstep)
- ✅ Can be used by external integrations (notebooks, scripts)
- ✅ No need to manage orchestrator/runtime lifecycle

### 2. DataSciBench Evaluator (`sabre/eval/datascibench.py`)

**Purpose**: Run DataSciBench tasks through SABRE and measure performance

**Key Classes**:
- `DataSciBenchEvaluator`: Loads tasks, runs through SABRE, calculates metrics
- `TaskOutput`: Results for single task
- `BenchmarkResult`: Aggregate metrics across all tasks

**Usage**:
```bash
# Start SABRE server first
uv run sabre

# Run evaluation (in another terminal)
uv run datascibench --limit 5
```

**Output**:
- `results/baseline/summary.json`: Aggregate metrics
- `results/baseline/results.jsonl`: Per-task detailed results

## Running Evaluations

### Setup

1. **Start SABRE server**:
   ```bash
   uv run sabre
   ```

2. **Verify server health**:
   ```bash
   curl http://localhost:8011/health
   ```

### Run DataSciBench Baseline

```bash
# Full benchmark (222 tasks)
uv run datascibench

# Limit to 10 tasks
uv run datascibench --limit 10

# Filter by task type (CSV/Excel tasks)
uv run datascibench --filter csv_excel --limit 5

# BCB tasks only
uv run datascibench --filter bcb --limit 5

# Custom output directory
uv run datascibench --output results/custom

# Remote server
uv run datascibench --server http://remote-server:8011
```

### Understanding Results

**Summary (`summary.json`)**:
```json
{
  "success_rate": 0.65,
  "total_tasks": 10,
  "successful_tasks": 6,
  "failed_tasks": 4,
  "avg_time": 12.3,
  "total_tokens": {
    "input": 15000,
    "output": 8000
  }
}
```

**Detailed Results (`results.jsonl`)**:
Each line is a task result:
```json
{
  "task_id": "csv_excel_0",
  "output_dir": "/path/to/results/csv_excel_0",
  "time_cost": 15.2,
  "steps": [
    {
      "step_id": 0,
      "code": "import pandas as pd\n...",
      "result": "DataFrame created successfully",
      "is_success": true,
      "error": null
    }
  ],
  "error_count": 0,
  "tokens_used": {"input": 1500, "output": 800, "reasoning": 0},
  "final_result": "CSV file saved to output.csv",
  "is_success": true
}
```

## Adding New Benchmarks

To add a new benchmark (e.g., CIBench):

1. Create `sabre/eval/cibench.py`
2. Reuse `SabreClient` for execution
3. Implement benchmark-specific metrics

Example structure:
```python
from sabre.sdk import SabreClient


class CIBenchEvaluator:
    def __init__(self, benchmark_path, output_dir):
        self.client = SabreClient()

    async def run_task(self, task):
        result = await self.client.run(task["prompt"])
        # Calculate CIBench-specific metrics
        return metrics
```

## Benefits of This Architecture

1. **Decoupled**: Eval harness doesn't depend on server internals
2. **Reusable**: SDK can be used for other evals, notebooks, scripts
3. **Flexible**: Can evaluate local or remote servers
4. **No modifications**: Server stays unchanged, clean separation
5. **Maintainable**: Changes to server don't break eval harness

## Client vs SDK

**Client** (`sabre/client/client.py`):
- Interactive TUI for humans
- Rich rendering, syntax highlighting
- Keyboard input, cancellation
- Pure presentation logic

**SDK** (`sabre/sdk.py`):
- Programmatic API for code
- Returns structured results
- No rendering, just data
- For eval harnesses, integrations

Both use the same `/message` endpoint and SSE stream, just different consumers.
