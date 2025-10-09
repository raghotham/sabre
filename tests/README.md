# LLMVM2 Tests

Test suite for llmvm2 using pytest.

## Setup

Make sure you have dev dependencies installed:

```bash
uv sync --dev
```

## Running Tests

### Run all tests

```bash
uv run pytest
```

### Run with verbose output

```bash
uv run pytest -v
```

### Run specific test file

```bash
uv run pytest tests/test_executor.py
```

### Run specific test

```bash
uv run pytest tests/test_executor.py::test_simple_execution
```

### Run with coverage

```bash
uv run pytest --cov=llmvm2 --cov-report=html
```

This generates a coverage report in `htmlcov/index.html`.

### Run with output capture disabled (see prints)

```bash
uv run pytest -s
```

## Test Structure

```
tests/
├── __init__.py
├── conftest.py           # Shared fixtures and configuration
├── test_executor.py      # ResponseExecutor tests
└── README.md            # This file
```

## Writing Tests

### Test Fixtures

Common fixtures are defined in `conftest.py`:

- `executor` - ResponseExecutor instance
- `tree_context` - Dummy execution tree context

### Async Tests

Use `@pytest.mark.asyncio` decorator for async tests:

```python
@pytest.mark.asyncio
async def test_something(executor):
    result = await executor.execute_simple("test")
    assert result is not None
```

### API Key

Tests require `OPENAI_API_KEY` environment variable to be set.

The `conftest.py` automatically skips all tests if the API key is not found.

## Test Categories

### Current Tests

- **test_executor.py** - Tests for ResponseExecutor
  - `test_simple_execution` - Basic non-streaming call
  - `test_streaming_execution` - Streaming with event callbacks
  - `test_response_continuation` - response_id continuation
  - `test_token_counting` - Token counting
  - `test_custom_model` - Custom model parameter

### Future Tests

- **test_orchestrator.py** - Continuation loop tests
- **test_runtime.py** - Python runtime tests
- **test_tree.py** - Execution tree tests
- **test_integration.py** - End-to-end integration tests

## Pytest Configuration

Configuration is in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
asyncio_mode = "auto"
addopts = [
    "-v",
    "--strict-markers",
    "--tb=short",
]
```

## Tips

1. **Fast feedback**: Run specific tests while developing
2. **Coverage**: Check coverage regularly with `--cov`
3. **Debugging**: Use `-s` to see print statements
4. **Verbose**: Use `-v` to see test names
5. **Parallel**: Use `pytest-xdist` for parallel execution (not yet installed)

## CI/CD

Tests should be run in CI/CD pipeline:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    uv sync --dev
    uv run pytest --cov=llmvm2
```
