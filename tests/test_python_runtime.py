"""
Tests for Python Runtime.

Run with: uv run pytest tests/test_python_runtime.py
"""

from sabre.server.python_runtime import PythonRuntime
from sabre.server.bash_helper import BashResult


def test_simple_execution():
    """Test basic Python execution."""
    runtime = PythonRuntime()

    result = runtime.execute("x = 2 + 2\nprint(x)")

    assert result.success
    assert "4" in result.output


def test_result_function():
    """Test result() capture."""
    runtime = PythonRuntime()

    result = runtime.execute("result('hello', 'world')")

    assert result.success
    assert len(result.results) == 2
    assert result.results[0] == "hello"
    assert result.results[1] == "world"


def test_bcl_bash():
    """Test Bash.execute() helper."""
    runtime = PythonRuntime()

    result = runtime.execute("""
bash_result = Bash.execute("echo 'test'")
result(bash_result)
""")

    assert result.success
    assert len(result.results) == 1
    assert isinstance(result.results[0], BashResult)
    assert "test" in result.results[0].stdout


def test_bash_result_get_str():
    """Test BashResult.get_str() in result output."""
    runtime = PythonRuntime()

    result = runtime.execute("""
bash_result = Bash.execute("echo 'hello'")
result(bash_result)
""")

    assert result.success
    assert "hello" in result.output


def test_bash_with_output_processing():
    """Test processing bash output in Python."""
    runtime = PythonRuntime()

    result = runtime.execute("""
files = Bash.execute("echo 'a.py\\nb.py\\nc.txt'").stdout.strip().split('\\n')
py_files = [f for f in files if f.endswith('.py')]
result(py_files)
""")

    assert result.success
    assert len(result.results) == 1
    assert result.results[0] == ["a.py", "b.py"]


def test_error_handling():
    """Test error capture."""
    runtime = PythonRuntime()

    result = runtime.execute("x = 1 / 0")

    assert not result.success
    assert "ZeroDivisionError" in result.error


def test_namespace_isolation():
    """Test that executions don't interfere."""
    runtime = PythonRuntime()

    # First execution
    result1 = runtime.execute("x = 10\nresult(x)")
    assert result1.success
    assert result1.results[0] == 10

    # Reset and second execution should not have x
    runtime.reset()
    result2 = runtime.execute("result(locals().get('x', 'not found'))")
    assert result2.success
    # x should not exist in new namespace
    assert result2.results[0] == "not found"


def test_multiple_result_calls():
    """Test multiple result() calls in one execution."""
    runtime = PythonRuntime()

    result = runtime.execute("""
result(1)
result(2)
result(3)
""")

    assert result.success
    assert result.results == [1, 2, 3]


def test_print_and_result_combined():
    """Test that both print and result() work together."""
    runtime = PythonRuntime()

    result = runtime.execute("""
print("Starting")
x = 2 + 2
print(f"x = {x}")
result(x)
""")

    assert result.success
    assert "Starting" in result.output
    assert "x = 4" in result.output
    assert result.results[0] == 4


def test_bash_timeout():
    """Test bash command timeout."""
    runtime = PythonRuntime()

    result = runtime.execute("""
bash_result = Bash.execute("sleep 20", timeout=100)
result(bash_result.stderr)
""")

    assert result.success
    assert "timed out" in result.output.lower()
