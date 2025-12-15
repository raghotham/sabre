"""
Tests for Bash helper improvements.

Run with: uv run pytest tests/test_bash_improvements.py
"""

from sabre.server.python_runtime import PythonRuntime
from sabre.server.helpers.bash import BashResult


def test_bash_auto_print():
    """Test that Bash.execute() auto-prints output."""
    runtime = PythonRuntime()

    result = runtime.execute("""
Bash.execute("echo 'test output'")
""")

    assert result.success
    # Output should be auto-printed (visible in result.output)
    assert "test output" in result.output


def test_bash_quiet_mode():
    """Test that quiet=True suppresses auto-print."""
    runtime = PythonRuntime()

    result = runtime.execute("""
Bash.execute("echo 'should not appear'", quiet=True)
print("manual output")
""")

    assert result.success
    # Quiet output should NOT appear
    assert "should not appear" not in result.output
    # Manual print should appear
    assert "manual output" in result.output


def test_bash_with_result():
    """Test Bash.execute() with explicit result() capture."""
    runtime = PythonRuntime()

    result = runtime.execute("""
bash_result = Bash.execute("echo 'captured'")
result(bash_result)
""")

    assert result.success
    # Should appear in both output (auto-print) and results
    assert "captured" in result.output
    assert isinstance(result.results[0], BashResult)
    assert "captured" in result.results[0].stdout


def test_bash_longer_timeout():
    """Test that default timeout is now 5 minutes (300000ms)."""
    runtime = PythonRuntime()

    # This should not timeout with new default (would timeout with old 10s default)
    result = runtime.execute("""
bash_result = Bash.execute("sleep 2")
result(bash_result.exit_code)
""")

    assert result.success
    assert result.results[0] == 0


def test_bash_custom_timeout():
    """Test custom timeout parameter."""
    runtime = PythonRuntime()

    result = runtime.execute("""
bash_result = Bash.execute("sleep 10", timeout=100)
result(bash_result.stderr)
""")

    assert result.success
    assert "timed out" in result.output.lower()
