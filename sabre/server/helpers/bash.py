"""
Bash helper for Python runtime.

Simple bash command execution with result capture.
"""

import subprocess
import time
from dataclasses import dataclass


@dataclass
class BashResult:
    """Result of executing a bash command."""

    stdout: str
    stderr: str
    exit_code: int
    command: str
    execution_time: float

    def get_str(self) -> str:
        """Return stdout and stderr for use with result()."""
        if self.stderr:
            return f"{self.stdout}\nSTDERR:\n{self.stderr}"
        return self.stdout


def execute_bash(command: str, timeout: int = 10000, cwd: str = None) -> BashResult:
    """
    Execute a bash command.

    Args:
        command: Bash command to execute
        timeout: Timeout in milliseconds (default 10000)
        cwd: Working directory (default: current)

    Returns:
        BashResult with execution details
    """
    start_time = time.time()

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout / 1000.0,  # Convert to seconds
            cwd=cwd,
        )

        execution_time = time.time() - start_time

        return BashResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
            command=command,
            execution_time=execution_time,
        )

    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        return BashResult(
            stdout="",
            stderr=f"Command timed out after {timeout}ms",
            exit_code=124,
            command=command,
            execution_time=execution_time,
        )

    except Exception as e:
        execution_time = time.time() - start_time
        return BashResult(
            stdout="", stderr=f"Execution error: {str(e)}", exit_code=1, command=command, execution_time=execution_time
        )


class Bash:
    """Bash command execution helper."""

    @staticmethod
    def execute(command: str, timeout: int = None, quiet: bool = False) -> BashResult:
        """
        Execute a bash command.

        Examples:
            Bash.execute("ls -la")  # Output is printed automatically
            result(Bash.execute("cat config.txt"))  # Explicit result capture
            files = Bash.execute("ls *.py", quiet=True).stdout.split('\\n')  # Suppress output

        Args:
            command: Bash command to execute
            timeout: Timeout in milliseconds (default 300000 = 5 minutes)
            quiet: If True, don't print output (default False)

        Returns:
            BashResult with stdout, stderr, exit_code
        """
        timeout = timeout if timeout is not None else 300000  # 5 minutes default
        result = execute_bash(command, timeout=timeout)

        # Auto-print output so LLM sees it in helpers_result
        # This ensures output is captured even without explicit result() call
        if not quiet:
            output = result.get_str()
            if output.strip():
                print(output)

        return result
