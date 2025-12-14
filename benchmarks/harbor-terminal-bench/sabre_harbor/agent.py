"""
SABRE Agent implementation for Harbor Terminal-Bench 2.0.

This module provides an AbstractInstalledAgent implementation that allows SABRE
to be evaluated on Terminal-Bench 2.0 via the Harbor framework.

Usage:
    harbor run -d terminal-bench@2.0 --agent-import-path sabre_harbor:SabreAgent
"""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from harbor.agents.installed.base import BaseInstalledAgent
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
    from harbor.models.exec import ExecInput
    from harbor.models.trial.result import AgentInfo, ModelInfo


# Import from harbor - these will be available when running in Harbor environment
HARBOR_AVAILABLE = False
try:
    from harbor.agents.installed.base import BaseInstalledAgent, ExecInput
    from harbor.models.agent.context import AgentContext
    from harbor.models.trajectories import Step, Trajectory, Metrics as TrajectoryMetrics
    from harbor.models.trial.result import AgentInfo, ModelInfo
    HARBOR_AVAILABLE = True
except ImportError as e:
    # Allow module to be imported for inspection even without harbor installed
    # Create stub classes that accept the same constructor arguments
    # Note: AgentInfo is NOT stubbed here - we'll create it dynamically in to_agent_info()
    import sys
    print(f"[sabre_harbor] Harbor import failed: {e}, using stubs", file=sys.stderr)

    class BaseInstalledAgentStub:
        """Stub class for when harbor is not installed."""

        def __init__(
            self,
            logs_dir: Path,
            prompt_template_path: Path | str | None = None,
            version: str | None = None,
            *args,
            **kwargs,
        ):
            self.logs_dir = logs_dir
            self._version = version
            self.model_name = kwargs.get('model_name')

    BaseInstalledAgent = BaseInstalledAgentStub

    class ExecInput:
        """Stub for ExecInput when harbor is not installed."""
        def __init__(self, command: str, timeout: int = 3600, **kwargs):
            self.command = command
            self.timeout = timeout
            for k, v in kwargs.items():
                setattr(self, k, v)

    AgentContext = Any
    AgentInfo = None  # Will be imported dynamically when needed

    # Stub classes for trajectory
    class Step:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class TrajectoryMetrics:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class Trajectory:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


class SabreAgent(BaseInstalledAgent):
    """
    SABRE agent for Terminal-Bench 2.0 evaluation.

    SABRE is a persona-driven AI agent using OpenAI's Responses API with a
    continuation-passing execution model. It provides helpers like Bash, Search,
    Web, and llm_call for recursive task delegation.

    This agent runs SABRE in script mode: `uv run sabre "message"`
    """

    SABRE_REPO_URL = "https://github.com/har-ki/sabre.git"
    SABRE_BRANCH = "terminal_bench"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        version: str | None = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the SABRE agent.

        Args:
            logs_dir: Directory for storing logs
            model_name: OpenAI model to use (e.g., "gpt-4o", "gpt-4o-mini")
            version: Optional SABRE version/branch/commit to install
        """
        super().__init__(logs_dir, version=version, *args, **kwargs)
        self._model_name = model_name or os.environ.get("OPENAI_MODEL", "gpt-4o")

    @staticmethod
    def name() -> str:
        """Return the agent name."""
        return "sabre"

    def version(self) -> str | None:
        """Return the agent version."""
        return self._version or "latest"

    @classmethod
    def import_path(cls) -> str:
        """Return the import path for this agent class."""
        return f"{cls.__module__}:{cls.__name__}"

    def to_agent_info(self):
        """Return agent information for Harbor."""
        # Import from the correct location in Harbor
        from harbor.models.trial.result import AgentInfo, ModelInfo

        # Parse model info if we have a model name
        model_info = None
        if self._model_name:
            # Model name might be in format "provider/model" or just "model"
            if "/" in self._model_name:
                provider, model = self._model_name.split("/", 1)
            else:
                provider = "openai"
                model = self._model_name
            model_info = ModelInfo(name=model, provider=provider)

        return AgentInfo(
            name=self.name(),
            version=self.version() or "latest",
            model_info=model_info,
        )

    @property
    def _install_agent_template_path(self) -> Path:
        """Return path to the installation script template."""
        return Path(__file__).parent / "install-sabre.sh.j2"

    @property
    def _template_variables(self) -> dict[str, str]:
        """Return template variables for the installation script."""
        return {
            "version": self._version or self.SABRE_BRANCH,
            "repo_url": self.SABRE_REPO_URL,
        }

    def _get_env_vars(self) -> dict[str, str]:
        """
        Get environment variables to pass to SABRE.

        Returns:
            Dictionary of environment variables
        """
        env = {}

        # Required: OpenAI API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        env["OPENAI_API_KEY"] = api_key

        # Optional: Custom base URL
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            env["OPENAI_BASE_URL"] = base_url

        # Model to use
        env["OPENAI_MODEL"] = self._model_name

        # Set log level to reduce noise
        env["LOG_LEVEL"] = os.environ.get("LOG_LEVEL", "WARNING")

        return env

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        """
        Create commands to run SABRE with the given instruction.

        Args:
            instruction: The task description/prompt to send to SABRE

        Returns:
            List of ExecInput commands to execute
        """
        # Add context about the environment for the agent
        augmented_instruction = f"""{instruction}

CRITICAL INSTRUCTIONS - YOU MUST FOLLOW THESE EXACTLY:

1. EXECUTION: You MUST wrap ALL code in <helpers>...</helpers> blocks.
   Markdown code blocks (```python or ```bash) will NOT execute.
   Only code inside <helpers>...</helpers> tags actually runs.

2. WORKING DIRECTORY: You are in /app. All files must be created there.

3. FILE OPERATIONS - Use Bash.execute() for ALL file I/O:
   - READ files:  Bash.execute('cat filename.txt')
   - WRITE files: Bash.execute('cat > filename.txt << "EOF"\\nyour content\\nEOF')
   - RUN scripts: Bash.execute('python script.py')

4. DO NOT USE write_file() or read_file() - they save to the wrong directory!

5. CORRECT EXAMPLE:
<helpers>
# Create a Python script in /app
Bash.execute('''cat > solution.py << "EOF"
def solve():
    return 42
print(solve())
EOF''')

# Run it
result = Bash.execute('python solution.py')
print(result.stdout)
</helpers>

6. WRONG (will not execute):
```python
print("This does nothing - it's just markdown!")
```

Remember: <helpers> blocks = code runs. Markdown blocks = code does NOT run."""

        # Shell-escape the instruction to handle special characters
        escaped_instruction = shlex.quote(augmented_instruction)

        # Build environment variable exports
        env_vars = self._get_env_vars()
        env_exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_vars.items())

        # SABRE script mode command
        # Run from /app (Harbor's task working directory) but use SABRE from ~/sabre
        # Tee output to log file for trajectory parsing
        command = f"""
cd /app && \\
{env_exports} \\
~/.local/bin/uv --project ~/sabre run sabre {escaped_instruction} 2>&1 | tee /logs/agent/sabre-output.txt
"""

        return [
            ExecInput(
                command=command.strip(),
                timeout=3600,  # 1 hour timeout for complex tasks
            )
        ]

    def _get_sabre_log_path(self) -> Path | None:
        """
        Find the SABRE log file path.

        Returns:
            Path to the log file, or None if not found
        """
        # SABRE logs to ~/.local/state/sabre/logs/server.log
        log_path = Path.home() / ".local" / "state" / "sabre" / "logs" / "server.log"
        if log_path.exists():
            return log_path
        return None

    def _parse_sabre_output(self, stdout: str, stderr: str) -> list[Step]:
        """
        Parse SABRE output into trajectory steps.

        SABRE outputs streaming text with <helpers> blocks for code execution.
        This method extracts the conversation flow.

        Args:
            stdout: Standard output from SABRE
            stderr: Standard error from SABRE

        Returns:
            List of trajectory steps
        """
        from harbor.models.trajectories import ToolCall, Observation, ObservationResult

        steps = []
        step_id = 1  # Step IDs start at 1

        # Combined output
        output = stdout + stderr

        # Pattern to match <helpers> blocks
        helpers_pattern = re.compile(
            r"<helpers>(.*?)</helpers>",
            re.DOTALL
        )

        # Pattern to match <helpers_result> blocks
        result_pattern = re.compile(
            r"<helpers_result>(.*?)</helpers_result>",
            re.DOTALL
        )

        # Split output into segments based on helpers blocks
        segments = helpers_pattern.split(output)

        current_pos = 0
        for i, segment in enumerate(segments):
            if i % 2 == 0:
                # This is text output (assistant message)
                text = segment.strip()
                if text:
                    # Remove helpers_result blocks from display text
                    display_text = result_pattern.sub("", text).strip()
                    if display_text:
                        steps.append(
                            Step(
                                step_id=step_id,
                                source="agent",
                                message=display_text,
                            )
                        )
                        step_id += 1
            else:
                # This is a helpers code block (tool call)
                code = segment.strip()
                if code:
                    # Look for corresponding result
                    result_match = result_pattern.search(output, current_pos)
                    result_text = None
                    if result_match:
                        result_text = result_match.group(1).strip()
                        current_pos = result_match.end()

                    # Create tool call with observation
                    tool_call = ToolCall(
                        tool_call_id=f"helpers_{step_id}",
                        function_name="python_runtime",
                        arguments={"code": code},
                    )

                    observation = None
                    if result_text:
                        observation = Observation(
                            results=[
                                ObservationResult(
                                    source_call_id=f"helpers_{step_id}",
                                    content=result_text,
                                )
                            ]
                        )

                    steps.append(
                        Step(
                            step_id=step_id,
                            source="agent",
                            message=f"Executing Python code",
                            tool_calls=[tool_call],
                            observation=observation,
                        )
                    )
                    step_id += 1

        return steps

    def populate_context_post_run(self, context: AgentContext) -> None:
        """
        Populate the agent context after execution.

        This method reads SABRE's output from log files and converts it to the ATIF trajectory format.

        Args:
            context: The agent context to populate
        """
        # Read output from log file if available
        log_file = self.logs_dir / "sabre-output.txt"
        stdout = ""
        if log_file.exists():
            try:
                stdout = log_file.read_text()
            except Exception as e:
                print(f"Failed to read SABRE log file: {e}")

        # Parse output into steps
        steps = self._parse_sabre_output(stdout, "")

        # If no steps were parsed but we have output, create a single message step
        if not steps and stdout:
            steps = [
                Step(
                    step_id=1,
                    source="agent",
                    message=stdout,
                )
            ]

        # Create trajectory and write to file
        if steps:
            from harbor.models.trajectories import Trajectory, FinalMetrics, Agent

            trajectory = Trajectory(
                schema_version="ATIF-v1.2",
                session_id=str(self.logs_dir),
                agent=Agent(
                    name=self.name(),
                    version=self.version() or "latest",
                    model_name=self._model_name,
                ),
                steps=steps,
                final_metrics=FinalMetrics(
                    total_prompt_tokens=None,
                    total_completion_tokens=None,
                    total_cached_tokens=None,
                    total_cost_usd=None,
                    total_steps=len(steps),
                ),
            )

            # Write trajectory to file
            trajectory_path = self.logs_dir / "trajectory.json"
            try:
                import json
                with open(trajectory_path, "w") as f:
                    json.dump(trajectory.to_json_dict(), f, indent=2)
                print(f"Wrote SABRE trajectory to {trajectory_path}")
            except Exception as e:
                print(f"Failed to write trajectory file: {e}")

        # SABRE doesn't expose token counts in CLI mode
        context.n_input_tokens = 0
        context.n_output_tokens = 0
        context.n_cache_tokens = 0
        context.cost_usd = None
