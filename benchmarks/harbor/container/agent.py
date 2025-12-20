"""
SABRE Agent implementation for Harbor benchmarks.

This agent installs SABRE in the Harbor container and runs it in command mode.

Architecture:
- Container builds from task's Dockerfile (e.g., ubuntu:24.04)
- install.sh installs SABRE and dependencies
- SABRE runs in command mode: `uv run sabre "task description"`
- SABRE executes commands inside the container's filesystem
- Harbor's verifier checks results in the same container

Usage:
    uvx harbor run -d hello-world@head --agent-import-path container:SabreAgent
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from harbor.agents.installed.base import BaseInstalledAgent
    from harbor.models.agent.context import AgentContext
    from harbor.models.exec import ExecInput
    from harbor.models.trial.result import AgentInfo, ModelInfo

# Import from harbor - these will be available when running in Harbor environment
# Import AgentInfo and ModelInfo separately since they exist in newer Harbor versions
HARBOR_AVAILABLE = False
AgentInfo = None
ModelInfo = None

try:
    from harbor.models.trial.result import AgentInfo, ModelInfo
except ImportError:
    pass  # Will use stubs below

try:
    from harbor.agents.installed.base import BaseInstalledAgent
    from harbor.models.agent.context import AgentContext

    HARBOR_AVAILABLE = True
except ImportError as e:
    import sys

    print(f"[sabre_agent] Harbor BaseInstalledAgent import failed: {e}, using stubs", file=sys.stderr)

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
            self.model_name = kwargs.get("model_name")

    BaseInstalledAgent = BaseInstalledAgentStub  # type: ignore

# Import ExecInput separately - it may not exist in all Harbor versions
try:
    from harbor.models.exec import ExecInput
except ImportError:

    class ExecInputStub:
        """Stub for ExecInput when harbor is not installed."""

        def __init__(self, command: str, timeout: int = 3600, **kwargs):
            self.command = command
            self.timeout = timeout
            self.timeout_sec = timeout  # Harbor expects timeout_sec attribute
            # Set default values for commonly expected attributes
            self.cwd = kwargs.get("cwd", None)
            self.env = kwargs.get("env", None)
            # Handle any additional kwargs
            for k, v in kwargs.items():
                if k not in ("cwd", "env"):  # Skip already handled
                    setattr(self, k, v)

    ExecInput = ExecInputStub  # type: ignore

AgentContext = Any  # type: ignore

# Only create stubs for AgentInfo/ModelInfo if they weren't imported above
if AgentInfo is None or ModelInfo is None:
    # Import pydantic for stub classes
    try:
        from pydantic import BaseModel

        if AgentInfo is None:

            class AgentInfoStub(BaseModel):
                """Stub for AgentInfo when harbor is not installed."""

                name: str
                version: str
                model_info: Any

            AgentInfo = AgentInfoStub  # type: ignore

        if ModelInfo is None:

            class ModelInfoStub(BaseModel):
                """Stub for ModelInfo when harbor is not installed."""

                name: str
                provider: str

            ModelInfo = ModelInfoStub  # type: ignore

    except ImportError:
        # Fallback if pydantic not available
        if AgentInfo is None:

            class AgentInfoStub:  # type: ignore
                """Stub for AgentInfo when harbor is not installed."""

                def __init__(self, name: str, version: str, model_info: Any):
                    self.name = name
                    self.version = version
                    self.model_info = model_info

            AgentInfo = AgentInfoStub  # type: ignore

        if ModelInfo is None:

            class ModelInfoStub:  # type: ignore
                """Stub for ModelInfo when harbor is not installed."""

                def __init__(self, name: str, provider: str):
                    self.name = name
                    self.provider = provider

            ModelInfo = ModelInfoStub  # type: ignore


class SabreAgent(BaseInstalledAgent):
    """
    SABRE agent for Harbor benchmarks.

    Installs SABRE from source in the container and runs it in command mode.
    """

    def __init__(
        self,
        logs_dir: Path,
        version: str | None = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the SABRE agent.

        Args:
            logs_dir: Directory for storing logs
            version: Optional SABRE version identifier
        """
        super().__init__(logs_dir, version=version, *args, **kwargs)

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
        # SABRE uses OpenAI model from environment
        model_name = os.environ.get("OPENAI_MODEL", "gpt-4o")
        model_info = ModelInfo(name=model_name, provider="openai")

        return AgentInfo(
            name=self.name(),
            version=self.version() or "latest",
            model_info=model_info,
        )

    @property
    def _install_agent_template_path(self) -> Path:
        """Return path to the installation script."""
        return Path(__file__).parent / "install.sh"

    async def setup(self, environment):
        """Setup the agent in the environment, including copying SABRE source."""
        # Copy SABRE source code to logs_dir so install.sh can access it
        import shutil
        import subprocess

        # Use current working directory as SABRE source (must be run from repo root)
        sabre_source_dir = Path.cwd()
        target_dir = self.logs_dir / "sabre_source"
        target_dir.mkdir(parents=True, exist_ok=True)

        # Get list of git-tracked files, excluding benchmarks/
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=sabre_source_dir,
            capture_output=True,
            text=True,
            check=True,
        )

        for file_path in result.stdout.strip().split("\n"):
            # Skip benchmarks/ and tests/ directories
            if file_path.startswith("benchmarks/") or file_path.startswith("tests/"):
                continue

            src = sabre_source_dir / file_path
            dst = target_dir / file_path

            # Create parent directory if needed
            dst.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            shutil.copy2(src, dst)

        # Call parent setup which runs install.sh
        await super().setup(environment)

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        """
        Create commands to run SABRE in command mode.

        Args:
            instruction: The task description/prompt to send to SABRE

        Returns:
            List of ExecInput commands to execute
        """
        # Run SABRE in command mode using -m flag
        # SABRE is installed at /tmp/sabre during setup
        # The OPENAI_API_KEY environment variable is passed via env dict
        # Execute from /app so files are created in the right place

        # Transform instruction to add @ prefix before file references
        # Pattern: "The file <filename>" -> "The file @/app/<filename>"
        # This helps SABRE's @filepath feature automatically load referenced files
        import re

        # Add @/app/ before filenames mentioned as "The file <filename>"
        # This pattern indicates an existing file that should be loaded
        # Files in Harbor are placed in /app/ directory
        transformed_instruction = re.sub(
            r"\bthe\s+file\s+([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+)\b", r"the file @/app/\1", instruction, flags=re.IGNORECASE
        )

        # Use shlex.quote for proper shell escaping
        import shlex

        # Build command as array, then join with proper quoting
        # Note: We want $HOME to be expanded, so we don't quote it
        uv_path = "$HOME/.local/bin/uv"

        cmd_parts = [
            "run",
            "--directory",
            "/tmp/sabre",
            "sabre",
            "-m",
            transformed_instruction,  # Use transformed instruction with @ prefixes
            "--export-atif",
        ]

        # Quote only the arguments that need quoting
        quoted_args = " ".join(shlex.quote(str(part)) for part in cmd_parts)

        # Construct final command with unquoted $HOME for expansion
        quoted_cmd = f"{uv_path} {quoted_args}"

        # Add error handling with fallback
        command = f"{quoted_cmd} 2>&1 || (echo '=== SABRE FAILED, checking server.log ===' && cat /root/.local/state/sabre/logs/server.log 2>/dev/null || echo 'No server.log found')"

        return [
            ExecInput(
                command=command,
                cwd="/app",  # Run from /app so Bash commands execute there
                timeout=3600,  # 1 hour timeout
                env={
                    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),  # Pass API key
                    "NO_COLOR": "1",  # Disable color output in script mode
                },
            ),
            # Copy files from SABRE's session directory to /app
            # SABRE's write_file() and download helpers save to ~/.local/state/sabre/logs/sessions/{session_id}/files/
            # Harbor verifiers expect files in /app, so copy them over
            ExecInput(
                command='bash -c \'echo "=== Checking for SABRE session files ==="; if [ -d "$HOME/.local/state/sabre/logs/sessions" ]; then find $HOME/.local/state/sabre/logs/sessions -type d -name files; find $HOME/.local/state/sabre/logs/sessions -type f \\( -name "*.txt" -o -name "*.csv" -o -name "*.json" -o -name "*.pdf" \\) | while read f; do echo "Copying: $f"; cp -v "$f" /app/ || true; done; else echo "SABRE sessions directory not found - no files to copy"; fi; echo "=== Files in /app ==="; ls -la /app/\'',
                cwd="/app",
                timeout=30,
            ),
            # Copy any PNG files from /app to logs directory for inspection
            ExecInput(
                command="sh -c 'cp /app/*.png /logs/agent/ 2>/dev/null || true'",
                cwd="/app",
                timeout=10,
            ),
            # Copy SABRE session directory to logs for debugging
            # Sessions are at ~/.local/share/sabre/sessions/
            ExecInput(
                command="sh -c 'cp -r $HOME/.local/share/sabre/sessions /logs/agent/sabre_sessions 2>/dev/null || true'",
                cwd="/app",
                timeout=30,
            ),
            # Copy SABRE server log to logs for debugging
            # Server logs are at ~/.local/state/sabre/logs/server.log
            ExecInput(
                command="sh -c 'cp $HOME/.local/state/sabre/logs/server.log /logs/agent/server.log 2>/dev/null || true'",
                cwd="/app",
                timeout=30,
            ),
        ]

    def populate_context_post_run(self, context: AgentContext) -> None:
        """
        Populate the agent context after execution.

        SABRE writes ATIF traces to ~/.local/share/sabre/sessions/{session_id}/atif.json
        when run with --export-atif flag.

        Args:
            context: The agent context to populate
        """
        # Try to find and parse ATIF file
        from pathlib import Path

        # ATIF files are in ~/.local/share/sabre/sessions/{session_id}/atif.json
        sabre_data_dir = Path.home() / ".local" / "share" / "sabre" / "sessions"

        # Find the most recent session directory (SABRE creates timestamped session IDs)
        if sabre_data_dir.exists():
            session_dirs = sorted(sabre_data_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)

            for session_dir in session_dirs:
                atif_file = session_dir / "atif.json"
                if atif_file.exists():
                    try:
                        atif_data = json.loads(atif_file.read_text())

                        # Extract token counts from ATIF
                        # ATIF format has usage data in the trajectory
                        total_input_tokens = 0
                        total_output_tokens = 0
                        total_cache_tokens = 0

                        # Parse trajectory for usage data
                        for step in atif_data.get("trajectory", []):
                            usage = step.get("usage", {})
                            total_input_tokens += usage.get("input_tokens", 0)
                            total_output_tokens += usage.get("output_tokens", 0)
                            # Cache tokens might be in cache_read_input_tokens or similar
                            total_cache_tokens += usage.get("cache_read_input_tokens", 0)

                        context.n_input_tokens = total_input_tokens
                        context.n_output_tokens = total_output_tokens
                        context.n_cache_tokens = total_cache_tokens
                        context.cost_usd = None  # Could calculate based on model pricing

                        # Successfully parsed ATIF
                        return

                    except Exception as e:
                        # Log error but continue - fall back to defaults
                        import sys

                        print(f"Warning: Could not parse ATIF file {atif_file}: {e}", file=sys.stderr)

        # Fallback: set default values if ATIF not found or parsing failed
        context.n_input_tokens = 0
        context.n_output_tokens = 0
        context.n_cache_tokens = 0
        context.cost_usd = None
