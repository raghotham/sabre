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

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.models.agent.context import AgentContext
from harbor.models.exec import ExecInput
from harbor.models.trial.result import AgentInfo, ModelInfo


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

        sabre_source_dir = Path(__file__).parent.parent.parent.parent  # Go up to sabre repo root
        target_dir = self.logs_dir / "sabre_source"

        # Copy SABRE source
        if sabre_source_dir.exists():
            shutil.copytree(
                sabre_source_dir,
                target_dir,
                symlinks=False,  # Don't copy symlinks, copy actual files
                ignore=shutil.ignore_patterns(
                    ".git",
                    ".venv",
                    "venv",
                    "__pycache__",
                    "*.pyc",
                    "benchmarks",
                    "jobs",
                    ".pytest_cache",
                    "tmp",
                    "*.egg-info",
                    "results",
                ),
            )

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

        # Escape the instruction for shell - replace double quotes with escaped quotes
        escaped_instruction = instruction.replace('"', '\\"')

        return [
            ExecInput(
                command=f'$HOME/.local/bin/uv run --directory /tmp/sabre sabre -m "{escaped_instruction}" --export-atif 2>&1 || (echo "=== SABRE FAILED, checking server.log ===" && cat /root/.local/state/sabre/logs/server.log 2>/dev/null || echo "No server.log found")',
                cwd="/app",  # Run from /app so Bash commands execute there
                timeout=3600,  # 1 hour timeout
                env={"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "")},  # Pass API key
            ),
            # Copy any PNG files from /app to logs directory for inspection
            ExecInput(
                command='cp /app/*.png /logs/agent/ 2>/dev/null || echo "No PNG files to copy"',
                cwd="/app",
                timeout=10,
            ),
            # Copy SABRE session directory to logs for debugging
            # Sessions are at ~/.local/share/sabre/sessions/
            ExecInput(
                command='cp -r $HOME/.local/share/sabre/sessions /logs/agent/sabre_sessions 2>/dev/null || echo "No SABRE sessions to copy"',
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
