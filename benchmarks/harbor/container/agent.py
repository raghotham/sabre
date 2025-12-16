"""
SABRE Agent implementation for Harbor benchmarks.

This agent communicates with a separately-running SABRE server via HTTP.
The server must be started before running Harbor benchmarks.

Architecture:
- SABRE server runs on host at http://host.docker.internal:8011
- Harbor container executes this agent code
- Agent creates session via POST /v1/sessions
- Agent sends task via POST /v1/sessions/{id}/messages
- Agent retrieves ATIF trace via GET /v1/sessions/{id}/atif

Usage:
    uvx harbor run -d terminal-bench@2.0 --agent-import-path container:SabreAgent
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
HARBOR_AVAILABLE = False
try:
    from harbor.agents.installed.base import BaseInstalledAgent, ExecInput
    from harbor.models.agent.context import AgentContext
    from harbor.models.trial.result import AgentInfo, ModelInfo

    HARBOR_AVAILABLE = True
except ImportError as e:
    import sys

    print(f"[sabre_agent] Harbor import failed: {e}, using stubs", file=sys.stderr)

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

    BaseInstalledAgent = BaseInstalledAgentStub

    class ExecInput:
        """Stub for ExecInput when harbor is not installed."""

        def __init__(self, command: str, timeout: int = 3600, **kwargs):
            self.command = command
            self.timeout = timeout
            for k, v in kwargs.items():
                setattr(self, k, v)

    AgentContext = Any
    AgentInfo = None
    ModelInfo = None


class SabreAgent(BaseInstalledAgent):
    """
    SABRE agent for Harbor benchmarks.

    Communicates with a separately-running SABRE server via HTTP.
    The server handles all LLM interactions and execution.
    """

    def __init__(
        self,
        logs_dir: Path,
        server_url: str | None = None,
        version: str | None = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the SABRE agent.

        Args:
            logs_dir: Directory for storing logs
            server_url: SABRE server URL (default: http://host.docker.internal:8011)
            version: Optional SABRE version identifier
        """
        super().__init__(logs_dir, version=version, *args, **kwargs)
        self._server_url = server_url or os.environ.get("SABRE_SERVER_URL", "http://host.docker.internal:8011")
        self._session_id: str | None = None

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
        if not HARBOR_AVAILABLE:
            return None

        from harbor.models.trial.result import AgentInfo, ModelInfo

        # Get model info from server health endpoint
        model_info = None
        try:
            import requests

            response = requests.get(
                f"{self._server_url}/v1/health",
                timeout=5.0,
                proxies={"http": None, "https": None},
            )
            if response.status_code == 200:
                health_data = response.json()
                if "model" in health_data:
                    model = health_data["model"]
                    # Parse provider from model name if present
                    if "/" in model:
                        provider, model_name = model.split("/", 1)
                    else:
                        provider = "openai"
                        model_name = model
                    model_info = ModelInfo(name=model_name, provider=provider)
        except Exception:
            pass

        return AgentInfo(
            name=self.name(),
            version=self.version() or "latest",
            model_info=model_info,
        )

    @property
    def _install_agent_template_path(self) -> Path:
        """Return path to the installation script."""
        return Path(__file__).parent / "install.sh"

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        """
        Create commands to run SABRE via HTTP API.

        This creates a uv script that:
        1. Creates a session via POST /v1/sessions
        2. Sends the task message via POST /v1/sessions/{id}/messages
        3. Saves session_id for later ATIF retrieval

        Args:
            instruction: The task description/prompt to send to SABRE

        Returns:
            List of ExecInput commands to execute
        """
        # Create uv script that communicates with SABRE server
        script = f'''#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests>=2.31.0",
# ]
# ///

import json
import os
import sys
import time
import requests

# Configuration
SERVER_URL = "{self._server_url}"
LOGS_DIR = "{self.logs_dir}"
SESSION_ID_FILE = os.path.join(LOGS_DIR, "session_id.txt")

# Bypass proxy for host.docker.internal
PROXIES = {{"http": None, "https": None}}

def create_session():
    """Create a new SABRE session."""
    print("Creating SABRE session...", file=sys.stderr)
    response = requests.post(
        f"{{SERVER_URL}}/v1/sessions",
        json={{}},
        timeout=30.0,
        proxies=PROXIES
    )
    response.raise_for_status()
    data = response.json()
    session_id = data["session_id"]
    print(f"Created session: {{session_id}}", file=sys.stderr)

    # Save session ID for ATIF retrieval
    with open(SESSION_ID_FILE, "w") as f:
        f.write(session_id)

    return session_id

def send_message(session_id, message):
    """Send message to SABRE session and wait for completion."""
    print(f"Sending task to session {{session_id}}...", file=sys.stderr)

    # Stream the response
    response = requests.post(
        f"{{SERVER_URL}}/v1/sessions/{{session_id}}/messages",
        json={{"message": message}},
        timeout=3600.0,  # 1 hour timeout
        stream=True,
        proxies=PROXIES
    )
    response.raise_for_status()

    # Process SSE stream
    for line in response.iter_lines():
        if not line:
            continue

        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue

        data_str = line[6:]  # Remove "data: " prefix
        if data_str == "[DONE]":
            break

        try:
            event = json.loads(data_str)
            event_type = event.get("type")

            # Print tokens as they arrive
            if event_type == "response_token":
                token = event.get("data", {{}}).get("token", "")
                print(token, end="", flush=True)
            elif event_type == "complete":
                print("\\n", file=sys.stderr)
                print(f"Task completed successfully", file=sys.stderr)
                return True
            elif event_type == "error":
                error_msg = event.get("data", {{}}).get("error_message", "Unknown error")
                print(f"\\nError: {{error_msg}}", file=sys.stderr)
                return False
        except json.JSONDecodeError:
            continue

    return True

def main():
    # Task instruction
    instruction = """{json.dumps(instruction)}"""

    try:
        # Create session
        session_id = create_session()

        # Send message
        success = send_message(session_id, instruction)

        if not success:
            sys.exit(1)

        print(f"\\nSession completed: {{session_id}}", file=sys.stderr)
        sys.exit(0)

    except Exception as e:
        print(f"Error: {{e}}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

        # Write script to file
        script_path = self.logs_dir / "run_sabre.py"
        script_path.write_text(script)
        script_path.chmod(0o755)

        return [
            ExecInput(
                command="$HOME/.local/bin/uv run ./run_sabre.py",
                timeout=3600,  # 1 hour timeout
            )
        ]

    def populate_context_post_run(self, context: AgentContext) -> None:
        """
        Populate the agent context after execution.

        This method retrieves the ATIF trace from the SABRE server.

        Args:
            context: The agent context to populate
        """
        # Read session ID from file
        session_id_file = self.logs_dir / "session_id.txt"
        if not session_id_file.exists():
            print(f"Session ID file not found: {session_id_file}")
            context.n_input_tokens = 0
            context.n_output_tokens = 0
            context.n_cache_tokens = 0
            context.cost_usd = None
            return

        try:
            session_id = session_id_file.read_text().strip()
            print(f"Retrieved session ID: {session_id}")
        except Exception as e:
            print(f"Failed to read session ID: {e}")
            context.n_input_tokens = 0
            context.n_output_tokens = 0
            context.n_cache_tokens = 0
            context.cost_usd = None
            return

        # Retrieve ATIF trace from server
        try:
            import requests

            print(f"Retrieving ATIF trace from {self._server_url}/v1/sessions/{session_id}/atif")
            response = requests.get(
                f"{self._server_url}/v1/sessions/{session_id}/atif",
                timeout=30.0,
                proxies={"http": None, "https": None},
            )
            response.raise_for_status()
            atif_data = response.json()

            # Write ATIF trajectory to file
            trajectory_path = self.logs_dir / "trajectory.json"
            with open(trajectory_path, "w") as f:
                json.dump(atif_data, f, indent=2)
            print(f"Wrote ATIF trajectory to {trajectory_path}")

            # Extract token counts from ATIF if available
            final_metrics = atif_data.get("final_metrics", {})
            context.n_input_tokens = final_metrics.get("total_prompt_tokens", 0)
            context.n_output_tokens = final_metrics.get("total_completion_tokens", 0)
            context.n_cache_tokens = final_metrics.get("total_cached_tokens", 0)
            context.cost_usd = final_metrics.get("total_cost_usd")

        except Exception as e:
            print(f"Failed to retrieve ATIF trace: {e}")
            import traceback

            traceback.print_exc()
            context.n_input_tokens = 0
            context.n_output_tokens = 0
            context.n_cache_tokens = 0
            context.cost_usd = None
