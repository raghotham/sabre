"""
ATIF (Agent Trajectory Interchange Format) export functionality.

Converts SABRE session logs to ATIF v1.2 format for benchmark compatibility
(Harbor, Terminal-Bench, etc.).

ATIF is a standardized format for representing agent execution traces with:
- Sequential steps (agent messages, tool calls, observations)
- Token usage and cost metrics
- Agent metadata (name, version, model)

SABRE's ExecutionTree contains richer information (hierarchical structure,
precise timing, status tracking) which is flattened to ATIF's sequential format.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def events_to_atif(
    events: list[dict],
    agent_name: str = "sabre",
    agent_version: str = "latest",
    model_name: Optional[str] = None,
) -> dict:
    """
    Convert SABRE session events to ATIF format.

    Args:
        events: List of event dicts from session.jsonl
        agent_name: Name of the agent (default: "sabre")
        agent_version: Version of the agent (default: "latest")
        model_name: Model name used (e.g., "gpt-4o")

    Returns:
        ATIF v1.2 formatted dict

    ATIF Format:
        {
            "schema_version": "ATIF-v1.2",
            "session_id": "...",
            "agent": {
                "name": "sabre",
                "version": "latest",
                "model_name": "gpt-4o"
            },
            "steps": [
                {
                    "step_id": 1,
                    "source": "agent",
                    "message": "...",
                    "tool_calls": [...],
                    "observation": {...}
                }
            ],
            "final_metrics": {
                "total_prompt_tokens": 1000,
                "total_completion_tokens": 500,
                "total_cached_tokens": 0,
                "total_cost_usd": null,
                "total_steps": 5
            }
        }
    """
    if not events:
        logger.warning("No events to convert to ATIF")
        return _empty_atif(agent_name, agent_version, model_name)

    # Extract session_id from first event
    session_id = events[0].get("session_id", "unknown")

    # Parse events into ATIF steps
    steps = []
    step_id = 1

    # Track metrics
    total_input_tokens = 0
    total_output_tokens = 0
    total_reasoning_tokens = 0

    # Track helper execution state
    # Map: node_id -> {code, result, started}
    helpers_state = {}

    for event in events:
        event_type = event.get("event_type")

        # User messages
        if event_type == "user_message":
            content = event.get("content", "")
            if content:
                steps.append(
                    {
                        "step_id": step_id,
                        "source": "user",
                        "message": content,
                    }
                )
                step_id += 1

        # Session start (contains initial user message)
        elif event_type == "session_start":
            message = event.get("message", "")
            if message:
                steps.append(
                    {
                        "step_id": step_id,
                        "source": "user",
                        "message": message,
                    }
                )
                step_id += 1

        # Assistant messages (LLM responses)
        elif event_type == "assistant_message":
            content = event.get("content", "")
            if content:
                steps.append(
                    {
                        "step_id": step_id,
                        "source": "agent",
                        "message": content,
                    }
                )
                step_id += 1

        # Node output can contain various types
        elif event_type == "node_output":
            output_type = event.get("output_type")
            content = event.get("content", "")

            # Helper execution started - track the code
            if output_type == "helper_code":
                node_id = event.get("node_id")
                if node_id:
                    helpers_state[node_id] = {
                        "code": content,
                        "result": None,
                        "started": True,
                    }

            # Helper result - combine with code into tool call
            elif output_type == "helper_result":
                node_id = event.get("node_id")
                if node_id and node_id in helpers_state:
                    helpers_state[node_id]["result"] = content

                    # Create tool call step
                    code = helpers_state[node_id]["code"]
                    result = content

                    tool_call = {
                        "tool_call_id": f"helper_{node_id[:8]}",
                        "function_name": "python_runtime",
                        "arguments": {"code": code},
                    }

                    observation = {
                        "results": [
                            {
                                "source_call_id": f"helper_{node_id[:8]}",
                                "content": result,
                            }
                        ]
                    }

                    steps.append(
                        {
                            "step_id": step_id,
                            "source": "agent",
                            "message": "Executing code",
                            "tool_calls": [tool_call],
                            "observation": observation,
                        }
                    )
                    step_id += 1

                    # Clean up
                    del helpers_state[node_id]

        # Node completion - extract token counts
        elif event_type == "node_complete":
            tokens = event.get("tokens", {})
            if tokens:
                total_input_tokens += tokens.get("input_tokens", 0)
                total_output_tokens += tokens.get("output_tokens", 0)
                total_reasoning_tokens += tokens.get("reasoning_tokens", 0)

    # Build final ATIF structure
    atif = {
        "schema_version": "ATIF-v1.2",
        "session_id": session_id,
        "agent": {
            "name": agent_name,
            "version": agent_version,
        },
        "steps": steps,
        "final_metrics": {
            "total_prompt_tokens": total_input_tokens,
            "total_completion_tokens": total_output_tokens,
            "total_cached_tokens": total_reasoning_tokens,  # Map reasoning to cached
            "total_cost_usd": None,  # SABRE doesn't track cost yet
            "total_steps": len(steps),
        },
    }

    # Add model_name if provided
    if model_name:
        atif["agent"]["model_name"] = model_name

    logger.info(
        f"Converted session {session_id} to ATIF: {len(steps)} steps, "
        f"{total_input_tokens} input tokens, {total_output_tokens} output tokens"
    )

    return atif


def _empty_atif(
    agent_name: str,
    agent_version: str,
    model_name: Optional[str] = None,
) -> dict:
    """Return empty ATIF structure for sessions with no events."""
    atif = {
        "schema_version": "ATIF-v1.2",
        "session_id": "unknown",
        "agent": {
            "name": agent_name,
            "version": agent_version,
        },
        "steps": [],
        "final_metrics": {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cached_tokens": 0,
            "total_cost_usd": None,
            "total_steps": 0,
        },
    }

    if model_name:
        atif["agent"]["model_name"] = model_name

    return atif


def export_session_to_atif_file(
    session_id: str,
    session_logger,
    output_path: str,
    agent_name: str = "sabre",
    agent_version: str = "latest",
    model_name: Optional[str] = None,
) -> bool:
    """
    Export a session to ATIF JSON file.

    Args:
        session_id: Session ID to export
        session_logger: SessionLogger instance
        output_path: Path to write ATIF JSON file
        agent_name: Name of the agent
        agent_version: Version of the agent
        model_name: Model name used

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load session events
        events = session_logger.get_session(session_id)

        # Convert to ATIF
        atif = events_to_atif(
            events,
            agent_name=agent_name,
            agent_version=agent_version,
            model_name=model_name,
        )

        # Write to file
        with open(output_path, "w") as f:
            json.dump(atif, f, indent=2)

        logger.info(f"Exported session {session_id} to ATIF file: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to export session {session_id} to ATIF: {e}")
        return False
