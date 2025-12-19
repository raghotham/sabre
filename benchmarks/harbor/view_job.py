#!/usr/bin/env -S uv run
"""
Harbor Job Viewer

Interactive viewer for Harbor benchmark job folders.
Displays job structure, trial results, command outputs, and error logs.

Usage:
    # View latest job
    uv run benchmarks/harbor/view_job.py

    # View specific job
    uv run benchmarks/harbor/view_job.py --job 2025-12-18__13-51-30

    # View specific trial
    uv run benchmarks/harbor/view_job.py --trial hello-world__VQAszMD

    # Show only errors
    uv run benchmarks/harbor/view_job.py --errors-only
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rich>=13.7.0",
# ]
# ///

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree

console = Console()


def get_repo_root() -> Path:
    """Get the repository root directory."""
    return Path(__file__).parent.parent.parent


def get_jobs_dir() -> Path:
    """Get the jobs directory."""
    return get_repo_root() / "tmp" / "harbor"


def find_latest_job(jobs_dir: Path) -> Path | None:
    """Find the most recent job directory."""
    job_dirs = sorted(jobs_dir.glob("2025-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return job_dirs[0] if job_dirs else None


def read_json_file(path: Path) -> dict | None:
    """Read and parse a JSON file."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        console.print(f"[yellow]Warning: Could not parse {path.name}: {e}[/yellow]")
        return None


def read_text_file(path: Path, max_lines: int | None = None) -> str | None:
    """Read a text file, optionally limiting lines."""
    if not path.exists():
        return None
    try:
        content = path.read_text()
        if max_lines:
            lines = content.split("\n")
            if len(lines) > max_lines:
                return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
        return content
    except Exception as e:
        console.print(f"[yellow]Warning: Could not read {path.name}: {e}[/yellow]")
        return None


def display_job_summary(job_dir: Path):
    """Display job-level summary."""
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]Job:[/bold cyan] {job_dir.name}\n[dim]Path: {job_dir}[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    # Read job result.json
    result_json = read_json_file(job_dir / "result.json")
    if result_json:
        stats = result_json.get("stats", {})
        console.print("[bold]Job Statistics:[/bold]")
        console.print(f"  Total trials: {stats.get('n_trials', 0)}")
        console.print(f"  Errors: {stats.get('n_errors', 0)}")
        console.print(f"  Started: {result_json.get('started_at', 'N/A')}")
        console.print(f"  Finished: {result_json.get('finished_at', 'N/A')}")
        console.print()

        # Display eval stats
        evals = stats.get("evals", {})
        if evals:
            for eval_name, eval_data in evals.items():
                console.print(f"[bold cyan]{eval_name}[/bold cyan]")
                metrics = eval_data.get("metrics", [])
                if metrics and len(metrics) > 0:
                    console.print(f"  Mean reward: {metrics[0].get('mean', 'N/A')}")

                # Show reward distribution
                reward_stats = eval_data.get("reward_stats", {}).get("reward", {})
                if reward_stats:
                    console.print("  Reward distribution:")
                    for reward, trials in reward_stats.items():
                        console.print(f"    {reward}: {len(trials)} trial(s)")
                console.print()


def display_trial_tree(trial_dir: Path) -> Tree:
    """Create a tree view of trial directory structure."""
    tree = Tree(f"[bold]{trial_dir.name}[/bold]")

    # Add agent directory
    agent_dir = trial_dir / "agent"
    if agent_dir.exists():
        agent_node = tree.add("[cyan]agent/[/cyan]")

        # Add command directories
        command_dirs = sorted(agent_dir.glob("command-*"))
        for cmd_dir in command_dirs:
            cmd_node = agent_node.add(f"[yellow]{cmd_dir.name}/[/yellow]")
            for file in sorted(cmd_dir.glob("*")):
                if file.is_file():
                    size = file.stat().st_size
                    cmd_node.add(f"{file.name} [dim]({size} bytes)[/dim]")

        # Add other important files
        for file in sorted(agent_dir.glob("*")):
            if file.is_file():
                agent_node.add(f"{file.name}")

    # Add verifier directory
    verifier_dir = trial_dir / "verifier"
    if verifier_dir.exists():
        verifier_node = tree.add("[magenta]verifier/[/magenta]")
        for file in sorted(verifier_dir.glob("*")):
            if file.is_file():
                verifier_node.add(f"{file.name}")

    # Add trial files
    for file in sorted(trial_dir.glob("*.json")):
        tree.add(f"{file.name}")

    return tree


def display_command_output(cmd_dir: Path, show_full: bool = False):
    """Display command execution details."""
    cmd_name = cmd_dir.name

    # Read command
    command_txt = read_text_file(cmd_dir / "command.txt")
    return_code_txt = read_text_file(cmd_dir / "return-code.txt")
    stdout_txt = read_text_file(cmd_dir / "stdout.txt", max_lines=None if show_full else 50)
    stderr_txt = read_text_file(cmd_dir / "stderr.txt", max_lines=None if show_full else 50)

    # Determine status
    return_code = int(return_code_txt.strip()) if return_code_txt else None
    status = "✓" if return_code == 0 else "✗"
    status_color = "green" if return_code == 0 else "red"

    console.print()
    console.print(f"[bold {status_color}]{status} {cmd_name}[/bold {status_color}]")

    if command_txt:
        console.print(
            Panel(
                Syntax(command_txt.strip(), "bash", theme="monokai", line_numbers=False),
                title="Command",
                border_style="blue",
            )
        )

    if return_code_txt:
        console.print(f"[dim]Return code: {return_code}[/dim]")

    if stdout_txt and stdout_txt.strip():
        console.print(
            Panel(
                stdout_txt.strip(),
                title="stdout",
                border_style="green",
            )
        )

    if stderr_txt and stderr_txt.strip():
        console.print(
            Panel(
                stderr_txt.strip(),
                title="stderr",
                border_style="red",
            )
        )


def display_trial(trial_dir: Path, show_full: bool = False, errors_only: bool = False):
    """Display trial details."""
    console.print()
    console.print("=" * 80)
    console.print(f"[bold cyan]Trial: {trial_dir.name}[/bold cyan]")
    console.print("=" * 80)

    # Read trial result
    result_json = read_json_file(trial_dir / "result.json")
    if result_json:
        console.print()
        console.print("[bold]Trial Result:[/bold]")
        console.print(f"  Reward: {result_json.get('reward', 'N/A')}")
        console.print(f"  Success: {result_json.get('success', 'N/A')}")

        error = result_json.get("error")
        if error:
            console.print(f"[red]  Error: {error}[/red]")

        console.print()

    # Show directory tree
    if not errors_only:
        console.print("[bold]Directory Structure:[/bold]")
        tree = display_trial_tree(trial_dir)
        console.print(tree)
        console.print()

    # Display command outputs
    agent_dir = trial_dir / "agent"
    if agent_dir.exists():
        command_dirs = sorted(agent_dir.glob("command-*"))

        if errors_only:
            # Only show commands with non-zero return codes
            error_commands = []
            for cmd_dir in command_dirs:
                return_code_txt = read_text_file(cmd_dir / "return-code.txt")
                return_code = int(return_code_txt.strip()) if return_code_txt else None
                if return_code and return_code != 0:
                    error_commands.append(cmd_dir)
            command_dirs = error_commands

        if command_dirs:
            console.print(f"[bold]Command Outputs ({len(command_dirs)} commands):[/bold]")
            for cmd_dir in command_dirs:
                display_command_output(cmd_dir, show_full=show_full)

    # Display SABRE client output (from command-0 which runs SABRE)
    if agent_dir.exists():
        sabre_cmd = agent_dir / "command-0"
        if sabre_cmd.exists():
            sabre_stdout = read_text_file(sabre_cmd / "stdout.txt", max_lines=None if show_full else 200)
            if sabre_stdout:
                console.print()
                console.print("[bold]SABRE Client Output:[/bold]")
                console.print(
                    Panel(
                        sabre_stdout.strip(),
                        title="SABRE execution (command-0/stdout.txt)",
                        border_style="cyan",
                    )
                )

    # Display SABRE server log if it exists (copied from container)
    if agent_dir.exists():
        server_log = agent_dir / "server.log"
        if server_log.exists():
            console.print()
            console.print("[bold]SABRE Server Log:[/bold]")
            log_content = read_text_file(server_log, max_lines=None if show_full else 100)
            if log_content:
                console.print(
                    Panel(
                        log_content.strip(),
                        title="server.log (from container)",
                        border_style="blue",
                    )
                )

    # Display SABRE sessions if they exist
    sabre_sessions_dir = agent_dir / "sabre_sessions" if agent_dir.exists() else None
    if sabre_sessions_dir and sabre_sessions_dir.exists():
        console.print()
        console.print("[bold]SABRE Sessions:[/bold]")

        # Find all session directories
        session_dirs = sorted([d for d in sabre_sessions_dir.iterdir() if d.is_dir()])

        for session_dir in session_dirs:
            console.print(f"\n[cyan]Session: {session_dir.name}[/cyan]")

            # Show server.log if exists
            server_log = session_dir / "server.log"
            if server_log.exists():
                log_content = read_text_file(server_log, max_lines=None if show_full else 100)
                if log_content:
                    console.print(
                        Panel(
                            log_content.strip(),
                            title=f"{session_dir.name}/server.log",
                            border_style="blue",
                        )
                    )

            # Show client output if exists (might be in different locations)
            client_log = session_dir / "client.log"
            if client_log.exists():
                log_content = read_text_file(client_log, max_lines=None if show_full else 100)
                if log_content:
                    console.print(
                        Panel(
                            log_content.strip(),
                            title=f"{session_dir.name}/client.log",
                            border_style="cyan",
                        )
                    )

            # Show ATIF file if exists
            atif_file = session_dir / "atif.json"
            if atif_file.exists():
                console.print(f"[dim]  ATIF trace available: {atif_file}[/dim]")

    # Display verifier output
    verifier_dir = trial_dir / "verifier"
    if verifier_dir.exists():
        console.print()
        console.print("[bold]Verifier Output:[/bold]")

        test_stdout = read_text_file(verifier_dir / "test-stdout.txt", max_lines=None if show_full else 50)
        if test_stdout:
            console.print(
                Panel(
                    test_stdout.strip(),
                    title="test-stdout.txt",
                    border_style="magenta",
                )
            )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="View Harbor job results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--job",
        "-j",
        help="Specific job directory name (e.g., 2025-12-18__13-51-30). Defaults to latest.",
    )

    parser.add_argument(
        "--trial",
        "-t",
        help="Specific trial name to view (e.g., hello-world__VQAszMD)",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Show full output (don't truncate long files)",
    )

    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="Only show failed commands and errors",
    )

    args = parser.parse_args()

    jobs_dir = get_jobs_dir()
    if not jobs_dir.exists():
        console.print(f"[red]Jobs directory not found: {jobs_dir}[/red]")
        sys.exit(1)

    # Find job directory
    if args.job:
        job_dir = jobs_dir / args.job
        if not job_dir.exists():
            console.print(f"[red]Job directory not found: {job_dir}[/red]")
            sys.exit(1)
    else:
        job_dir = find_latest_job(jobs_dir)
        if not job_dir:
            console.print(f"[red]No job directories found in {jobs_dir}[/red]")
            sys.exit(1)

    # Display job summary
    display_job_summary(job_dir)

    # Find trial directories
    if args.trial:
        trial_dirs = list(job_dir.glob(f"*{args.trial}*"))
        trial_dirs = [d for d in trial_dirs if d.is_dir() and "__" in d.name]
        if not trial_dirs:
            console.print(f"[red]Trial not found: {args.trial}[/red]")
            sys.exit(1)
    else:
        trial_dirs = [d for d in job_dir.iterdir() if d.is_dir() and "__" in d.name]

    # Display trials
    for trial_dir in sorted(trial_dirs):
        display_trial(trial_dir, show_full=args.full, errors_only=args.errors_only)


if __name__ == "__main__":
    main()
