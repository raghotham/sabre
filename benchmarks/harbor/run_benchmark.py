#!/usr/bin/env -S uv run
"""
SABRE Harbor Benchmark Runner.

Simplified wrapper for running SABRE with Harbor benchmarks.

Usage:
    # From repo root
    OPENAI_API_KEY=`cat ~/.openai/key` uv run benchmarks/harbor/run_benchmark.py --dataset terminal-bench@2.0
    OPENAI_API_KEY=`cat ~/.openai/key` uv run benchmarks/harbor/run_benchmark.py --dataset terminal-bench@2.0 --task chess-best-move

    # Run all tasks in dataset
    OPENAI_API_KEY=`cat ~/.openai/key` uv run benchmarks/harbor/run_benchmark.py --dataset hello-world@head

Examples:
    # Run all tasks in terminal-bench v2.0
    uv run benchmarks/harbor/run_benchmark.py --dataset terminal-bench@2.0

    # Run specific task
    uv run benchmarks/harbor/run_benchmark.py --dataset terminal-bench@2.0 --task chess-best-move

    # Run with debug logging
    uv run benchmarks/harbor/run_benchmark.py --dataset terminal-bench@2.0 --debug

    # Custom results directory
    uv run benchmarks/harbor/run_benchmark.py --dataset terminal-bench@2.0 --results-dir my-results
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rich>=13.7.0",
# ]
# ///

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()


def get_repo_root() -> Path:
    """Get the repository root directory."""
    # This script is at benchmarks/harbor/run_benchmark.py
    return Path(__file__).parent.parent.parent


def ensure_directories() -> tuple[Path, Path]:
    """Ensure required directories exist and return their paths."""
    repo_root = get_repo_root()

    # Jobs output directory (temporary files, logs)
    jobs_dir = repo_root / "tmp" / "harbor"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    # Results directory (final results to keep)
    results_dir = repo_root / "benchmarks" / "harbor" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    return jobs_dir, results_dir


def check_prerequisites() -> bool:
    """Check that all prerequisites are met."""
    errors = []

    # Check OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY environment variable not set")

    # Check Docker
    if not shutil.which("docker"):
        errors.append("Docker not found in PATH")
    else:
        # Check if Docker daemon is running
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                errors.append("Docker daemon not running")
        except Exception as e:
            errors.append(f"Error checking Docker: {e}")

    if errors:
        console.print("[bold red]Prerequisites check failed:[/bold red]")
        for error in errors:
            console.print(f"  ✗ {error}", style="red")
        console.print()
        console.print("[yellow]Run the setup script first:[/yellow]")
        console.print("  [cyan]uv run benchmarks/harbor/setup_benchmark.py[/cyan]")
        return False

    return True


def run_harbor_benchmark(
    dataset: str,
    task: str | None = None,
    debug: bool = False,
    results_dir: Path | None = None,
) -> int:
    """
    Run Harbor benchmark with SABRE agent.

    Args:
        dataset: Dataset name@version (e.g., terminal-bench@2.0)
        task: Optional task name to run (runs all if not specified)
        debug: Enable debug logging
        results_dir: Custom results directory (optional)

    Returns:
        Exit code from Harbor command
    """
    jobs_dir, default_results_dir = ensure_directories()
    results_dir = results_dir or default_results_dir

    repo_root = get_repo_root()

    # Build Harbor command - use uvx with local package
    cmd = [
        "uvx",
        "--with",
        ".",
        "harbor",
        "run",
        "-d",
        dataset,
        "--agent-import-path",
        "container:SabreAgent",
        "--env",
        "docker",
        "--ek",
        f"OPENAI_API_KEY={os.getenv('OPENAI_API_KEY')}",
        "--jobs-dir",
        str(jobs_dir),
    ]

    # Add task filter if specified
    if task:
        cmd.extend(["-t", task])

    # Add debug flag if requested
    if debug:
        cmd.append("--debug")

    # Display what we're running
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]Running SABRE on {dataset}[/bold cyan]"
            + (f"\n[dim]Task: {task}[/dim]" if task else "\n[dim]All tasks[/dim]"),
            border_style="cyan",
        )
    )
    console.print()
    console.print(f"[dim]Jobs output:[/dim] {jobs_dir}")
    console.print(f"[dim]Results:[/dim] {results_dir}")

    # Print command in debug mode
    if debug:
        # Mask API key in debug output
        debug_cmd = [part if not part.startswith("OPENAI_API_KEY=") else "OPENAI_API_KEY=***" for part in cmd]
        console.print(f"\n[dim]Command:[/dim] {' '.join(debug_cmd)}")
        console.print(f"[dim]Working directory:[/dim] {repo_root / 'benchmarks' / 'harbor'}")

    console.print()

    # Run Harbor from benchmarks/harbor directory
    # The --with . flag in the command ensures container package is available
    harbor_dir = repo_root / "benchmarks" / "harbor"

    try:
        result = subprocess.run(
            cmd,
            cwd=harbor_dir,
            env={**os.environ},
        )

        # Copy results to results directory
        if result.returncode == 0:
            copy_results_to_permanent(jobs_dir, results_dir, dataset, task)

        return result.returncode

    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[bold red]Error running Harbor:[/bold red] {e}")
        return 1


def copy_results_to_permanent(jobs_dir: Path, results_dir: Path, dataset: str, task: str | None) -> None:
    """Copy results from jobs directory to permanent results directory."""
    # Find the most recent job directory
    job_dirs = sorted(jobs_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not job_dirs:
        console.print("[yellow]No job results found to copy[/yellow]")
        return

    latest_job = job_dirs[0]

    # Create results subdirectory based on dataset
    dataset_name = dataset.split("@")[0]
    dataset_version = dataset.split("@")[1] if "@" in dataset else "latest"

    dest_dir = results_dir / f"{dataset_name}_{dataset_version}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy result.json if it exists
    result_json = latest_job / "result.json"
    if result_json.exists():
        shutil.copy2(result_json, dest_dir / f"{latest_job.name}_result.json")
        console.print(f"\n[green]✓ Results saved to:[/green] {dest_dir}")

    # If specific task, also copy the task directory
    if task:
        task_dirs = list(latest_job.glob(f"{task}__*"))
        if task_dirs:
            task_dir = task_dirs[0]
            dest_task_dir = dest_dir / task_dir.name
            if dest_task_dir.exists():
                shutil.rmtree(dest_task_dir)
            shutil.copytree(task_dir, dest_task_dir)
            console.print(f"[green]✓ Task results saved to:[/green] {dest_task_dir}")

            # Print SABRE session path if it exists
            sabre_sessions = task_dir / "agent" / "sabre_sessions"
            if sabre_sessions.exists():
                session_dirs = list(sabre_sessions.iterdir())
                if session_dirs:
                    console.print("\n[cyan]📝 SABRE session(s):[/cyan]")
                    for session_dir in session_dirs:
                        console.print(f"  {session_dir}")
                        # Check for ATIF file
                        atif_file = session_dir / "atif.json"
                        if atif_file.exists():
                            console.print("    [dim]└─ atif.json found[/dim]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run SABRE Harbor benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tasks in terminal-bench v2.0
  uv run benchmarks/harbor/run_benchmark.py --dataset terminal-bench@2.0

  # Run specific task
  uv run benchmarks/harbor/run_benchmark.py --dataset terminal-bench@2.0 --task chess-best-move

  # With debug logging
  uv run benchmarks/harbor/run_benchmark.py --dataset terminal-bench@2.0 --debug

Note: Set OPENAI_API_KEY environment variable before running.
        """,
    )

    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        help="Dataset name@version (e.g., terminal-bench@2.0, hello-world@head)",
    )

    parser.add_argument(
        "--task",
        "-t",
        help="Specific task name to run (runs all tasks if not specified)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging from Harbor",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Custom results directory (default: benchmarks/harbor/results)",
    )

    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip prerequisites check",
    )

    args = parser.parse_args()

    # Check prerequisites unless skipped
    if not args.skip_check and not check_prerequisites():
        sys.exit(1)

    # Run benchmark
    exit_code = run_harbor_benchmark(
        dataset=args.dataset,
        task=args.task,
        debug=args.debug,
        results_dir=args.results_dir,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
