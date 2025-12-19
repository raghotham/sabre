#!/usr/bin/env -S uv run
"""
SABRE Harbor Benchmark Setup and Prerequisites Checker.

This script verifies that all prerequisites are met for running SABRE with Harbor:
1. Docker is installed and running
2. Harbor CLI is installed and accessible (via uvx)
3. uv is installed (for running Harbor)
4. OPENAI_API_KEY is set

Note: SABRE runs inside the Docker container in command mode, so no local
SABRE server is required.

Usage:
    uv run setup_benchmark.py
    uv run setup_benchmark.py --quiet
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

from rich.console import Console
from rich.panel import Panel

console = Console()


class PrerequisiteChecker:
    """Check all prerequisites for running SABRE Harbor benchmark."""

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.errors = []
        self.warnings = []
        self.checks_passed = 0
        self.checks_total = 0

    def check_all(self) -> bool:
        """Run all prerequisite checks."""
        if not self.quiet:
            console.print(
                Panel.fit(
                    "[bold cyan]SABRE Harbor Benchmark Prerequisites Checker[/bold cyan]",
                    border_style="cyan",
                )
            )
            console.print()

        # Run all checks
        self.check_docker()
        self.check_harbor_cli()
        self.check_env_vars()

        # Display summary
        if not self.quiet:
            self._display_summary()

        return len(self.errors) == 0

    def check_docker(self):
        """Check if Docker is installed and running."""
        self.checks_total += 1

        if not self.quiet:
            console.print("[bold]1. Checking Docker...[/bold]")

        # Check if docker command exists
        docker_path = shutil.which("docker")
        if not docker_path:
            self.errors.append("Docker not found in PATH")
            if not self.quiet:
                console.print("   ✗ Docker not found in PATH", style="red")
                console.print(
                    "   → Install Docker: [cyan]https://docs.docker.com/get-docker/[/cyan]",
                    style="yellow",
                )
                console.print()
            return

        # Check if Docker daemon is running
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                self.checks_passed += 1
                if not self.quiet:
                    console.print(f"   ✓ Docker installed at {docker_path}", style="green")
                    console.print("   ✓ Docker daemon is running", style="green")

                    # Parse docker info for version
                    for line in result.stdout.split("\n"):
                        if "Server Version:" in line:
                            version = line.split(":")[-1].strip()
                            console.print(f"   ✓ Docker version: {version}", style="green")
                            break
                    console.print()
            else:
                self.errors.append("Docker daemon not running")
                if not self.quiet:
                    console.print("   ✗ Docker daemon not running", style="red")
                    console.print("   → Start Docker Desktop or Docker daemon", style="yellow")
                    console.print()
        except subprocess.TimeoutExpired:
            self.errors.append("Docker command timed out")
            if not self.quiet:
                console.print("   ✗ Docker command timed out", style="red")
                console.print()
        except Exception as e:
            self.errors.append(f"Error checking Docker: {e}")
            if not self.quiet:
                console.print(f"   ✗ Error: {e}", style="red")
                console.print()

    def check_harbor_cli(self):
        """Check if Harbor CLI is installed, and install if needed."""
        self.checks_total += 1

        if not self.quiet:
            console.print("[bold]2. Checking Harbor CLI...[/bold]")

        # Check if harbor is already in PATH
        harbor_path = shutil.which("harbor")
        if harbor_path:
            try:
                result = subprocess.run(
                    [harbor_path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    self.checks_passed += 1
                    if not self.quiet:
                        version = result.stdout.strip()
                        console.print(f"   ✓ Harbor installed at {harbor_path}", style="green")
                        console.print(f"   ✓ Version: {version}", style="green")
                        console.print()
                    return
            except Exception:
                pass

        # Harbor not found - install it
        if not self.quiet:
            console.print("   ⚠ Harbor not found in PATH", style="yellow")
            console.print("   → Installing Harbor via uv pip...", style="cyan")

        try:
            result = subprocess.run(
                ["uv", "pip", "install", "harbor"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                self.checks_passed += 1
                if not self.quiet:
                    console.print("   ✓ Harbor installed successfully", style="green")
                    console.print()
            else:
                self.errors.append(f"Failed to install Harbor: {result.stderr}")
                if not self.quiet:
                    console.print("   ✗ Failed to install Harbor", style="red")
                    console.print(f"   Error: {result.stderr}", style="red")
                    console.print()
        except Exception as e:
            self.errors.append(f"Error installing Harbor: {e}")
            if not self.quiet:
                console.print(f"   ✗ Error: {e}", style="red")
                console.print()

    def check_env_vars(self):
        """Check environment variables."""
        self.checks_total += 1

        if not self.quiet:
            console.print("[bold]3. Checking Environment Variables...[/bold]")

        # Check for OPENAI_API_KEY (required)
        api_key = os.getenv("OPENAI_API_KEY")

        if api_key:
            self.checks_passed += 1
            if not self.quiet:
                # Mask the API key for security
                masked_key = api_key[:10] + "..." + api_key[-4:] if len(api_key) > 14 else "***"
                console.print(f"   ✓ OPENAI_API_KEY is set ({masked_key})", style="green")
                console.print()
        else:
            self.errors.append("OPENAI_API_KEY not set")
            if not self.quiet:
                console.print("   ✗ OPENAI_API_KEY not set", style="red")
                console.print(
                    "   → Set it with: [cyan]export OPENAI_API_KEY=your-key-here[/cyan]",
                    style="yellow",
                )
                console.print(
                    "   → Or pass it when running: [cyan]--ek OPENAI_API_KEY=$OPENAI_API_KEY[/cyan]",
                    style="yellow",
                )
                console.print()

    def _display_summary(self):
        """Display summary of all checks."""
        console.print()
        console.print("=" * 60)

        if len(self.errors) == 0:
            console.print(f"[bold green]✓ All checks passed ({self.checks_passed}/{self.checks_total})[/bold green]")
            if self.warnings:
                console.print(f"\n[yellow]⚠ {len(self.warnings)} warning(s):[/yellow]")
                for warning in self.warnings:
                    console.print(f"  • {warning}", style="yellow")
            console.print("\n[green]Ready to run Harbor benchmarks![/green]")
        else:
            console.print(
                f"[bold red]✗ {len(self.errors)} error(s) found ({self.checks_passed}/{self.checks_total} checks passed)[/bold red]"
            )
            console.print()
            for error in self.errors:
                console.print(f"  • {error}", style="red")

        console.print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check prerequisites for SABRE Harbor benchmark")
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output - only show errors",
    )

    args = parser.parse_args()

    # Ensure tmp/harbor directory exists
    from pathlib import Path

    repo_root = Path(__file__).parent.parent.parent
    tmp_harbor = repo_root / "tmp" / "harbor"
    tmp_harbor.mkdir(parents=True, exist_ok=True)

    results_dir = repo_root / "benchmarks" / "harbor" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    checker = PrerequisiteChecker(quiet=args.quiet)
    success = checker.check_all()

    if success and not args.quiet:
        console.print()
        console.print("[dim]Output directories created:[/dim]")
        console.print(f"  [cyan]Jobs:[/cyan] {tmp_harbor}")
        console.print(f"  [cyan]Results:[/cyan] {results_dir}")
        console.print()
        console.print("[green]✓ Ready to run Harbor benchmarks![/green]")
        console.print()
        console.print("[dim]Note: Harbor will download dataset tasks automatically on first run.[/dim]")
        console.print("[dim]Use --list-tasks to see available tasks after running a benchmark.[/dim]")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
