#!/usr/bin/env -S uv run
"""
SABRE Harbor Benchmark Setup and Prerequisites Checker.

This script verifies that all prerequisites are met for running SABRE with Harbor:
1. SABRE server is running (http://localhost:8011)
2. Docker is installed and running
3. Harbor CLI is installed and accessible
4. Required environment variables are set

Usage:
    uv run setup_benchmark.py
    uv run setup_benchmark.py --test-server
    uv run setup_benchmark.py --quiet
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests>=2.31.0",
#     "rich>=13.7.0",
# ]
# ///

import argparse
import os
import shutil
import subprocess
import sys

import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class PrerequisiteChecker:
    """Check all prerequisites for running SABRE Harbor benchmark."""

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.errors = []
        self.warnings = []
        self.checks_passed = 0
        self.checks_total = 0

    def check_all(self, test_server: bool = False) -> bool:
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
        self.check_sabre_server(test_server)
        self.check_docker()
        self.check_harbor_cli()
        self.check_env_vars()

        # Display summary
        if not self.quiet:
            self._display_summary()

        return len(self.errors) == 0

    def check_sabre_server(self, test_connection: bool = False):
        """Check if SABRE server is running."""
        self.checks_total += 1
        server_url = "http://localhost:8011"

        if not self.quiet:
            console.print("[bold]1. Checking SABRE Server...[/bold]")

        try:
            # Bypass proxy for localhost
            response = requests.get(
                f"{server_url}/v1/health",
                timeout=5.0,
                proxies={"http": None, "https": None},  # Disable proxy for localhost
            )
            if response.status_code == 200:
                self.checks_passed += 1
                if not self.quiet:
                    health_data = response.json()
                    console.print(f"   ✓ SABRE server running at {server_url}", style="green")
                    console.print(f"   ✓ Status: {health_data.get('status', 'unknown')}", style="green")
                    if "model" in health_data:
                        console.print(f"   ✓ Model: {health_data['model']}", style="green")
                    console.print()

                # Optional: test full connection
                if test_connection:
                    self._test_server_connection(server_url)
            else:
                self.errors.append(f"SABRE server returned status {response.status_code}")
                if not self.quiet:
                    console.print(f"   ✗ Server returned status {response.status_code}", style="red")
                    console.print()
        except requests.exceptions.ConnectionError:
            self.errors.append("SABRE server not running")
            if not self.quiet:
                console.print(f"   ✗ SABRE server not running at {server_url}", style="red")
                console.print("   → Start it with: [cyan]uv run sabre-server[/cyan]", style="yellow")
                console.print()
        except Exception as e:
            self.errors.append(f"Error connecting to SABRE server: {e}")
            if not self.quiet:
                console.print(f"   ✗ Error: {e}", style="red")
                console.print()

    def _test_server_connection(self, server_url: str):
        """Test SABRE server with a simple message."""
        if not self.quiet:
            console.print("   [dim]Testing server connection...[/dim]")

        try:
            response = requests.post(
                f"{server_url}/v1/messages",
                json={
                    "message": "Hello SABRE! This is a test.",
                    "stream": False,
                },
                timeout=30.0,
                proxies={"http": None, "https": None},
            )
            if response.status_code == 200:
                console.print("   ✓ Server connection test passed", style="green")
            else:
                self.warnings.append(f"Server test returned status {response.status_code}")
                console.print(f"   ⚠ Test returned status {response.status_code}", style="yellow")
        except Exception as e:
            self.warnings.append(f"Server connection test failed: {e}")
            console.print(f"   ⚠ Connection test failed: {e}", style="yellow")

        console.print()

    def check_docker(self):
        """Check if Docker is installed and running."""
        self.checks_total += 1

        if not self.quiet:
            console.print("[bold]2. Checking Docker...[/bold]")

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
        """Check if Harbor CLI is installed."""
        self.checks_total += 1

        if not self.quiet:
            console.print("[bold]3. Checking Harbor CLI...[/bold]")

        # Try uvx harbor first (recommended)
        try:
            result = subprocess.run(
                ["uvx", "harbor", "--help"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                self.checks_passed += 1
                if not self.quiet:
                    console.print("   ✓ Harbor accessible via uvx", style="green")
                    console.print("   ✓ Run with: [cyan]uvx harbor run[/cyan]", style="green")
                    console.print()
                return
        except Exception:
            pass

        # Fall back to checking if harbor is in PATH
        harbor_path = shutil.which("harbor")
        if not harbor_path:
            self.errors.append("Harbor CLI not found")
            if not self.quiet:
                console.print("   ✗ Harbor CLI not found", style="red")
                console.print(
                    "   → Install: [cyan]pip install harbor-bench[/cyan] or [cyan]uv pip install harbor-bench[/cyan]",
                    style="yellow",
                )
                console.print("   → Or use: [cyan]uvx harbor[/cyan] (no install needed)", style="yellow")
                console.print()
            return

        # Harbor is in PATH - test it
        try:
            result = subprocess.run(
                [harbor_path, "--help"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self.checks_passed += 1
                if not self.quiet:
                    console.print(f"   ✓ Harbor installed at {harbor_path}", style="green")
                    console.print()
            else:
                self.warnings.append("Harbor executable found but failed to run")
                if not self.quiet:
                    console.print("   ⚠ Harbor found but failed to run", style="yellow")
                    console.print()
        except Exception as e:
            self.errors.append(f"Error checking Harbor: {e}")
            if not self.quiet:
                console.print(f"   ✗ Error: {e}", style="red")
                console.print()

    def check_env_vars(self):
        """Check environment variables (all optional - SABRE server handles auth)."""
        self.checks_total += 1

        if not self.quiet:
            console.print("[bold]4. Checking Environment Variables...[/bold]")

        # All optional - SABRE server already has OPENAI_API_KEY
        optional_vars = {
            "SABRE_SERVER_URL": "SABRE server URL (default: http://localhost:8011)",
            "LOG_LEVEL": "Logging level (default: WARNING)",
        }

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Variable", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Value/Description")

        # Check optional
        for var, desc in optional_vars.items():
            value = os.getenv(var)
            if value:
                table.add_row(var, "✓", value, style="green")
            else:
                table.add_row(var, "-", f"Optional: {desc}", style="dim")

        if not self.quiet:
            console.print(table)
            console.print()

        # Always pass - all vars are optional
        self.checks_passed += 1

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
        "--test-server",
        action="store_true",
        help="Test SABRE server connection with a sample message",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output - only show errors",
    )

    args = parser.parse_args()

    checker = PrerequisiteChecker(quiet=args.quiet)
    success = checker.check_all(test_server=args.test_server)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
