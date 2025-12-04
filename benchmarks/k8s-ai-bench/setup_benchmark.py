#!/usr/bin/env python3
"""
Setup script for k8s-ai-bench benchmark with SABRE.

This script:
- Checks all prerequisites (kubectl-ai, kubectl, k8s-ai-bench, etc.)
- Sets up MCP configuration for kubectl-ai
- Verifies SABRE server connectivity

Usage:
    python setup_benchmark.py              # Check prerequisites and setup
    python setup_benchmark.py --check      # Only check prerequisites
    python setup_benchmark.py --setup-mcp  # Only setup MCP config
    python setup_benchmark.py --force      # Force overwrite MCP config
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_DIR = Path.home() / ".config" / "sabre"
MCP_CONFIG = CONFIG_DIR / "mcp.yaml"
MCP_TEMPLATE = SCRIPT_DIR / "configs" / "mcp.yaml"
SABRE_PORT = 8011

# K8S_AI_BENCH_DIR must be set by user
K8S_AI_BENCH_DIR_ENV = os.environ.get("K8S_AI_BENCH_DIR")
if K8S_AI_BENCH_DIR_ENV:
    K8S_AI_BENCH_DIR = Path(K8S_AI_BENCH_DIR_ENV)
else:
    K8S_AI_BENCH_DIR = None

# Colors for terminal output
class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    NC = "\033[0m"  # No Color


def print_success(msg: str):
    print(f"{Colors.GREEN}✅{Colors.NC} {msg}")


def print_fail(msg: str):
    print(f"{Colors.RED}❌{Colors.NC} {msg}")


def print_warn(msg: str):
    print(f"{Colors.YELLOW}⚠️{Colors.NC} {msg}")


def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ️{Colors.NC} {msg}")


def print_header(msg: str):
    print(f"\n{Colors.BLUE}{'=' * 50}{Colors.NC}")
    print(f"{Colors.BLUE}{msg}{Colors.NC}")
    print(f"{Colors.BLUE}{'=' * 50}{Colors.NC}\n")


def command_exists(cmd: str) -> bool:
    """Check if a command exists in PATH."""
    return shutil.which(cmd) is not None


def run_command(cmd: str, capture: bool = True) -> tuple[bool, str]:
    """Run a shell command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=capture,
            text=True,
            timeout=10
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def check_prerequisites() -> tuple[int, int]:
    """
    Check all prerequisites for running k8s-ai-bench.

    Returns:
        Tuple of (passed_count, failed_count)
    """
    print_header("k8s-ai-bench Prerequisites Check")

    passed = 0
    failed = 0

    def check(name: str, condition: bool, required: bool = True):
        nonlocal passed, failed
        if condition:
            print_success(name)
            passed += 1
        elif required:
            print_fail(name)
            failed += 1
        else:
            print_warn(f"{name} (optional)")

    # Required checks
    print("Required:")
    check("kubectl-ai binary", command_exists("kubectl-ai"))
    check("kubectl binary", command_exists("kubectl"))

    # Check K8S_AI_BENCH_DIR
    if K8S_AI_BENCH_DIR is None:
        print_fail("K8S_AI_BENCH_DIR environment variable not set")
        print_info("  Set it with: export K8S_AI_BENCH_DIR=/path/to/kubectl-ai/k8s-ai-bench")
        failed += 1
    else:
        check("k8s-ai-bench directory", K8S_AI_BENCH_DIR.exists())
        check("k8s-ai-bench binary", (K8S_AI_BENCH_DIR / "k8s-ai-bench").exists())

    check("OPENAI_API_KEY set", bool(os.environ.get("OPENAI_API_KEY")))

    # SABRE checks
    print("\nSABRE:")
    check("SABRE MCP config exists", MCP_CONFIG.exists())

    # Check if kubectl-ai is in MCP config
    kubectl_ai_configured = False
    if MCP_CONFIG.exists():
        try:
            content = MCP_CONFIG.read_text()
            kubectl_ai_configured = "kubectl-ai" in content
        except:
            pass
    check("kubectl-ai in MCP config", kubectl_ai_configured, required=False)

    # Check if SABRE server is running
    sabre_running, _ = run_command(f"curl -s http://localhost:{SABRE_PORT}/health")
    check("SABRE server running", sabre_running, required=False)

    # Kubernetes checks
    print("\nKubernetes:")
    k8s_accessible, _ = run_command("kubectl cluster-info")
    check("Kubernetes cluster accessible", k8s_accessible, required=False)
    check("kind available", command_exists("kind"), required=False)

    # Summary
    print_header("Summary")
    print(f"Passed: {Colors.GREEN}{passed}{Colors.NC}")
    print(f"Failed: {Colors.RED}{failed}{Colors.NC}")

    if failed > 0:
        print(f"\n{Colors.RED}Fix the failed checks before running the benchmark.{Colors.NC}")
    else:
        print(f"\n{Colors.GREEN}All required checks passed!{Colors.NC}")

    return passed, failed


def setup_mcp_config(force: bool = False) -> bool:
    """
    Setup MCP configuration for kubectl-ai.

    Args:
        force: Force overwrite existing config

    Returns:
        True if setup was successful
    """
    print_header("MCP Configuration Setup")

    # Check if kubectl-ai is installed
    if not command_exists("kubectl-ai"):
        print_fail("kubectl-ai not found in PATH")
        print_info("Install kubectl-ai first:")
        print_info("  cd /path/to/kubectl-ai && go build -o kubectl-ai ./cmd")
        print_info("  sudo mv kubectl-ai /usr/local/bin/")
        return False

    kubectl_ai_path = shutil.which("kubectl-ai")
    print_success(f"kubectl-ai found at: {kubectl_ai_path}")

    # Create config directory if needed
    if not CONFIG_DIR.exists():
        print_info(f"Creating config directory: {CONFIG_DIR}")
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Check if MCP config already exists
    if MCP_CONFIG.exists():
        print_warn(f"MCP config already exists: {MCP_CONFIG}")

        try:
            content = MCP_CONFIG.read_text()
        except Exception as e:
            print_fail(f"Failed to read config: {e}")
            return False

        # Check if kubectl-ai is already configured
        if "kubectl-ai" in content:
            print_info("kubectl-ai is already configured in MCP config")
            print_info("Current configuration:")

            # Show kubectl-ai section
            lines = content.split("\n")
            in_section = False
            for line in lines:
                if "kubectl-ai:" in line:
                    in_section = True
                if in_section:
                    print(f"  {line}")
                    if line.strip() and not line.startswith(" ") and "kubectl-ai:" not in line:
                        break

            if not force:
                response = input("\nDo you want to overwrite with fresh config? [y/N] ").strip().lower()
                if response != "y":
                    print_info("Keeping existing configuration")
                    return True
        else:
            print_info("kubectl-ai not found in existing config")
            print_info("Appending kubectl-ai configuration...")

            # Append kubectl-ai config
            kubectl_ai_config = """
  # kubectl-ai MCP server (added by setup_benchmark.py)
  kubectl-ai:
    type: stdio
    command: kubectl-ai
    args: ["--mcp-server"]
    enabled: true
    timeout: 60
"""
            with open(MCP_CONFIG, "a") as f:
                f.write(kubectl_ai_config)

            print_success(f"kubectl-ai configuration appended to {MCP_CONFIG}")
            return True

    # Check if template exists
    if not MCP_TEMPLATE.exists():
        print_warn(f"MCP template not found: {MCP_TEMPLATE}")
        print_info("Creating MCP config from scratch...")

        mcp_content = """# SABRE MCP Server Configuration for kubectl-ai
#
# This configuration enables kubectl-ai as an MCP server, which provides:
#   - kubectl_ai.kubectl - Execute kubectl commands against the cluster
#   - kubectl_ai.bash - Execute bash commands
#
mcp_servers:
  kubectl-ai:
    type: stdio
    command: kubectl-ai
    args: ["--mcp-server"]
    enabled: true
    timeout: 60
"""
        MCP_CONFIG.write_text(mcp_content)
    else:
        # Copy template
        print_info("Installing MCP configuration from template...")
        shutil.copy(MCP_TEMPLATE, MCP_CONFIG)

    print_success(f"MCP config installed: {MCP_CONFIG}")

    # Show configuration
    print_info("Configuration:")
    print(MCP_CONFIG.read_text())

    # Print next steps
    print_header("Setup Complete!")
    print("Next steps:")
    print("  1. Restart SABRE server:")
    print("     uv run sabre --stop && uv run sabre-server")
    print("  2. Verify tools are available:")
    print("     uv run sabre list")
    print("  3. Run benchmark:")
    print("     ./run_benchmark.sh")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Setup k8s-ai-bench benchmark with SABRE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              # Check prerequisites and setup MCP
  %(prog)s --check      # Only check prerequisites
  %(prog)s --setup-mcp  # Only setup MCP config
  %(prog)s --force      # Force overwrite MCP config
"""
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check prerequisites, don't setup MCP"
    )
    parser.add_argument(
        "--setup-mcp",
        action="store_true",
        help="Only setup MCP config, skip prerequisite checks"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing MCP config"
    )

    args = parser.parse_args()

    exit_code = 0

    # Determine what to do
    if args.check:
        # Only check prerequisites
        _, failed = check_prerequisites()
        exit_code = 1 if failed > 0 else 0
    elif args.setup_mcp:
        # Only setup MCP
        success = setup_mcp_config(force=args.force)
        exit_code = 0 if success else 1
    else:
        # Do both: check prerequisites and setup MCP
        _, failed = check_prerequisites()

        if failed > 0:
            print_warn("\nSome prerequisites are missing. Setup may not work correctly.")
            response = input("Continue with MCP setup anyway? [y/N] ").strip().lower()
            if response != "y":
                sys.exit(1)

        print()  # Add spacing
        success = setup_mcp_config(force=args.force)
        exit_code = 0 if success else 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
