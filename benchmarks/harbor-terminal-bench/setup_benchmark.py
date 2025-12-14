#!/usr/bin/env python3
"""
Setup script for Terminal-Bench 2.0 benchmark with SABRE via Harbor.

This script:
- Checks all prerequisites (harbor CLI, Docker, API keys)
- Verifies the sabre_harbor package is properly configured
- Tests the agent can be imported by Harbor

Usage:
    python setup_benchmark.py              # Check prerequisites and setup
    python setup_benchmark.py --check      # Only check prerequisites
    python setup_benchmark.py --test-agent # Test agent import
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent.resolve()
SABRE_HARBOR_DIR = SCRIPT_DIR / "sabre_harbor"

# Colors for terminal output
class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    NC = "\033[0m"  # No Color


def print_success(msg: str):
    print(f"{Colors.GREEN}[OK]{Colors.NC} {msg}")


def print_fail(msg: str):
    print(f"{Colors.RED}[FAIL]{Colors.NC} {msg}")


def print_warn(msg: str):
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {msg}")


def print_info(msg: str):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}")


def print_header(msg: str):
    print(f"\n{Colors.BLUE}{'=' * 50}{Colors.NC}")
    print(f"{Colors.BLUE}{msg}{Colors.NC}")
    print(f"{Colors.BLUE}{'=' * 50}{Colors.NC}\n")


def command_exists(cmd: str) -> bool:
    """Check if a command exists in PATH."""
    return shutil.which(cmd) is not None


def run_command(cmd: str, capture: bool = True, timeout: int = 30) -> tuple[bool, str]:
    """Run a shell command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=capture,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def check_prerequisites() -> tuple[int, int]:
    """
    Check all prerequisites for running Terminal-Bench 2.0 with SABRE.

    Returns:
        Tuple of (passed_count, failed_count)
    """
    print_header("Terminal-Bench 2.0 Prerequisites Check")

    passed = 0
    failed = 0

    def check(name: str, condition: bool, required: bool = True, help_text: str = ""):
        nonlocal passed, failed
        if condition:
            print_success(name)
            passed += 1
        elif required:
            print_fail(name)
            if help_text:
                print_info(f"  {help_text}")
            failed += 1
        else:
            print_warn(f"{name} (optional)")
            if help_text:
                print_info(f"  {help_text}")

    # Required: harbor CLI
    print("Required:")
    check(
        "harbor CLI installed",
        command_exists("harbor"),
        help_text="Install: pip install harbor-bench"
    )

    # Check harbor version
    if command_exists("harbor"):
        success, output = run_command("harbor --version")
        if success:
            print_info(f"  Version: {output.strip()}")

    # Docker
    docker_ok = command_exists("docker")
    check(
        "docker installed",
        docker_ok,
        help_text="Install Docker from https://docs.docker.com/get-docker/"
    )

    if docker_ok:
        docker_running, _ = run_command("docker info")
        check(
            "Docker daemon running",
            docker_running,
            help_text="Start Docker daemon"
        )

    # Python environment
    check(
        "Python 3.10+",
        sys.version_info >= (3, 10),
        help_text=f"Current: {sys.version}"
    )

    # API Keys
    print("\nAPI Keys:")
    check(
        "OPENAI_API_KEY set",
        bool(os.environ.get("OPENAI_API_KEY")),
        help_text="export OPENAI_API_KEY='sk-...'"
    )

    # Optional: Custom base URL
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        print_info(f"  OPENAI_BASE_URL: {base_url}")

    # SABRE Agent Package
    print("\nSABRE Agent:")
    check(
        "sabre_harbor package exists",
        SABRE_HARBOR_DIR.exists(),
        help_text=f"Expected at: {SABRE_HARBOR_DIR}"
    )

    check(
        "agent.py exists",
        (SABRE_HARBOR_DIR / "agent.py").exists(),
        help_text="Missing agent implementation"
    )

    check(
        "install-sabre.sh.j2 exists",
        (SABRE_HARBOR_DIR / "install-sabre.sh.j2").exists(),
        help_text="Missing installation script template"
    )

    # Try importing the agent
    agent_import_ok = False
    try:
        sys.path.insert(0, str(SCRIPT_DIR))
        from sabre_harbor import SabreAgent
        agent_import_ok = True
        print_success("SabreAgent imports successfully")
        passed += 1

        # Verify agent class
        if hasattr(SabreAgent, 'name') and callable(getattr(SabreAgent, 'name')):
            print_info(f"  Agent name: {SabreAgent.name()}")
    except ImportError as e:
        print_fail(f"SabreAgent import failed: {e}")
        failed += 1

    # Summary
    print_header("Summary")
    print(f"Passed: {Colors.GREEN}{passed}{Colors.NC}")
    print(f"Failed: {Colors.RED}{failed}{Colors.NC}")

    if failed > 0:
        print(f"\n{Colors.RED}Fix the failed checks before running the benchmark.{Colors.NC}")
    else:
        print(f"\n{Colors.GREEN}All required checks passed!{Colors.NC}")

    return passed, failed


def test_agent_import() -> bool:
    """
    Test that the agent can be imported and has required methods.

    Returns:
        True if agent is valid
    """
    print_header("Testing SABRE Agent")

    try:
        sys.path.insert(0, str(SCRIPT_DIR))
        from sabre_harbor import SabreAgent

        # Check required methods
        required_methods = ['name', 'version', 'create_run_agent_commands', 'populate_context_post_run']
        missing = []

        for method in required_methods:
            if hasattr(SabreAgent, method):
                print_success(f"Has method: {method}")
            else:
                print_fail(f"Missing method: {method}")
                missing.append(method)

        # Check required properties
        required_properties = ['_install_agent_template_path']
        for prop in required_properties:
            if hasattr(SabreAgent, prop):
                print_success(f"Has property: {prop}")
            else:
                print_fail(f"Missing property: {prop}")
                missing.append(prop)

        # Test instantiation (with mock logs_dir)
        print("\nTesting instantiation...")
        try:
            agent = SabreAgent(logs_dir=Path("/tmp/test-logs"))
            print_success("Agent instantiated successfully")
            print_info(f"  Name: {agent.name()}")
            print_info(f"  Version: {agent.version()}")

            # Check template path exists
            template_path = agent._install_agent_template_path
            if template_path.exists():
                print_success(f"Installation template exists: {template_path}")
            else:
                print_fail(f"Installation template missing: {template_path}")
                missing.append("install template")

        except Exception as e:
            print_fail(f"Agent instantiation failed: {e}")
            return False

        if missing:
            print(f"\n{Colors.RED}Agent is missing required components: {missing}{Colors.NC}")
            return False

        print(f"\n{Colors.GREEN}Agent is valid and ready for use!{Colors.NC}")
        return True

    except ImportError as e:
        print_fail(f"Failed to import SabreAgent: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Setup Terminal-Bench 2.0 benchmark with SABRE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              # Check prerequisites
  %(prog)s --check      # Only check prerequisites
  %(prog)s --test-agent # Test agent import and validation

Running the benchmark:
  ./run_benchmark.sh --model gpt-4o --dataset hello-world@head  # Debug task

  # Run full Terminal-Bench 2.0
  ./run_benchmark.sh --model gpt-4o --n-concurrent 4
"""
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check prerequisites"
    )
    parser.add_argument(
        "--test-agent",
        action="store_true",
        help="Test agent import and validation"
    )

    args = parser.parse_args()

    exit_code = 0

    if args.test_agent:
        success = test_agent_import()
        exit_code = 0 if success else 1
    else:
        # Default: check prerequisites
        _, failed = check_prerequisites()
        exit_code = 1 if failed > 0 else 0

        if exit_code == 0:
            # Also test agent if prerequisites pass
            print()
            success = test_agent_import()
            exit_code = 0 if success else 1

    # Print next steps
    if exit_code == 0:
        print_header("Next Steps")
        print("Run debug task (hello-world dataset):")
        print("  ./run_benchmark.sh --model gpt-4o --dataset hello-world@head")
        print()
        print("Run full Terminal-Bench 2.0:")
        print("  ./run_benchmark.sh --model gpt-4o --n-concurrent 4")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
