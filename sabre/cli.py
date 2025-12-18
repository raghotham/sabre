"""
SABRE CLI Entry Point.

Starts the server in background and launches the client.
"""

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from sabre.common.paths import get_logs_dir, get_pid_file, ensure_dirs, migrate_from_old_structure, cleanup_all


def start_server():
    """Start the SABRE server in background"""
    # Migrate from old structure if needed
    migrate_from_old_structure()

    # Ensure all directories exist
    ensure_dirs()

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Try to read from ~/.openai/key
        key_file = Path.home() / ".openai" / "key"
        if key_file.exists():
            api_key = key_file.read_text().strip()
        else:
            print("Error: OPENAI_API_KEY not set and ~/.openai/key not found", file=sys.stderr)
            sys.exit(1)

    # Get optional configuration from environment
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL")
    port = os.getenv("PORT", "8011")

    # Get logs directory
    log_dir = get_logs_dir()
    server_log = log_dir / "server.log"

    # Start server process
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key

    # Pass optional config to server
    if base_url:
        env["OPENAI_BASE_URL"] = base_url
    if model:
        env["OPENAI_MODEL"] = model
    env["PORT"] = port

    # Open log file (keep it open for subprocess)
    print("Starting server...")
    log_f = open(server_log, "w")

    try:
        server_process = subprocess.Popen(
            [sys.executable, "-m", "sabre.server"],
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Proper process management
        )

        # Write PID file
        pid_file = get_pid_file()
        pid_file.write_text(str(server_process.pid))

        # Wait for server to start (give it more time for initialization)
        time.sleep(3)

        # Check if process crashed
        if server_process.poll() is not None:
            print("Error: Server failed to start", file=sys.stderr)
            print(f"Check {server_log} for details", file=sys.stderr)
            log_f.close()
            pid_file.unlink(missing_ok=True)
            sys.exit(1)

        # Check health endpoint
        import requests

        max_retries = 40  # Increased to 20 seconds (40 * 0.5s)
        for i in range(max_retries):
            try:
                response = requests.get(f"http://localhost:{port}/v1/health", timeout=2)
                if response.status_code == 200:
                    print(f"Server started (PID: {server_process.pid}, port: {port})")
                    print(f"Server logs: {server_log}")
                    return server_process
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                pass

            # Check if process died
            if server_process.poll() is not None:
                print("Error: Server process died during startup", file=sys.stderr)
                print(f"Check {server_log} for details", file=sys.stderr)
                log_f.close()
                pid_file.unlink(missing_ok=True)
                sys.exit(1)

            time.sleep(0.5)

        # Timeout
        print("Error: Server did not become healthy within timeout", file=sys.stderr)
        print(f"Check {server_log} for details", file=sys.stderr)
        server_process.terminate()
        log_f.close()
        pid_file.unlink(missing_ok=True)
        sys.exit(1)

    except Exception as e:
        print(f"Error: Failed to start server: {e}", file=sys.stderr)
        log_f.close()
        sys.exit(1)


def stop_server():
    """Stop the SABRE server using PID file or process search"""
    pid_file = get_pid_file()
    pid = None

    # Try PID file first
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
        except ValueError:
            print("Invalid PID in file", file=sys.stderr)
            pid_file.unlink(missing_ok=True)

    # If no PID from file, search for running server process
    if pid is None:
        try:
            result = subprocess.run(["pgrep", "-f", "sabre.server"], capture_output=True, text=True)
            if result.returncode == 0:
                pids = [int(p) for p in result.stdout.strip().split("\n") if p]
                if pids:
                    pid = pids[0]  # Take first match
                    print(f"Found server process via search (PID: {pid})")
        except Exception as e:
            print(f"Could not search for server process: {e}", file=sys.stderr)

    if pid is None:
        print("No server PID file found and no running server process detected.", file=sys.stderr)
        return 1

    # Try to kill the process
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Stopping server (PID: {pid})...")

        # Wait up to 5 seconds for graceful shutdown
        for _ in range(50):
            try:
                os.kill(pid, 0)  # Check if process exists
                time.sleep(0.1)
            except ProcessLookupError:
                # Process is gone
                break
        else:
            # Force kill if still running
            print("Server didn't stop gracefully, force killing...")
            os.kill(pid, signal.SIGKILL)

        print("Server stopped")
        pid_file.unlink(missing_ok=True)
        return 0

    except ProcessLookupError:
        print(f"Server process {pid} not found. Cleaning up PID file.", file=sys.stderr)
        pid_file.unlink(missing_ok=True)
        return 1
    except PermissionError:
        print(f"Permission denied when trying to kill process {pid}", file=sys.stderr)
        return 1


def cleanup(force: bool = False):
    """Clean up all SABRE XDG directories"""

    # Get cleanup info (without actually removing)
    result = cleanup_all(force=False)

    if not result["directories"]:
        print("No SABRE directories found to clean up.")
        return 0

    # Show what will be removed
    print("The following directories will be removed:")
    print()

    def format_size(size_bytes):
        """Format size in human-readable form"""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    for directory in result["directories"]:
        size = result["sizes"][directory]
        print(f"  {directory} ({format_size(size)})")

    print()
    print(f"Total size: {format_size(result['total_size'])}")
    print()

    # Ask for confirmation unless force is True
    if not force:
        try:
            response = input("Are you sure you want to delete these directories? [y/N]: ")
            if response.lower() not in ("y", "yes"):
                print("Cleanup cancelled.")
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\nCleanup cancelled.")
            return 0

    # Perform cleanup
    try:
        cleanup_all(force=True)
        print("Cleanup complete!")
        return 0
    except Exception as e:
        print(f"Error during cleanup: {e}", file=sys.stderr)
        return 1


async def run_client(message: str | None = None, export_atif: bool = False):
    """Run the client, optionally with a single message (script mode)"""
    from sabre.client.client import main

    return await main(message, export_atif=export_atif)


def mcp_list():
    """List configured MCP servers"""
    try:
        from sabre.server.mcp import MCPConfigLoader

        configs = MCPConfigLoader.load()

        if not configs:
            print("No MCP servers configured.")
            print(f"Create config at: {MCPConfigLoader.get_config_path()}")
            print("Run: sabre mcp init  to create example config")
            return 0

        print("Configured MCP Servers:")
        print()

        for config in configs:
            status = "enabled" if config.enabled else "disabled"
            print(f"  {config.name} ({config.type.value}) - {status}")
            if config.type.value == "stdio":
                print(f"    Command: {config.command} {' '.join(config.args)}")
            elif config.type.value == "sse":
                print(f"    URL: {config.url}")
            print()

        return 0

    except Exception as e:
        print(f"Error listing MCP servers: {e}", file=sys.stderr)
        return 1


def mcp_init(force=False):
    """Initialize MCP configuration"""
    try:
        from sabre.server.mcp import MCPConfigLoader

        config_path = MCPConfigLoader.get_config_path()

        if config_path.exists() and not force:
            print(f"MCP config already exists at: {config_path}")
            print("Use --force to overwrite")
            return 1

        MCPConfigLoader.create_example_config()
        print(f"Created example MCP config at: {config_path}")
        print()
        print("Edit this file to add MCP servers, then restart SABRE.")
        return 0

    except Exception as e:
        print(f"Error creating MCP config: {e}", file=sys.stderr)
        return 1


def mcp_config():
    """Show MCP configuration file path"""
    from sabre.server.mcp import MCPConfigLoader

    config_path = MCPConfigLoader.get_config_path()
    print(f"MCP configuration file: {config_path}")

    if config_path.exists():
        print("Status: File exists")
    else:
        print("Status: File not found")
        print("Run: sabre mcp init  to create example config")

    return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="SABRE - Persona-driven AI agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sabre                    # Start interactive client
  sabre --message "task"   # Run single task (script mode)
  sabre mcp list           # List MCP servers
  sabre mcp init           # Initialize MCP config
  sabre --stop             # Stop server
  sabre --clean            # Clean up data
        """,
    )

    # Script mode
    parser.add_argument("-m", "--message", type=str, help="Run single message in script mode (non-interactive)")
    parser.add_argument("--export-atif", action="store_true", help="Export ATIF trace after execution")

    # Server management
    parser.add_argument("--stop", action="store_true", help="Stop the running server")
    parser.add_argument("--clean", action="store_true", help="Clean up all SABRE XDG directories")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt (for --clean)")

    # MCP subcommands
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommands")

    # MCP command group
    mcp_parser = subparsers.add_parser("mcp", help="MCP server management")
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command", help="MCP commands")
    mcp_subparsers.add_parser("list", help="List configured MCP servers")
    mcp_subparsers.add_parser("init", help="Create example MCP configuration")
    mcp_subparsers.add_parser("config", help="Show MCP config file path")

    args = parser.parse_args()

    # Handle --clean flag
    if args.clean:
        return cleanup(force=args.force)

    # Handle --stop flag
    if args.stop:
        return stop_server()

    # Handle MCP subcommands
    if args.subcommand == "mcp":
        if args.mcp_command == "list":
            return mcp_list()
        elif args.mcp_command == "init":
            return mcp_init(force=args.force)
        elif args.mcp_command == "config":
            return mcp_config()
        else:
            # No MCP subcommand provided
            mcp_parser.print_help()
            return 1

    # Normal operation: start server and run client
    server_process = None

    try:
        # Start server
        server_process = start_server()

        # Run client (with optional message for script mode)
        exit_code = asyncio.run(run_client(args.message, export_atif=args.export_atif))

        return exit_code

    except KeyboardInterrupt:
        print("\nShutting down...")
        return 0

    finally:
        # Cleanup: kill server and remove PID file
        if server_process:
            print("Stopping server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()

            # Remove PID file
            get_pid_file().unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
