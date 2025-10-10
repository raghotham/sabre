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

from sabre.common.paths import get_logs_dir, get_pid_file, ensure_dirs, migrate_from_old_structure


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
                response = requests.get(f"http://localhost:{port}/health", timeout=2)
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


async def run_client():
    """Run the client"""
    from sabre.client.client import main

    return await main()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="SABRE CLI")
    parser.add_argument("--stop", action="store_true", help="Stop the running server")
    args = parser.parse_args()

    # Handle --stop flag
    if args.stop:
        return stop_server()

    # Normal operation: start server and run client
    server_process = None

    try:
        # Start server
        server_process = start_server()

        # Run client
        exit_code = asyncio.run(run_client())

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
