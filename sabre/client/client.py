"""
Minimal SABRE Client with HTTP SSE communication.

Connects to server via HTTP SSE and provides chat interface with tree visualization.
"""

import asyncio
import json
import logging
import os
import sys
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PromptStyle
import httpx
import jsonpickle

from sabre.client.slash_commands import SlashCommandHandler
from sabre.client.tui import TUI

# Initialize logger
logger = logging.getLogger(__name__)


class Client:
    """HTTP SSE chat client with rich rendering"""

    def __init__(self, server_url: str = "http://localhost:8011", history_file: str | None = None):
        self.server_url = server_url

        # Setup history using XDG-compliant paths
        if history_file is None:
            from sabre.common.paths import get_logs_dir, ensure_dirs

            ensure_dirs()
            log_dir = get_logs_dir()
            history_file = str(log_dir / "history")

        # Initialize TUI (defaults to dark theme)
        self.tui = TUI()

        # Track if we're processing a message (for cancellation)
        self.processing = False
        self.cancel_requested = False  # Flag for cancellation
        self.current_request_id: str | None = None
        self.conversation_id: str | None = None

        # Track session info (displayed at start of each conversation)
        self.session_id: str | None = None
        self.workspace_dir: str | None = None
        self.session_info_displayed = False

        # Create styled prompt session
        prompt_style = PromptStyle.from_dict(
            {
                "": f"{self.tui.colors['user_input']}",  # User input
            }
        )

        self.session = PromptSession(history=FileHistory(history_file), style=prompt_style)
        self.tree_depth = 0

        # Initialize slash command handler
        self.slash_handler = SlashCommandHandler(self)

    async def cancel_processing(self):
        """Cancel the current processing task via cancel endpoint"""
        logger.info(
            f"cancel_processing called: processing={self.processing}, request_id={self.current_request_id}, cancel_requested={self.cancel_requested}"
        )
        if self.processing and self.current_request_id and not self.cancel_requested:
            self.cancel_requested = True
            try:
                logger.info(f"Sending cancel request for {self.current_request_id}")
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.server_url}/v1/cancel/{self.current_request_id}",
                        timeout=5.0,
                    )
                    if response.status_code == 200:
                        logger.info("Cancel request sent successfully")
                        self.tui.print(f'\n<style fg="{self.tui.colors["warning"]}">Cancelling...</style>')
                    else:
                        logger.warning(f"Cancel request returned {response.status_code}")
            except Exception as e:
                logger.error(f"Failed to send cancel: {e}")
                self.tui.print(f'\n<style fg="{self.tui.colors["error"]}">Failed to send cancel: {e}</style>')
        else:
            logger.info("Cancel not sent - conditions not met")

    async def _handle_escape(self):
        """Handle escape key press - show message and cancel"""
        self.tui.print(
            f'\n<style fg="{self.tui.colors["warning"]}">[Escape detected - requesting cancellation...]</style>'
        )
        await self.cancel_processing()

    async def _monitor_escape_key(self):
        """Background task to monitor for Escape key"""
        import sys
        import termios
        import tty

        # Only run if stdin is a TTY
        if not sys.stdin.isatty():
            return

        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            # Set terminal to cbreak mode (no line buffering, but keep signal handling)
            tty.setcbreak(sys.stdin.fileno())

            while self.processing:
                # Non-blocking read with select
                import select

                if select.select([sys.stdin], [], [], 0.1)[0]:
                    # Peek at input without consuming it
                    char = sys.stdin.read(1)

                    # Only consume ESC key
                    if char == "\x1b":
                        if self.processing and not self.cancel_requested:
                            # Restore terminal before showing message
                            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                            await self._handle_escape()
                            return
                    else:
                        # Put the character back by simulating typing it
                        # (This is a hack - we can't truly "put it back")
                        # During processing, we just discard non-ESC input since
                        # the user shouldn't be typing at the prompt anyway
                        pass

                await asyncio.sleep(0.05)
        finally:
            # Always restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def _parse_sse_line(self, line: str) -> dict | None:
        """Parse SSE line into event dict."""
        if not line or not line.startswith("data: "):
            return None

        if line == "data: [DONE]":
            logger.info("Stream complete ([DONE] received)")
            return None

        json_str = line[6:]  # Remove "data: " prefix

        # Check for request ID
        if json_str.startswith("__REQUEST_ID__:"):
            self.current_request_id = json_str.split(":", 1)[1]
            logger.info(f"Got request ID: {self.current_request_id}")
            return None

        # Decode with jsonpickle
        try:
            event = jsonpickle.decode(json_str)
            return event
        except Exception as e:
            # Try regular JSON as fallback
            try:
                event = json.loads(json_str)
                return event
            except Exception:
                logger.error(f"Failed to decode event: {e}")
                return None

    async def send_message(self, client: httpx.AsyncClient, user_input: str):
        """Send message and stream response via SSE."""
        from sabre.client.file_loader import FileLoadError, FileLoader

        # Parse and load @filepath references
        loader = FileLoader()
        attachments = []
        message_text = user_input  # Default to original input

        try:
            clean_text, filepaths = loader.parse_message(user_input)

            # Load files if any were referenced
            if filepaths:
                for filepath in filepaths:
                    self.tui.print(f'<style fg="ansiblue">Loading: {filepath}</style>')
                    content = loader.load_file(filepath)
                    attachments.append(content)
                    self.tui.print(f'<style fg="ansigreen">✓ Loaded: {filepath}</style>')

                # Use clean text (with @ references removed) if files were loaded
                message_text = clean_text
        except FileLoadError as e:
            self.tui.print(f'<style fg="ansired">Error loading file:</style> {e}')
            return

        self.processing = True
        self.cancel_requested = False
        self.current_request_id = None

        # Start Esc key monitoring task (only if running in a TTY)
        esc_monitor_task = asyncio.create_task(self._monitor_escape_key())

        # Initialize state before try block to avoid UnboundLocalError in except block
        thinking_shown = False

        try:
            # Build request payload
            import jsonpickle

            payload = {
                "type": "message",
                "content": message_text,
                "conversation_id": self.conversation_id,
            }

            # Add session_id if we have one
            if self.session_id:
                payload["session_id"] = self.session_id

            # Add attachments if any were loaded
            if attachments:
                payload["attachments"] = jsonpickle.encode(attachments, unpicklable=True)

            # POST request with streaming response
            async with client.stream(
                "POST",
                f"{self.server_url}/v1/message",
                json=payload,
                headers={"Accept": "text/event-stream"},
                timeout=httpx.Timeout(None),  # No timeout for streaming
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    self.tui.print(
                        f'<style fg="ansired">Server error ({response.status_code}):</style> {error_text.decode()}'
                    )
                    return

                # Reset state for new response
                self.tree_depth = 0
                response_started = False

                # Stream SSE lines
                async for line in response.aiter_lines():
                    if not self.processing or self.cancel_requested:
                        # Cancellation requested
                        break

                    event = self._parse_sse_line(line)
                    if event is None:
                        continue

                    # Handle Event objects (deserialized by jsonpickle)
                    from sabre.common.models.events import (
                        ResponseStartEvent,
                        ResponseTextEvent,
                        ResponseRetryEvent,
                        HelpersExecutionStartEvent,
                        HelpersExecutionEndEvent,
                        CompleteEvent,
                        CancelledEvent,
                        ErrorEvent,
                    )

                    # Extract conversation_id
                    self.conversation_id = event.conversation_id

                    # Log event with timestamp and details
                    import datetime

                    timestamp_str = datetime.datetime.now().isoformat()
                    event_type = event.type.value if hasattr(event, "type") else type(event).__name__
                    node_id = event.node_id[:8] if hasattr(event, "node_id") and event.node_id else "N/A"
                    depth = event.depth if hasattr(event, "depth") else "N/A"
                    logger.info(f"← [{timestamp_str}] RECV event: {event_type} (node={node_id}, depth={depth})")

                    # Clear thinking animation when any event (other than ResponseStartEvent) arrives
                    if thinking_shown and not isinstance(event, ResponseStartEvent):
                        self.tui.clear_thinking()
                        thinking_shown = False

                    if isinstance(event, ResponseStartEvent):
                        if not response_started:
                            self.tui.print()  # Blank line before first assistant response

                            # Display session info at the start if we haven't yet
                            if not self.session_info_displayed and self.session_id:
                                dim_color = "#666666"  # Dim gray color
                                self.tui.print(f'<style fg="{dim_color}">Session ID: {self.session_id}</style>')
                                if self.workspace_dir:
                                    self.tui.print(f'<style fg="{dim_color}">Workspace: {self.workspace_dir}</style>')
                                self.tui.print()  # Blank line after session info
                                self.session_info_displayed = True

                            response_started = True

                        # Show "Thinking..." animation instead of tree node
                        if not thinking_shown:
                            self.tui.print_thinking()
                            thinking_shown = True

                    elif isinstance(event, ResponseRetryEvent):
                        attempt = event.data["attempt"]
                        max_retries = event.data["max_retries"]
                        wait_seconds = event.data["wait_seconds"]
                        self.tui.print_tree_node(
                            "RATE_LIMIT",
                            f"retrying in {wait_seconds:.1f}s (attempt {attempt}/{max_retries})",
                            depth=event.depth,
                            path=event.path,
                        )
                        self.tui.render_response_retry(event.data, event.depth)

                    elif isinstance(event, HelpersExecutionStartEvent):
                        block_number = event.data["block_number"]
                        helper_name = event.data.get("helper_name", "unknown")
                        label = (
                            f"{helper_name} #{block_number}" if helper_name != "unknown" else f"helper #{block_number}"
                        )
                        logger.info(f"  └─ EXECUTING {label}")
                        self.tui.print_tree_node("EXECUTING", f"{label}...", depth=event.depth, path=event.path)

                    elif isinstance(event, HelpersExecutionEndEvent):
                        duration_ms = event.data["duration_ms"]
                        success = event.data["success"]
                        block_number = event.data["block_number"]
                        helper_name = event.data.get("helper_name", "unknown")
                        result = event.data["result"]

                        label = (
                            f"{helper_name} #{block_number}" if helper_name != "unknown" else f"helper #{block_number}"
                        )
                        status = "✓" if success else "✗"
                        logger.info(f"  └─ RESULT {label} {status} {duration_ms:.0f}ms")
                        color = self.tui.colors["complete"] if success else self.tui.colors["error"]
                        self.tui.print_tree_node(
                            "RESULT",
                            f'{label} <style fg="{color}">{status} {duration_ms:.0f}ms</style>',
                            depth=event.depth,
                            path=event.path,
                        )

                        # Display result content (text output + image URLs)
                        self.tui.render_helpers_end({"result": result}, event.depth)

                    elif isinstance(event, ResponseTextEvent):
                        text_length = event.data["text_length"]
                        has_helpers = event.data["has_helpers"]
                        helper_count = event.data["helper_count"]
                        full_text = event.data.get("text", "")

                        # Build token info string
                        input_tokens = event.data.get("input_tokens", 0)
                        output_tokens = event.data.get("output_tokens", 0)
                        reasoning_tokens = event.data.get("reasoning_tokens", 0)
                        token_info = ""
                        if input_tokens > 0:
                            token_info = (
                                f" • Tokens: {input_tokens} in, {output_tokens} out, {reasoning_tokens} reasoning"
                            )

                        # Log response details
                        if has_helpers:
                            logger.info(
                                f"  └─ RESPONSE_TEXT: {text_length} chars, {helper_count} helper(s){token_info}"
                            )
                        else:
                            logger.info(f"  └─ RESPONSE_TEXT: {text_length} chars, no helpers{token_info}")

                        # Show response summary with tokens on same line
                        if has_helpers:
                            self.tui.print_tree_node(
                                "RESPONSE_TEXT",
                                f"{text_length} chars, {helper_count} helper(s){token_info}",
                                depth=event.depth,
                                path=event.path,
                            )
                        else:
                            self.tui.print_tree_node(
                                "RESPONSE_TEXT",
                                f"{text_length} chars, no helpers{token_info}",
                                depth=event.depth,
                                path=event.path,
                            )

                        # Parse and display response text, <helpers> blocks, and <helpers_result> blocks
                        import re

                        # Extract <helpers> blocks and display them
                        helpers_pattern = r"<helpers>(.*?)</helpers>"
                        helpers_blocks = re.findall(helpers_pattern, full_text, re.DOTALL)

                        # Extract <helpers_result> blocks
                        results_pattern = r"<helpers_result>(.*?)</helpers_result>"
                        results_blocks = re.findall(results_pattern, full_text, re.DOTALL)

                        # Remove <helpers> and <helpers_result> blocks from text for display
                        display_text = re.sub(helpers_pattern, "", full_text, flags=re.DOTALL)
                        display_text = re.sub(results_pattern, "", display_text, flags=re.DOTALL)

                        # Display the response text (without helpers/results blocks)
                        if display_text.strip():
                            self.tui.render_response_text({"text": display_text.strip()}, event.depth)

                        # Display extracted <helpers> blocks as code
                        for i, code in enumerate(helpers_blocks):
                            self.tui.print_tree_node(
                                "HELPER_BLOCK", f"#{i + 1}", depth=event.depth + 1, path=event.path
                            )
                            if code.strip():
                                self.tui.print_code_block(code.strip(), event.depth + 1)

                        # Display extracted <helpers_result> blocks
                        for i, result in enumerate(results_blocks):
                            self.tui.print_tree_node(
                                f"RESULT #{i + 1}",
                                "",
                                depth=event.depth + 1,
                                path=event.path,
                            )
                            # Render result (contains text and markdown image URLs)
                            from sabre.common.models.messages import TextContent

                            result_content = [TextContent(result.strip())]
                            self.tui.render_helpers_end({"result": result_content}, event.depth + 1)

                    elif isinstance(event, CompleteEvent):
                        final_message = event.data["final_message"]

                        # Store session info (only from top-level completes)
                        if event.depth == 1:
                            session_id = event.data.get("session_id", "")
                            workspace_dir = event.data.get("workspace_dir", "")
                            if session_id:
                                self.session_id = session_id
                            if workspace_dir:
                                self.workspace_dir = workspace_dir

                        if final_message.strip():
                            self.tui.print()  # Blank line
                            self.tui.print_tree_node("COMPLETE", "", depth=event.depth, path=event.path)
                            self.tui.render_complete(event.data, event.depth)
                        # Only break on top-level complete (depth=1)
                        # Nested complete events (depth > 1) are from sabre_call/llm_call
                        if event.depth == 1:
                            self.tui.print()  # Blank line after response
                            self.processing = False
                            break
                        # For nested completes, just continue processing events
                        elif final_message.strip():
                            self.tui.print()  # Blank line after nested complete

                    elif isinstance(event, CancelledEvent):
                        cancel_msg = event.data["message"]
                        warning_color = self.tui.colors["warning"]
                        self.tui.print_tree_node(
                            "CANCELLED",
                            f'<style fg="{warning_color}">{cancel_msg}</style>',
                            depth=event.depth,
                            path=event.path,
                        )
                        self.tui.print()  # Blank line
                        self.processing = False
                        break

                    elif isinstance(event, ErrorEvent):
                        error_msg = event.data["error_message"]
                        error_color = self.tui.colors["error"]
                        self.tui.print_tree_node(
                            "ERROR",
                            f'<style fg="{error_color}">{error_msg}</style>',
                            depth=event.depth,
                            path=event.path,
                        )
                        self.processing = False
                        break

                # Stream ended - clear thinking animation if still shown
                if thinking_shown:
                    self.tui.clear_thinking()
                    thinking_shown = False

        except httpx.ConnectError:
            # Clear thinking animation on connection error
            if thinking_shown:
                self.tui.clear_thinking()
            self.tui.print(f'<style fg="ansired">Cannot connect to server at {self.server_url}</style>')
            self.tui.print("Make sure the server is running with: uv run python -m sabre.server")
        except Exception as e:
            # Clear thinking animation on error
            if thinking_shown:
                self.tui.clear_thinking()
            if self.processing:  # Only show error if not cancelled
                self.tui.print(f'<style fg="ansired">Error:</style> {e}')
                logger.error(f"Error in send_message: {e}", exc_info=True)
        finally:
            self.processing = False
            self.current_request_id = None
            # Cancel the Esc monitor task
            esc_monitor_task.cancel()
            try:
                await esc_monitor_task
            except asyncio.CancelledError:
                pass

    async def run_once(self, message: str, export_atif: bool = False):
        """Run client in non-interactive mode with a single message"""
        try:
            async with httpx.AsyncClient() as client:
                # Show user message (escape HTML to prevent parsing errors)
                user_color = self.tui.colors["user_input"]
                escaped_message = self.tui.html_escape(message)
                self.tui.print(f'<style fg="{user_color}">&gt; {escaped_message}</style>')
                self.tui.print()

                # Send message
                await self.send_message(client, message)

                # Export ATIF if requested
                if export_atif and self.conversation_id:
                    await self._export_atif(client)

        except Exception as e:
            escaped_error = self.tui.html_escape(str(e))
            self.tui.print(f'<style fg="ansired">Error:</style> {escaped_error}')
            logger.error(f"Error in run_once: {e}", exc_info=True)
            return 1

        return 0

    async def _export_atif(self, client: httpx.AsyncClient):
        """Request ATIF export from server"""
        try:
            response = await client.get(f"{self.server_url}/v1/sessions/{self.conversation_id}/atif", timeout=10.0)
            if response.status_code == 200:
                atif_data = response.json()
                # ATIF is returned as JSON, we can optionally save it
                from sabre.common.paths import get_data_dir

                data_dir = get_data_dir()
                sessions_dir = data_dir / "sessions" / self.conversation_id
                sessions_dir.mkdir(parents=True, exist_ok=True)
                atif_path = sessions_dir / "atif.json"
                atif_path.write_text(json.dumps(atif_data, indent=2))
                self.tui.print(f'\n<style fg="ansigreen">ATIF exported to: {atif_path}</style>')
            else:
                logger.warning(f"ATIF export failed: {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not export ATIF: {e}")

    async def run(self):
        """Main client loop"""
        self.tui.print('\n<b><style fg="ansicyan">SABRE Client</style></b>')
        self.tui.print(f"Server: {self.server_url}\n")

        try:
            # Create HTTP client (no persistent connection needed for SSE)
            async with httpx.AsyncClient() as client:
                # Fetch server health (including model) at startup
                try:
                    health_response = await client.get(f"{self.server_url}/v1/health", timeout=5.0)
                    if health_response.status_code == 200:
                        health_data = health_response.json()
                        model = health_data.get("model", "unknown")
                        self.tui.print(f'Model: <style fg="ansibrightcyan">{model}</style>\n')
                    else:
                        logger.warning(f"Failed to fetch server health: {health_response.status_code}")
                except Exception as e:
                    logger.warning(f"Could not fetch server health: {e}")

                self.tui.print("<style fg=\"ansigreen\">Ready!</style> Type your message (Ctrl+D or 'exit' to quit)")
                self.tui.print('<style fg="ansibrightblack">Press Esc to cancel execution</style>\n')

                while True:
                    # Get user input
                    try:
                        user_input = await asyncio.get_event_loop().run_in_executor(None, self.session.prompt, "> ")
                    except (EOFError, KeyboardInterrupt):
                        self.tui.print('\n<style fg="ansiyellow">Exiting...</style>')
                        break

                    if not user_input.strip():
                        continue

                    # Check for old-style exit commands (still supported)
                    if user_input.strip().lower() in ("exit", "quit"):
                        break

                    # Check if this is a slash command
                    if self.slash_handler.is_slash_command(user_input):
                        result = await self.slash_handler.execute_command(user_input)

                        # Display result
                        if result.success:
                            if result.data and result.data.get("action") == "exit":
                                # Special case: exit command
                                self.tui.print('\n<style fg="ansiyellow">Exiting...</style>')
                                break
                            elif result.data and result.data.get("action") == "clear_conversation":
                                # Special case: clear conversation
                                if result.message:
                                    self.tui.print(f'\n<style fg="ansicyan">{result.message}</style>')
                                # Clear conversation_id locally
                                self.conversation_id = None
                                self.tui.print('<style fg="ansigreen">Conversation cleared. Starting fresh.</style>\n')
                                continue
                            elif result.data and result.data.get("action") == "request_helpers":
                                # Special case: request helpers from server
                                # Send as regular message "helpers()" to execute the helper
                                user_input = "helpers()"
                                # Fall through to normal message processing
                            elif result.message:
                                self.tui.print(f'\n<style fg="ansicyan">{result.message}</style>\n')
                                continue  # Don't send to server
                        else:
                            self.tui.print(f'\n<style fg="ansired">{result.message}</style>\n')
                            continue  # Don't send to server

                        # If we didn't continue above, fall through to send message to server

                    # Send message to server (normal message or request_helpers action)
                    await self.send_message(client, user_input)

        except Exception as e:
            self.tui.print(f'<style fg="ansired">Error:</style> {e}')
            logger.error(f"Error in run: {e}", exc_info=True)
            return 1

        return 0


async def main(message: str | None = None, export_atif: bool = False):
    """Entry point for client"""
    import argparse
    from sabre.common.paths import get_logs_dir, ensure_dirs

    # Parse command line arguments (only if message not provided programmatically)
    if message is None:
        parser = argparse.ArgumentParser(description="SABRE Client")
        parser.add_argument("message", nargs="?", help="Message to send (non-interactive mode)")
        parser.add_argument("--port", default=os.getenv("PORT", "8011"), help="Server port")
        parser.add_argument("--export-atif", action="store_true", help="Export ATIF trace after execution")
        args = parser.parse_args()
        message = args.message
        port = args.port
        export_atif = args.export_atif
    else:
        # Message provided programmatically (from CLI --message flag)
        port = os.getenv("PORT", "8011")

    # Setup logging using XDG-compliant paths
    ensure_dirs()
    log_dir = get_logs_dir()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            # Only write to file, not to screen
            logging.FileHandler(log_dir / "client.log")
        ],
    )

    # Get server URL
    server_url = f"http://localhost:{port}"

    # Create client (use /theme to toggle light/dark mode)
    client = Client(server_url=server_url)

    # If message provided, run in non-interactive mode
    if message:
        return await client.run_once(message, export_atif=export_atif)
    else:
        return await client.run()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
