"""
Minimal SABRE Client with WebSocket communication.

Connects to server via WebSocket and provides chat interface with tree visualization.
"""

import asyncio
import json
import logging
import os
import sys
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PromptStyle
import websockets

from sabre.client.slash_commands import SlashCommandHandler
from sabre.client.tui import TUI

# Initialize logger
logger = logging.getLogger(__name__)


class Client:
    """WebSocket chat client with rich rendering"""

    def __init__(self, server_url: str = "ws://localhost:8011/message", history_file: str | None = None):
        self.server_url = server_url

        # Setup history using XDG-compliant paths
        if history_file is None:
            from sabre.common.paths import get_logs_dir, ensure_dirs

            ensure_dirs()
            log_dir = get_logs_dir()
            history_file = str(log_dir / "history")

        # Initialize TUI (auto-detects theme)
        self.tui = TUI()

        # Track if we're processing a message (for cancellation)
        self.processing = False
        self.websocket = None
        self.cancel_requested = False  # Flag for cancellation

        # Create styled prompt session
        prompt_style = PromptStyle.from_dict(
            {
                "": f"{self.tui.colors['user_input']} bg:ansigray",  # User input with gray background
            }
        )

        self.session = PromptSession(history=FileHistory(history_file), style=prompt_style)
        self.tree_depth = 0

        # Initialize slash command handler
        self.slash_handler = SlashCommandHandler(self)

    async def cancel_processing(self):
        """Cancel the current processing task"""
        logger.info(
            f"cancel_processing called: processing={self.processing}, websocket={self.websocket is not None}, cancel_requested={self.cancel_requested}"
        )
        if self.processing and self.websocket and not self.cancel_requested:
            self.cancel_requested = True
            try:
                logger.info("Sending cancel message to server")
                await self.websocket.send(json.dumps({"type": "cancel"}))
                logger.info("Cancel message sent successfully")
                self.tui.print(f'\n<style fg="{self.tui.colors["warning"]}">Cancelling...</style>')
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

    async def run(self):
        """Main client loop"""
        self.tui.print('\n<b><style fg="ansicyan">SABRE Client</style></b>')
        self.tui.print(f"Connecting to {self.server_url}...\n")

        try:
            # Use longer ping timeout for LLM responses that can take time
            async with websockets.connect(
                self.server_url,
                ping_interval=30,  # Send ping every 30 seconds
                ping_timeout=None,  # No timeout for pong (allow long operations)
            ) as websocket:
                self.websocket = websocket  # Store for cancellation
                self.tui.print(
                    "<style fg=\"ansigreen\">Connected!</style> Type your message (Ctrl+D or 'exit' to quit)"
                )
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
                                # Send clear message to server
                                await websocket.send(json.dumps({"type": "clear_conversation"}))
                                # Wait for confirmation
                                response = await websocket.recv()
                                data = json.loads(response)
                                if data.get("type") == "conversation_cleared":
                                    self.tui.print(
                                        '<style fg="ansigreen">Conversation cleared. Starting fresh.</style>\n'
                                    )
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

                    # Send message to server
                    self.processing = True  # Track that we're processing
                    self.cancel_requested = False  # Reset cancel flag

                    # Start Esc key monitoring task
                    esc_monitor_task = asyncio.create_task(self._monitor_escape_key())

                    await websocket.send(json.dumps({"type": "message", "content": user_input}))

                    # Reset state for new response
                    self.tree_depth = 0
                    response_started = False

                    # Receive and render response
                    async for message in websocket:
                        data = json.loads(message)
                        msg_type = data.get("type")

                        # Log event receipt
                        logger.info(f"← RECV event: type={msg_type}")

                        # Debug: print all received messages
                        # self.tui.print(f'<style fg="ansiyellow">[DEBUG] {msg_type or data.get("type", "unknown")}</style>')

                        # Handle raw events (for tree visualization)
                        if "node_id" in data:  # This is an Event
                            event_type = data.get("type")
                            event_data = data.get("data", {})
                            logger.info(f"  Processing event type: {event_type}")
                            depth = data.get("depth", 0)
                            path = data.get("path", [])
                            timestamp = data.get("timestamp", "N/A")
                            node_id = data.get("node_id", "N/A")
                            path_summary = data.get("path_summary", "")

                            if event_type == "response_start":
                                logger.info("  → Handling response_start event")
                                if not response_started:
                                    self.tui.print()  # Blank line before first assistant response
                                    response_started = True

                                # Always show response rounds
                                round_id = event_data.get("round_id", "?")
                                self.tui.print_tree_node(
                                    "RESPONSE_ROUND", f"iteration {round_id}", depth=depth, path=path
                                )
                                self.tui.render_response_start(event_data, depth)
                                logger.info("  ✓ Completed response_start handling")

                            elif event_type == "response_retry":
                                attempt = event_data.get("attempt", 0)
                                max_retries = event_data.get("max_retries", 0)
                                wait_seconds = event_data.get("wait_seconds", 0)
                                self.tui.print_tree_node(
                                    "RATE_LIMIT",
                                    f"retrying in {wait_seconds:.1f}s (attempt {attempt}/{max_retries})",
                                    depth=depth,
                                    path=path,
                                )
                                self.tui.render_response_retry(event_data, depth)

                            elif event_type == "helpers_extracted":
                                logger.info("  → Handling helpers_extracted event")
                                code = event_data.get("code", "")
                                block_num = event_data.get("block_count", 0)
                                logger.info(f"  → block_num={block_num}, code_length={len(code)}")
                                self.tui.print_tree_node("HELPER_BLOCK", f"#{block_num}", depth=depth, path=path)
                                # Print code
                                if code.strip():
                                    logger.info("  → About to print_code_block")
                                    self.tui.print_code_block(code.strip(), depth)
                                    logger.info("  → Finished print_code_block")
                                logger.info("  ✓ Completed helpers_extracted handling")

                            elif event_type == "helpers_start":
                                logger.info("  → Handling helpers_start event")
                                self.tui.print_tree_node("EXECUTING", "...", depth=depth, path=path)
                                logger.info("  ✓ Completed helpers_start handling")

                            elif event_type == "response_text":
                                logger.info("  → Handling response_text event")
                                text_length = event_data.get("text_length", 0)
                                has_helpers = event_data.get("has_helpers", False)
                                helper_count = event_data.get("helper_count", 0)

                                # Build token info string
                                input_tokens = event_data.get("input_tokens", 0)
                                output_tokens = event_data.get("output_tokens", 0)
                                reasoning_tokens = event_data.get("reasoning_tokens", 0)
                                token_info = ""
                                if input_tokens > 0:
                                    token_info = f" • Tokens: {input_tokens} in, {output_tokens} out, {reasoning_tokens} reasoning"

                                # Show response summary with tokens on same line
                                if has_helpers:
                                    self.tui.print_tree_node(
                                        "RESPONSE_TEXT",
                                        f"{text_length} chars, {helper_count} helper(s){token_info}",
                                        depth=depth,
                                        path=path,
                                    )
                                else:
                                    self.tui.print_tree_node(
                                        "RESPONSE_TEXT",
                                        f"{text_length} chars, no helpers{token_info}",
                                        depth=depth,
                                        path=path,
                                    )

                                self.tui.render_response_text(event_data, depth)
                                logger.info("  ✓ Completed response_text handling")

                            elif event_type == "helpers_end":
                                logger.info("  → Handling helpers_end event")
                                duration = event_data.get("duration_ms", 0)
                                block_number = event_data.get("block_number", 0)
                                code_preview = event_data.get("code_preview", "")

                                # Show helper identifier with code preview
                                preview_text = code_preview.replace("\n", " ")[:50]
                                if len(code_preview) > 50:
                                    preview_text += "..."

                                self.tui.print_tree_node(
                                    f"RESULT #{block_number}",
                                    f"{duration:.0f}ms - {preview_text}",
                                    depth=depth,
                                    path=path,
                                )

                                self.tui.render_helpers_end(event_data, depth)
                                logger.info("  ✓ Completed helpers_end handling")

                            elif event_type == "complete":
                                logger.info("  → Handling complete event")
                                final_message = event_data.get("final_message", "")
                                if final_message.strip():
                                    self.tui.print()  # Blank line
                                    self.tui.print_tree_node("COMPLETE", "", depth=depth, path=path)
                                    self.tui.render_complete(event_data, depth)
                                self.tui.print()  # Blank line after response
                                self.processing = False
                                logger.info("  ✓ Completed complete handling")
                                # Cancel the Esc monitor task
                                if "esc_monitor_task" in locals():
                                    esc_monitor_task.cancel()
                                break

                            elif event_type == "cancelled":
                                logger.info("  → Handling cancelled event")
                                cancel_msg = event_data.get("message", "Execution cancelled")
                                warning_color = self.tui.colors["warning"]
                                self.tui.print_tree_node(
                                    "CANCELLED",
                                    f'<style fg="{warning_color}">{cancel_msg}</style>',
                                    depth=depth,
                                    path=path,
                                )
                                self.tui.print()  # Blank line
                                self.processing = False
                                logger.info("  ✓ Completed cancelled handling")
                                # Cancel the Esc monitor task
                                if "esc_monitor_task" in locals():
                                    esc_monitor_task.cancel()
                                break

                            elif event_type == "error":
                                logger.info("  → Handling error event")
                                error_msg = event_data.get("error_message", "Unknown error")
                                error_color = self.tui.colors["error"]
                                self.tui.print_tree_node(
                                    "ERROR", f'<style fg="{error_color}">{error_msg}</style>', depth=depth, path=path
                                )
                                self.processing = False
                                logger.info("  ✓ Completed error handling")
                                # Cancel the Esc monitor task
                                if "esc_monitor_task" in locals():
                                    esc_monitor_task.cancel()
                                break

        except websockets.exceptions.WebSocketException as e:
            self.tui.print(f'<style fg="ansired">Connection error:</style> {e}')
            return 1
        except Exception as e:
            self.tui.print(f'<style fg="ansired">Error:</style> {e}')
            return 1

        return 0


async def main():
    """Entry point for client"""
    from sabre.common.paths import get_logs_dir, ensure_dirs

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

    # Get server port from environment
    port = os.getenv("PORT", "8011")
    server_url = f"ws://localhost:{port}/message"

    # Create client (theme auto-detected by TUI)
    client = Client(server_url=server_url)
    return await client.run()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
