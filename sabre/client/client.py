"""
Minimal LLMVM2 Client with prompt_toolkit rendering.

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
from prompt_toolkit.formatted_text import FormattedText, HTML
from prompt_toolkit import print_formatted_text
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import Terminal256Formatter
import websockets

from sabre.client.slash_commands import SlashCommandHandler

# Initialize logger
logger = logging.getLogger(__name__)


# Colors for terminal rendering (ANSI color names for prompt_toolkit)
COLORS = {
    "light": {
        "user_input": "ansiblue",
        "node_label": "ansimagenta",
        "code": "ansiblack",
        "result": "ansigreen",
        "complete": "ansiblack",
        "final_answer": "ansimagenta",  # More prominent for final answer
        "error": "ansired",
        "warning": "ansiyellow",
        "pygments_style": "github-light",  # Pygments theme for code highlighting
    },
    "dark": {
        "user_input": "ansibrightblack",
        "node_label": "ansibrightcyan",
        "code": "ansibrightblack",
        "result": "ansibrightgreen",
        "complete": "ansiwhite",
        "final_answer": "ansiwhite",  # Brighter for final answer
        "error": "ansibrightred",
        "warning": "ansibrightyellow",
        "pygments_style": "monokai",  # Pygments theme for code highlighting
    }
}


class Client:
    """WebSocket chat client with rich rendering"""

    def __init__(self, server_url: str = "ws://localhost:8011/message", theme: str = "dark", history_file: str | None = None):
        self.server_url = server_url

        # Setup history using XDG-compliant paths
        if history_file is None:
            from sabre.common.paths import get_logs_dir, ensure_dirs
            ensure_dirs()
            log_dir = get_logs_dir()
            history_file = str(log_dir / "history")

        # Set up prompt styling based on theme
        self.theme = theme
        self.colors = COLORS[theme]

        # Track if we're processing a message (for cancellation)
        self.processing = False
        self.websocket = None
        self.cancel_requested = False  # Flag for cancellation

        # Create styled prompt session
        prompt_style = PromptStyle.from_dict({
            '': f"{self.colors['user_input']} bg:ansigray",  # User input with gray background
        })

        self.session = PromptSession(
            history=FileHistory(history_file),
            style=prompt_style
        )
        self.tree_depth = 0

        # Initialize slash command handler
        self.slash_handler = SlashCommandHandler(self)

    def print(self, text: str = ""):
        """Print text using prompt_toolkit"""
        print_formatted_text(HTML(text))
        sys.stdout.flush()

    def html_escape(self, text: str) -> str:
        """Escape HTML/XML special characters for safe rendering"""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#x27;'))

    def print_content_block(self, lines: list[str], depth: int, color: str = 'result'):
        """Print a block of content with single tree connector and consistent indentation"""
        if not lines:
            return

        indent = ""
        col = self.colors.get(color, 'result')

        # First line gets the connector
        first_line = self.html_escape(lines[0])
        self.print(f'{indent}  <style fg="{col}">⎿  {first_line}</style>')

        # Subsequent lines just get indented (aligned with first line content)
        for line in lines[1:]:
            escaped_line = self.html_escape(line)
            self.print(f'{indent}     <style fg="{col}">{escaped_line}</style>')

    # ==================== Event Rendering Methods ====================

    def render_response_start(self, event_data: dict, depth: int):
        """Render RESPONSE_ROUND with model and token info"""
        lines = [
            f"Model: {event_data.get('model', '?')}",
            f"Prompt tokens: {event_data.get('prompt_tokens', 0)}"
        ]
        self.print_content_block(lines, depth, color='result')

    def render_response_retry(self, event_data: dict, depth: int):
        """Render RATE_LIMIT retry info"""
        lines = [f"Reason: {event_data.get('reason', 'Unknown')}"]
        self.print_content_block(lines, depth, color='warning')

    def render_response_text(self, event_data: dict, depth: int):
        """Render RESPONSE_TEXT content (tokens shown in tree node)"""
        lines = []

        # Add cleaned text content (without helpers blocks)
        text = event_data.get("text", "")
        if text.strip():
            import re
            text_clean = text
            text_clean = re.sub(r'<helpers>.*?</helpers>', '', text_clean, flags=re.DOTALL)
            text_clean = re.sub(r'<helpers_result>.*?</helpers_result>', '', text_clean, flags=re.DOTALL)

            if text_clean.strip():
                # Add each line of cleaned text
                for line in text_clean.strip().split('\n'):
                    if line.strip():
                        lines.append(line.strip())

        if lines:
            self.print_content_block(lines, depth, color='result')

    def render_helpers_end(self, event_data: dict, depth: int):
        """Render RESULT with execution results (text and images)"""
        lines = []
        result = event_data.get("result", [])

        # Process all result items
        for content_obj in result:
            content_type = content_obj.get("type", "text")
            content_data = content_obj.get("data", "")

            if content_type == "text" and content_data.strip():
                # Add each line of result text
                for line in content_data.strip().split('\n'):
                    if line.strip():
                        lines.append(line.strip())
            elif content_type == "image":
                # Render accumulated text first
                if lines:
                    self.print_content_block(lines, depth, color='result')
                    lines = []

                # Handle image (URL or base64)
                image_data = content_data.get("image", "")
                if image_data:
                    if image_data.startswith('http'):
                        # URL - either imgcat (WezTerm) or show URL
                        self._render_image_url(image_data)
                    else:
                        # Base64 (shouldn't happen anymore, but handle it)
                        mime_type = content_data.get("mime_type", "image/png")
                        self._render_image(f"data:{mime_type};base64,{image_data}", "")

        # Render any remaining text lines
        if lines:
            self.print_content_block(lines, depth, color='result')

    def render_complete(self, event_data: dict, depth: int):
        """Render COMPLETE with final message (handles markdown images)"""
        final_message = event_data.get("final_message", "")
        if final_message.strip():
            # Use message renderer to handle markdown images
            self._render_message_with_code_blocks(final_message)

    def print_tree_node(self, label: str, text: str, depth: int = 0, path: list = None):
        """Print a tree node with execution label and content"""
        label_color = self.colors['node_label']

        # Format: ⏺ [LABEL] #depth content [path]
        depth_str = f'<style fg="ansibrightblack">#{depth}</style>' if depth > 0 else ''

        # Format path to show last 2 nodes for readability
        path_suffix = ""
        if path and len(path) > 0:
            # Show last 2 nodes in path (shortened)
            display_path = path[-2:] if len(path) > 2 else path
            # Take first 6 chars of each node ID
            short_path = [node_id[:6] for node_id in display_path]
            path_str = " > ".join(short_path)
            path_suffix = f' <style fg="ansibrightblack">[{path_str}]</style>'

        # Build output: ⏺ [LABEL] #depth content [path]
        parts = ['⏺', f'<style fg="{label_color}">[{label}]</style>']
        if depth_str:
            parts.append(depth_str)
        if text:
            parts.append(text)
        if path_suffix:
            parts.append(path_suffix)

        self.print(' '.join(parts))

    def print_node_content(self, content: str, depth: int = 0):
        """Print content indented under a tree node with line wrapping"""
        try:
            import shutil

            indent = "  "
            content_color = self.colors['result']

            # Get terminal width
            terminal_width = shutil.get_terminal_size().columns

            # Calculate available width for content
            # indent + "   " + "⎿ " = indent + 5 chars
            base_indent_len = len(indent) + 5
            available_width = max(40, terminal_width - base_indent_len)  # Ensure minimum width

            # Continuation line indent (aligns with content after ⎿ )
            continuation_indent = indent + "   "  # 3 spaces to align

            lines = [line for line in content.split('\n') if line.strip()]
            if not lines:
                return

            is_first_line = True
            for line in lines:
                line_text = line.strip()

                # Wrap long lines
                while line_text:
                    if is_first_line:
                        # First line with tree connector
                        if len(line_text) <= available_width:
                            self.print(f'{indent}  <style fg="{content_color}">⎿  {line_text}</style>')
                            line_text = ""
                            is_first_line = False
                        else:
                            # Split at available width
                            chunk = line_text[:available_width]
                            self.print(f'{indent}  <style fg="{content_color}">⎿  {chunk}</style>')
                            line_text = line_text[available_width:]
                            is_first_line = False
                    else:
                        # Continuation lines
                        if len(line_text) <= available_width:
                            self.print(f'{continuation_indent}<style fg="{content_color}">{line_text}</style>')
                            line_text = ""
                        else:
                            # Split at available width
                            chunk = line_text[:available_width]
                            self.print(f'{continuation_indent}<style fg="{content_color}">{chunk}</style>')
                            line_text = line_text[available_width:]
        except Exception as e:
            # Fallback to simple print if wrapping fails
            logger.error(f"Error in print_node_content: {e}")
            indent = "  " * depth
            content_color = self.colors['result']
            self.print(f'{indent}  <style fg="{content_color}">⎿  {content}</style>')

    def print_code_block(self, code: str, depth: int = 0):
        """Print code block with syntax highlighting and line numbers"""
        indent = "  "

        # Use Pygments for syntax highlighting (without built-in line numbers)
        formatter = Terminal256Formatter(
            style=self.colors['pygments_style'],
        )

        # Highlight the code
        highlighted = highlight(code, PythonLexer(), formatter)

        # Add line numbers manually (without padding)
        lines = highlighted.rstrip('\n').split('\n')
        for i, line in enumerate(lines, 1):
            # No padding - just line number, space, and code
            print(f'{indent}  {i} {line}')
            sys.stdout.flush()

    def _render_image_url(self, url: str):
        """Render image URL - imgcat in WezTerm, otherwise show URL"""
        import os
        import subprocess
        import urllib.request

        term_program = os.environ.get('TERM_PROGRAM', '')
        is_wezterm = term_program.lower() == 'wezterm'

        if is_wezterm:
            try:
                # Download image and pipe to wezterm imgcat
                with urllib.request.urlopen(url) as response:
                    image_bytes = response.read()
                    subprocess.run(
                        ['wezterm', 'imgcat'],
                        input=image_bytes,
                        check=True
                    )
                    logger.info(f"Rendered image via wezterm imgcat: {url}")
            except Exception as e:
                # Fall back to showing URL
                logger.error(f"Failed to imgcat {url}: {e}")
                self.print(f'<style fg="{self.colors["result"]}">{url}</style>')
        else:
            # Just show the URL
            self.print(f'<style fg="{self.colors["result"]}">{url}</style>')

    def _render_image(self, image_data: str, alt_text: str = ""):
        """Render an image based on terminal capabilities"""
        import os
        import base64
        import subprocess

        # Check if terminal is WezTerm
        term_program = os.environ.get('TERM_PROGRAM', '')
        is_wezterm = term_program.lower() == 'wezterm'

        # Decode base64 image data (handle data URL format)
        if image_data.startswith('data:image/'):
            # Extract base64 portion from data URL: data:image/png;base64,<data>
            _, encoded = image_data.split(',', 1)
        else:
            encoded = image_data

        try:
            image_bytes = base64.b64decode(encoded)

            if is_wezterm:
                # Use wezterm imgcat for inline rendering in WezTerm
                try:
                    subprocess.run(
                        ['wezterm', 'imgcat'],
                        input=image_bytes,
                        check=True
                    )
                    if alt_text:
                        self.print(f'<style fg="{self.colors["complete"]}">{alt_text}</style>')
                except FileNotFoundError:
                    # wezterm not found, fall back to system viewer
                    self._open_image_with_viewer(image_bytes, alt_text)
            else:
                # Open with system viewer
                self._open_image_with_viewer(image_bytes, alt_text)
        except Exception as e:
            self.print(f'<style fg="{self.colors["error"]}">Failed to render image: {e}</style>')

    def _open_image_with_viewer(self, image_bytes: bytes, alt_text: str = ""):
        """Save image to temp file and open with system viewer"""
        import tempfile
        import subprocess
        import platform

        # Create temp file (don't auto-delete so viewer can open it)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        # Open with system viewer in background
        system = platform.system()
        try:
            if system == 'Darwin':  # macOS
                subprocess.Popen(['open', tmp_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif system == 'Linux':
                subprocess.Popen(['xdg-open', tmp_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:  # Windows
                subprocess.Popen(['start', tmp_path], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if alt_text:
                self.print(f'<style fg="{self.colors["complete"]}">[Image opened in viewer: {alt_text}]</style>')
            else:
                self.print(f'<style fg="{self.colors["complete"]}">[Image opened in viewer]</style>')
        except Exception as e:
            self.print(f'<style fg="{self.colors["error"]}">Failed to open image: {e}</style>')

    def _render_message_with_code_blocks(self, message: str):
        """Parse and render message with <helpers>, <helpers_result> blocks and images"""
        import re
        import html

        # Pattern to match markdown images: ![alt text](url)
        image_pattern = re.compile(
            r'!\[([^\]]*)\]\(([^)]+)\)',
            re.IGNORECASE
        )

        # First, extract and render images
        def replace_image(match):
            alt_text = match.group(1)
            url = match.group(2)
            logger.info(f"Found markdown image: alt='{alt_text}', url='{url}'")
            # Render based on URL type
            if url.startswith('http'):
                self._render_image_url(url)
            elif url.startswith('data:image'):
                self._render_image(url, alt_text)
            return ""  # Remove from text

        # Replace images with empty strings and render them
        message = image_pattern.sub(replace_image, message)

        # Pattern to match <helpers>...</helpers> and <helpers_result>...</helpers_result>
        code_block_pattern = re.compile(
            r'(<helpers>|<helpers_result>)(.*?)(</helpers>|</helpers_result>)',
            re.DOTALL
        )

        last_end = 0
        complete_color = self.colors['complete']

        for match in code_block_pattern.finditer(message):
            # Print text before code block
            text_before = message[last_end:match.start()]
            if text_before.strip():
                # Escape HTML entities and print as block
                escaped_text = html.escape(text_before)
                self.print(f'<style fg="{complete_color}">{escaped_text}</style>')

            # Print code block with highlighting
            code = match.group(2).strip()
            if code:
                self.print_code_block(code)

            last_end = match.end()

        # Print remaining text after last code block
        text_after = message[last_end:]
        if text_after.strip():
            # Escape HTML entities and print as block
            escaped_text = html.escape(text_after)
            self.print(f'<style fg="{complete_color}">{escaped_text}</style>')

    async def cancel_processing(self):
        """Cancel the current processing task"""
        logger.info(f"cancel_processing called: processing={self.processing}, websocket={self.websocket is not None}, cancel_requested={self.cancel_requested}")
        if self.processing and self.websocket and not self.cancel_requested:
            self.cancel_requested = True
            try:
                logger.info("Sending cancel message to server")
                await self.websocket.send(json.dumps({
                    "type": "cancel"
                }))
                logger.info("Cancel message sent successfully")
                self.print(f'\n<style fg="{self.colors["warning"]}">Cancelling...</style>')
            except Exception as e:
                logger.error(f"Failed to send cancel: {e}")
                self.print(f'\n<style fg="{self.colors["error"]}">Failed to send cancel: {e}</style>')
        else:
            logger.info(f"Cancel not sent - conditions not met")

    async def _handle_escape(self):
        """Handle escape key press - show message and cancel"""
        self.print(f'\n<style fg="{self.colors["warning"]}">[Escape detected - requesting cancellation...]</style>')
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
                    if char == '\x1b':
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
        self.print('\n<b><style fg="ansicyan">LLMVM2 Client</style></b>')
        self.print(f'Connecting to {self.server_url}...\n')

        try:
            # Use longer ping timeout for LLM responses that can take time
            async with websockets.connect(
                self.server_url,
                ping_interval=30,      # Send ping every 30 seconds
                ping_timeout=None,     # No timeout for pong (allow long operations)
            ) as websocket:
                self.websocket = websocket  # Store for cancellation
                self.print('<style fg="ansigreen">Connected!</style> Type your message (Ctrl+D or \'exit\' to quit)')
                self.print('<style fg="ansibrightblack">Press Esc to cancel execution</style>\n')

                while True:
                    # Get user input
                    try:
                        user_input = await asyncio.get_event_loop().run_in_executor(
                            None, self.session.prompt, "> "
                        )
                    except (EOFError, KeyboardInterrupt):
                        self.print('\n<style fg="ansiyellow">Exiting...</style>')
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
                                self.print('\n<style fg="ansiyellow">Exiting...</style>')
                                break
                            elif result.data and result.data.get("action") == "clear_conversation":
                                # Special case: clear conversation
                                if result.message:
                                    self.print(f'\n<style fg="ansicyan">{result.message}</style>')
                                # Send clear message to server
                                await websocket.send(json.dumps({
                                    "type": "clear_conversation"
                                }))
                                # Wait for confirmation
                                response = await websocket.recv()
                                data = json.loads(response)
                                if data.get("type") == "conversation_cleared":
                                    self.print('<style fg="ansigreen">Conversation cleared. Starting fresh.</style>\n')
                                continue
                            elif result.data and result.data.get("action") == "request_helpers":
                                # Special case: request helpers from server
                                # Send as regular message "helpers()" to execute the helper
                                user_input = "helpers()"
                                # Fall through to normal message processing
                            elif result.message:
                                self.print(f'\n<style fg="ansicyan">{result.message}</style>\n')
                                continue  # Don't send to server
                        else:
                            self.print(f'\n<style fg="ansired">{result.message}</style>\n')
                            continue  # Don't send to server

                        # If we didn't continue above, fall through to send message to server

                    # Send message to server (normal message or request_helpers action)

                    # Send message to server
                    self.processing = True  # Track that we're processing
                    self.cancel_requested = False  # Reset cancel flag

                    # Start Esc key monitoring task
                    esc_monitor_task = asyncio.create_task(self._monitor_escape_key())

                    await websocket.send(json.dumps({
                        "type": "message",
                        "content": user_input
                    }))

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
                        # self.print(f'<style fg="ansiyellow">[DEBUG] {msg_type or data.get("type", "unknown")}</style>')

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
                                logger.info(f"  → Handling response_start event")
                                if not response_started:
                                    self.print()  # Blank line before first assistant response
                                    response_started = True

                                # Always show response rounds
                                round_id = event_data.get('round_id', '?')
                                self.print_tree_node(
                                    "RESPONSE_ROUND",
                                    f"iteration {round_id}",
                                    depth=depth,
                                    path=path
                                )
                                self.render_response_start(event_data, depth)
                                logger.info(f"  ✓ Completed response_start handling")

                            elif event_type == "response_retry":
                                attempt = event_data.get("attempt", 0)
                                max_retries = event_data.get("max_retries", 0)
                                wait_seconds = event_data.get("wait_seconds", 0)
                                self.print_tree_node(
                                    "RATE_LIMIT",
                                    f"retrying in {wait_seconds:.1f}s (attempt {attempt}/{max_retries})",
                                    depth=depth,
                                    path=path
                                )
                                self.render_response_retry(event_data, depth)

                            elif event_type == "helpers_extracted":
                                logger.info(f"  → Handling helpers_extracted event")
                                code = event_data.get("code", "")
                                block_num = event_data.get("block_count", 0)
                                logger.info(f"  → block_num={block_num}, code_length={len(code)}")
                                self.print_tree_node(
                                    "HELPER_BLOCK",
                                    f"#{block_num}",
                                    depth=depth,
                                    path=path
                                )
                                # Print code
                                if code.strip():
                                    logger.info(f"  → About to print_code_block")
                                    self.print_code_block(code.strip(), depth)
                                    logger.info(f"  → Finished print_code_block")
                                logger.info(f"  ✓ Completed helpers_extracted handling")

                            elif event_type == "helpers_start":
                                logger.info(f"  → Handling helpers_start event")
                                self.print_tree_node(
                                    "EXECUTING",
                                    "...",
                                    depth=depth,
                                    path=path
                                )
                                logger.info(f"  ✓ Completed helpers_start handling")

                            elif event_type == "response_text":
                                logger.info(f"  → Handling response_text event")
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
                                    self.print_tree_node(
                                        "RESPONSE_TEXT",
                                        f"{text_length} chars, {helper_count} helper(s){token_info}",
                                        depth=depth,
                                        path=path
                                    )
                                else:
                                    self.print_tree_node(
                                        "RESPONSE_TEXT",
                                        f"{text_length} chars, no helpers{token_info}",
                                        depth=depth,
                                        path=path
                                    )

                                self.render_response_text(event_data, depth)
                                logger.info(f"  ✓ Completed response_text handling")

                            elif event_type == "helpers_end":
                                logger.info(f"  → Handling helpers_end event")
                                duration = event_data.get("duration_ms", 0)
                                block_number = event_data.get("block_number", 0)
                                code_preview = event_data.get("code_preview", "")

                                # Show helper identifier with code preview
                                preview_text = code_preview.replace('\n', ' ')[:50]
                                if len(code_preview) > 50:
                                    preview_text += "..."

                                self.print_tree_node(
                                    f"RESULT #{block_number}",
                                    f"{duration:.0f}ms - {preview_text}",
                                    depth=depth,
                                    path=path
                                )

                                self.render_helpers_end(event_data, depth)
                                logger.info(f"  ✓ Completed helpers_end handling")

                            elif event_type == "complete":
                                logger.info(f"  → Handling complete event")
                                final_message = event_data.get("final_message", "")
                                if final_message.strip():
                                    self.print()  # Blank line
                                    self.print_tree_node(
                                        "COMPLETE",
                                        "",
                                        depth=depth,
                                        path=path
                                    )
                                    self.render_complete(event_data, depth)
                                self.print()  # Blank line after response
                                self.processing = False
                                logger.info(f"  ✓ Completed complete handling")
                                # Cancel the Esc monitor task
                                if 'esc_monitor_task' in locals():
                                    esc_monitor_task.cancel()
                                break

                            elif event_type == "cancelled":
                                logger.info(f"  → Handling cancelled event")
                                cancel_msg = event_data.get("message", "Execution cancelled")
                                warning_color = self.colors['warning']
                                self.print_tree_node(
                                    "CANCELLED",
                                    f'<style fg="{warning_color}">{cancel_msg}</style>',
                                    depth=depth,
                                    path=path
                                )
                                self.print()  # Blank line
                                self.processing = False
                                logger.info(f"  ✓ Completed cancelled handling")
                                # Cancel the Esc monitor task
                                if 'esc_monitor_task' in locals():
                                    esc_monitor_task.cancel()
                                break

                            elif event_type == "error":
                                logger.info(f"  → Handling error event")
                                error_msg = event_data.get("error_message", "Unknown error")
                                error_color = self.colors['error']
                                self.print_tree_node(
                                    "ERROR",
                                    f'<style fg="{error_color}">{error_msg}</style>',
                                    depth=depth,
                                    path=path
                                )
                                self.processing = False
                                logger.info(f"  ✓ Completed error handling")
                                # Cancel the Esc monitor task
                                if 'esc_monitor_task' in locals():
                                    esc_monitor_task.cancel()
                                break

        except websockets.exceptions.WebSocketException as e:
            self.print(f'<style fg="ansired">Connection error:</style> {e}')
            return 1
        except Exception as e:
            self.print(f'<style fg="ansired">Error:</style> {e}')
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
        ]
    )

    # Get server port from environment
    port = os.getenv("LLMVM_PORT", "8011")
    server_url = f"ws://localhost:{port}/message"

    # Detect theme (could be made configurable)
    theme = "dark"  # Default to dark theme
    client = Client(server_url=server_url, theme=theme)
    return await client.run()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
