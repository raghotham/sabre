"""
Terminal UI utilities for SABRE client.

Handles theme management, color rendering, and all display logic.
"""

import logging
import os
import sys
from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import HTML
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import Terminal256Formatter

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
        "pygments_style": "friendly",  # Pygments theme for code highlighting (light)
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
        "pygments_style": "monokai",  # Pygments theme for code highlighting (dark)
    },
}


class TUI:
    """Terminal User Interface - handles theme management and color rendering."""

    def __init__(self):
        """Initialize TUI with dark theme (default)."""
        self.theme = "dark"  # Default to dark
        self.colors = COLORS[self.theme]

    # ==================== Theme Management ====================

    def set_theme(self, theme: str):
        """
        Set the theme (light or dark).

        Args:
            theme: "light" or "dark"
        """
        if theme not in ("light", "dark"):
            raise ValueError(f"Invalid theme: {theme}. Must be 'light' or 'dark'")
        self.theme = theme
        self.colors = COLORS[theme]
        logger.info(f"Theme set to: {theme}")

    def toggle_theme(self):
        """Toggle between light and dark theme."""
        new_theme = "light" if self.theme == "dark" else "dark"
        self.set_theme(new_theme)
        return new_theme

    # ==================== Rendering Methods ====================

    def print(self, text: str = ""):
        """Print text using prompt_toolkit."""
        print_formatted_text(HTML(text))
        sys.stdout.flush()

    def print_thinking(self):
        """Print thinking animation (stays on same line)."""
        # Use ANSI color code for cyan
        # Print without newline - stays on same line
        print("\r\033[96mThinking...\033[0m", end="", flush=True)

    def clear_thinking(self):
        """Clear the thinking animation line."""
        # Clear line and move cursor to beginning
        print("\r\033[K", end="", flush=True)

    def html_escape(self, text: str) -> str:
        """Escape HTML/XML special characters for safe rendering."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )

    def print_content_block(self, lines: list[str], depth: int, color: str = "result"):
        """Print a block of content with single tree connector and consistent indentation."""
        if not lines:
            return

        indent = ""
        col = self.colors.get(color, "result")

        # First line gets the connector
        first_line = self.html_escape(lines[0])
        self.print(f'{indent}  <style fg="{col}">⎿  {first_line}</style>')

        # Subsequent lines just get indented (aligned with first line content)
        for line in lines[1:]:
            escaped_line = self.html_escape(line)
            self.print(f'{indent}     <style fg="{col}">{escaped_line}</style>')

    def print_tree_node(self, label: str, text: str, depth: int = 0, path: list = None):
        """Print a tree node with execution label and content."""
        label_color = self.colors["node_label"]

        # Format: ⏺ [LABEL] #depth content [path]
        depth_str = f'<style fg="ansibrightblack">#{depth}</style>' if depth > 0 else ""

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
        parts = ["⏺", f'<style fg="{label_color}">[{label}]</style>']
        if depth_str:
            parts.append(depth_str)
        if text:
            parts.append(text)
        if path_suffix:
            parts.append(path_suffix)

        self.print(" ".join(parts))

    def print_node_content(self, content: str, depth: int = 0):
        """Print content indented under a tree node with line wrapping."""
        try:
            import shutil

            indent = "  "
            content_color = self.colors["result"]

            # Get terminal width
            terminal_width = shutil.get_terminal_size().columns

            # Calculate available width for content
            # indent + "   " + "⎿ " = indent + 5 chars
            base_indent_len = len(indent) + 5
            available_width = max(40, terminal_width - base_indent_len)  # Ensure minimum width

            # Continuation line indent (aligns with content after ⎿ )
            continuation_indent = indent + "   "  # 3 spaces to align

            lines = [line for line in content.split("\n") if line.strip()]
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
            content_color = self.colors["result"]
            self.print(f'{indent}  <style fg="{content_color}">⎿  {content}</style>')

    def print_code_block(self, code: str, depth: int = 0):
        """Print code block with syntax highlighting and line numbers."""
        indent = "  "

        # Check for NO_COLOR environment variable
        no_color = os.environ.get("NO_COLOR")

        if no_color:
            # No syntax highlighting - just print plain code with line numbers
            lines = code.rstrip("\n").split("\n")
            max_line_num = len(lines)
            # Calculate width needed for line numbers (minimum 2 characters)
            line_num_width = max(2, len(str(max_line_num)))

            for i, line in enumerate(lines, 1):
                # Right-align line number with padding, then add spacing before code
                line_num_str = str(i).rjust(line_num_width)
                print(f"{indent}  {line_num_str}  {line}")
                sys.stdout.flush()
        else:
            # Use Pygments for syntax highlighting (without built-in line numbers)
            formatter = Terminal256Formatter(
                style=self.colors["pygments_style"],
            )

            # Highlight the code
            highlighted = highlight(code, PythonLexer(), formatter)

            # Add line numbers manually with right-alignment
            lines = highlighted.rstrip("\n").split("\n")
            max_line_num = len(lines)
            # Calculate width needed for line numbers (minimum 2 characters)
            line_num_width = max(2, len(str(max_line_num)))

            for i, line in enumerate(lines, 1):
                # Right-align line number with padding, then add spacing before code
                line_num_str = str(i).rjust(line_num_width)
                print(f"{indent}  {line_num_str}  {line}")
                sys.stdout.flush()

    # ==================== Event Rendering Methods ====================

    def render_response_start(self, event_data: dict, depth: int):
        """Render RESPONSE_ROUND with model and token info."""
        lines = [f"Model: {event_data.get('model', '?')}", f"Prompt tokens: {event_data.get('prompt_tokens', 0)}"]
        self.print_content_block(lines, depth, color="result")

    def render_response_retry(self, event_data: dict, depth: int):
        """Render RATE_LIMIT retry info."""
        lines = [f"Reason: {event_data.get('reason', 'Unknown')}"]
        self.print_content_block(lines, depth, color="warning")

    def render_response_text(self, event_data: dict, depth: int):
        """Render RESPONSE_TEXT content (tokens shown in tree node)."""
        lines = []

        # Add cleaned text content (without helpers blocks)
        text = event_data.get("text", "")
        if text.strip():
            import re

            text_clean = text
            text_clean = re.sub(r"<helpers>.*?</helpers>", "", text_clean, flags=re.DOTALL)
            text_clean = re.sub(r"<helpers_result>.*?</helpers_result>", "", text_clean, flags=re.DOTALL)

            if text_clean.strip():
                # Add each line of cleaned text
                for line in text_clean.strip().split("\n"):
                    if line.strip():
                        lines.append(line.strip())

        if lines:
            self.print_content_block(lines, depth, color="result")

    def render_helpers_end(self, event_data: dict, depth: int):
        """Render RESULT with execution results (text and images)."""
        import re

        lines = []
        result = event_data.get("result", [])

        # Pattern to match markdown images: ![alt text](url)
        image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)", re.IGNORECASE)

        # Process all result items (Content objects)
        for content_obj in result:
            content_type = content_obj.type.value
            content_data = content_obj.data

            if content_type == "text" and content_data.strip():
                # Extract markdown images from text
                images = []
                for match in image_pattern.finditer(content_data):
                    alt_text = match.group(1)
                    url = match.group(2)
                    images.append((alt_text, url))

                # Remove image markdown from text
                text_without_images = image_pattern.sub("", content_data)

                # Add each line of result text (without images)
                if text_without_images.strip():
                    for line in text_without_images.strip().split("\n"):
                        if line.strip():
                            lines.append(line.strip())

                # Render accumulated text first
                if lines:
                    self.print_content_block(lines, depth, color="result")
                    lines = []

                # Render images
                for alt_text, url in images:
                    if url.startswith("http"):
                        self._render_image_url(url)
                    elif url.startswith("data:image"):
                        self._render_image(url, alt_text)

            elif content_type == "image":
                # Render accumulated text first
                if lines:
                    self.print_content_block(lines, depth, color="result")
                    lines = []

                # Handle image (URL or base64)
                image_data = content_data.get("image", "")
                if image_data:
                    if image_data.startswith("http"):
                        # URL - either imgcat (WezTerm) or show URL
                        self._render_image_url(image_data)
                    else:
                        # Base64 (shouldn't happen anymore, but handle it)
                        mime_type = content_data.get("mime_type", "image/png")
                        self._render_image(f"data:{mime_type};base64,{image_data}", "")

        # Render any remaining text lines
        if lines:
            self.print_content_block(lines, depth, color="result")

    def render_complete(self, event_data: dict, depth: int):
        """Render COMPLETE with final message (handles markdown images)."""
        final_message = event_data.get("final_message", "")
        if final_message.strip():
            # Parse and render final message with proper indentation
            import re

            # Pattern to match markdown images: ![alt text](url)
            image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)", re.IGNORECASE)

            # Extract images and render them
            images = []
            for match in image_pattern.finditer(final_message):
                alt_text = match.group(1)
                url = match.group(2)
                images.append((alt_text, url))

            # Remove image markdown from text
            text_without_images = image_pattern.sub("", final_message)

            # Pattern to match <helpers> and <helpers_result> blocks
            code_block_pattern = re.compile(
                r"(<helpers>|<helpers_result>)(.*?)(</helpers>|</helpers_result>)", re.DOTALL
            )

            # Check if there are code blocks
            has_code_blocks = bool(code_block_pattern.search(text_without_images))

            if has_code_blocks:
                # If there are code blocks, use the existing renderer (no indentation)
                self._render_message_with_code_blocks(final_message)
            else:
                # No code blocks - render as indented content block
                lines = []
                for line in text_without_images.strip().split("\n"):
                    if line.strip():
                        lines.append(line.strip())

                if lines:
                    self.print_content_block(lines, depth, color="complete")

                # Render images after text
                for alt_text, url in images:
                    if url.startswith("http"):
                        self._render_image_url(url)
                    elif url.startswith("data:image"):
                        self._render_image(url, alt_text)

        # Session info is now displayed at the beginning of the session
        # (see client.py ResponseStartEvent handler)

    # ==================== Image Rendering ====================

    def _render_image_url(self, url: str):
        """Render image URL - imgcat in WezTerm, otherwise show URL."""
        import subprocess
        import urllib.request
        import re

        term_program = os.environ.get("TERM_PROGRAM", "")
        is_wezterm = term_program.lower() == "wezterm"

        # Convert localhost URLs to local file paths
        # URL format: http://localhost:8011/v1/sessions/{session_id}/files/{filename}
        localhost_pattern = r"http://localhost:\d+/v1/sessions/([^/]+)/files/(.+)"
        match = re.match(localhost_pattern, url)

        if match:
            session_id = match.group(1)
            filename = match.group(2)
            # Construct local file path
            from sabre.common.paths import get_session_files_dir

            files_dir = get_session_files_dir(session_id)
            local_path = files_dir / filename

            if local_path.exists():
                # Use local file path directly
                if is_wezterm:
                    try:
                        # Use local file path with wezterm imgcat
                        subprocess.run(["wezterm", "imgcat", str(local_path)], check=True)
                        logger.info(f"Rendered image via wezterm imgcat: {local_path}")
                        return
                    except Exception as e:
                        logger.error(f"Failed to imgcat {local_path}: {e}")
                        self.print(f'<style fg="{self.colors["result"]}">{url}</style>')
                        return
                else:
                    # Show local path instead of URL
                    self.print(f'<style fg="{self.colors["result"]}">{local_path}</style>')
                    return
            else:
                logger.warning(f"Local file not found: {local_path}, falling back to URL")

        # Fallback: try to download from URL (for remote images)
        if is_wezterm:
            try:
                # Download image and pipe to wezterm imgcat
                with urllib.request.urlopen(url) as response:
                    image_bytes = response.read()
                    subprocess.run(["wezterm", "imgcat"], input=image_bytes, check=True)
                    logger.info(f"Rendered image via wezterm imgcat: {url}")
            except Exception as e:
                # Fall back to showing URL
                logger.error(f"Failed to imgcat {url}: {e}")
                self.print(f'<style fg="{self.colors["result"]}">{url}</style>')
        else:
            # Just show the URL
            self.print(f'<style fg="{self.colors["result"]}">{url}</style>')

    def _render_image(self, image_data: str, alt_text: str = ""):
        """Render an image based on terminal capabilities."""
        import base64
        import subprocess

        # Check if terminal is WezTerm
        term_program = os.environ.get("TERM_PROGRAM", "")
        is_wezterm = term_program.lower() == "wezterm"

        # Decode base64 image data (handle data URL format)
        if image_data.startswith("data:image/"):
            # Extract base64 portion from data URL: data:image/png;base64,<data>
            _, encoded = image_data.split(",", 1)
        else:
            encoded = image_data

        try:
            image_bytes = base64.b64decode(encoded)

            if is_wezterm:
                # Use wezterm imgcat for inline rendering in WezTerm
                try:
                    subprocess.run(["wezterm", "imgcat"], input=image_bytes, check=True)
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
        """Save image to temp file and open with system viewer."""
        import tempfile
        import subprocess
        import platform

        # Create temp file (don't auto-delete so viewer can open it)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        # Open with system viewer in background
        system = platform.system()
        try:
            if system == "Darwin":  # macOS
                subprocess.Popen(["open", tmp_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif system == "Linux":
                subprocess.Popen(["xdg-open", tmp_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:  # Windows
                subprocess.Popen(["start", tmp_path], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if alt_text:
                self.print(f'<style fg="{self.colors["complete"]}">[Image opened in viewer: {alt_text}]</style>')
            else:
                self.print(f'<style fg="{self.colors["complete"]}">[Image opened in viewer]</style>')
        except Exception as e:
            self.print(f'<style fg="{self.colors["error"]}">Failed to open image: {e}</style>')

    def _render_message_with_code_blocks(self, message: str):
        """Parse and render message with <helpers>, <helpers_result> blocks and images."""
        import re
        import html

        # Pattern to match markdown images: ![alt text](url)
        image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)", re.IGNORECASE)

        # First, extract and render images
        def replace_image(match):
            alt_text = match.group(1)
            url = match.group(2)
            logger.info(f"Found markdown image: alt='{alt_text}', url='{url}'")
            # Render based on URL type
            if url.startswith("http"):
                self._render_image_url(url)
            elif url.startswith("data:image"):
                self._render_image(url, alt_text)
            return ""  # Remove from text

        # Replace images with empty strings and render them
        message = image_pattern.sub(replace_image, message)

        # Pattern to match <helpers>...</helpers> and <helpers_result>...</helpers_result>
        code_block_pattern = re.compile(r"(<helpers>|<helpers_result>)(.*?)(</helpers>|</helpers_result>)", re.DOTALL)

        last_end = 0
        complete_color = self.colors["complete"]

        for match in code_block_pattern.finditer(message):
            # Print text before code block
            text_before = message[last_end : match.start()]
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
