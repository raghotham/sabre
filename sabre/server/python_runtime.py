"""
Python Runtime for executing <helpers> blocks.

Provides isolated execution environment with helper functions.
"""

import sys
import io
import os
import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import httpx
import pandas as pd

from sabre.server.helpers.bash import Bash
from sabre.server.helpers.search import Search
from sabre.server.helpers.web import Web, download, download_csv
from sabre.server.helpers.llm_call import LLMCall
from sabre.server.helpers.llm_bind import LLMBind
from sabre.server.helpers.coerce import Coerce
from sabre.server.helpers.llm_list_bind import LLMListBind
from sabre.server.helpers.pandas_bind import PandasBind
from sabre.server.helpers.matplotlib_helpers import matplotlib_to_image, generate_graph_image
from sabre.server.helpers.introspection import get_helper_signatures
from sabre.server.helpers.fs import write_file, read_file
from sabre.server.helpers.sabre_call import SabreCall
from sabre.server.helpers.database import DatabaseHelpers
from sabre.server.helpers.semantic_database import SemanticDatabaseHelpers
from sabre.common.models.messages import Content, ImageContent, TextContent, PdfContent, FileContent

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of Python code execution."""

    success: bool
    output: str  # Text output for display
    error: str | None
    results: list[Any]  # Captured via result() calls
    content: list[Content] = field(default_factory=list)  # Structured content (images, etc.)


class PythonRuntime:
    """
    Python code execution engine.

    Executes Python code in isolated namespace with helper functions.
    """

    def __init__(self, mcp_adapter=None):
        """
        Initialize runtime with clean namespace.

        Args:
            mcp_adapter: Optional MCPHelperAdapter for MCP tool integration
        """
        self.orchestrator = None  # Set by orchestrator after init
        self.openai_client = None  # Lazy init in _llm_call_async
        self.mcp_adapter = mcp_adapter  # MCP tool adapter

        # Configure matplotlib to use non-interactive backend
        try:
            import matplotlib

            matplotlib.use("Agg")  # Non-interactive backend
            logger.info("Configured matplotlib to use Agg backend")
        except ImportError:
            logger.debug("matplotlib not available")

        self.reset()

    def reset(self):
        """
        Reset the runtime namespace.

        This reinitializes all helpers and re-injects MCP tools from the adapter.
        Call this after connector changes (add/remove/update) to refresh available tools.
        """
        # Results collected via result() calls
        self._results = []

        # Initialize helpers that need runtime context
        # Use lambda to get orchestrator dynamically (set later via set_orchestrator)
        llm_call = LLMCall(lambda: self.orchestrator, self._get_openai_client)
        llm_bind = LLMBind(lambda: self.orchestrator, self._get_openai_client, lambda: self.namespace)
        coerce = Coerce(lambda: self.orchestrator, self._get_openai_client)
        llm_list_bind = LLMListBind(lambda: self.orchestrator, self._get_openai_client)
        pandas_bind = PandasBind(lambda: self.orchestrator, self._get_openai_client)

        # Helper for recursive execution - needs orchestrator, tree, and event callback
        # These are accessed via execution_context
        def get_tree():
            from sabre.common.execution_context import get_execution_context

            ctx = get_execution_context()
            return ctx.tree if ctx else None

        def get_event_callback():
            from sabre.common.execution_context import get_execution_context

            ctx = get_execution_context()
            return ctx.event_callback if ctx else None

        sabre_call = SabreCall(lambda: self.orchestrator, get_tree, get_event_callback)

        # Build namespace with helpers
        self.namespace = {
            # Class-based helpers (stateless)
            "Bash": Bash,
            "Search": Search,
            "Web": Web,
            "download": download,
            "download_csv": download_csv,
            "DatabaseHelpers": DatabaseHelpers,
            "SemanticDatabaseHelpers": SemanticDatabaseHelpers,
            # Instance-based helpers (need runtime context)
            "llm_call": llm_call,
            "llm_bind": llm_bind,
            "coerce": coerce,
            "llm_list_bind": llm_list_bind,
            "pandas_bind": pandas_bind,
            "sabre_call": sabre_call,  # Recursive execution
            "result": self._result,
            "capture_figures": self._capture_figures_for_user,
            # File I/O helpers (stateless)
            "write_file": write_file,
            "read_file": read_file,
            # Matplotlib helpers
            "matplotlib_to_image": matplotlib_to_image,
            "generate_graph_image": generate_graph_image,
            # Standard library (minimal set)
            "print": print,
            "pd": pd,  # pandas for user code
        }

        # Add datetime module
        import datetime

        self.namespace["datetime"] = datetime

        # Storage for captured figures
        self._captured_figures: list[ImageContent] = []

        # Add matplotlib if available
        try:
            import matplotlib.pyplot as plt

            self.namespace["plt"] = plt
            logger.debug("Added matplotlib.pyplot to namespace")
        except ImportError:
            logger.debug("matplotlib not available, skipping")

        # Add MCP tools if adapter is available
        if self.mcp_adapter:
            mcp_tools = self.mcp_adapter.get_available_tools()

            # Group tools by server to create namespace objects
            # e.g., remote_test.echo becomes remote_test object with echo attribute
            # Hyphens are replaced with underscores to create valid Python identifiers
            servers = {}
            for tool_name, tool_func in mcp_tools.items():
                if "." in tool_name:
                    server_name, method_name = tool_name.split(".", 1)
                    # Sanitize server name: replace hyphens with underscores for valid Python identifiers
                    safe_server_name = server_name.replace("-", "_")
                    if safe_server_name not in servers:
                        servers[safe_server_name] = type(safe_server_name, (), {})()
                    setattr(servers[safe_server_name], method_name, tool_func)
                else:
                    # Flat tool name (no server prefix)
                    self.namespace[tool_name] = tool_func

            # Add server objects to namespace
            for server_name, server_obj in servers.items():
                self.namespace[server_name] = server_obj

            logger.info(f"Added {len(mcp_tools)} MCP tools to namespace: {list(mcp_tools.keys())}")
            logger.info(f"Created {len(servers)} MCP server namespaces: {list(servers.keys())}")

    def set_orchestrator(self, orchestrator):
        """
        Set orchestrator reference for recursive calls.

        Called by orchestrator after initialization.

        Args:
            orchestrator: Orchestrator instance
        """
        self.orchestrator = orchestrator

    def get_available_functions(self) -> str:
        """
        Get list of all available functions in the runtime.

        Returns:
            Formatted string with function signatures
        """
        # Get built-in helper signatures
        signatures = get_helper_signatures(self.namespace)

        # Add MCP tools documentation if available
        if self.mcp_adapter:
            mcp_docs = self.mcp_adapter.generate_documentation()
            if mcp_docs:
                signatures += "\n\n" + mcp_docs

        return signatures

    def _get_openai_client(self):
        """
        Get or create AsyncOpenAI client with proper configuration.

        Respects OPENAI_API_KEY and OPENAI_BASE_URL environment variables.

        Returns:
            AsyncOpenAI client instance
        """
        if not self.openai_client:
            from openai import AsyncOpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            skip_ssl = os.getenv("OPENAI_SKIP_SSL_VERIFY", "").lower() in ("true", "1", "yes")

            # Create httpx client with SSL settings
            http_client = None
            if skip_ssl:
                logger.warning("⚠️  SSL certificate verification is DISABLED - use only for testing!")
                http_client = httpx.AsyncClient(verify=False)

            # Create client with optional base_url and http_client
            client_kwargs = {"api_key": api_key}
            if base_url:
                logger.info(f"Creating OpenAI client with custom base URL: {base_url}")
                client_kwargs["base_url"] = base_url
            if http_client:
                client_kwargs["http_client"] = http_client

            self.openai_client = AsyncOpenAI(**client_kwargs)

        return self.openai_client

    def _get_nested_call_instructions(self) -> str:
        """
        System instructions for nested LLM calls.

        Different from main conversation - simpler, focused on data processing.

        Returns:
            System instructions string
        """
        return """You are processing a delegated subtask from a parent conversation.

You will receive:
- One or more context sections (data, text, etc.)
- A specific task to complete

Complete the task based on the provided context. Be direct and concise.

You may use <helpers> blocks if computation is needed, but prefer to return
results directly when possible."""

    def _result(self, *args: Any) -> None:
        """
        Collect results from helper execution.

        Called by user code like: result(value)

        Args:
            *args: Values to collect as results
        """
        for arg in args:
            self._results.append(arg)

    def _capture_figures_for_user(self) -> list[ImageContent]:
        """
        Capture current matplotlib figures for use in user code.

        This allows helper code to explicitly capture figures and pass them
        to llm_call() for analysis:

        Example:
            plt.plot(data)
            plt.title("Sales over time")

            figures = capture_figures()
            analysis = llm_call(figures, "Analyze this chart and explain the trend")

        Returns:
            List of ImageContent objects (one per figure)
        """
        # Capture figures and store them so they're not lost
        self._captured_figures = self._capture_matplotlib_figures()
        return self._captured_figures

    def _capture_matplotlib_figures(self) -> list[ImageContent]:
        """
        Capture any open matplotlib figures as ImageContent objects.

        Returns:
            List of ImageContent objects with base64 encoded PNG images
        """
        try:
            import matplotlib.pyplot as plt
            import base64
            from io import BytesIO

            figures = []
            # Get all figure numbers
            fig_nums = plt.get_fignums()

            if not fig_nums:
                return []

            logger.info(f"Capturing {len(fig_nums)} matplotlib figure(s)")

            for fig_num in fig_nums:
                try:
                    fig = plt.figure(fig_num)
                    logger.info(f"Processing figure {fig_num}, axes: {len(fig.axes)}")

                    # Save figure to BytesIO buffer
                    buf = BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                    buf.seek(0)

                    # Encode as base64
                    img_data = buf.read()
                    logger.info(f"Figure {fig_num} saved, size: {len(img_data)} bytes")
                    img_base64 = base64.b64encode(img_data).decode("utf-8")

                    # Create ImageContent object
                    img_content = ImageContent(image_data=img_base64, mime_type="image/png")
                    figures.append(img_content)

                    buf.close()
                    logger.info(f"Successfully captured figure {fig_num}")
                except Exception as fig_error:
                    logger.error(f"Error capturing figure {fig_num}: {fig_error}", exc_info=True)

            # Close all figures to free memory
            try:
                plt.close("all")
                logger.info(f"Closed all matplotlib figures, returning {len(figures)} captured images")
            except Exception as close_error:
                logger.error(f"Error closing figures: {close_error}", exc_info=True)

            return figures

        except ImportError:
            # matplotlib not available
            logger.debug("matplotlib not available")
            return []
        except Exception as e:
            logger.error(f"Error capturing matplotlib figures: {e}", exc_info=True)
            return []

    def _write_file(self, filename: str, content: Any) -> str:
        """
        Write content to file in conversation directory.

        Args:
            filename: Basename only (e.g., "data.csv", "plot.png")
            content: Content to write. Can be:
                     - str: Text content
                     - list[Content]: Multiple content objects
                     - ImageContent: Image data
                     - matplotlib figure: Figure with savefig() method
                     - list/dict: Will be converted to JSON
                     - bytes: Binary data

        Returns:
            HTTP URL to access the file (e.g., "http://localhost:5000/files/{conv_id}/data.csv")

        Raises:
            RuntimeError: If filename contains path separators (security)
            RuntimeError: If conversation_id not available (no context)
        """
        from sabre.common.execution_context import get_execution_context
        import json
        import base64

        # Step 1: Security - basename only
        if os.path.basename(filename) != filename:
            raise RuntimeError(
                f"write_file() requires basename only (got '{filename}'). Use 'data.csv', not 'path/to/data.csv'"
            )

        # Step 2: Get session ID from context
        ctx = get_execution_context()
        if not ctx or not ctx.session_id:
            raise RuntimeError("write_file() requires execution context with session_id")

        session_id = ctx.session_id

        # Step 3: Create directory structure using session-based paths
        from sabre.common.paths import get_session_files_dir

        files_dir = get_session_files_dir(session_id)
        files_dir.mkdir(parents=True, exist_ok=True)

        file_path = files_dir / filename

        # Step 4: Detect mode (binary vs text)
        is_binary = False

        # Check file extension
        binary_extensions = (
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".webp",
            ".bmp",
            ".pdf",
            ".zip",
            ".tar",
            ".gz",
            ".mp4",
            ".mp3",
        )
        if filename.lower().endswith(binary_extensions):
            is_binary = True

        # Check content type
        if isinstance(content, ImageContent):
            is_binary = True
        elif hasattr(content, "savefig"):  # matplotlib figure
            is_binary = True
        elif isinstance(content, bytes):
            is_binary = True

        mode = "wb" if is_binary else "w"

        # Step 5: Handle different content types
        try:
            with open(file_path, mode) as f:
                if isinstance(content, ImageContent):
                    # Decode base64 image data
                    img_bytes = base64.b64decode(content.image_data)
                    f.write(img_bytes)
                    logger.info(f"Wrote ImageContent to {file_path} ({len(img_bytes)} bytes)")

                elif hasattr(content, "savefig"):
                    # matplotlib figure
                    content.savefig(f, format="png", bbox_inches="tight", dpi=100)
                    logger.info(f"Saved matplotlib figure to {file_path}")

                elif isinstance(content, list) and all(isinstance(c, Content) for c in content):
                    # List of Content objects - extract text
                    text_parts = [c.get_str() for c in content]
                    f.write("\n\n".join(text_parts))
                    logger.info(f"Wrote {len(content)} Content objects to {file_path}")

                elif isinstance(content, (list, dict)):
                    # JSON-serializable data
                    if is_binary:
                        # Shouldn't happen, but fallback to text mode
                        file_path.write_text(json.dumps(content, indent=2))
                    else:
                        json.dump(content, f, indent=2)
                    logger.info(f"Wrote JSON data to {file_path}")

                elif isinstance(content, bytes):
                    # Binary data
                    f.write(content)
                    logger.info(f"Wrote binary data to {file_path} ({len(content)} bytes)")

                elif isinstance(content, str):
                    # Text content
                    if is_binary:
                        # Write as UTF-8 bytes
                        f.write(content.encode("utf-8"))
                    else:
                        f.write(content)
                    logger.info(f"Wrote text to {file_path} ({len(content)} chars)")

                else:
                    # Unknown type - convert to string
                    text = str(content)
                    if is_binary:
                        f.write(text.encode("utf-8"))
                    else:
                        f.write(text)
                    logger.info(f"Wrote {type(content).__name__} as text to {file_path}")

            # Step 6: Return HTTP URL
            # Use PORT env var or default to 8011 (SABRE's default port)
            port = os.getenv("PORT", "8011")
            url = f"http://localhost:{port}/files/{session_id}/{filename}"
            logger.info(f"File accessible at: {url}")
            return url

        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to write file: {e}")

    def _read_file(self, filename: str) -> Content:
        """
        Read a file from the conversation directory or absolute path.

        Args:
            filename: Basename (searches conversation dir) or full path

        Returns:
            Content object based on file type:
            - Text files (.txt, .md, .csv, etc.) → TextContent
            - Images (.png, .jpg, etc.) → ImageContent
            - PDFs (.pdf) → PdfContent
            - Other binary files → FileContent

        Raises:
            RuntimeError: If file not found
            RuntimeError: If conversation_id not available (for basename)
            RuntimeError: If file cannot be read
        """
        from sabre.common.execution_context import get_execution_context
        from pathlib import Path
        import base64

        # Step 1: Determine path type
        path = Path(filename)

        if path.is_absolute():
            # Full path provided
            file_path = path
            logger.info(f"Reading from absolute path: {file_path}")
        else:
            # Basename - use session files directory
            ctx = get_execution_context()
            if not ctx or not ctx.session_id:
                raise RuntimeError(
                    "read_file() with basename requires execution context with session_id. "
                    "Use absolute path or ensure write_file() was called first."
                )

            session_id = ctx.session_id
            from sabre.common.paths import get_session_files_dir

            files_dir = get_session_files_dir(session_id)
            file_path = files_dir / filename
            logger.info(f"Reading from session files directory: {file_path}")

        # Step 2: Check file exists
        if not file_path.exists():
            raise RuntimeError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise RuntimeError(f"Not a file: {file_path}")

        # Step 3: Detect file type and read accordingly
        extension = file_path.suffix.lower()

        # Text file extensions
        text_extensions = {
            ".txt",
            ".md",
            ".csv",
            ".json",
            ".xml",
            ".yaml",
            ".yml",
            ".py",
            ".js",
            ".ts",
            ".java",
            ".c",
            ".cpp",
            ".h",
            ".go",
            ".rs",
            ".sh",
            ".bash",
            ".zsh",
            ".sql",
            ".html",
            ".css",
            ".scss",
            ".log",
            ".conf",
            ".cfg",
            ".ini",
            ".toml",
        }

        # Image file extensions
        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg"}

        try:
            if extension == ".pdf":
                # PDF file
                with open(file_path, "rb") as f:
                    pdf_data = f.read()
                logger.info(f"Read PDF file: {file_path} ({len(pdf_data)} bytes)")
                return PdfContent(pdf_data=pdf_data, url=str(file_path))

            elif extension in image_extensions:
                # Image file
                with open(file_path, "rb") as f:
                    img_data = f.read()

                # Detect MIME type
                mime_type = f"image/{extension[1:]}"  # .png → image/png
                if extension == ".jpg":
                    mime_type = "image/jpeg"
                elif extension == ".svg":
                    mime_type = "image/svg+xml"

                # Encode as base64
                img_base64 = base64.b64encode(img_data).decode("utf-8")

                logger.info(f"Read image file: {file_path} ({len(img_data)} bytes, {mime_type})")
                return ImageContent(image_data=img_base64, mime_type=mime_type)

            elif extension in text_extensions or extension == "":
                # Text file (or no extension - assume text)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    logger.info(f"Read text file: {file_path} ({len(text)} chars)")
                    return TextContent(text)
                except UnicodeDecodeError:
                    # Not UTF-8 - treat as binary
                    with open(file_path, "rb") as f:
                        binary_data = f.read()
                    logger.info(f"Read binary file (decode failed): {file_path} ({len(binary_data)} bytes)")
                    return FileContent(file_data=binary_data, url=str(file_path), filename=file_path.name)
            else:
                # Unknown extension - treat as binary
                with open(file_path, "rb") as f:
                    binary_data = f.read()
                logger.info(f"Read binary file: {file_path} ({len(binary_data)} bytes)")
                return FileContent(file_data=binary_data, url=str(file_path), filename=file_path.name)

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to read file: {e}")

    def execute(self, code: str) -> ExecutionResult:
        """
        Execute Python code.

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with output and results
        """
        # Clear captured figures from previous execution
        self._captured_figures.clear()

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            # Execute code in namespace
            exec(code, self.namespace)

            # Get captured output
            output = sys.stdout.getvalue()

            # Check if user code already captured figures, otherwise auto-capture
            if self._captured_figures:
                logger.info(f"Using {len(self._captured_figures)} figures captured by user code")
                figures = self._captured_figures
            else:
                # Capture any matplotlib figures before building output
                figures = self._capture_matplotlib_figures()

            # Build result output
            result_output = []
            for r in self._results:
                if hasattr(r, "get_str"):
                    result_output.append(r.get_str())
                else:
                    result_output.append(str(r))

            # Build structured content list
            content = []

            # Add text output if present
            if output.strip():
                content.append(TextContent(output.strip()))

            # Add result outputs as text
            if result_output:
                content.append(TextContent("\n".join(result_output)))

            # Add figure content
            if figures:
                # Add text indicator
                content.append(TextContent(f"\n[Generated {len(figures)} matplotlib figure(s)]"))
                # Add actual image content
                content.extend(figures)

            # Build display output (for terminal/client)
            display_output = output
            if result_output:
                display_output += "\n" + "\n".join(result_output)

            # Add figure indicators to display output
            # NOTE: Only add summary, NOT full base64 data - that goes in content list
            # This prevents huge token consumption when result goes into <helpers_result>
            if figures:
                display_output += f"\n[Generated {len(figures)} matplotlib figure(s)]"
                for i, fig_content in enumerate(figures, 1):
                    # Just indicate image was created - actual image is in content list
                    display_output += f"\n[Figure {i}: PNG image]"

            logger.info(f"Executed code successfully, output length: {len(display_output)}, figures: {len(figures)}")

            return ExecutionResult(
                success=True, output=display_output.strip(), error=None, results=self._results.copy(), content=content
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Execution error: {error_msg}")

            # Still try to capture figures even on error
            try:
                self._capture_matplotlib_figures()
            except:
                pass

            return ExecutionResult(success=False, output=sys.stdout.getvalue(), error=error_msg, results=[], content=[])

        finally:
            # Restore stdout
            sys.stdout = old_stdout
            # Clear results for next execution
            self._results = []
