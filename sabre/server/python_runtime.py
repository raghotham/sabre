"""
Python Runtime for executing <helpers> blocks.

Provides isolated execution environment with helper functions.
"""
import sys
import io
import os
import logging
import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING, Union, Type

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
from sabre.common.utils.prompt_loader import PromptLoader
from sabre.common.models.messages import Content, ImageContent, TextContent

if TYPE_CHECKING:
    from sabre.common.executors.response import ResponseExecutor

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

    def __init__(self):
        """Initialize runtime with clean namespace."""
        self.orchestrator = None  # Set by orchestrator after init
        self.openai_client = None  # Lazy init in _llm_call_async

        # Configure matplotlib to use non-interactive backend
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            logger.info("Configured matplotlib to use Agg backend")
        except ImportError:
            logger.debug("matplotlib not available")

        self.reset()

    def reset(self):
        """Reset the runtime namespace."""
        # Results collected via result() calls
        self._results = []

        # Initialize helpers that need runtime context
        # Use lambda to get orchestrator dynamically (set later via set_orchestrator)
        llm_call = LLMCall(lambda: self.orchestrator, self._get_openai_client)
        llm_bind = LLMBind(lambda: self.orchestrator, self._get_openai_client, lambda: self.namespace)
        coerce = Coerce(lambda: self.orchestrator, self._get_openai_client)
        llm_list_bind = LLMListBind(lambda: self.orchestrator, self._get_openai_client)
        pandas_bind = PandasBind(lambda: self.orchestrator, self._get_openai_client)

        # Build namespace with helpers
        self.namespace = {
            # Class-based helpers (stateless)
            'Bash': Bash,
            'Search': Search,
            'Web': Web,
            'download': download,
            'download_csv': download_csv,

            # Instance-based helpers (need runtime context)
            'llm_call': llm_call,
            'llm_bind': llm_bind,
            'coerce': coerce,
            'llm_list_bind': llm_list_bind,
            'pandas_bind': pandas_bind,
            'result': self._result,
            'capture_figures': self._capture_figures_for_user,

            # Matplotlib helpers
            'matplotlib_to_image': matplotlib_to_image,
            'generate_graph_image': generate_graph_image,

            # Standard library (minimal set)
            'print': print,
            'pd': pd,  # pandas for user code
        }

        # Add datetime module
        import datetime
        self.namespace['datetime'] = datetime

        # Storage for captured figures
        self._captured_figures: list[ImageContent] = []

        # Add matplotlib if available
        try:
            import matplotlib.pyplot as plt
            self.namespace['plt'] = plt
            logger.debug("Added matplotlib.pyplot to namespace")
        except ImportError:
            logger.debug("matplotlib not available, skipping")

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
        return get_helper_signatures(self.namespace)

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

            if base_url:
                logger.info(f"Creating OpenAI client with custom base URL: {base_url}")
                self.openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            else:
                self.openai_client = AsyncOpenAI(api_key=api_key)

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
                    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                    buf.seek(0)

                    # Encode as base64
                    img_data = buf.read()
                    logger.info(f"Figure {fig_num} saved, size: {len(img_data)} bytes")
                    img_base64 = base64.b64encode(img_data).decode('utf-8')

                    # Create ImageContent object
                    img_content = ImageContent(image_data=img_base64, mime_type="image/png")
                    figures.append(img_content)

                    buf.close()
                    logger.info(f"Successfully captured figure {fig_num}")
                except Exception as fig_error:
                    logger.error(f"Error capturing figure {fig_num}: {fig_error}", exc_info=True)

            # Close all figures to free memory
            try:
                plt.close('all')
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
                if hasattr(r, 'get_str'):
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
                success=True,
                output=display_output.strip(),
                error=None,
                results=self._results.copy(),
                content=content
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Execution error: {error_msg}")

            # Still try to capture figures even on error
            try:
                self._capture_matplotlib_figures()
            except:
                pass

            return ExecutionResult(
                success=False,
                output=sys.stdout.getvalue(),
                error=error_msg,
                results=[],
                content=[]
            )

        finally:
            # Restore stdout
            sys.stdout = old_stdout
            # Clear results for next execution
            self._results = []
