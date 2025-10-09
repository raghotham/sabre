"""
Streaming parser for <helpers> blocks.

Parses LLM response tokens as they arrive to extract <helpers> blocks in real-time.
"""

import logging

logger = logging.getLogger(__name__)


class StreamingHelperParser:
    """
    State machine parser for extracting <helpers> blocks from streaming tokens.

    As tokens arrive one by one, this parser detects opening/closing tags
    and extracts the code inside <helpers>...</helpers> blocks.

    States:
    - NORMAL: Not currently inside a <helpers> block
    - IN_HELPERS: Currently accumulating content inside a <helpers> block

    Example:
        parser = StreamingHelperParser()
        parser.feed("Let me ")
        parser.feed("help. ")
        parser.feed("<helpers>")
        parser.feed("print('hello')")
        parser.feed("</helpers>")
        helpers = parser.get_helpers()  # ["print('hello')"]
    """

    def __init__(self):
        """Initialize parser."""
        self.buffer = ""  # Accumulates incoming tokens for tag detection
        self.helpers = []  # Completed helper blocks
        self.current_helper = None  # Currently accumulating helper (or None)

        # Tag strings
        self.open_tag = "<helpers>"
        self.close_tag = "</helpers>"

    def feed(self, token: str):
        """
        Feed a token to the parser.

        Detects <helpers> and </helpers> tags as they appear.

        Args:
            token: Text token from LLM stream
        """
        self.buffer += token

        # Process buffer for tags
        self._process_buffer()

    def _process_buffer(self):
        """
        Process buffer to detect and extract helpers blocks.

        This handles the state machine logic:
        - Look for opening tag
        - If found, start accumulating helper content
        - Look for closing tag
        - If found, save completed helper
        """
        while True:
            # Check if we're currently in a helpers block
            if self.current_helper is None:
                # NORMAL state - look for opening tag
                if self.open_tag in self.buffer:
                    # Found opening tag - split on it
                    parts = self.buffer.split(self.open_tag, 1)
                    # Everything before the tag stays in buffer (normal text)
                    # Everything after starts a new helper
                    self.current_helper = ""
                    self.buffer = parts[1] if len(parts) > 1 else ""
                    logger.debug("StreamingParser: Found opening <helpers> tag")
                else:
                    # No opening tag yet - keep accumulating
                    # But trim buffer to avoid unbounded growth
                    # Keep last 20 chars in case tag is split across tokens
                    if len(self.buffer) > 20:
                        self.buffer = self.buffer[-20:]
                    break
            else:
                # IN_HELPERS state - look for closing tag
                if self.close_tag in self.buffer:
                    # Found closing tag - extract helper content
                    parts = self.buffer.split(self.close_tag, 1)
                    self.current_helper += parts[0]

                    # Save completed helper
                    helper_code = self.current_helper.strip()
                    if helper_code:  # Only save non-empty helpers
                        self.helpers.append(helper_code)
                        logger.debug(f"StreamingParser: Extracted helper block ({len(helper_code)} chars)")

                    # Reset state
                    self.current_helper = None
                    self.buffer = parts[1] if len(parts) > 1 else ""

                    # Continue processing in case there are more tags
                else:
                    # No closing tag yet - accumulate to current helper
                    # Keep last 20 chars in buffer in case tag is split
                    if len(self.buffer) > 20:
                        self.current_helper += self.buffer[:-20]
                        self.buffer = self.buffer[-20:]
                    break

    def finalize(self):
        """
        Finalize parsing (call when stream is complete).

        If we're still in a helpers block, it's incomplete.
        This flushes any remaining buffer content.
        """
        if self.current_helper is not None:
            # We were in a helpers block but never saw closing tag
            # This is an incomplete/malformed helper
            # Add remaining buffer to current helper for logging
            incomplete_content = self.current_helper + self.buffer
            logger.warning(
                f"StreamingParser: Incomplete <helpers> block (no closing tag). "
                f"Content so far ({len(incomplete_content)} chars): {incomplete_content[:500]}"
            )
            # Don't save incomplete helpers - they're malformed

        # Clear buffer
        self.buffer = ""

    def get_helpers(self) -> list[str]:
        """
        Get all parsed helper blocks.

        Returns:
            List of code strings extracted from <helpers> blocks
        """
        return self.helpers.copy()

    def reset(self):
        """Reset parser state."""
        self.buffer = ""
        self.helpers = []
        self.current_helper = None
