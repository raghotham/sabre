"""
File loader for SABRE client.

Enables users to attach files to messages using @filepath syntax.

Examples:
    sabre "Analyze this @data.txt"
    sabre "What's in @screenshot.png?"
    sabre "Compare @file1.txt and @file2.json"
    sabre "Review @\"path with spaces.txt\""
"""

import base64
import os
import re
from pathlib import Path
from typing import Tuple

from sabre.common.models.messages import Content, FileContent, ImageContent, TextContent


class FileLoadError(Exception):
    """Error loading file from @filepath reference."""

    pass


class FileLoader:
    """
    Load files referenced in messages using @filepath syntax.

    Pattern matching:
    - @filepath - Simple file path
    - @"path with spaces.txt" - Quoted path for spaces
    - Ignores email addresses like user@domain.com

    File type detection (extension-based):
    - Images: .png, .jpg, .jpeg, .gif, .bmp, .webp → ImageContent (base64)
    - Text: .txt, .md, .py, .json, .csv, .log, etc. → TextContent
    - Other: Binary → FileContent

    Size limits:
    - Text files: 10MB
    - Images: 20MB
    - Other files: 50MB
    """

    # Pattern matches @filepath at word boundaries, supports quoted paths
    # (?:^|(?<=\s)) - Start of string or preceded by whitespace
    # @ - Literal @ sign
    # (?:"([^"]+)"|(\S+)) - Either quoted string or non-whitespace sequence
    FILEPATH_PATTERN = re.compile(r'(?:^|(?<=\s))@(?:"([^"]+)"|(\S+))', re.MULTILINE)

    # Image file extensions
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

    # Text file extensions (common ones)
    TEXT_EXTENSIONS = {
        ".txt",
        ".md",
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".xml",
        ".html",
        ".css",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".rs",
        ".go",
        ".java",
        ".kt",
        ".swift",
        ".rb",
        ".php",
        ".r",
        ".sql",
        ".csv",
        ".log",
    }

    # MIME types for images
    MIME_TYPES = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }

    # Size limits (in bytes)
    MAX_TEXT_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

    def parse_message(self, message: str) -> Tuple[str, list[str]]:
        """
        Parse message for @filepath patterns.

        Args:
            message: User message potentially containing @filepath references

        Returns:
            Tuple of (clean_text, filepaths)
            - clean_text: Message with @ references removed
            - filepaths: List of file paths found in message
        """
        filepaths = []
        clean_text = message

        # Find all matches
        for match in self.FILEPATH_PATTERN.finditer(message):
            # Extract path from either quoted or unquoted group
            quoted_path = match.group(1)
            unquoted_path = match.group(2)
            filepath = quoted_path if quoted_path else unquoted_path

            # Check if this looks like an email address (has @ before it or after it)
            # Already filtered by word boundary in regex, but extra check
            start_pos = match.start()
            if start_pos > 0 and message[start_pos - 1].isalnum():
                # Part of an email like "user@domain.com"
                continue

            filepaths.append(filepath)

            # Remove @ reference from message
            clean_text = clean_text.replace(match.group(0), "", 1)

        # Clean up extra whitespace from removals
        clean_text = " ".join(clean_text.split())

        return clean_text, filepaths

    def resolve_path(self, filepath: str) -> Path:
        """
        Resolve file path to absolute path.

        Args:
            filepath: Relative, absolute, or home path

        Returns:
            Resolved absolute Path object

        Raises:
            FileLoadError: If path cannot be resolved
        """
        # Expand user home directory (~)
        expanded = os.path.expanduser(filepath)

        # Convert to Path object
        path = Path(expanded)

        # Resolve to absolute path (handles relative paths, symlinks)
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as e:
            raise FileLoadError(f"Could not resolve path '{filepath}': {e}")

        # Check if file exists
        if not resolved.exists():
            raise FileLoadError(f"File not found: {filepath}")

        # Check if it's a file (not directory)
        if not resolved.is_file():
            raise FileLoadError(f"Not a file: {filepath}")

        return resolved

    def detect_file_type(self, path: Path) -> str:
        """
        Detect file type based on extension.

        Args:
            path: Path object

        Returns:
            One of: "image", "text", "binary"
        """
        ext = path.suffix.lower()

        if ext in self.IMAGE_EXTENSIONS:
            return "image"
        elif ext in self.TEXT_EXTENSIONS:
            return "text"
        else:
            return "binary"

    def load_file(self, filepath: str) -> Content:
        """
        Load file and create appropriate Content object.

        Args:
            filepath: File path to load

        Returns:
            Content object (TextContent, ImageContent, or FileContent)

        Raises:
            FileLoadError: If file cannot be loaded
        """
        # Resolve path
        path = self.resolve_path(filepath)

        # Check file size
        file_size = path.stat().st_size
        file_type = self.detect_file_type(path)

        # Check size limits
        if file_type == "text" and file_size > self.MAX_TEXT_SIZE:
            raise FileLoadError(
                f"Text file too large: {filepath} ({file_size / 1024 / 1024:.1f}MB, max {self.MAX_TEXT_SIZE / 1024 / 1024}MB)"
            )
        elif file_type == "image" and file_size > self.MAX_IMAGE_SIZE:
            raise FileLoadError(
                f"Image file too large: {filepath} ({file_size / 1024 / 1024:.1f}MB, max {self.MAX_IMAGE_SIZE / 1024 / 1024}MB)"
            )
        elif file_size > self.MAX_FILE_SIZE:
            raise FileLoadError(
                f"File too large: {filepath} ({file_size / 1024 / 1024:.1f}MB, max {self.MAX_FILE_SIZE / 1024 / 1024}MB)"
            )

        # Load based on type
        try:
            if file_type == "text":
                return self._load_text_file(path)
            elif file_type == "image":
                return self._load_image_file(path)
            else:
                return self._load_binary_file(path)
        except UnicodeDecodeError as e:
            raise FileLoadError(f"Invalid UTF-8 encoding in {filepath}: {e}")
        except PermissionError as e:
            raise FileLoadError(f"Permission denied reading {filepath}: {e}")
        except Exception as e:
            raise FileLoadError(f"Error loading {filepath}: {e}")

    def _load_text_file(self, path: Path) -> TextContent:
        """
        Load text file as TextContent.

        Args:
            path: Resolved path to text file

        Returns:
            TextContent with file contents wrapped in markers
        """
        content = path.read_text(encoding="utf-8")

        # Wrap content with file markers for clarity
        wrapped = f"--- File: {path.name} ---\n{content}\n--- End of {path.name} ---"

        return TextContent(wrapped)

    def _load_image_file(self, path: Path) -> ImageContent:
        """
        Load image file as ImageContent (base64).

        Args:
            path: Resolved path to image file

        Returns:
            ImageContent with base64 encoded image data
        """
        # Read image as bytes
        image_bytes = path.read_bytes()

        # Encode as base64
        image_data = base64.b64encode(image_bytes).decode("utf-8")

        # Get MIME type from extension
        mime_type = self.MIME_TYPES.get(path.suffix.lower(), "image/png")

        return ImageContent(image_data=image_data, mime_type=mime_type)

    def _load_binary_file(self, path: Path) -> FileContent:
        """
        Load binary file as FileContent.

        Args:
            path: Resolved path to binary file

        Returns:
            FileContent with file data
        """
        # Read file as bytes
        file_data = path.read_bytes()

        return FileContent(file_data=file_data, filename=path.name)
