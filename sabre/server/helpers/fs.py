"""
File system helpers for reading and writing files.

Provides write_file() and read_file() functions for persistent file storage
within conversations.
"""

import os
import logging
import json
import base64
from typing import Any
from pathlib import Path

from sabre.common.models.messages import (
    Content,
    ImageContent,
    TextContent,
    PdfContent,
    FileContent,
)
from sabre.common.execution_context import get_execution_context
from sabre.common.paths import get_session_files_dir

logger = logging.getLogger(__name__)


def write_file(filename: str, content: Any) -> str:
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
        HTTP URL to access the file (e.g., "http://localhost:8011/files/{conv_id}/data.csv")

    Raises:
        RuntimeError: If filename contains path separators (security)
        RuntimeError: If conversation_id not available (no context)
    """
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

        # Step 6: Return HTTP URL (with file path for script mode)
        # Use PORT env var or default to 8011 (SABRE's default port)
        # URL uses session_id for routing via /v1/sessions/{session_id}/files/{filename}
        port = os.getenv("PORT", "8011")
        url = f"http://localhost:{port}/v1/sessions/{session_id}/files/{filename}"

        # Print file path to stdout (useful in script mode where URL doesn't work)
        print(f"[write_file] Wrote: {file_path}")
        logger.info(f"File written to: {file_path}")
        logger.info(f"File accessible at: {url}")

        return url

    except Exception as e:
        logger.error(f"Error writing file {file_path}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to write file: {e}")


def read_file(filename: str) -> Content:
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
    # Step 1: Determine path type
    path = Path(filename)

    if path.is_absolute():
        # Full path provided
        file_path = path
        logger.info(f"Reading from absolute path: {file_path}")
    else:
        # Basename - use session directory
        ctx = get_execution_context()
        if not ctx or not ctx.session_id:
            raise RuntimeError(
                "read_file() with basename requires execution context with session_id. "
                "Use absolute path or ensure write_file() was called first."
            )

        session_id = ctx.session_id
        files_dir = get_session_files_dir(session_id)
        file_path = files_dir / filename
        logger.info(f"Reading from session directory: {file_path}")

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
