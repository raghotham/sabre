"""
Message data models for SABRE.

Simple, clean message representation:
- Content: Base class for different content types
- Message: Base message class
- User, Assistant, System: Role-specific message types
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class ContentType(Enum):
    """Types of content that can be in a message"""

    TEXT = "text"
    IMAGE = "image"
    CODE = "code"
    PDF = "pdf"
    FILE = "file"


@dataclass
class Content:
    """
    Base class for message content.

    All content has a type and raw data.
    """

    type: ContentType
    data: Any

    def get_str(self) -> str:
        """Get string representation of content"""
        return str(self.data)


@dataclass
class TextContent(Content):
    """Text content"""

    def __init__(self, text: str):
        super().__init__(type=ContentType.TEXT, data=text)

    @property
    def text(self) -> str:
        return self.data

    def get_str(self) -> str:
        return self.text


@dataclass
class SearchResult(TextContent):
    """Search result with metadata"""

    url: str = ""
    title: str = ""
    snippet: str = ""
    engine: str = ""

    def __init__(self, url: str = "", title: str = "", snippet: str = "", engine: str = ""):
        # Use snippet as the text content
        super().__init__(text=snippet)
        self.url = url
        self.title = title
        self.snippet = snippet
        self.engine = engine

    def __repr__(self) -> str:
        """Custom repr for better display"""
        return f'SearchResult(url="{self.url}", title="{self.title}", snippet="{self.snippet[:50]}...", engine="{self.engine}")'


@dataclass
class ImageContent(Content):
    """
    Image content (base64, URL, or OpenAI file_id).

    Can represent:
    - Base64 data: image_data="iVBORw0KGgo..."
    - File ID: file_id="file-abc123..." (from OpenAI Files API)
    """

    def __init__(self, image_data: str = None, mime_type: str = "image/png", file_id: str = None):
        """
        Create image content.

        Args:
            image_data: Base64 encoded image data (mutually exclusive with file_id)
            mime_type: MIME type (e.g., "image/png")
            file_id: OpenAI file_id (mutually exclusive with image_data)
        """
        if image_data and file_id:
            raise ValueError("Cannot specify both image_data and file_id")
        if not image_data and not file_id:
            raise ValueError("Must specify either image_data or file_id")

        super().__init__(type=ContentType.IMAGE, data={"image": image_data, "mime_type": mime_type, "file_id": file_id})

    @property
    def image_data(self) -> str:
        return self.data.get("image")

    @property
    def mime_type(self) -> str:
        return self.data.get("mime_type", "image/png")

    @property
    def file_id(self) -> str:
        return self.data.get("file_id")

    @property
    def is_file_reference(self) -> bool:
        """True if this is a file_id reference, False if base64 data"""
        return self.file_id is not None

    def __repr__(self) -> str:
        """Custom repr that doesn't include full base64 (for logging)"""
        if self.file_id:
            return f"ImageContent(file_id={self.file_id!r})"
        elif self.image_data:
            # Check if it's a URL (http://) or base64
            if self.image_data.startswith("http://") or self.image_data.startswith("https://"):
                # Extract file path from URL if it's a local file
                # URL format: http://localhost:8011/files/{conversation_id}/filename.png
                return f"ImageContent(url={self.image_data!r})"
            else:
                # Truncate base64 for logging
                preview = self.image_data[:50] + "..." if len(self.image_data) > 50 else self.image_data
                return f"ImageContent(base64={preview!r} [{len(self.image_data)} chars])"
        else:
            return "ImageContent(empty)"


@dataclass
class CodeContent(Content):
    """Code content with language"""

    def __init__(self, code: str, language: str = "python"):
        super().__init__(type=ContentType.CODE, data={"code": code, "language": language})

    @property
    def code(self) -> str:
        return self.data["code"]

    @property
    def language(self) -> str:
        return self.data["language"]

    def get_str(self) -> str:
        return self.code


@dataclass
class PdfContent(Content):
    """PDF document content."""

    def __init__(self, pdf_data: bytes, url: str = ""):
        super().__init__(type=ContentType.PDF, data={"pdf_data": pdf_data, "url": url})

    @property
    def pdf_data(self) -> bytes:
        return self.data["pdf_data"]

    @property
    def url(self) -> str:
        return self.data.get("url", "")

    def get_str(self) -> str:
        """Extract text from PDF or return summary."""
        # TODO: Implement PDF text extraction using PyPDF2 or similar
        size = len(self.pdf_data)
        url_info = f", url={self.url}" if self.url else ""
        return f"[PDF document: {size} bytes{url_info}]"


@dataclass
class FileContent(Content):
    """Generic file content (binary)."""

    def __init__(self, file_data: bytes, url: str = "", filename: str = ""):
        super().__init__(type=ContentType.FILE, data={"file_data": file_data, "url": url, "filename": filename})

    @property
    def file_data(self) -> bytes:
        return self.data["file_data"]

    @property
    def url(self) -> str:
        return self.data.get("url", "")

    @property
    def filename(self) -> str:
        return self.data.get("filename", "")

    def get_str(self) -> str:
        """Return file info."""
        size = len(self.file_data)
        name_info = f"{self.filename}, " if self.filename else ""
        url_info = f", url={self.url}" if self.url else ""
        return f"[File: {name_info}{size} bytes{url_info}]"


@dataclass
class Message:
    """
    Base message class.

    A message has:
    - role: "user", "assistant", or "system"
    - content: List of Content objects
    """

    role: str
    content: list[Content] = field(default_factory=list)

    def get_str(self) -> str:
        """Get full text of message"""
        return " ".join(c.get_str() for c in self.content)

    def add_content(self, content: Content) -> None:
        """Add content to message"""
        self.content.append(content)


@dataclass
class User(Message):
    """User message"""

    def __init__(self, content: list[Content] | None = None):
        super().__init__(role="user", content=content or [])


@dataclass
class Assistant(Message):
    """
    Assistant message.

    Includes optional response_id for conversation continuation and token usage stats.
    """

    response_id: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    def __init__(
        self,
        content: list[Content] | None = None,
        response_id: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        reasoning_tokens: int = 0,
    ):
        super().__init__(role="assistant", content=content or [])
        self.response_id = response_id
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.reasoning_tokens = reasoning_tokens


@dataclass
class System(Message):
    """System message"""

    def __init__(self, content: list[Content] | None = None):
        super().__init__(role="system", content=content or [])


# Helper functions


def text_message(role: str, text: str) -> Message:
    """Create a simple text message"""
    content_classes = {
        "user": User,
        "assistant": Assistant,
        "system": System,
    }
    msg_class = content_classes.get(role, Message)
    return msg_class(content=[TextContent(text)])


def user_text(text: str) -> User:
    """Create user message with text"""
    return User(content=[TextContent(text)])


def assistant_text(text: str, response_id: str | None = None) -> Assistant:
    """Create assistant message with text"""
    return Assistant(content=[TextContent(text)], response_id=response_id)


def system_text(text: str) -> System:
    """Create system message with text"""
    return System(content=[TextContent(text)])
