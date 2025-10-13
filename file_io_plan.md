# File I/O Implementation Plan

## Overview

Implement `write_file()` and `read_file()` helpers in SABRE to enable persistent file storage and retrieval within conversations. These functions mirror the behavior from old LLMVM but use SABRE's XDG-compliant directory structure.

## Directory Structure

Files are stored in conversation-specific directories:
```
~/.local/share/sabre/files/{conversation_id}/{filename}
```

This structure:
- ✅ Follows XDG Base Directory Specification
- ✅ Isolates files by conversation (privacy/security)
- ✅ Makes cleanup easy (delete conversation folder)
- ✅ Enables HTTP serving via `/files/{conversation_id}/{filename}` endpoint

## Part 1: write_file() Implementation

### Function Signature

```python
def write_file(filename: str, content: Any) -> str:
    """
    Write content to a file in the conversation directory.

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
```

### Implementation Location

**File**: `sabre/server/python_runtime.py`

Add as private method `_write_file()` and expose in namespace:

```python
def reset(self):
    """Reset the runtime namespace."""
    # ... existing code ...

    self.namespace = {
        # ... existing helpers ...
        "write_file": self._write_file,
        "read_file": self._read_file,
    }
```

### Core Implementation

```python
def _write_file(self, filename: str, content: Any) -> str:
    """
    Write content to file in conversation directory.

    Implementation steps:
    1. Validate filename is basename only (security)
    2. Get conversation_id from ExecutionContext
    3. Create conversation files directory if needed
    4. Detect binary vs text mode
    5. Handle different content types
    6. Write file
    7. Return HTTP URL
    """
    from sabre.common.execution_context import get_execution_context
    from sabre.common.models.messages import ImageContent
    from pathlib import Path
    import json
    import base64

    # Step 1: Security - basename only
    if os.path.basename(filename) != filename:
        raise RuntimeError(
            f"write_file() requires basename only (got '{filename}'). "
            "Use 'data.csv', not 'path/to/data.csv'"
        )

    # Step 2: Get conversation ID from context
    ctx = get_execution_context()
    if not ctx or not ctx.conversation_id:
        raise RuntimeError("write_file() requires conversation context")

    conversation_id = ctx.conversation_id

    # Step 3: Create directory structure
    files_dir = Path.home() / ".local" / "share" / "sabre" / "files" / conversation_id
    files_dir.mkdir(parents=True, exist_ok=True)

    file_path = files_dir / filename

    # Step 4: Detect mode (binary vs text)
    is_binary = False

    # Check file extension
    binary_extensions = (
        '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp',
        '.pdf', '.zip', '.tar', '.gz', '.mp4', '.mp3'
    )
    if filename.lower().endswith(binary_extensions):
        is_binary = True

    # Check content type
    if isinstance(content, ImageContent):
        is_binary = True
    elif hasattr(content, 'savefig'):  # matplotlib figure
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

            elif hasattr(content, 'savefig'):
                # matplotlib figure
                content.savefig(f, format='png', bbox_inches='tight', dpi=100)
                logger.info(f"Saved matplotlib figure to {file_path}")

            elif isinstance(content, list) and all(isinstance(c, Content) for c in content):
                # List of Content objects - extract text
                text_parts = [c.get_str() for c in content]
                f.write('\n\n'.join(text_parts))
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
                    f.write(content.encode('utf-8'))
                else:
                    f.write(content)
                logger.info(f"Wrote text to {file_path} ({len(content)} chars)")

            else:
                # Unknown type - convert to string
                text = str(content)
                if is_binary:
                    f.write(text.encode('utf-8'))
                else:
                    f.write(text)
                logger.info(f"Wrote {type(content).__name__} as text to {file_path}")

        # Step 6: Return HTTP URL
        # Assume server runs on localhost:5000 (get from config in production)
        port = os.getenv("PORT", "5000")
        url = f"http://localhost:{port}/files/{conversation_id}/{filename}"
        logger.info(f"File accessible at: {url}")
        return url

    except Exception as e:
        logger.error(f"Error writing file {file_path}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to write file: {e}")
```

### HTTP Endpoint

The files need to be accessible via HTTP. Add to `sabre/server/routes.py`:

```python
@app.get("/files/{conversation_id}/{filename}")
async def get_file(conversation_id: str, filename: str):
    """
    Serve files written by write_file().

    Security:
    - Only serves files from conversation directories
    - Validates filename is basename only
    - Returns 404 for non-existent files
    """
    from pathlib import Path
    from fastapi.responses import FileResponse
    import os

    # Security: basename only
    if os.path.basename(filename) != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Build file path
    files_dir = Path.home() / ".local" / "share" / "sabre" / "files" / conversation_id
    file_path = files_dir / filename

    # Check file exists
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Serve file
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )
```

## Part 2: read_file() Implementation

### Function Signature

```python
def read_file(filename: str) -> Content:
    """
    Read a file from the conversation directory.

    Args:
        filename: Basename or full path to file.
                  If basename, searches conversation directory.
                  If full path, reads from that location.

    Returns:
        Content object based on file type:
        - Text files (.txt, .md, .csv, .json, .py, etc.) → TextContent
        - Images (.png, .jpg, .webp, etc.) → ImageContent
        - PDFs (.pdf) → PdfContent
        - Other binary files → FileContent

    Raises:
        RuntimeError: If file not found
        RuntimeError: If conversation_id not available (for basename)
        RuntimeError: If file cannot be read
    """
```

### Implementation

```python
def _read_file(self, filename: str) -> Content:
    """
    Read file from conversation directory or absolute path.

    Implementation steps:
    1. Determine if filename is basename or full path
    2. If basename, build path from conversation directory
    3. If full path, use as-is (with security checks)
    4. Detect file type from extension
    5. Read file with appropriate handler
    6. Return appropriate Content object
    """
    from sabre.common.execution_context import get_execution_context
    from sabre.common.models.messages import ImageContent, TextContent, FileContent, PdfContent
    from pathlib import Path
    import base64

    # Step 1: Determine path type
    path = Path(filename)

    if path.is_absolute():
        # Full path provided
        file_path = path
        logger.info(f"Reading from absolute path: {file_path}")
    else:
        # Basename - use conversation directory
        ctx = get_execution_context()
        if not ctx or not ctx.conversation_id:
            raise RuntimeError(
                "read_file() with basename requires conversation context. "
                "Use absolute path or ensure write_file() was called first."
            )

        conversation_id = ctx.conversation_id
        files_dir = Path.home() / ".local" / "share" / "sabre" / "files" / conversation_id
        file_path = files_dir / filename
        logger.info(f"Reading from conversation directory: {file_path}")

    # Step 2: Check file exists
    if not file_path.exists():
        raise RuntimeError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise RuntimeError(f"Not a file: {file_path}")

    # Step 3: Detect file type and read accordingly
    extension = file_path.suffix.lower()

    # Text file extensions
    text_extensions = {
        '.txt', '.md', '.csv', '.json', '.xml', '.yaml', '.yml',
        '.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.go', '.rs',
        '.sh', '.bash', '.zsh', '.sql', '.html', '.css', '.scss',
        '.log', '.conf', '.cfg', '.ini', '.toml'
    }

    # Image file extensions
    image_extensions = {
        '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.svg'
    }

    try:
        if extension == '.pdf':
            # PDF file
            with open(file_path, 'rb') as f:
                pdf_data = f.read()
            logger.info(f"Read PDF file: {file_path} ({len(pdf_data)} bytes)")
            return PdfContent(pdf_data=pdf_data, url=str(file_path))

        elif extension in image_extensions:
            # Image file
            with open(file_path, 'rb') as f:
                img_data = f.read()

            # Detect MIME type
            mime_type = f"image/{extension[1:]}"  # .png → image/png
            if extension == '.jpg':
                mime_type = 'image/jpeg'
            elif extension == '.svg':
                mime_type = 'image/svg+xml'

            # Encode as base64
            img_base64 = base64.b64encode(img_data).decode('utf-8')

            logger.info(f"Read image file: {file_path} ({len(img_data)} bytes, {mime_type})")
            return ImageContent(image_data=img_base64, mime_type=mime_type)

        elif extension in text_extensions or extension == '':
            # Text file (or no extension - assume text)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                logger.info(f"Read text file: {file_path} ({len(text)} chars)")
                return TextContent(full_text=text)
            except UnicodeDecodeError:
                # Not UTF-8 - treat as binary
                with open(file_path, 'rb') as f:
                    binary_data = f.read()
                logger.info(f"Read binary file (decode failed): {file_path} ({len(binary_data)} bytes)")
                return FileContent(file_data=binary_data, url=str(file_path), filename=file_path.name)
        else:
            # Unknown extension - treat as binary
            with open(file_path, 'rb') as f:
                binary_data = f.read()
            logger.info(f"Read binary file: {file_path} ({len(binary_data)} bytes)")
            return FileContent(file_data=binary_data, url=str(file_path), filename=file_path.name)

    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to read file: {e}")
```

### Missing Content Types

Need to add `PdfContent` and `FileContent` to `sabre/common/models/messages.py`:

```python
@dataclass
class PdfContent(Content):
    """PDF document content."""
    pdf_data: bytes
    url: str = ""

    def get_str(self) -> str:
        """Extract text from PDF."""
        # TODO: Implement PDF text extraction using PyPDF2 or similar
        return f"[PDF document: {len(self.pdf_data)} bytes, url={self.url}]"


@dataclass
class FileContent(Content):
    """Generic file content (binary)."""
    file_data: bytes
    url: str = ""
    filename: str = ""

    def get_str(self) -> str:
        """Return file info."""
        return f"[File: {self.filename}, {len(self.file_data)} bytes, url={self.url}]"
```

## Part 3: Integration & Testing

### Update System Prompt

Add to `sabre/prompts/system.prompt`:

```markdown
## File I/O Functions

You can read and write files within the conversation:

### write_file(filename, content) -> str

Write content to a file in the conversation directory.

**Arguments**:
- `filename` (str): Basename only (e.g., "data.csv", "plot.png")
- `content` (Any): Content to write:
  - str: Text content
  - ImageContent: Image data
  - matplotlib figure: Chart/plot
  - list/dict: Will be saved as JSON

**Returns**: HTTP URL to access the file

**Example**:
```python
# Save analysis results
url = write_file("results.txt", "Analysis: Sales increased 23%")
result(f"Saved to {url}")

# Save plot
plt.plot(data)
url = write_file("chart.png", plt.gcf())
result(f"Chart saved to {url}")

# Save JSON data
url = write_file("data.json", {"revenue": 1000, "profit": 200})
```

### read_file(filename) -> Content

Read a file from the conversation directory.

**Arguments**:
- `filename` (str): Basename (searches conversation dir) or full path

**Returns**: Content object (TextContent, ImageContent, PdfContent, or FileContent)

**Example**:
```python
# Read previous analysis
content = read_file("results.txt")
print(content.get_str())

# Read image for processing
image = read_file("screenshot.png")
analysis = llm_call([image], "What's in this image?")

# Read with full path
config = read_file("/etc/myapp/config.json")
```

**Common Pattern - Save and Reuse**:
```python
# Generate and save
plt.plot(sales_data)
url = write_file("sales.png", plt.gcf())

# Read back later
chart = read_file("sales.png")
insights = llm_call([chart], "What trends do you see?")
```
```

### Test Cases

Create `tests/test_file_io.py`:

```python
"""Test write_file() and read_file() helpers."""

import os
import base64
from pathlib import Path
from sabre.server.python_runtime import PythonRuntime
from sabre.common.models.messages import ImageContent, TextContent
from sabre.common.execution_context import set_execution_context


def test_write_read_text_file():
    """Test writing and reading text files."""
    runtime = PythonRuntime()

    # Set up context
    set_execution_context(
        event_callback=None,
        tree=None,
        tree_context={},
        conversation_id="test-conv-1"
    )

    # Write text file
    url = runtime._write_file("test.txt", "Hello, world!")
    assert "test-conv-1" in url
    assert "test.txt" in url

    # Read it back
    content = runtime._read_file("test.txt")
    assert isinstance(content, TextContent)
    assert content.get_str() == "Hello, world!"

    # Cleanup
    files_dir = Path.home() / ".local" / "share" / "sabre" / "files" / "test-conv-1"
    (files_dir / "test.txt").unlink()


def test_write_read_json_file():
    """Test writing and reading JSON files."""
    runtime = PythonRuntime()

    set_execution_context(None, None, {}, "test-conv-2")

    # Write JSON
    data = {"name": "Alice", "age": 30}
    url = runtime._write_file("data.json", data)

    # Read it back
    content = runtime._read_file("data.json")
    assert isinstance(content, TextContent)
    assert "Alice" in content.get_str()

    # Cleanup
    files_dir = Path.home() / ".local" / "share" / "sabre" / "files" / "test-conv-2"
    (files_dir / "data.json").unlink()


def test_write_read_image():
    """Test writing and reading images."""
    runtime = PythonRuntime()

    set_execution_context(None, None, {}, "test-conv-3")

    # Create test image
    img_data = b"\x89PNG\r\n\x1a\n"  # PNG header
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    img_content = ImageContent(image_data=img_base64, mime_type="image/png")

    # Write image
    url = runtime._write_file("test.png", img_content)

    # Read it back
    content = runtime._read_file("test.png")
    assert isinstance(content, ImageContent)
    assert content.mime_type == "image/png"

    # Cleanup
    files_dir = Path.home() / ".local" / "share" / "sabre" / "files" / "test-conv-3"
    (files_dir / "test.png").unlink()


def test_security_no_path_traversal():
    """Test that path traversal is blocked."""
    runtime = PythonRuntime()

    set_execution_context(None, None, {}, "test-conv-4")

    # Try to write with path separator
    try:
        runtime._write_file("../etc/passwd", "hacker")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "basename only" in str(e).lower()


def test_read_absolute_path():
    """Test reading from absolute path."""
    runtime = PythonRuntime()

    set_execution_context(None, None, {}, "test-conv-5")

    # Create temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Absolute path test")
        temp_path = f.name

    try:
        # Read with absolute path
        content = runtime._read_file(temp_path)
        assert isinstance(content, TextContent)
        assert content.get_str() == "Absolute path test"
    finally:
        os.unlink(temp_path)
```

## Part 4: Usage Examples

### Example 1: Save Analysis Results

```python
# User: "Analyze this sales data and save a summary report"

<helpers>
# Load and analyze data
data = download_csv("https://example.com/sales.csv")
total_sales = data['revenue'].sum()
avg_sale = data['revenue'].mean()

# Generate report
report = f"""
Sales Analysis Report
====================

Total Sales: ${total_sales:,.2f}
Average Sale: ${avg_sale:,.2f}
Number of Transactions: {len(data)}

Top Products:
{data.groupby('product')['revenue'].sum().sort_values(ascending=False).head(5)}
"""

# Save report
url = write_file("sales_report.txt", report)
result(f"Report saved to {url}")
</helpers>
```

### Example 2: Generate and Save Charts

```python
# User: "Create a sales trend chart and save it"

<helpers>
import matplotlib.pyplot as plt

# Create chart
plt.figure(figsize=(10, 6))
plt.plot(dates, sales)
plt.title("Sales Trend - Q4 2024")
plt.xlabel("Date")
plt.ylabel("Sales ($)")

# Save chart
url = write_file("sales_trend.png", plt.gcf())
result(f"Chart saved to {url}")
</helpers>
```

### Example 3: Read Previous Analysis

```python
# User: "Compare today's results with the previous report"

<helpers>
# Read previous report
prev_report = read_file("sales_report.txt")
prev_data = prev_report.get_str()

# Get today's data
today_sales = calculate_today_sales()

# Compare
comparison = llm_call(
    [prev_report, f"Today's sales: ${today_sales}"],
    "Compare today's results with the previous report. Are we up or down?"
)

result(comparison)
</helpers>
```

### Example 4: Multi-Step Workflow

```python
# User: "Download the data, analyze it, save results, and create a summary"

<helpers>
# Step 1: Download
raw_data = download_csv("https://example.com/data.csv")
url1 = write_file("raw_data.csv", raw_data.to_csv(index=False))
result(f"Saved raw data to {url1}")

# Step 2: Analyze
analysis = raw_data.describe()
url2 = write_file("analysis.txt", str(analysis))
result(f"Saved analysis to {url2}")

# Step 3: Visualize
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.hist(raw_data['value'])
plt.title("Distribution")

plt.subplot(2, 2, 2)
plt.plot(raw_data['date'], raw_data['value'])
plt.title("Trend")

url3 = write_file("charts.png", plt.gcf())
result(f"Saved charts to {url3}")

# Step 4: Generate summary with LLM
chart_img = read_file("charts.png")
summary = llm_call(
    [analysis, chart_img],
    "Create an executive summary of this analysis with key insights"
)

url4 = write_file("summary.txt", summary)
result(f"Saved summary to {url4}")
result(f"\nAll analysis files saved. View at:\n- {url1}\n- {url2}\n- {url3}\n- {url4}")
</helpers>
```

## Part 5: Implementation Checklist

### Phase 1: Core Implementation
- [ ] Add `PdfContent` and `FileContent` to `sabre/common/models/messages.py`
- [ ] Implement `_write_file()` in `sabre/server/python_runtime.py`
- [ ] Implement `_read_file()` in `sabre/server/python_runtime.py`
- [ ] Add both functions to runtime namespace in `reset()`
- [ ] Add HTTP endpoint `/files/{conversation_id}/{filename}` to `sabre/server/routes.py`

### Phase 2: Testing
- [ ] Create `tests/test_file_io.py` with test cases
- [ ] Test text files (read/write)
- [ ] Test JSON files
- [ ] Test images (ImageContent)
- [ ] Test security (path traversal prevention)
- [ ] Test absolute paths
- [ ] Test matplotlib figures
- [ ] Run full test suite: `uv run pytest tests/test_file_io.py`

### Phase 3: Documentation
- [ ] Update `sabre/prompts/system.prompt` with write_file() and read_file() docs
- [ ] Add usage examples to system prompt
- [ ] Update CLAUDE.md with file I/O patterns

### Phase 4: Integration Testing
- [ ] Test end-to-end workflow with actual LLM
- [ ] Test file persistence across conversation
- [ ] Test HTTP file serving
- [ ] Test with various file types (PDF, CSV, images, text)

### Phase 5: Edge Cases
- [ ] Handle file name conflicts (overwrite vs error)
- [ ] Handle large files (streaming?)
- [ ] Handle disk full errors
- [ ] Add file size limits
- [ ] Add directory size limits per conversation

## Future Enhancements

1. **File versioning** - Keep history of file modifications
2. **File listing** - `list_files()` to see all files in conversation
3. **File deletion** - `delete_file(filename)` for cleanup
4. **File metadata** - Track creation time, size, type
5. **Compression** - Auto-compress large files
6. **Cloud storage** - Sync to S3/GCS for persistence
7. **File sharing** - Generate shareable links
8. **File templates** - Pre-populate with templates
9. **File search** - `search_files(pattern)` to find files
10. **File conversion** - Auto-convert formats (CSV↔JSON, PNG↔JPEG)
