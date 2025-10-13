# Parse Paths Feature Plan

## Overview

The `-p` / `--path` option from old LLMVM allows users to attach files to their messages, automatically detecting file types and formatting them appropriately for the LLM.

## Example Usage

```bash
llmvm -p docs/turnbull-speech.pdf "what is Malcolm Turnbull advocating for?"
```

This would:
1. Read `docs/turnbull-speech.pdf`
2. Create a `User(PdfContent(...))` message
3. Add the question as another `User(TextContent(...))` message
4. Send both to the LLM

## How It Works in Old LLMVM

### 1. CLI Option Definition

Location: `llmvm/client/cli.py`

```python
@click.option(
    "--path",
    "-p",
    callback=parse_path,
    required=False,
    multiple=True,
    help='Adds a single file, multiple files, directory of files, glob, or url to the User message stack. If using a glob, put inside "" quotes.',
)
```

### 2. Path Parsing (`parse_path()`)

Location: `llmvm/client/parsing.py`, lines 169-233

**Features**:
- Single files, directories, globs, URLs
- Brace expansion (`{a,b}.py`)
- Exclusions (prefix with `!`)
- Recursive directory walking
- HTTP/HTTPS URLs

**Algorithm**:
1. Normalize input to flat list of tokens (handles tuples, lists, strings)
2. Use `shlex.split()` to honor quotes/backslashes like POSIX shell
3. Expand brace globs (`{a,b}.py` → `a.py`, `b.py`)
4. For each token:
   - Expand `~` or `~user` paths
   - Handle exclusions (tokens starting with `!`)
   - Pass through HTTP/HTTPS URLs unchanged
   - Recursively walk directories
   - Handle plain files
   - Expand glob patterns (`*.py`, `**/*.py`)
   - Keep literal tokens that don't match
5. Apply exclusions and deduplicate while preserving order

### 3. File to Message Conversion (`get_path_as_messages()`)

Location: `llmvm/client/parsing.py`, lines 388-497

**Supported File Types**:

| File Type | Content Type | Processing |
|-----------|-------------|------------|
| **PDFs** (`.pdf`) | `PdfContent` | Read binary, create `User(PdfContent(f.read(), url=path))` |
| **Images** (`.jpg`, `.png`, `.webp`) | `ImageContent` | Read binary, resize/compress with `Helpers.load_resize_save()`, create `User(ImageContent(...))` |
| **Markdown** (`.md`) | `MarkdownContent` | Read text, create `User(MarkdownContent([TextContent(...)], url=path))` |
| **HTML** (`.htm`, `.html`) | `MarkdownContent` | Read text, convert to markdown via `WebHelpers.convert_html_to_markdown()`, create `User(MarkdownContent([TextContent(...)], url=path))` |
| **Other text files** | `FileContent` | Read as UTF-8 text, create `User(FileContent(content.encode('utf-8'), url=path))` |
| **URLs** (http/https) | (varies) | Download with requests library, save to tempfile, detect type from extension, process accordingly |

**URL Handling** (lines 396-417):
- Uses `User-Agent` header to avoid 403 errors
- Downloads content to temporary file
- Detects file type from extension
- Processes as appropriate file type

**Error Handling**:
- Skips directories when processing
- Raises `ValueError` for non-UTF8 files that aren't PDFs or images
- Filters by `allowed_file_types` if provided

## Implementation Notes

### Key Dependencies

From `llmvm/client/parsing.py`:
```python
from llmvm.common.objects import (
    Content,
    ImageContent,
    MarkdownContent,
    MessageModel,
    PdfContent,
    FileContent,
    SessionThreadModel,
    TextContent,
    User,
    Message,
)
from llmvm.common.helpers import Helpers
```

### Helper Functions Used

1. **`Helpers.glob_brace()`** - Expands brace patterns
2. **`Helpers.is_glob_recursive()`** - Checks if glob should be recursive
3. **`Helpers.is_glob_pattern()`** - Checks if string is a glob pattern
4. **`Helpers.classify_image()`** - Detects image MIME type
5. **`Helpers.load_resize_save()`** - Resizes/compresses images
6. **`Helpers.late_bind()`** - Dynamically loads WebHelpers for HTML→markdown conversion

### Content Type Classes

These need to be compatible with SABRE's content model:

**Old LLMVM**:
- `TextContent` - Plain text
- `ImageContent` - Images (JPEG, PNG, WebP)
- `PdfContent` - PDF documents
- `FileContent` - Generic text files
- `MarkdownContent` - Markdown documents (wraps TextContent)

**SABRE** (from `sabre/common/models/messages.py`):
- `TextContent` - Plain text with `full_text` field
- `ImageContent` - Images with `image_data` (base64) and `mime_type`
- (Need to add `PdfContent`, `FileContent`, `MarkdownContent`)

## Migration Plan to SABRE

### Phase 1: Add Missing Content Types

Add to `sabre/common/models/messages.py`:
```python
@dataclass
class PdfContent(Content):
    """PDF document content."""
    pdf_data: bytes
    url: str = ""

    def get_str(self) -> str:
        # Extract text from PDF using PyPDF2
        pass

@dataclass
class FileContent(Content):
    """Generic file content."""
    file_data: bytes
    url: str = ""
    filename: str = ""

    def get_str(self) -> str:
        return self.file_data.decode('utf-8')

@dataclass
class MarkdownContent(Content):
    """Markdown document content."""
    content: list[Content]
    url: str = ""

    def get_str(self) -> str:
        return '\n'.join(c.get_str() for c in self.content)
```

### Phase 2: Port Helper Functions

Create `sabre/common/path_helpers.py` with:
- `glob_brace()` - Brace expansion
- `is_glob_recursive()` - Recursive glob detection
- `is_glob_pattern()` - Glob pattern detection
- `classify_image()` - Image MIME type detection
- `load_resize_save()` - Image resizing

### Phase 3: Port Parsing Functions

Create `sabre/client/file_parsing.py` with:
- `parse_path()` - Path parsing with glob/brace expansion
- `get_path_as_messages()` - File to message conversion

### Phase 4: Update CLI

Update `sabre/cli.py` (or new client CLI):
```python
parser.add_argument(
    "--path",
    "-p",
    action="append",
    help='Attach file(s) to message. Supports files, directories, globs, URLs.',
)
```

Process paths before sending message:
```python
if args.path:
    from sabre.client.file_parsing import parse_path, get_path_as_messages

    paths = []
    for p in args.path:
        paths.extend(parse_path(None, None, p))

    file_messages = get_path_as_messages(paths)
    # Add file_messages to the message stack
```

### Phase 5: Update Message API

Ensure `/message` endpoint accepts array of content items:
```json
{
  "message": [
    {"type": "pdf", "data": "...", "url": "docs/speech.pdf"},
    {"type": "text", "text": "What is this about?"}
  ]
}
```

## Future Enhancements

1. **Streaming large files** - Don't load entire PDFs into memory
2. **File type validation** - Reject dangerous file types
3. **Size limits** - Prevent uploading huge files
4. **Caching** - Cache downloaded URLs
5. **Progress indicators** - Show download/processing progress
6. **OCR support** - Extract text from images
7. **Archive support** - Extract files from .zip, .tar.gz
8. **Code file syntax** - Syntax highlighting for code files
9. **DOCX/XLSX support** - Microsoft Office documents
10. **Video/audio support** - Frame extraction, transcription

## Testing Plan

1. Test single file attachment
2. Test multiple files with `-p file1.pdf -p file2.txt`
3. Test glob patterns: `-p "docs/*.pdf"`
4. Test brace expansion: `-p "docs/{a,b,c}.pdf"`
5. Test recursive globs: `-p "**/*.py"`
6. Test exclusions: `-p "*.py" -p "!test_*.py"`
7. Test URLs: `-p https://example.com/document.pdf`
8. Test directory recursion: `-p docs/`
9. Test mixed types: PDFs, images, text files together
10. Test error cases: non-existent files, binary files, permission errors
