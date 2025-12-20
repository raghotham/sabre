"""
Unit tests for FileLoader.

Tests pattern matching, path resolution, file loading, and error handling.
"""

import base64
import os
from pathlib import Path

import pytest

from sabre.client.file_loader import FileLoadError, FileLoader
from sabre.common.models.messages import FileContent, ImageContent, TextContent


class TestFileLoaderPatternMatching:
    """Test @filepath pattern matching"""

    def test_parse_single_file(self):
        """Test parsing single @filepath"""
        loader = FileLoader()
        clean_text, filepaths = loader.parse_message("Analyze @data.txt")

        assert clean_text == "Analyze"
        assert filepaths == ["data.txt"]

    def test_parse_multiple_files(self):
        """Test parsing multiple @filepath references"""
        loader = FileLoader()
        clean_text, filepaths = loader.parse_message("Compare @file1.txt and @file2.json")

        assert clean_text == "Compare and"
        assert filepaths == ["file1.txt", "file2.json"]

    def test_parse_quoted_path(self):
        """Test parsing quoted path with spaces"""
        loader = FileLoader()
        clean_text, filepaths = loader.parse_message('Review @"my document.txt"')

        assert clean_text == "Review"
        assert filepaths == ["my document.txt"]

    def test_parse_ignores_email(self):
        """Test that email addresses are ignored"""
        loader = FileLoader()
        clean_text, filepaths = loader.parse_message("Email user@domain.com about the report")

        assert clean_text == "Email user@domain.com about the report"
        assert filepaths == []

    def test_parse_complex_message(self):
        """Test parsing complex message with multiple references"""
        loader = FileLoader()
        clean_text, filepaths = loader.parse_message(
            'Analyze @data.csv and @"report 2024.pdf" then email results@team.com'
        )

        assert clean_text == "Analyze and then email results@team.com"
        assert filepaths == ["data.csv", "report 2024.pdf"]

    def test_parse_empty_message(self):
        """Test parsing message with no @filepath"""
        loader = FileLoader()
        clean_text, filepaths = loader.parse_message("Just a regular message")

        assert clean_text == "Just a regular message"
        assert filepaths == []

    def test_parse_absolute_path(self):
        """Test parsing absolute path"""
        loader = FileLoader()
        clean_text, filepaths = loader.parse_message("Load @/tmp/data.txt")

        assert clean_text == "Load"
        assert filepaths == ["/tmp/data.txt"]

    def test_parse_home_path(self):
        """Test parsing home directory path"""
        loader = FileLoader()
        clean_text, filepaths = loader.parse_message("Load @~/documents/file.txt")

        assert clean_text == "Load"
        assert filepaths == ["~/documents/file.txt"]


class TestFileLoaderPathResolution:
    """Test path resolution"""

    def test_resolve_absolute_path(self, tmp_path):
        """Test resolving absolute path"""
        # Create temp file
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        loader = FileLoader()
        resolved = loader.resolve_path(str(test_file))

        assert resolved == test_file.resolve()

    def test_resolve_relative_path(self, tmp_path):
        """Test resolving relative path"""
        # Create temp file in current directory
        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            test_file = tmp_path / "test.txt"
            test_file.write_text("content")

            loader = FileLoader()
            resolved = loader.resolve_path("test.txt")

            assert resolved == test_file.resolve()
        finally:
            os.chdir(original_cwd)

    def test_resolve_home_path(self, tmp_path, monkeypatch):
        """Test resolving home directory path"""
        # Create temp file
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Mock home directory
        monkeypatch.setenv("HOME", str(tmp_path))

        loader = FileLoader()
        resolved = loader.resolve_path("~/test.txt")

        assert resolved == test_file.resolve()

    def test_resolve_nonexistent_file(self):
        """Test resolving non-existent file raises error"""
        loader = FileLoader()

        with pytest.raises(FileLoadError, match="Could not resolve path"):
            loader.resolve_path("/nonexistent/file.txt")

    def test_resolve_directory(self, tmp_path):
        """Test resolving directory raises error"""
        # Create directory
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        loader = FileLoader()

        with pytest.raises(FileLoadError, match="Not a file"):
            loader.resolve_path(str(test_dir))


class TestFileLoaderFileTypeDetection:
    """Test file type detection"""

    def test_detect_image_type(self):
        """Test detecting image files"""
        loader = FileLoader()

        assert loader.detect_file_type(Path("test.png")) == "image"
        assert loader.detect_file_type(Path("test.jpg")) == "image"
        assert loader.detect_file_type(Path("test.jpeg")) == "image"
        assert loader.detect_file_type(Path("test.gif")) == "image"
        assert loader.detect_file_type(Path("test.bmp")) == "image"
        assert loader.detect_file_type(Path("test.webp")) == "image"

    def test_detect_text_type(self):
        """Test detecting text files"""
        loader = FileLoader()

        assert loader.detect_file_type(Path("test.txt")) == "text"
        assert loader.detect_file_type(Path("test.md")) == "text"
        assert loader.detect_file_type(Path("test.py")) == "text"
        assert loader.detect_file_type(Path("test.json")) == "text"
        assert loader.detect_file_type(Path("test.csv")) == "text"
        assert loader.detect_file_type(Path("test.log")) == "text"

    def test_detect_binary_type(self):
        """Test detecting binary files"""
        loader = FileLoader()

        assert loader.detect_file_type(Path("test.pdf")) == "binary"
        assert loader.detect_file_type(Path("test.zip")) == "binary"
        assert loader.detect_file_type(Path("test.exe")) == "binary"


class TestFileLoaderTextFiles:
    """Test loading text files"""

    def test_load_text_file(self, tmp_path):
        """Test loading text file"""
        # Create temp file
        test_file = tmp_path / "test.txt"
        test_content = "Hello, world!\nThis is a test."
        test_file.write_text(test_content)

        loader = FileLoader()
        content = loader.load_file(str(test_file))

        assert isinstance(content, TextContent)
        assert test_content in content.text
        assert "test.txt" in content.text
        assert "---" in content.text  # File markers

    def test_load_text_file_too_large(self, tmp_path):
        """Test loading text file that exceeds size limit"""
        # Create large file (>10MB)
        test_file = tmp_path / "large.txt"
        large_content = "x" * (11 * 1024 * 1024)  # 11MB
        test_file.write_text(large_content)

        loader = FileLoader()

        with pytest.raises(FileLoadError, match="Text file too large"):
            loader.load_file(str(test_file))

    def test_load_text_file_invalid_utf8(self, tmp_path):
        """Test loading file with invalid UTF-8"""
        # Create file with invalid UTF-8
        test_file = tmp_path / "invalid.txt"
        test_file.write_bytes(b"\xff\xfe\xfd")

        loader = FileLoader()

        with pytest.raises(FileLoadError, match="Invalid UTF-8"):
            loader.load_file(str(test_file))


class TestFileLoaderImageFiles:
    """Test loading image files"""

    def test_load_image_file(self, tmp_path):
        """Test loading image file"""
        # Create simple PNG file (1x1 red pixel)
        test_file = tmp_path / "test.png"
        # Minimal valid PNG (1x1 red pixel)
        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        )
        test_file.write_bytes(png_data)

        loader = FileLoader()
        content = loader.load_file(str(test_file))

        assert isinstance(content, ImageContent)
        assert content.image_data is not None
        assert content.mime_type == "image/png"
        # Verify it's base64 encoded
        base64.b64decode(content.image_data)  # Should not raise

    def test_load_jpeg_file(self, tmp_path):
        """Test loading JPEG file"""
        # Create JPEG file
        test_file = tmp_path / "test.jpg"
        # Use PNG data for simplicity (we're testing mime type detection)
        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        )
        test_file.write_bytes(png_data)

        loader = FileLoader()
        content = loader.load_file(str(test_file))

        assert isinstance(content, ImageContent)
        assert content.mime_type == "image/jpeg"

    def test_load_image_file_too_large(self, tmp_path):
        """Test loading image file that exceeds size limit"""
        # Create large image (>20MB)
        test_file = tmp_path / "large.png"
        large_image = b"\x00" * (21 * 1024 * 1024)  # 21MB of zeros
        test_file.write_bytes(large_image)

        loader = FileLoader()

        with pytest.raises(FileLoadError, match="Image file too large"):
            loader.load_file(str(test_file))


class TestFileLoaderBinaryFiles:
    """Test loading binary files"""

    def test_load_binary_file(self, tmp_path):
        """Test loading binary file"""
        # Create binary file
        test_file = tmp_path / "test.bin"
        binary_data = b"\x00\x01\x02\x03\x04\x05"
        test_file.write_bytes(binary_data)

        loader = FileLoader()
        content = loader.load_file(str(test_file))

        assert isinstance(content, FileContent)
        assert content.file_data == binary_data
        assert content.filename == "test.bin"

    def test_load_binary_file_too_large(self, tmp_path):
        """Test loading binary file that exceeds size limit"""
        # Create large file (>50MB)
        test_file = tmp_path / "large.bin"
        large_data = b"\x00" * (51 * 1024 * 1024)  # 51MB
        test_file.write_bytes(large_data)

        loader = FileLoader()

        with pytest.raises(FileLoadError, match="File too large"):
            loader.load_file(str(test_file))


class TestFileLoaderErrors:
    """Test error handling"""

    def test_load_nonexistent_file(self):
        """Test loading non-existent file"""
        loader = FileLoader()

        with pytest.raises(FileLoadError, match="Could not resolve path"):
            loader.load_file("/nonexistent/file.txt")

    def test_load_permission_denied(self, tmp_path):
        """Test loading file with no read permission"""
        # Create file
        test_file = tmp_path / "noperm.txt"
        test_file.write_text("content")

        # Remove read permission
        test_file.chmod(0o000)

        loader = FileLoader()

        try:
            with pytest.raises(FileLoadError, match="Permission denied"):
                loader.load_file(str(test_file))
        finally:
            # Restore permission for cleanup
            test_file.chmod(0o644)
