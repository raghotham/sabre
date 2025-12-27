"""Test write_file() and read_file() helpers."""

import os
import base64
import tempfile

# Test imports
from sabre.server.helpers.fs import write_file, read_file
from sabre.common.models.messages import ImageContent, TextContent
from sabre.common.execution_context import set_execution_context, clear_execution_context
from sabre.common.paths import get_session_files_dir


def test_write_read_text_file():
    """Test writing and reading text files."""
    print("\n=== Test: Write and Read Text File ===")

    # Set up context
    test_conv_id = "test-conv-text-file"
    set_execution_context(
        event_callback=None, tree=None, tree_context={}, conversation_id=test_conv_id, session_id=test_conv_id
    )

    try:
        # Write text file
        url = write_file("test.txt", "Hello, world!")
        print(f"✓ Wrote file, URL: {url}")
        assert test_conv_id in url
        assert "test.txt" in url

        # Read it back
        content = read_file("test.txt")
        print(f"✓ Read file, content type: {type(content).__name__}")
        assert isinstance(content, TextContent)
        assert content.get_str() == "Hello, world!"
        print(f"✓ Content matches: {content.get_str()}")

        # Cleanup
        files_dir = get_session_files_dir(test_conv_id)
        (files_dir / "test.txt").unlink()
        files_dir.rmdir()
        print("✓ Cleanup successful\n")

    finally:
        clear_execution_context()


def test_write_read_json_file():
    """Test writing and reading JSON files."""
    print("\n=== Test: Write and Read JSON File ===")

    test_conv_id = "test-conv-json"
    set_execution_context(None, None, {}, test_conv_id, test_conv_id)

    try:
        # Write JSON
        data = {"name": "Alice", "age": 30, "city": "Brisbane"}
        url = write_file("data.json", data)
        print(f"✓ Wrote JSON, URL: {url}")

        # Read it back
        content = read_file("data.json")
        print(f"✓ Read file, content type: {type(content).__name__}")
        assert isinstance(content, TextContent)
        assert "Alice" in content.get_str()
        assert "Brisbane" in content.get_str()
        print("✓ Content contains expected data")

        # Cleanup
        files_dir = get_session_files_dir(test_conv_id)
        (files_dir / "data.json").unlink()
        files_dir.rmdir()
        print("✓ Cleanup successful\n")

    finally:
        clear_execution_context()


def test_write_read_image():
    """Test writing and reading images."""
    print("\n=== Test: Write and Read Image ===")

    test_conv_id = "test-conv-image"
    set_execution_context(None, None, {}, test_conv_id, test_conv_id)

    try:
        # Create test image (minimal PNG header)
        img_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        img_base64 = base64.b64encode(img_data).decode("utf-8")
        img_content = ImageContent(image_data=img_base64, mime_type="image/png")

        # Write image
        url = write_file("test.png", img_content)
        print(f"✓ Wrote image, URL: {url}")

        # Read it back
        content = read_file("test.png")
        print(f"✓ Read file, content type: {type(content).__name__}")
        assert isinstance(content, ImageContent)
        assert content.mime_type == "image/png"
        print("✓ Content is ImageContent with correct MIME type")

        # Cleanup
        files_dir = get_session_files_dir(test_conv_id)
        (files_dir / "test.png").unlink()
        files_dir.rmdir()
        print("✓ Cleanup successful\n")

    finally:
        clear_execution_context()


def test_security_no_path_traversal():
    """Test that path traversal is blocked."""
    print("\n=== Test: Security - Path Traversal Prevention ===")

    test_conv_id = "test-conv-security"
    set_execution_context(None, None, {}, test_conv_id, test_conv_id)

    try:
        # Try to write with path separator
        try:
            write_file("../etc/passwd", "hacker")
            print("✗ Should have raised RuntimeError")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "basename only" in str(e).lower()
            print(f"✓ Path traversal blocked: {e}")

        # Cleanup (just in case)
        files_dir = get_session_files_dir(test_conv_id)
        if files_dir.exists():
            files_dir.rmdir()
        print("✓ Security test passed\n")

    finally:
        clear_execution_context()


def test_read_absolute_path():
    """Test reading from absolute path."""
    print("\n=== Test: Read from Absolute Path ===")

    test_conv_id = "test-conv-absolute"
    set_execution_context(None, None, {}, test_conv_id, test_conv_id)

    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Absolute path test content")
            temp_path = f.name

        print(f"✓ Created temp file: {temp_path}")

        try:
            # Read with absolute path
            content = read_file(temp_path)
            assert isinstance(content, TextContent)
            assert content.get_str() == "Absolute path test content"
            print("✓ Read file with absolute path successfully")
        finally:
            os.unlink(temp_path)
            print("✓ Cleaned up temp file")

    finally:
        clear_execution_context()


def test_write_file_no_context():
    """Test that write_file fails without execution context."""
    print("\n=== Test: Write File Without Context ===")

    # Don't set context
    try:
        write_file("test.txt", "This should fail")
        print("✗ Should have raised RuntimeError")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "execution context" in str(e).lower() or "session_id" in str(e).lower()
        print(f"✓ Correctly rejected write without context: {e}")


def main():
    """Run all tests."""
    print("=" * 70)
    print("  File I/O Tests")
    print("=" * 70)

    tests = [
        test_write_read_text_file,
        test_write_read_json_file,
        test_write_read_image,
        test_security_no_path_traversal,
        test_read_absolute_path,
        test_write_file_no_context,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()

    print("=" * 70)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
