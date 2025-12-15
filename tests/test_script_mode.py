"""
Tests for CLI script mode functionality.

Run with: uv run pytest tests/test_script_mode.py
"""

from unittest.mock import patch
import asyncio


def test_run_client_with_message():
    """Test run_client() with message parameter (script mode)."""
    from sabre.cli import run_client

    # Mock the client main function
    with patch("sabre.client.client.main") as mock_main:
        mock_main.return_value = 0

        asyncio.run(run_client("test message"))

        # Should have called main
        mock_main.assert_called_once()


def test_run_client_without_message():
    """Test run_client() without message (interactive mode)."""
    from sabre.cli import run_client

    # Mock the client main function
    with patch("sabre.client.client.main") as mock_main:
        mock_main.return_value = 0

        asyncio.run(run_client(None))

        # Should call main without modifying sys.argv
        mock_main.assert_called_once()


def test_mcp_subcommand_structure():
    """Test MCP subcommand parsing."""
    import argparse

    # Simulate the argument structure
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--message", type=str)
    subparsers = parser.add_subparsers(dest="subcommand")
    mcp_parser = subparsers.add_parser("mcp")
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command")
    mcp_subparsers.add_parser("list")
    mcp_subparsers.add_parser("init")
    mcp_subparsers.add_parser("config")

    # Test MCP list
    args = parser.parse_args(["mcp", "list"])
    assert args.subcommand == "mcp"
    assert args.mcp_command == "list"

    # Test MCP init
    args = parser.parse_args(["mcp", "init"])
    assert args.subcommand == "mcp"
    assert args.mcp_command == "init"

    # Test script mode
    args = parser.parse_args(["--message", "hello world"])
    assert args.message == "hello world"
    assert args.subcommand is None


def test_cli_message_flag():
    """Test that --message/-m flag works correctly."""
    from sabre.cli import main

    # Test with --message flag
    with patch("sys.argv", ["sabre", "--message", "test task"]):
        with patch("sabre.cli.start_server") as mock_server:
            with patch("asyncio.run") as mock_run:
                mock_server.return_value = None
                mock_run.return_value = 0

                try:
                    main()
                except (SystemExit, KeyboardInterrupt):
                    pass

                # Should have called asyncio.run with run_client
                mock_run.assert_called_once()
                # The first argument to asyncio.run should be the coroutine from run_client
                call_args = mock_run.call_args
                assert call_args is not None


def test_mcp_command_execution():
    """Test that MCP commands are executed correctly."""
    from sabre.cli import main

    # Test MCP list command
    with patch("sys.argv", ["sabre", "mcp", "list"]):
        with patch("sabre.cli.mcp_list") as mock_list:
            mock_list.return_value = 0
            try:
                main()
            except SystemExit:
                pass
            mock_list.assert_called_once()
