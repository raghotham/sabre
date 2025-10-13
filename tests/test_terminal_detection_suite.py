#!/usr/bin/env python3
"""
Comprehensive terminal detection test suite.

Tests all terminal theme detection methods in various scenarios.
Run with: uv run python tests/test_terminal_detection_suite.py
Or in WezTerm PTY: ./tests/run_wezterm_pty_test.sh tests/test_terminal_detection_suite.py
"""

import os
import sys
from sabre.client.tui import TUI


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(name: str, value: str, status: str = "info"):
    """Print a test result."""
    colors = {
        "pass": "\033[92m✓\033[0m",  # Green checkmark
        "fail": "\033[91m✗\033[0m",  # Red X
        "info": "\033[94mℹ\033[0m",  # Blue info
        "warn": "\033[93m⚠\033[0m",  # Yellow warning
    }
    symbol = colors.get(status, colors["info"])
    print(f"  {symbol} {name}: {value}")


def test_tty_status():
    """Test TTY status."""
    print_section("TTY Status")

    stdin_tty = sys.stdin.isatty()
    stdout_tty = sys.stdout.isatty()
    stderr_tty = sys.stderr.isatty()

    print_result("stdin.isatty()", str(stdin_tty), "pass" if stdin_tty else "fail")
    print_result("stdout.isatty()", str(stdout_tty), "pass" if stdout_tty else "fail")
    print_result("stderr.isatty()", str(stderr_tty), "pass" if stderr_tty else "fail")

    if not (stdin_tty and stdout_tty):
        print_result("Overall", "Not in TTY - OSC 11 will not work", "warn")
    else:
        print_result("Overall", "Full TTY support available", "pass")


def test_environment_variables():
    """Test environment variables."""
    print_section("Environment Variables")

    env_vars = {
        "TERM": os.getenv("TERM", "not set"),
        "TERM_PROGRAM": os.getenv("TERM_PROGRAM", "not set"),
        "COLORTERM": os.getenv("COLORTERM", "not set"),
        "COLORFGBG": os.getenv("COLORFGBG", "not set"),
        "ITERM_PROFILE": os.getenv("ITERM_PROFILE", "not set"),
        "SABRE_THEME": os.getenv("SABRE_THEME", "not set"),
    }

    for name, value in env_vars.items():
        status = "pass" if value != "not set" else "info"
        print_result(name, value, status)


def test_colorfgbg_detection():
    """Test COLORFGBG detection."""
    print_section("COLORFGBG Detection")

    colorfgbg = os.getenv("COLORFGBG", "")
    if not colorfgbg:
        print_result("COLORFGBG", "not set - cannot detect theme", "warn")
        return

    print_result("COLORFGBG value", colorfgbg, "info")

    parts = colorfgbg.split(";")
    if len(parts) >= 2:
        try:
            fg = int(parts[0]) if len(parts) > 0 else None
            bg = int(parts[-1])

            print_result("Foreground color", str(fg) if fg is not None else "unknown", "info")
            print_result("Background color", str(bg), "info")

            # Background colors 0-7 are dark, 8-15 are light
            detected_theme = "light" if bg >= 8 else "dark"
            print_result("Detected theme", detected_theme, "pass")

        except ValueError as e:
            print_result("Parse error", str(e), "fail")
    else:
        print_result("Format", "Invalid COLORFGBG format", "fail")


def test_osc11_detection():
    """Test OSC 11 background color query."""
    print_section("OSC 11 Background Color Query")

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print_result("Status", "Not in TTY - OSC 11 unavailable", "warn")
        return

    print_result("Status", "Sending OSC 11 query...", "info")

    try:
        import select
        import termios
        import tty
        import time

        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            # Set terminal to raw mode
            tty.setcbreak(sys.stdin.fileno())

            # Query background color with OSC 11
            sys.stdout.write("\033]11;?\033\\")
            sys.stdout.flush()

            # Read response with timeout (200ms - longer for slower terminals)
            response = ""
            start_time = time.time()
            timeout = 0.2  # 200ms timeout

            while True:
                # Check for timeout
                if time.time() - start_time > timeout:
                    print_result("Response", "Timeout - no response received", "fail")
                    break

                # Check if data available
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    char = sys.stdin.read(1)
                    response += char

                    # Look for terminator (BEL or ST)
                    if char == "\a" or response.endswith("\033\\"):
                        break

            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

            if response:
                print_result("Raw response", repr(response), "info")
                print_result("Response length", str(len(response)), "info")

                # Parse response format: \e]11;rgb:RRRR/GGGG/BBBB\a
                if "rgb:" in response:
                    import re

                    match = re.search(r"rgb:([0-9a-f]+)/([0-9a-f]+)/([0-9a-f]+)", response, re.IGNORECASE)
                    if match:
                        # Parse hex values (can be 2, 4, or 8 digits)
                        r_hex, g_hex, b_hex = match.groups()

                        print_result("RGB hex", f"R={r_hex}, G={g_hex}, B={b_hex}", "info")

                        # Normalize to 0-255 range
                        r = int(r_hex[:2], 16) if len(r_hex) >= 2 else int(r_hex, 16) * 17
                        g = int(g_hex[:2], 16) if len(g_hex) >= 2 else int(g_hex, 16) * 17
                        b = int(b_hex[:2], 16) if len(b_hex) >= 2 else int(b_hex, 16) * 17

                        print_result("RGB 8-bit", f"R={r}, G={g}, B={b}", "info")

                        # Calculate relative luminance (ITU-R BT.709)
                        luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0

                        print_result("Luminance", f"{luminance:.4f}", "info")

                        detected_theme = "light" if luminance > 0.5 else "dark"
                        print_result("Detected theme", detected_theme, "pass")
                    else:
                        print_result("Parse error", "Could not extract RGB values", "fail")
                else:
                    print_result("Parse error", "No 'rgb:' found in response", "fail")

        except Exception as e:
            # Restore terminal on error
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except:
                pass
            print_result("Error", str(e), "fail")

    except Exception as e:
        print_result("Setup error", str(e), "fail")


def test_os_dark_mode():
    """Test OS-level dark mode detection."""
    print_section("OS Dark Mode Detection")

    import subprocess
    import platform

    system = platform.system()
    print_result("Platform", system, "info")

    if system == "Darwin":  # macOS
        try:
            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"], capture_output=True, text=True, timeout=1
            )

            if result.returncode == 0:
                apple_style = result.stdout.strip()
                print_result("AppleInterfaceStyle", apple_style, "info")

                detected_theme = "dark" if "dark" in apple_style.lower() else "light"
                print_result("Detected theme", detected_theme, "pass")
            else:
                print_result("AppleInterfaceStyle", "not set (Light Mode)", "info")
                print_result("Detected theme", "light", "pass")

        except subprocess.TimeoutExpired:
            print_result("Error", "Command timeout", "fail")
        except Exception as e:
            print_result("Error", str(e), "fail")
    else:
        print_result("Status", "OS dark mode detection only available on macOS", "warn")


def test_tui_detection():
    """Test actual TUI theme detection."""
    print_section("TUI Theme Detection (Integrated)")

    try:
        tui = TUI()
        term_info = tui.get_terminal_info()

        print_result("Detected theme", term_info["detected_theme"], "pass")
        print_result("Detection method", term_info["detection_method"] or "none", "info")
        print_result("Terminal", term_info["term_program"], "info")
        print_result("TERM", term_info["term"], "info")
        print_result("Terminal size", f"{term_info['size'].columns}x{term_info['size'].lines}", "info")

    except Exception as e:
        print_result("Error", str(e), "fail")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  SABRE Terminal Detection Test Suite")
    print("=" * 70)

    # Run all test sections
    test_tty_status()
    test_environment_variables()
    test_colorfgbg_detection()
    test_osc11_detection()
    test_os_dark_mode()
    test_tui_detection()

    print("\n" + "=" * 70)
    print("  Test Suite Complete")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
