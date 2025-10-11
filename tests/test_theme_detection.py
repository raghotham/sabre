#!/usr/bin/env python3
"""
Test theme detection to debug why it's detecting light mode on dark terminal.
"""

import os
import sys
import select
import termios
import tty

def test_osc11():
    """Test OSC 11 background color query."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("Not running in a TTY")
        return

    print("Testing OSC 11 background color query...")
    print("This will query your terminal for its background color.\n")

    try:
        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            # Set terminal to raw mode
            tty.setcbreak(sys.stdin.fileno())

            # Query background color with OSC 11
            sys.stdout.write("\033]11;?\033\\")
            sys.stdout.flush()

            # Read response with timeout (100ms)
            response = ""
            start_time = __import__("time").time()
            while True:
                # Check for timeout (100ms)
                if __import__("time").time() - start_time > 0.1:
                    print("Timeout waiting for response")
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

            print(f"Raw response: {repr(response)}")
            print(f"Response length: {len(response)}")

            # Parse response format: \e]11;rgb:RRRR/GGGG/BBBB\a
            if "rgb:" in response:
                import re

                match = re.search(r"rgb:([0-9a-f]+)/([0-9a-f]+)/([0-9a-f]+)", response, re.IGNORECASE)
                if match:
                    # Parse hex values (can be 2, 4, or 8 digits)
                    r_hex, g_hex, b_hex = match.groups()

                    print(f"\nParsed RGB hex: R={r_hex}, G={g_hex}, B={b_hex}")

                    # Normalize to 0-255 range
                    r = int(r_hex[:2], 16) if len(r_hex) >= 2 else int(r_hex, 16) * 17
                    g = int(g_hex[:2], 16) if len(g_hex) >= 2 else int(g_hex, 16) * 17
                    b = int(b_hex[:2], 16) if len(b_hex) >= 2 else int(b_hex, 16) * 17

                    print(f"RGB values: R={r}, G={g}, B={b}")

                    # Calculate relative luminance (ITU-R BT.709)
                    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0

                    print(f"Luminance: {luminance:.4f}")
                    print(f"Theme: {'light' if luminance > 0.5 else 'dark'}")
                else:
                    print("Could not parse RGB values from response")
            else:
                print("No 'rgb:' found in response")

        except Exception as e:
            # Restore terminal on error
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except:
                pass
            print(f"Error during query: {e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"Failed to set up terminal: {e}")
        import traceback
        traceback.print_exc()


def test_other_methods():
    """Test other detection methods."""
    print("\n" + "="*60)
    print("Other detection methods:")
    print("="*60)

    # COLORFGBG
    colorfgbg = os.getenv("COLORFGBG", "")
    print(f"\nCOLORFGBG: {colorfgbg if colorfgbg else 'not set'}")
    if colorfgbg:
        parts = colorfgbg.split(";")
        if len(parts) >= 2:
            try:
                bg = int(parts[-1])
                print(f"  Background value: {bg}")
                print(f"  Theme: {'light' if bg >= 8 else 'dark'}")
            except ValueError:
                print(f"  Could not parse background value")

    # iTerm profile
    iterm_profile = os.getenv("ITERM_PROFILE", "")
    print(f"\niTERM_PROFILE: {iterm_profile if iterm_profile else 'not set'}")
    if iterm_profile:
        if "light" in iterm_profile.lower():
            print(f"  Theme: light")
        elif "dark" in iterm_profile.lower():
            print(f"  Theme: dark")
        else:
            print(f"  No theme hint in profile name")

    # macOS dark mode
    import subprocess
    import platform
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode == 0:
                print(f"\nmacOS AppleInterfaceStyle: {result.stdout.strip()}")
                print(f"  Theme: {'dark' if 'dark' in result.stdout.lower() else 'light'}")
            else:
                print(f"\nmacOS AppleInterfaceStyle: not set (light mode)")
                print(f"  Theme: light")
        except Exception as e:
            print(f"\nmacOS dark mode check failed: {e}")


if __name__ == "__main__":
    test_osc11()
    test_other_methods()
