#!/usr/bin/env python3
"""Test to verify instructions are being sent correctly."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test():
    from sabre.server.api.server import SYSTEM_PROMPT

    print("=" * 80)
    print("SYSTEM PROMPT LENGTH:", len(SYSTEM_PROMPT))
    print("=" * 80)
    print("\nFirst 1000 characters:")
    print(SYSTEM_PROMPT[:1000])
    print("\n...")
    print("\nSearch for <helpers> tag:")
    if "<helpers>" in SYSTEM_PROMPT:
        idx = SYSTEM_PROMPT.find("<helpers>")
        print(f"Found at position {idx}")
        print("Context around <helpers>:")
        print(SYSTEM_PROMPT[max(0, idx - 200) : idx + 200])
    else:
        print("NOT FOUND!")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(test())
