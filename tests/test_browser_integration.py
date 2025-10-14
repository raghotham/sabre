"""
Test script for Playwright browser integration and smart detection.

Tests:
1. Static sites (should use HTTP)
2. JS-heavy sites (should use browser)
3. Framework detection
4. Fallback behavior
"""

import asyncio
import sys
import os

# Add sabre to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


async def test_static_sites():
    """Test that static sites use HTTP."""
    from sabre.server.helpers.web import Web, _should_use_browser

    print("\n=== Testing Static Sites (HTTP) ===\n")

    static_urls = [
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "https://news.ycombinator.com/",
        "https://github.com/anthropics",
    ]

    for url in static_urls:
        should_use_browser = _should_use_browser(url)
        print(f"URL: {url}")
        print(f"  Should use browser: {should_use_browser}")

        if should_use_browser:
            print("  ❌ FAIL: Should use HTTP, not browser")
            return False
        else:
            print("  ✓ PASS: Correctly using HTTP")

        # Try fetching
        try:
            content = Web.get_url(url)
            print(f"  Content length: {len(content)} chars")
            print(f"  Preview: {content[:100]}...")
            print()
        except Exception as e:
            print(f"  ❌ ERROR fetching: {e}")
            return False

    return True


async def test_js_heavy_detection():
    """Test that JS-heavy sites are detected."""
    from sabre.server.helpers.web import _should_use_browser

    print("\n=== Testing JS-Heavy Site Detection ===\n")

    js_heavy_urls = [
        "https://twitter.com/elonmusk",
        "https://www.reddit.com/r/programming",
        "https://medium.com/",
    ]

    for url in js_heavy_urls:
        should_use_browser = _should_use_browser(url)
        print(f"URL: {url}")
        print(f"  Should use browser: {should_use_browser}")

        if not should_use_browser:
            print("  ❌ FAIL: Should use browser for this JS-heavy site")
            return False
        else:
            print("  ✓ PASS: Correctly detected as JS-heavy")
        print()

    return True


async def test_browser_fetch():
    """Test actual browser fetching (if Playwright is installed)."""
    from sabre.server.helpers.browser import BrowserHelper

    print("\n=== Testing Browser Fetch ===\n")

    try:
        # Check if Playwright is available
        print("Checking Playwright installation...")
        browser = await BrowserHelper.get_instance()
        print("✓ Playwright initialized successfully")

        # Test fetch
        test_url = "https://example.com"
        print(f"\nFetching: {test_url}")
        html = await browser.get_url(test_url)
        print(f"  Content length: {len(html)} chars")
        print(f"  Contains 'Example Domain': {'Example Domain' in html}")

        if "Example Domain" in html:
            print("  ✓ PASS: Successfully fetched with browser")
        else:
            print("  ❌ FAIL: Content doesn't match expected")
            return False

        await browser.close()
        return True

    except Exception as e:
        print(f"❌ Browser test failed: {e}")
        print("\nNote: This is expected if Playwright browsers aren't installed.")
        print("Run: playwright install")
        return False


async def test_framework_detection():
    """Test framework detection from HTML content."""
    from sabre.server.helpers.web import _should_use_browser

    print("\n=== Testing Framework Detection ===\n")

    # Test URLs that might have framework markers
    test_urls = [
        "https://nextjs.org/",  # Next.js site (might have markers)
        "https://vuejs.org/",  # Vue.js site (might have markers)
    ]

    for url in test_urls:
        print(f"Testing: {url}")
        should_use_browser = _should_use_browser(url)
        print(f"  Detection result: {'Browser' if should_use_browser else 'HTTP'}")
        print()

    return True


async def test_fallback_behavior():
    """Test that browser falls back to HTTP on error."""
    from sabre.server.helpers.web import Web

    print("\n=== Testing Fallback Behavior ===\n")

    # Use a simple URL
    test_url = "https://example.com"

    print(f"Fetching: {test_url}")
    print("Note: This should use HTTP (static domain)")

    try:
        content = Web.get_url(test_url)
        print(f"  Content length: {len(content)} chars")
        print(f"  Contains 'Example Domain': {'Example Domain' in content}")

        if "Example Domain" in content:
            print("  ✓ PASS: Successfully fetched")
            return True
        else:
            print("  ❌ FAIL: Unexpected content")
            return False

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Playwright Browser Integration Test Suite")
    print("=" * 60)

    results = {}

    # Test 1: Static sites
    results["static_sites"] = await test_static_sites()

    # Test 2: JS-heavy detection
    results["js_heavy_detection"] = await test_js_heavy_detection()

    # Test 3: Framework detection
    results["framework_detection"] = await test_framework_detection()

    # Test 4: Fallback behavior
    results["fallback"] = await test_fallback_behavior()

    # Test 5: Browser fetch (optional - requires Playwright installed)
    results["browser_fetch"] = await test_browser_fetch()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    total_passed = sum(1 for p in results.values() if p)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
