"""
Browser helper for rendering JavaScript-heavy websites.

Uses Playwright to handle dynamic content that requires JS execution.
"""

import asyncio
import logging
from typing import Optional
from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright

logger = logging.getLogger(__name__)


class BrowserHelper:
    """
    Playwright-based browser for dynamic web content.

    Singleton pattern - reuses browser instance across requests for performance.
    """

    _instance: Optional["BrowserHelper"] = None
    _lock = asyncio.Lock()

    def __init__(self):
        """Initialize browser helper (use get_instance() instead)."""
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self._initialized = False

    @classmethod
    async def get_instance(cls) -> "BrowserHelper":
        """
        Get or create singleton browser instance.

        Returns:
            BrowserHelper instance
        """
        async with cls._lock:
            if cls._instance is None:
                cls._instance = BrowserHelper()
                await cls._instance._initialize()
            return cls._instance

    async def _initialize(self):
        """Initialize Playwright and browser."""
        if self._initialized:
            return

        try:
            logger.info("Initializing Playwright browser...")
            self.playwright = await async_playwright().start()

            # Launch headless Chromium
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                ],
            )

            # Create persistent context
            self.context = await self.browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )

            self._initialized = True
            logger.info("Playwright browser initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            await self._cleanup()
            raise

    async def get_url(self, url: str, wait_for: str = "networkidle", timeout: int = 30000) -> str:
        """
        Navigate to URL and return rendered HTML.

        Args:
            url: URL to fetch
            wait_for: Wait condition - 'networkidle', 'load', or 'domcontentloaded'
            timeout: Timeout in milliseconds (default 30000)

        Returns:
            Rendered HTML content
        """
        if not self._initialized:
            await self._initialize()

        page: Optional[Page] = None
        try:
            logger.info(f"Fetching URL with browser: {url}")

            # Create new page
            page = await self.context.new_page()

            # Navigate to URL
            await page.goto(url, wait_until=wait_for, timeout=timeout)

            # Get rendered HTML
            html = await page.content()

            logger.info(f"Successfully fetched {url} ({len(html)} chars)")
            return html

        except Exception as e:
            logger.error(f"Browser failed to fetch {url}: {e}")
            raise
        finally:
            if page:
                await page.close()

    async def screenshot(self, url: str, timeout: int = 30000) -> bytes:
        """
        Take screenshot of URL.

        Args:
            url: URL to screenshot
            timeout: Timeout in milliseconds

        Returns:
            PNG screenshot as bytes
        """
        if not self._initialized:
            await self._initialize()

        page: Optional[Page] = None
        try:
            logger.info(f"Taking screenshot of: {url}")

            page = await self.context.new_page()
            await page.goto(url, wait_until="networkidle", timeout=timeout)

            screenshot_bytes = await page.screenshot(full_page=True)

            logger.info(f"Screenshot captured ({len(screenshot_bytes)} bytes)")
            return screenshot_bytes

        except Exception as e:
            logger.error(f"Failed to screenshot {url}: {e}")
            raise
        finally:
            if page:
                await page.close()

    async def download_file(self, url: str, timeout: int = 30000) -> str:
        """
        Download file (PDF, CSV, etc.) and return local path.

        Args:
            url: URL to download
            timeout: Timeout in milliseconds

        Returns:
            Path to downloaded file
        """
        if not self._initialized:
            await self._initialize()

        page: Optional[Page] = None
        try:
            logger.info(f"Downloading file from: {url}")

            page = await self.context.new_page()

            # Set up download handler
            async with page.expect_download() as download_info:
                await page.goto(url, timeout=timeout)

            download = await download_info.value
            file_path = f"/tmp/{download.suggested_filename}"
            await download.save_as(file_path)

            logger.info(f"Downloaded file to: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            raise
        finally:
            if page:
                await page.close()

    async def _cleanup(self):
        """Clean up browser resources."""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()

            self._initialized = False
            logger.info("Browser resources cleaned up")

        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    async def close(self):
        """Close browser instance."""
        await self._cleanup()
        BrowserHelper._instance = None

    @classmethod
    async def check_installed(cls) -> bool:
        """
        Check if Playwright browsers are installed.

        Returns:
            True if browsers are available
        """
        try:
            helper = await cls.get_instance()
            await helper.close()
            return True
        except Exception as e:
            logger.debug(f"Playwright check failed: {e}")
            return False
