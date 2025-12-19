"""
Browser helper for rendering JavaScript-heavy websites.

Uses Playwright to handle dynamic content that requires JS execution.
"""

import asyncio
import logging
from typing import Optional
from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright

from sabre.common.execution_context import get_execution_context
from sabre.common.paths import get_session_files_dir

logger = logging.getLogger(__name__)


class BrowserHelper:
    """
    Playwright-based browser for dynamic web content.

    Singleton pattern - reuses browser instance across requests for performance.
    """

    _instances: dict[int, "BrowserHelper"] = {}
    _init_locks: dict[int, asyncio.Lock] = {}

    def __init__(self):
        """Initialize browser helper (use get_instance() instead)."""
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self._initialized = False
        self._owning_loop_id: Optional[int] = None

    @classmethod
    async def get_instance(cls) -> "BrowserHelper":
        """
        Get or create browser instance scoped to the current event loop.

        Returns:
            BrowserHelper instance
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.error("BrowserHelper.get_instance() requires a running event loop")
            raise

        loop_id = id(loop)

        instance = cls._instances.get(loop_id)
        if instance and instance._initialized:
            return instance

        lock = cls._init_locks.get(loop_id)
        if lock is None:
            lock = asyncio.Lock()
            cls._init_locks[loop_id] = lock

        async with lock:
            instance = cls._instances.get(loop_id)
            if instance is None:
                instance = BrowserHelper()
                instance._owning_loop_id = loop_id
                cls._instances[loop_id] = instance
            if not instance._initialized:
                try:
                    await instance._initialize()
                except Exception:
                    # Initialization failed; remove broken instance so a future call can retry
                    cls._instances.pop(loop_id, None)
                    instance._owning_loop_id = None
                    raise
            return instance

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
            await page.goto(url, wait_until="load", timeout=timeout)

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
        Download file (PDF, CSV, etc.) and return local path in session directory.

        Args:
            url: URL to download
            timeout: Timeout in milliseconds

        Returns:
            Path to downloaded file in session directory
        """
        if not self._initialized:
            await self._initialize()

        # Get session context
        ctx = get_execution_context()
        if not ctx or not ctx.session_id:
            raise RuntimeError("download_file() requires execution context with session_id")

        # Get session files directory
        files_dir = get_session_files_dir(ctx.session_id)
        files_dir.mkdir(parents=True, exist_ok=True)

        page: Optional[Page] = None
        try:
            logger.info(f"Downloading file from: {url}")

            page = await self.context.new_page()

            # Set up download handler
            async with page.expect_download() as download_info:
                await page.goto(url, timeout=timeout)

            download = await download_info.value
            filename = download.suggested_filename
            file_path = files_dir / filename
            await download.save_as(str(file_path))

            logger.info(f"Downloaded file to: {file_path}")
            print(f"[browser.download_file] Downloaded: {url}")
            print(f"[browser.download_file] Saved to: {file_path}")

            return str(file_path)

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
        loop_id = self._owning_loop_id
        if loop_id is not None:
            BrowserHelper._instances.pop(loop_id, None)
            BrowserHelper._init_locks.pop(loop_id, None)
        self._owning_loop_id = None

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
