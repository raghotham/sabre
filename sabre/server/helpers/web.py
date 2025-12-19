"""
Web helper - download and fetch web content.

Provides web content downloading with:
- Simple HTTP for static content
- Playwright browser for JavaScript-heavy sites
- HTML to markdown conversion
"""

import asyncio
import logging
import re
import unicodedata
import warnings
import base64
import requests
import io
import concurrent.futures
from typing import Any
from pathlib import Path
from urllib.parse import urlparse
from markdownify import MarkdownConverter
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from sabre.server.helpers.browser import BrowserHelper
from sabre.server.helpers.llm_call import run_async_from_sync
from sabre.common.execution_context import get_execution_context
from sabre.common.paths import get_session_files_dir
from sabre.common.models.messages import Content, ImageContent, TextContent
from sabre.common.models import SearchResult

# PyPDF2 is optional - imported where needed
try:
    import PyPDF2

    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

logger = logging.getLogger(__name__)


async def _get_browser():
    """Get the browser helper instance for the current event loop."""
    return await BrowserHelper.get_instance()


def _clean_markdown(markdown_text: str) -> str:
    """
    Clean up markdown text by removing extra blank lines and unwanted elements.

    Args:
        markdown_text: Raw markdown text

    Returns:
        Cleaned markdown text
    """
    lines = []
    blank_counter = 0

    for line in markdown_text.splitlines():
        if line == "" and blank_counter == 0:
            blank_counter += 1
            lines.append(line)
        elif line == "" and blank_counter >= 1:
            # Skip multiple consecutive blank lines
            continue
        elif line in ("<div>", "</div>", "[]", "[[]]"):
            # Skip unwanted tags
            continue
        elif line in ("*", "* ", " *", "&starf;", "&star;", "&nbsp;"):
            # Skip unwanted symbols
            continue
        elif "(data:image" in line and ")" in line:
            # Remove data:image links
            lines.append(re.sub(r"\(data:image[^\)]+\)", "", line))
            blank_counter = 0
        else:
            lines.append(line)
            blank_counter = 0

    # Strip whitespace and filter empty lines
    lines = [line.strip() for line in lines]
    return "\n".join([line for line in lines if line])


class VerboseConverter(MarkdownConverter):
    """
    Custom HTML to Markdown converter.

    Strips script and style tags completely for cleaner output.
    """

    def convert_script(self, el, text, parent_tags):
        """Strip script tags"""
        return ""

    def convert_style(self, el, text, parent_tags):
        """Strip style tags"""
        return ""


def download_csv(url: str) -> str:
    """
    Download CSV file from URL and return path to file in session directory.

    Handles SSL properly. Use with pd.read_csv() to avoid SSL certificate errors.

    Example:
        csv_path = download_csv("https://example.com/data.csv")
        df = pd.read_csv(csv_path)

    Args:
        url: URL to CSV file

    Returns:
        Path to downloaded file in session directory (string)
    """
    logger.info(f"Downloading CSV from: {url}")

    # Get session context
    ctx = get_execution_context()
    if not ctx or not ctx.session_id:
        raise RuntimeError("download_csv() requires execution context with session_id")

    # Extract filename from URL, or use default
    parsed = urlparse(url)
    filename = Path(parsed.path).name
    if not filename or not filename.endswith(".csv"):
        filename = "downloaded.csv"

    # Get session files directory
    files_dir = get_session_files_dir(ctx.session_id)
    files_dir.mkdir(parents=True, exist_ok=True)
    file_path = files_dir / filename

    # Set a proper User-Agent to avoid 403 Forbidden errors
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, timeout=30, headers=headers)
        response.raise_for_status()

        # Save to session files directory
        file_path.write_text(response.text)

        logger.info(f"Downloaded CSV to: {file_path}")
        print(f"[download_csv] Downloaded: {url}")
        print(f"[download_csv] Saved to: {file_path}")

        return str(file_path)
    except Exception as e:
        error_msg = f"ERROR: Failed to download CSV from {url}: {e}"
        logger.error(error_msg)
        print(error_msg)
        raise


def download(urls_or_results: Any, max_urls: int = 10) -> list:
    """
    Download web content as screenshots or files.

    Returns list of Content objects:
    - Web pages → ImageContent (full page screenshot via Playwright)
    - PDFs → TextContent (extracted text)
    - CSVs → TextContent (file path to downloaded temp file)

    This is the PRIMARY way to fetch web content. Screenshots allow the LLM
    to see the page visually (layout, images, styling). Use llm_call() after
    download() if you need to extract specific information from the content.

    Args:
        urls_or_results: URL string, list of URLs, or list of search result dicts
        max_urls: Maximum number of URLs to download (default 10)

    Returns:
        list[Content]: List of ImageContent (screenshots) or TextContent (files)

    Examples:
        # Download single URL
        content = download("https://example.com")

        # Download search results
        results = Search.web_search("topic")
        content = download(results[:3])

        # Extract info with llm_call
        pages = download(["https://example.com"])
        info = llm_call(pages, "extract key facts")
    """

    # Run async implementation
    return run_async_from_sync(_download_async(urls_or_results, max_urls))


async def _download_async(urls_or_results: Any, max_urls: int = 10) -> list:
    """
    Async implementation of download().

    This is the actual implementation that handles async browser operations properly.
    """

    logger.info(f"download() called with type: {type(urls_or_results).__name__}")
    logger.debug(f"download() input value: {urls_or_results}")

    # Handle different input types - extract URLs
    urls = []

    # Import SearchResult to handle type checks

    if isinstance(urls_or_results, str):
        # Single URL string
        urls = [urls_or_results]
        logger.info(f"Extracted single URL from string: {urls_or_results}")
    elif isinstance(urls_or_results, SearchResult):
        # Single SearchResult object
        urls = [urls_or_results.url]
        logger.info(f"Extracted URL from SearchResult: {urls_or_results.url}")
    elif isinstance(urls_or_results, list):
        # List of URLs or search results
        logger.info(f"Processing list with {len(urls_or_results)} items")
        for idx, item in enumerate(urls_or_results):
            if isinstance(item, str):
                urls.append(item)
                logger.debug(f"  [{idx}] String URL: {item}")
            elif isinstance(item, SearchResult):
                # SearchResult object with .url attribute
                urls.append(item.url)
                logger.debug(f"  [{idx}] SearchResult URL: {item.url} (title: {getattr(item, 'title', 'N/A')})")
            elif isinstance(item, dict) and "link" in item:
                # Search result dict with 'link' key (legacy support)
                urls.append(item["link"])
                logger.debug(f"  [{idx}] Dict with 'link': {item['link']}")
            elif isinstance(item, dict) and "url" in item:
                # Search result dict with 'url' key (legacy support)
                urls.append(item["url"])
                logger.debug(f"  [{idx}] Dict with 'url': {item['url']}")
            else:
                logger.warning(f"  [{idx}] Skipped unknown item type: {type(item).__name__}")

    if not urls:
        error_msg = f"ERROR: Could not extract URLs from input type {type(urls_or_results).__name__}: {urls_or_results}"
        logger.error(error_msg)
        print(error_msg)
        return [TextContent(error_msg)]

    # Filter out empty URLs
    urls = [url for url in urls if url and url.strip()]

    if not urls:
        error_msg = "ERROR: All extracted URLs were empty or invalid"
        logger.error(error_msg)
        print(error_msg)
        return [TextContent(error_msg)]

    logger.info(f"Successfully extracted {len(urls)} valid URL(s) for download")

    # Limit number of URLs
    if len(urls) > max_urls:
        logger.warning(f"Limiting download from {len(urls)} to {max_urls} URLs")
        print(f"Note: Limiting download to first {max_urls} of {len(urls)} URLs")
        urls = urls[:max_urls]

    logger.info(f"Downloading {len(urls)} URL(s) as screenshots/files")

    # Download all URLs concurrently using asyncio.gather
    async def download_one(url: str) -> Content:
        """Download single URL and return Content object"""
        logger.info(f"▶ Starting download: {url}")
        try:
            url_lower = url.lower()

            # Handle PDFs
            if url_lower.endswith(".pdf") or "pdf" in url_lower:
                logger.info(f"  📄 Detected PDF format: {url}")
                # Use simple HTTP for PDFs (run in thread to avoid blocking)

                def _download_pdf():
                    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
                    response = requests.get(url, timeout=30, headers=headers)
                    response.raise_for_status()

                    # Extract text from PDF

                    pdf_bytes = io.BytesIO(response.content)
                    pdf_reader = PyPDF2.PdfReader(pdf_bytes)

                    text_parts = [f"=== PDF: {url} ==="]
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        text = page.extract_text()
                        if text.strip():
                            text_parts.append(f"--- Page {page_num} ---\n{text}")

                    full_text = "\n\n".join(text_parts)
                    logger.info(f"Extracted PDF text: {len(full_text)} chars from {len(pdf_reader.pages)} pages")
                    print(f"Downloaded PDF: {url} ({len(pdf_reader.pages)} pages)")

                    return TextContent(full_text)

                # Run blocking PDF download in thread
                return await asyncio.to_thread(_download_pdf)

            # Handle CSVs
            elif url_lower.endswith(".csv") or "csv" in url_lower:
                logger.info(f"  📊 Detected CSV format: {url}")

                def _download_csv_wrapper():
                    csv_path = download_csv(url)
                    logger.info(f"  ✓ CSV downloaded to: {csv_path}")
                    print(f"Downloaded CSV: {url} -> {csv_path}")
                    return TextContent(f"CSV file downloaded to: {csv_path}\nUse pd.read_csv('{csv_path}') to load it.")

                # Run blocking CSV download in thread
                return await asyncio.to_thread(_download_csv_wrapper)

            # Handle web pages - take screenshot
            else:
                logger.info(f"  🌐 Web page (will screenshot): {url}")
                # Get browser instance and take screenshot (properly async)
                logger.debug("  Getting browser instance...")
                browser = await _get_browser()
                logger.debug("  Taking screenshot...")
                screenshot_bytes = await browser.screenshot(url)

                # Convert to base64
                screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

                logger.info(
                    f"  ✓ Screenshot captured: {len(screenshot_bytes):,} bytes ({len(screenshot_b64):,} base64 chars)"
                )
                print(f"Downloaded (screenshot): {url}")

                return ImageContent(image_data=screenshot_b64, mime_type="image/png")

        except Exception as e:
            # Check if this is the HTTP2 protocol error (bot detection)
            if "ERR_HTTP2_PROTOCOL_ERROR" in str(e):
                logger.warning(f"  ⚠️  Bot detection (HTTP2 error) for {url}, trying HTTP fallback...")
                try:
                    # Fallback to simple HTTP request (synchronous, force HTTP not browser)
                    markdown_text = Web.get_url(url, use_browser=False)
                    logger.info(f"  ✓ HTTP fallback succeeded for {url}")
                    print(f"Downloaded (HTTP fallback): {url}")
                    return TextContent(markdown_text)
                except Exception as fallback_error:
                    error_msg = f"✗ Failed to download {url} (screenshot and HTTP fallback both failed): {type(fallback_error).__name__}: {fallback_error}"
                    logger.error(error_msg, exc_info=True)
                    print(error_msg)
                    return TextContent(error_msg)
            else:
                error_msg = f"✗ Failed to download {url}: {type(e).__name__}: {e}"
                logger.error(error_msg, exc_info=True)
                print(error_msg)
                return TextContent(error_msg)

    # Use asyncio.gather for concurrent downloads
    results = await asyncio.gather(*[download_one(url) for url in urls])

    # Count success vs errors
    successes = sum(1 for r in results if not (isinstance(r, TextContent) and "ERROR" in str(r.text)))
    errors = len(results) - successes

    if errors > 0:
        logger.warning(f"Download complete: {successes} succeeded, {errors} failed out of {len(results)} total")
    else:
        logger.info(f"Download complete: All {len(results)} items downloaded successfully")

    return list(results)


# =============================================================================
# UNUSED HELPERS - Text-based web fetching (kept for future use)
# =============================================================================
# The functions below convert web content to text/markdown instead of screenshots.
# They are not currently registered or used, but kept for potential future needs.


def _should_use_browser(url: str) -> bool:
    """
    Determine if URL should be fetched with browser or simple HTTP.

    Uses heuristics to detect JavaScript-heavy sites that require browser rendering.

    Args:
        url: URL to check

    Returns:
        True if browser should be used
    """
    url_lower = url.lower()

    # Known static content - use HTTP for speed
    static_domains = [
        "arxiv.org",  # Handles PDFs specially
        "wikipedia.org",  # Server-side rendered
        "github.com",  # Server-side rendered
        "stackoverflow.com",  # Server-side rendered
        "news.ycombinator.com",  # Server-side rendered
    ]

    # Check if it's a static file type
    static_extensions = [".pdf", ".csv", ".xml", ".txt", ".json"]

    if any(url_lower.endswith(ext) for ext in static_extensions):
        return False

    if any(domain in url_lower for domain in static_domains):
        return False

    # Known JS-heavy domains that require browser rendering
    js_heavy_domains = [
        "twitter.com",
        "x.com",
        "instagram.com",
        "facebook.com",
        "linkedin.com",
        "tiktok.com",
        "reddit.com",  # New Reddit UI is React-based
        "medium.com",  # Often requires JS
        "substack.com",  # JS-heavy
        "notion.so",  # Fully client-side
        "figma.com",  # Fully client-side
        "airbnb.com",  # React SPA
        "netflix.com",  # React SPA
        "docs.google.com",  # JS-heavy
        "app.",  # Common SPA subdomain pattern
        "dashboard.",  # Common SPA subdomain pattern
        "console.",  # Common SPA subdomain pattern
    ]

    if any(domain in url_lower for domain in js_heavy_domains):
        logger.info(f"Detected JS-heavy domain in {url}, using browser")
        return True

    # Check for SPA URL patterns
    spa_patterns = [
        "#/",  # Hash-based routing (Angular, Vue Router)
        "#!/",  # Hash-bang routing
    ]

    if any(pattern in url for pattern in spa_patterns):
        logger.info(f"Detected SPA routing pattern in {url}, using browser")
        return True

    # Try to detect frameworks from a quick HTTP request
    try:
        # Quick HEAD request to check headers (fast)
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

        # Use HEAD request first (faster)
        head_response = requests.head(url, timeout=3, headers=headers, allow_redirects=True)

        # Check for common SPA/framework headers
        server_header = head_response.headers.get("X-Powered-By", "").lower()
        if any(framework in server_header for framework in ["next.js", "nuxt", "gatsby"]):
            logger.info(f"Detected framework in headers: {server_header}, using browser")
            return True

        # If HEAD didn't give us enough info, do a quick GET for the HTML
        # Only fetch first 8KB to detect framework markers
        get_response = requests.get(url, timeout=5, headers=headers, stream=True)

        # Read only first chunk (8KB should contain <head> and framework markers)
        content = next(get_response.iter_content(8192), b"").decode("utf-8", errors="ignore")

        # Framework detection patterns in HTML
        framework_markers = [
            "react",  # React apps often have this in scripts
            "ng-app",  # Angular
            "ng-version",  # Angular
            "vue",  # Vue.js
            "__NEXT_DATA__",  # Next.js
            "__nuxt",  # Nuxt.js
            "gatsby",  # Gatsby
            "ember",  # Ember.js
            "backbone",  # Backbone.js
            'root"></div><script',  # Common SPA pattern: empty root div with scripts
        ]

        content_lower = content.lower()
        for marker in framework_markers:
            if marker in content_lower:
                logger.info(f"Detected framework marker '{marker}' in HTML, using browser")
                return True

        # Check if there's very little content in the initial HTML
        # SPAs typically have minimal HTML and load everything via JS
        if len(content.strip()) < 1000 and "<script" in content_lower:
            logger.info("Detected minimal HTML with scripts (likely SPA), using browser")
            return True

    except Exception as e:
        # If detection fails, default to HTTP (safe fallback)
        logger.debug(f"Framework detection failed for {url}: {e}")
        pass

    # Default to HTTP for unknown sites (faster, works for most content)
    return False


class Web:
    """Web content retrieval helpers."""

    @staticmethod
    def get_url(url: str, use_browser: bool | None = None) -> str:
        """
        Download webpage content and convert to appropriate format.

        Handles HTML (converts to markdown) and PDF (extracts text).
        Automatically detects when to use browser vs HTTP, or can be forced.

        Args:
            url: URL to fetch
            use_browser: Force browser (True) or HTTP (False), or auto-detect (None)

        Returns:
            Page content as string
        """
        # Determine strategy
        if use_browser is None:
            use_browser = _should_use_browser(url)

        # Try browser first if requested
        if use_browser:
            try:
                logger.info(f"Attempting to fetch with browser: {url}")
                return Web._get_url_with_browser(url)
            except Exception as browser_error:
                logger.warning(f"Browser fetch failed, falling back to HTTP: {browser_error}")
                # Fall through to HTTP method

        # Use HTTP (or browser fallback)
        try:
            logger.info(f"Fetching URL with HTTP: {url}")

            # Set a proper User-Agent to avoid 403 Forbidden errors
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }

            response = requests.get(url, timeout=30, headers=headers)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("Content-Type", "").lower()
            logger.info(f"Content-Type: {content_type}")

            if "application/pdf" in content_type or url.lower().endswith(".pdf"):
                # Handle PDF content
                try:
                    pdf_bytes = io.BytesIO(response.content)
                    pdf_reader = PyPDF2.PdfReader(pdf_bytes)

                    # Extract text from all pages
                    text_parts = []
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        text = page.extract_text()
                        if text.strip():
                            text_parts.append(f"=== Page {page_num} ===\n{text}")

                    content = "\n\n".join(text_parts)
                    logger.info(f"Extracted text from PDF ({len(pdf_reader.pages)} pages, {len(content)} chars)")

                    # Print summary to stdout
                    print(f"Downloaded PDF: {url}")
                    print(f"Pages: {len(pdf_reader.pages)}")
                    print(f"Content length: {len(content)} chars")
                    if content:
                        print(f"First 500 chars:\n{content[:500]}...")

                    return content
                except ImportError:
                    error_msg = "ERROR: PyPDF2 not installed. Install with: pip install PyPDF2"
                    logger.error(error_msg)
                    print(error_msg)
                    return error_msg
                except Exception as pdf_error:
                    error_msg = f"ERROR: Failed to extract text from PDF: {pdf_error}"
                    logger.error(error_msg)
                    print(error_msg)
                    return error_msg
            else:
                # Handle HTML/text content
                html_content = response.text

                # Convert HTML to markdown using proper converter
                try:
                    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

                    # Parse HTML with BeautifulSoup
                    soup = BeautifulSoup(html_content, features="lxml")

                    # Remove script and style tags
                    for data in soup(["style", "script"]):
                        data.decompose()

                    # Convert to markdown
                    markdown_content = VerboseConverter().convert_soup(soup)

                    # Clean markdown
                    content = _clean_markdown(markdown_content)

                    # Normalize Unicode
                    content = unicodedata.normalize("NFKD", content)

                    logger.info(f"Converted HTML to markdown ({len(html_content)} -> {len(content)} chars)")
                except Exception as md_error:
                    # If markdown conversion fails, return raw HTML
                    logger.warning(f"Failed to convert HTML to markdown: {md_error}. Returning raw HTML.")
                    content = html_content

                # Print summary to stdout
                print(f"Downloaded {url}")
                print(f"Content length: {len(content)} chars")
                print(f"First 500 chars:\n{content[:500]}...")

                return content
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            error_msg = f"ERROR: Failed to download {url}: {e}"
            print(error_msg)
            return error_msg

    @staticmethod
    def _get_url_with_browser(url: str) -> str:
        """
        Fetch URL using Playwright browser (for JavaScript-heavy sites).

        Args:
            url: URL to fetch

        Returns:
            Page content as markdown
        """
        # Run async browser fetch in sync context
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're already in an async context - need to create task

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, Web._async_get_url_with_browser(url))
                return future.result()
        else:
            # Not in async context - can run directly
            return asyncio.run(Web._async_get_url_with_browser(url))

    @staticmethod
    async def _async_get_url_with_browser(url: str) -> str:
        """
        Async implementation of browser-based URL fetching.

        Args:
            url: URL to fetch

        Returns:
            Page content as markdown
        """
        browser = await _get_browser()

        # Get rendered HTML from browser
        html_content = await browser.get_url(url)

        # Convert to markdown
        try:
            warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, features="lxml")

            # Remove script and style tags
            for data in soup(["style", "script"]):
                data.decompose()

            # Convert to markdown
            markdown_content = VerboseConverter().convert_soup(soup)

            # Clean markdown
            content = _clean_markdown(markdown_content)

            # Normalize Unicode
            content = unicodedata.normalize("NFKD", content)

            logger.info(f"Converted browser HTML to markdown ({len(html_content)} -> {len(content)} chars)")

            # Print summary
            print(f"Downloaded (browser): {url}")
            print(f"Content length: {len(content)} chars")
            print(f"First 500 chars:\n{content[:500]}...")

            return content

        except Exception as e:
            logger.error(f"Failed to convert browser HTML to markdown: {e}")
            # Return raw HTML on conversion failure
            return html_content
