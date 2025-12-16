"""
PDF Downloader Module
=====================

Downloads PDF files from URLs with fallback HTML-to-PDF conversion.
Designed for attaching PDFs to Zotero items from Publish or Perish imports.

Features:
- Direct PDF download with content-type and magic bytes detection
- HTML-to-PDF conversion via pdfkit (wkhtmltopdf) or Playwright
- File size limits and timeout handling
- MD5 hash calculation for Zotero upload API
- Retry logic with exponential backoff

Author: RAGpy Team
Date: 2025-12-11
"""

import asyncio
import hashlib
import logging
import shutil
import time
import random
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse, unquote

import aiohttp

# PDF magic bytes
PDF_MAGIC_BYTES = b'%PDF-'

# Try importing pdfkit (optional dependency)
try:
    import pdfkit
    WKHTMLTOPDF_PATH = shutil.which("wkhtmltopdf")
    PDFKIT_AVAILABLE = WKHTMLTOPDF_PATH is not None
except ImportError:
    PDFKIT_AVAILABLE = False
    WKHTMLTOPDF_PATH = None

# Try importing playwright (optional dependency)
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Configure logger
logger = logging.getLogger(__name__)

# List of 10 credible, modern User-Agents (Chrome, Firefox, Safari on Windows/Mac/Linux)
USER_AGENTS = [
    # Chrome on Windows
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    # Chrome on macOS
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    # Firefox on Windows
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0',
    # Firefox on macOS
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:123.0) Gecko/20100101 Firefox/123.0',
    # Safari on macOS
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15',
    # Edge on Windows
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0'
]


@dataclass
class PDFDownloadResult:
    """
    Result of a PDF download attempt.

    Attributes:
        success: Whether download was successful
        pdf_bytes: Raw PDF file content (None if failed)
        filename: Suggested filename for the PDF
        md5_hash: MD5 hash of pdf_bytes (for Zotero upload API)
        filesize: Size of pdf_bytes in bytes
        source: Source of PDF ("direct", "html_conversion", "failed")
        error: Error message if download failed
        mtime: Modification time in milliseconds (current time)
    """
    success: bool
    pdf_bytes: Optional[bytes] = None
    filename: str = "document.pdf"
    md5_hash: str = ""
    filesize: int = 0
    source: str = "direct"
    error: Optional[str] = None
    mtime: int = 0

    def __post_init__(self):
        """Calculate mtime if not set."""
        if self.mtime == 0 and self.pdf_bytes:
            self.mtime = int(time.time() * 1000)


async def download_pdf(
    fulltext_url: str,
    timeout: int = 60,
    max_size_mb: int = 50,
    max_retries: int = 2,
    retry_delay: float = 2.0,
    convert_html: bool = True
) -> PDFDownloadResult:
    """
    Download PDF from URL with HTML-to-PDF fallback.

    Strategy:
    1. Attempt direct PDF download (check content-type and magic bytes)
    2. If HTML received and convert_html=True, convert to PDF
    3. Return result with PDF bytes and metadata

    Args:
        fulltext_url: URL to download PDF from
        timeout: Download timeout in seconds
        max_size_mb: Maximum file size in MB (default: 50)
        max_retries: Number of retry attempts
        retry_delay: Base delay between retries in seconds
        convert_html: Whether to convert HTML to PDF if HTML is returned

    Returns:
        PDFDownloadResult with PDF bytes if successful

    Examples:
        >>> result = await download_pdf("https://arxiv.org/pdf/2024.12345.pdf")
        >>> if result.success:
        ...     print(f"Downloaded {result.filesize} bytes, MD5: {result.md5_hash}")
    """
    max_size_bytes = max_size_mb * 1024 * 1024

    logger.info(f"Downloading PDF from: {fulltext_url[:100]}...")

    for attempt in range(max_retries):
        try:
            # Select a random user agent for this attempt
            current_user_agent = random.choice(USER_AGENTS)

            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': current_user_agent,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Referer': 'https://www.google.com/',
                    'Upgrade-Insecure-Requests': '1',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'cross-site',
                    'Sec-Fetch-User': '?1',
                    'Sec-Ch-Ua': '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
                    'Sec-Ch-Ua-Mobile': '?0',
                    'Sec-Ch-Ua-Platform': '"macOS"'
                }

                async with session.get(
                    fulltext_url,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                    headers=headers,
                    allow_redirects=True
                ) as response:

                    # Check HTTP status
                    if response.status == 403:
                        logger.warning(f"Access forbidden (403) for: {fulltext_url[:100]} - falling back to browser/conversion")
                        if convert_html:
                            return await _convert_html_to_pdf(fulltext_url, timeout)
                        
                        return PDFDownloadResult(
                            success=False,
                            source="failed",
                            error=f"Access forbidden (HTTP 403) - likely paywall"
                        )

                    if response.status == 404:
                        return PDFDownloadResult(
                            success=False,
                            source="failed",
                            error=f"File not found (HTTP 404)"
                        )

                    if response.status != 200:
                        raise aiohttp.ClientError(f"HTTP {response.status}")

                    # Check content type
                    content_type = response.headers.get('Content-Type', '').lower()
                    is_pdf_content_type = 'application/pdf' in content_type

                    # Check content length
                    content_length = response.headers.get('Content-Length')
                    if content_length and int(content_length) > max_size_bytes:
                        return PDFDownloadResult(
                            success=False,
                            source="failed",
                            error=f"File too large ({int(content_length) / 1024 / 1024:.1f} MB > {max_size_mb} MB limit)"
                        )

                    # Read content
                    content = await response.read()

                    # Check actual size
                    if len(content) > max_size_bytes:
                        return PDFDownloadResult(
                            success=False,
                            source="failed",
                            error=f"File too large ({len(content) / 1024 / 1024:.1f} MB > {max_size_mb} MB limit)"
                        )

                    # Check magic bytes
                    is_pdf_magic = content[:5] == PDF_MAGIC_BYTES

                    if is_pdf_content_type or is_pdf_magic:
                        # Direct PDF download successful
                        filename = _extract_filename(fulltext_url, response, "document.pdf")
                        md5_hash = hashlib.md5(content).hexdigest()

                        logger.info(f"PDF downloaded successfully: {len(content)} bytes, MD5: {md5_hash[:8]}...")

                        return PDFDownloadResult(
                            success=True,
                            pdf_bytes=content,
                            filename=filename,
                            md5_hash=md5_hash,
                            filesize=len(content),
                            source="direct",
                            mtime=int(time.time() * 1000)
                        )

                    # Content is HTML - try conversion if enabled
                    if convert_html and ('text/html' in content_type or content[:1] == b'<'):
                        logger.info("Received HTML content, attempting conversion to PDF...")
                        return await _convert_html_to_pdf(
                            fulltext_url,
                            timeout=timeout
                        )

                    # Unknown content type
                    return PDFDownloadResult(
                        success=False,
                        source="failed",
                        error=f"Unexpected content type: {content_type}"
                    )

        except asyncio.TimeoutError:
            logger.warning(f"Timeout downloading PDF (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))

        except aiohttp.ClientError as e:
            logger.warning(f"Network error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))

        except Exception as e:
            logger.error(f"Unexpected error downloading PDF: {str(e)}")
            return PDFDownloadResult(
                success=False,
                source="failed",
                error=f"Unexpected error: {str(e)}"
            )

    # All retries failed
    return PDFDownloadResult(
        success=False,
        source="failed",
        error=f"Failed after {max_retries} attempts"
    )


async def _convert_html_to_pdf(
    url: str,
    timeout: int = 60
) -> PDFDownloadResult:
    """
    Convert HTML page to PDF using pdfkit or Playwright.

    Args:
        url: URL of HTML page to convert
        timeout: Conversion timeout in seconds

    Returns:
        PDFDownloadResult with converted PDF bytes
    """
    # Try pdfkit first (faster if available)
    if PDFKIT_AVAILABLE:
        try:
            logger.info("Converting HTML to PDF using pdfkit (wkhtmltopdf)...")

            # Run pdfkit in executor to not block async loop
            loop = asyncio.get_event_loop()
            pdf_bytes = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: _pdfkit_convert(url)
                ),
                timeout=timeout
            )

            if pdf_bytes:
                md5_hash = hashlib.md5(pdf_bytes).hexdigest()
                filename = _url_to_filename(url)

                logger.info(f"HTML converted to PDF successfully: {len(pdf_bytes)} bytes")

                return PDFDownloadResult(
                    success=True,
                    pdf_bytes=pdf_bytes,
                    filename=filename,
                    md5_hash=md5_hash,
                    filesize=len(pdf_bytes),
                    source="html_conversion",
                    mtime=int(time.time() * 1000)
                )

        except asyncio.TimeoutError:
            logger.warning("pdfkit conversion timed out")
        except Exception as e:
            logger.warning(f"pdfkit conversion failed: {str(e)}")

    # Fallback to Playwright
    if PLAYWRIGHT_AVAILABLE:
        try:
            logger.info("Converting HTML to PDF using Playwright...")

            pdf_bytes = await _playwright_convert(url, timeout)

            if pdf_bytes:
                md5_hash = hashlib.md5(pdf_bytes).hexdigest()
                filename = _url_to_filename(url)

                logger.info(f"HTML converted to PDF successfully: {len(pdf_bytes)} bytes")

                return PDFDownloadResult(
                    success=True,
                    pdf_bytes=pdf_bytes,
                    filename=filename,
                    md5_hash=md5_hash,
                    filesize=len(pdf_bytes),
                    source="html_conversion",
                    mtime=int(time.time() * 1000)
                )

        except Exception as e:
            logger.warning(f"Playwright conversion failed: {str(e)}")

    # Both methods failed
    logger.warning("HTML to PDF conversion failed - no available converters")
    return PDFDownloadResult(
        success=False,
        source="failed",
        error="HTML to PDF conversion failed (pdfkit and playwright unavailable or failed)"
    )


def _pdfkit_convert(url: str) -> Optional[bytes]:
    """
    Convert URL to PDF using pdfkit (synchronous).

    Args:
        url: URL to convert

    Returns:
        PDF bytes or None if failed
    """
    options = {
        'page-size': 'A4',
        'encoding': 'UTF-8',
        'no-outline': None,
        'quiet': '',
        'print-media-type': '',
        'no-stop-slow-scripts': '',
        'javascript-delay': '2000'
    }

    try:
        pdf_bytes = pdfkit.from_url(
            url,
            False,  # Return bytes, don't write to file
            options=options,
            configuration=pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)
        )
        return pdf_bytes
    except Exception as e:
        logger.warning(f"pdfkit error: {str(e)}")
        return None


# Common cookie consent button selectors (multi-language support)
COOKIE_ACCEPT_SELECTORS = [
    # Text-based selectors (English)
    'button:has-text("Accept")',
    'button:has-text("Accept all")',
    'button:has-text("Accept All")',
    'button:has-text("I agree")',
    'button:has-text("Agree")',
    'button:has-text("OK")',
    'button:has-text("Got it")',
    'button:has-text("Allow")',
    'button:has-text("Allow all")',
    # Text-based selectors (French)
    'button:has-text("Accepter")',
    'button:has-text("Tout accepter")',
    'button:has-text("J\'accepte")',
    'button:has-text("Autoriser")',
    # Text-based selectors (German)
    'button:has-text("Akzeptieren")',
    'button:has-text("Alle akzeptieren")',
    # ARIA attributes
    '[aria-label*="accept" i]',
    '[aria-label*="Accept" i]',
    '[aria-label*="consent" i]',
    '[aria-label*="cookie" i]',
    # Common class/ID patterns
    '.cookie-accept',
    '.accept-cookies',
    '.cookie-consent-accept',
    '#accept-cookies',
    '#cookie-accept',
    '#onetrust-accept-btn-handler',
    '#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll',
    '.cc-accept',
    '.cc-allow',
    # Data attributes
    '[data-action="accept"]',
    '[data-consent="accept"]',
    '[data-cookieconsent="accept"]',
    # Generic patterns for cookie/consent containers
    '[class*="cookie"] button:has-text("Accept")',
    '[class*="consent"] button:has-text("Accept")',
    '[class*="cookie"] button:has-text("OK")',
    '[class*="gdpr"] button:has-text("Accept")',
]


async def _dismiss_popups(page, timeout_ms: int = 3000) -> bool:
    """
    Attempt to dismiss cookie consent banners and other popups.

    Uses a layered strategy:
    1. Try clicking common accept buttons (with short timeout per selector)
    2. Fallback to hiding overlay elements via CSS injection

    Args:
        page: Playwright page object
        timeout_ms: Max time to wait for each selector in milliseconds

    Returns:
        True if any popup was dismissed, False otherwise
    """
    dismissed = False

    # Layer 1: Try clicking common accept buttons
    for selector in COOKIE_ACCEPT_SELECTORS:
        try:
            # Check if element exists and is visible (short timeout)
            locator = page.locator(selector).first
            if await locator.is_visible(timeout=500):
                await locator.click(timeout=timeout_ms)
                logger.debug(f"Clicked cookie consent button: {selector}")
                dismissed = True
                # Wait for popup to disappear
                await asyncio.sleep(0.3)
                break
        except Exception:
            # Element not found or not clickable, try next selector
            continue

    # Layer 2: CSS injection to hide remaining overlays
    try:
        hidden_count = await page.evaluate("""
            () => {
                let hiddenCount = 0;
                const hideSelectors = [
                    '[class*="cookie"]',
                    '[id*="cookie"]',
                    '[class*="consent"]',
                    '[id*="consent"]',
                    '[class*="gdpr"]',
                    '[id*="gdpr"]',
                    '[class*="cc-"]',
                    '[class*="CookieConsent"]',
                    '[class*="cookie-banner"]',
                    '[class*="cookie-notice"]',
                    '[role="dialog"][aria-modal="true"]',
                ];

                hideSelectors.forEach(sel => {
                    document.querySelectorAll(sel).forEach(el => {
                        // Only hide if it's an overlay-like element (large, covering viewport)
                        const rect = el.getBoundingClientRect();
                        const isOverlay = (
                            rect.width > window.innerWidth * 0.3 ||
                            rect.height > 100 ||
                            el.style.position === 'fixed' ||
                            window.getComputedStyle(el).position === 'fixed'
                        );
                        if (isOverlay && el.offsetParent !== null) {
                            el.style.display = 'none';
                            hiddenCount++;
                        }
                    });
                });

                // Restore body scroll if blocked by overlay
                document.body.style.overflow = 'auto';
                document.documentElement.style.overflow = 'auto';

                return hiddenCount;
            }
        """)

        if hidden_count > 0:
            logger.debug(f"Hidden {hidden_count} overlay elements via CSS injection")
            dismissed = True

    except Exception as e:
        logger.debug(f"CSS injection for popups failed: {e}")

    return dismissed


async def _playwright_convert(url: str, timeout: int = 60) -> Optional[bytes]:
    """
    Convert URL to PDF using Playwright (async).

    Includes automatic handling of:
    - JavaScript alert/confirm/prompt dialogs
    - Cookie consent banners and GDPR popups
    - Modal overlays

    Args:
        url: URL to convert
        timeout: Timeout in seconds

    Returns:
        PDF bytes or None if failed
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)

            try:
                # Use a random user agent for the browser context
                ua = random.choice(USER_AGENTS)
                logger.info(f"Playwright using User-Agent: {ua[:50]}...")

                context = await browser.new_context(user_agent=ua)
                page = await context.new_page()

                # Auto-dismiss JavaScript dialogs (alert, confirm, prompt)
                page.on('dialog', lambda dialog: asyncio.create_task(dialog.dismiss()))

                # Navigate with timeout
                await page.goto(url, wait_until='networkidle', timeout=timeout * 1000)

                # Wait a bit for any remaining JS to settle
                await asyncio.sleep(0.5)

                # Attempt to dismiss cookie consent banners and other popups
                popup_dismissed = await _dismiss_popups(page)
                if popup_dismissed:
                    logger.debug("Cookie consent or popup was dismissed")
                    # Wait for DOM to stabilize after dismissing popups
                    await asyncio.sleep(0.5)

                # Generate PDF
                pdf_bytes = await page.pdf(
                    format='A4',
                    print_background=True,
                    margin={
                        'top': '20mm',
                        'bottom': '20mm',
                        'left': '15mm',
                        'right': '15mm'
                    }
                )

                return pdf_bytes

            finally:
                await browser.close()

    except Exception as e:
        logger.warning(f"Playwright error: {str(e)}")
        return None


def _extract_filename(url: str, response: aiohttp.ClientResponse, default: str = "document.pdf") -> str:
    """
    Extract filename from response headers or URL.

    Args:
        url: Source URL
        response: HTTP response
        default: Default filename if extraction fails

    Returns:
        Extracted or default filename
    """
    # Try Content-Disposition header
    content_disposition = response.headers.get('Content-Disposition', '')
    if 'filename=' in content_disposition:
        try:
            # Handle both filename= and filename*=
            if 'filename*=' in content_disposition:
                # RFC 5987 format: filename*=UTF-8''encoded_name
                parts = content_disposition.split('filename*=')[1].split("'")
                if len(parts) >= 3:
                    filename = unquote(parts[2].strip(' "'))
                    if filename:
                        return _sanitize_filename(filename)

            # Standard format: filename="name.pdf"
            parts = content_disposition.split('filename=')[1]
            filename = parts.split(';')[0].strip(' "\'')
            if filename:
                return _sanitize_filename(filename)

        except Exception:
            pass

    # Fall back to URL parsing
    return _url_to_filename(url, default)


def _url_to_filename(url: str, default: str = "document.pdf") -> str:
    """
    Extract filename from URL path.

    Args:
        url: Source URL
        default: Default filename if extraction fails

    Returns:
        Extracted or default filename
    """
    try:
        parsed = urlparse(url)
        path = unquote(parsed.path)

        if path and '/' in path:
            filename = path.split('/')[-1]
            if filename and '.' in filename:
                return _sanitize_filename(filename)

        # Use domain + path hash for unique name
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        domain = parsed.netloc.replace('.', '_')
        return f"{domain}_{url_hash}.pdf"

    except Exception:
        return default


def _sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system use.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove or replace dangerous characters
    dangerous_chars = '<>:"/\\|?*'
    for char in dangerous_chars:
        filename = filename.replace(char, '_')

    # Limit length
    if len(filename) > 200:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, 'pdf')
        filename = f"{name[:190]}.{ext}"

    # Ensure PDF extension
    if not filename.lower().endswith('.pdf'):
        filename = f"{filename}.pdf"

    return filename


# Utility function to check conversion capabilities
def get_conversion_capabilities() -> dict:
    """
    Get available HTML-to-PDF conversion capabilities.

    Returns:
        Dictionary with availability status:
        - pdfkit_available: bool
        - playwright_available: bool
        - wkhtmltopdf_path: str or None
    """
    return {
        'pdfkit_available': PDFKIT_AVAILABLE,
        'playwright_available': PLAYWRIGHT_AVAILABLE,
        'wkhtmltopdf_path': WKHTMLTOPDF_PATH
    }
