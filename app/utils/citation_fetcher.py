"""
Citation Web Content Fetcher Module
====================================

This module handles fetching web content for citations from PDF and HTML sources.
It implements a fallback strategy to ensure the filtering process never blocks:
1. Try fulltext_url (PDF) → extract text via PyMuPDF
2. Try article_url (HTML) → extract text via BeautifulSoup
3. Fallback to empty content (use metadata only for LLM filtering)

Features:
- Automatic retry with exponential backoff
- Content truncation to control token usage
- Timeout handling
- Comprehensive error logging

Author: RAGpy Team
Date: 2025-12-05
"""

import asyncio
import logging
from typing import Tuple, Optional
from io import BytesIO
import re

import aiohttp
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import html2text

# Configure logger
logger = logging.getLogger(__name__)


class ContentFetchError(Exception):
    """Custom exception for content fetching errors."""
    pass


async def fetch_citation_content(
    article_url: Optional[str],
    fulltext_url: Optional[str],
    timeout: int = 30,
    max_chars: int = 10000,
    max_retries: int = 2,
    retry_delay: float = 2.0
) -> Tuple[str, str]:
    """
    Fetch web content for a citation with fallback strategy.

    This function attempts to extract text content from citation URLs using
    a priority-based approach:
    1. PDF extraction from fulltext_url
    2. HTML extraction from article_url
    3. Empty content (never blocks the process)

    Args:
        article_url: URL to article HTML page
        fulltext_url: URL to full-text PDF
        timeout: Request timeout in seconds (default: 30)
        max_chars: Maximum content length to return (default: 10000)
        max_retries: Number of retry attempts per URL (default: 2)
        retry_delay: Delay between retries in seconds (default: 2.0)

    Returns:
        Tuple of (content: str, source: str) where source is one of:
        - "pdf": Content extracted from PDF
        - "html": Content extracted from HTML
        - "none": No content fetched (both URLs failed or missing)

    Examples:
        >>> content, source = await fetch_citation_content(
        ...     article_url="https://example.com/article",
        ...     fulltext_url="https://arxiv.org/pdf/2024.12345.pdf"
        ... )
        >>> print(f"Fetched {len(content)} chars from {source}")
        Fetched 8532 chars from pdf

    Notes:
        - Never raises exceptions (returns empty content on failure)
        - Automatically retries failed requests
        - Truncates content to max_chars for token control
        - Logs all fetch attempts and failures
    """
    # Priority 1: Try PDF extraction
    if fulltext_url:
        logger.info(f"Attempting PDF fetch from: {fulltext_url[:100]}...")
        for attempt in range(max_retries):
            try:
                content = await _fetch_pdf_content(
                    fulltext_url,
                    timeout=timeout,
                    max_chars=max_chars
                )
                if content and len(content.strip()) > 100:
                    logger.info(f"Successfully fetched PDF content ({len(content)} chars)")
                    return (content[:max_chars], "pdf")
                else:
                    logger.warning(f"PDF content too short ({len(content)} chars)")
            except Exception as e:
                logger.warning(f"PDF fetch attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))

    # Priority 2: Try HTML extraction
    if article_url:
        logger.info(f"Attempting HTML fetch from: {article_url[:100]}...")
        for attempt in range(max_retries):
            try:
                content = await _fetch_html_content(
                    article_url,
                    timeout=timeout,
                    max_chars=max_chars
                )
                if content and len(content.strip()) > 100:
                    logger.info(f"Successfully fetched HTML content ({len(content)} chars)")
                    return (content[:max_chars], "html")
                else:
                    logger.warning(f"HTML content too short ({len(content)} chars)")
            except Exception as e:
                logger.warning(f"HTML fetch attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))

    # Fallback: Return empty content (use metadata only)
    logger.info("All fetch attempts failed - returning empty content for metadata-only filtering")
    return ("", "none")


async def _fetch_pdf_content(
    url: str,
    timeout: int = 30,
    max_chars: int = 10000
) -> str:
    """
    Fetch and extract text from a PDF URL.

    Args:
        url: PDF file URL
        timeout: Request timeout in seconds
        max_chars: Maximum characters to extract

    Returns:
        Extracted text content

    Raises:
        ContentFetchError: If PDF fetch or extraction fails
    """
    try:
        # Fetch PDF
        async with aiohttp.ClientSession() as session:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; RAGpy/1.0; +https://github.com/ragpy)'
            }
            async with session.get(url, timeout=timeout, headers=headers) as response:
                if response.status != 200:
                    raise ContentFetchError(f"HTTP {response.status} for PDF URL")

                pdf_bytes = await response.read()

        # Extract text with PyMuPDF
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []
        total_chars = 0

        for page_num in range(pdf_document.page_count):
            if total_chars >= max_chars:
                break

            page = pdf_document[page_num]
            page_text = page.get_text()

            # Clean text
            page_text = _clean_text(page_text)

            text_parts.append(page_text)
            total_chars += len(page_text)

        pdf_document.close()

        full_text = "\n".join(text_parts)
        return full_text[:max_chars]

    except aiohttp.ClientError as e:
        raise ContentFetchError(f"Network error fetching PDF: {str(e)}")
    except Exception as e:
        # Catch all PDF-related errors (PyMuPDF exceptions vary by version)
        raise ContentFetchError(f"PDF parsing error: {str(e)}")


async def _fetch_html_content(
    url: str,
    timeout: int = 30,
    max_chars: int = 10000
) -> str:
    """
    Fetch and extract text from an HTML URL.

    Args:
        url: HTML page URL
        timeout: Request timeout in seconds
        max_chars: Maximum characters to extract

    Returns:
        Extracted text content

    Raises:
        ContentFetchError: If HTML fetch or extraction fails
    """
    try:
        # Fetch HTML
        async with aiohttp.ClientSession() as session:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; RAGpy/1.0; +https://github.com/ragpy)'
            }
            async with session.get(url, timeout=timeout, headers=headers) as response:
                if response.status != 200:
                    raise ContentFetchError(f"HTTP {response.status} for HTML URL")

                html_content = await response.text()

        # Parse with BeautifulSoup
        soup = BeautifulSoup(html_content, 'lxml')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()

        # Extract main content
        # Try common article containers
        main_content = (
            soup.find('article') or
            soup.find('main') or
            soup.find('div', class_=re.compile(r'content|article|body', re.I)) or
            soup.find('body')
        )

        if not main_content:
            raise ContentFetchError("Could not locate main content in HTML")

        # Convert to markdown for better structure
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.ignore_emphasis = False
        h.body_width = 0  # Don't wrap lines

        text = h.handle(str(main_content))

        # Clean text
        text = _clean_text(text)

        return text[:max_chars]

    except aiohttp.ClientError as e:
        raise ContentFetchError(f"Network error fetching HTML: {str(e)}")
    except Exception as e:
        raise ContentFetchError(f"HTML parsing error: {str(e)}")


def _clean_text(text: str) -> str:
    """
    Clean extracted text content.

    Removes excessive whitespace, control characters, and normalizes formatting.

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text
    """
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    # Normalize whitespace
    text = re.sub(r'\t+', ' ', text)  # Tabs to spaces
    text = re.sub(r' +', ' ', text)  # Multiple spaces to one
    text = re.sub(r'\n\n+', '\n\n', text)  # Max 2 consecutive newlines

    # Remove leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


async def fetch_multiple_citations(
    citations: list,
    timeout: int = 30,
    max_chars: int = 10000,
    max_concurrent: int = 5
) -> list:
    """
    Fetch content for multiple citations concurrently.

    Args:
        citations: List of citation dicts with 'article_url' and 'fulltext_url'
        timeout: Request timeout per citation
        max_chars: Maximum chars per citation
        max_concurrent: Maximum concurrent fetches

    Returns:
        List of tuples [(content, source), ...] in same order as citations

    Examples:
        >>> citations = [
        ...     {"article_url": "https://example.com/1", "fulltext_url": None},
        ...     {"article_url": None, "fulltext_url": "https://arxiv.org/pdf/1.pdf"}
        ... ]
        >>> results = await fetch_multiple_citations(citations)
        >>> for i, (content, source) in enumerate(results):
        ...     print(f"Citation {i}: {len(content)} chars from {source}")
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_with_semaphore(citation):
        async with semaphore:
            return await fetch_citation_content(
                article_url=citation.get('article_url'),
                fulltext_url=citation.get('fulltext_url'),
                timeout=timeout,
                max_chars=max_chars
            )

    tasks = [fetch_with_semaphore(c) for c in citations]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions (replace with empty content)
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Citation {i} fetch failed: {str(result)}")
            processed_results.append(("", "none"))
        else:
            processed_results.append(result)

    return processed_results


def get_fetch_statistics(results: list) -> dict:
    """
    Generate statistics for fetch results.

    Args:
        results: List of (content, source) tuples

    Returns:
        Dictionary with statistics:
        - total: Total citations
        - from_pdf: Fetched from PDF
        - from_html: Fetched from HTML
        - failed: No content fetched
        - avg_length: Average content length

    Examples:
        >>> results = [("text1", "pdf"), ("text2", "html"), ("", "none")]
        >>> stats = get_fetch_statistics(results)
        >>> print(f"Success rate: {(stats['from_pdf'] + stats['from_html']) / stats['total']:.1%}")
    """
    if not results:
        return {
            'total': 0,
            'from_pdf': 0,
            'from_html': 0,
            'failed': 0,
            'avg_length': 0
        }

    source_counts = {'pdf': 0, 'html': 0, 'none': 0}
    total_length = 0

    for content, source in results:
        source_counts[source] += 1
        total_length += len(content)

    return {
        'total': len(results),
        'from_pdf': source_counts['pdf'],
        'from_html': source_counts['html'],
        'failed': source_counts['none'],
        'avg_length': total_length / len(results) if results else 0
    }
