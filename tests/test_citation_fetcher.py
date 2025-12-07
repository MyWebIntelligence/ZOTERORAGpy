"""
Tests for Citation Fetcher Module
==================================

Test suite for citation_fetcher.py covering:
- PDF content fetching
- HTML content fetching
- Fallback strategies
- Retry logic
- Error handling

Author: RAGpy Team
Date: 2025-12-05
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from io import BytesIO

from app.utils.citation_fetcher import (
    fetch_citation_content,
    fetch_multiple_citations,
    get_fetch_statistics,
    _fetch_pdf_content,
    _fetch_html_content,
    _clean_text,
    ContentFetchError
)


# Mock PDF content
MOCK_PDF_BYTES = b'%PDF-1.4\n%Test PDF\n'  # Minimal valid PDF header


class TestFetchCitationContent:
    """Test suite for main fetch_citation_content function."""

    @pytest.mark.asyncio
    async def test_fetch_pdf_success(self):
        """Test successful PDF content fetching."""
        with patch('app.utils.citation_fetcher._fetch_pdf_content') as mock_pdf:
            mock_pdf.return_value = "This is PDF content from the paper."

            content, source = await fetch_citation_content(
                article_url=None,
                fulltext_url="https://example.com/paper.pdf"
            )

            assert source == "pdf"
            assert "PDF content" in content
            assert len(content) > 0

    @pytest.mark.asyncio
    async def test_fetch_html_success(self):
        """Test successful HTML content fetching."""
        with patch('app.utils.citation_fetcher._fetch_html_content') as mock_html:
            mock_html.return_value = "This is HTML content from the article."

            content, source = await fetch_citation_content(
                article_url="https://example.com/article",
                fulltext_url=None
            )

            assert source == "html"
            assert "HTML content" in content
            assert len(content) > 0

    @pytest.mark.asyncio
    async def test_fallback_pdf_to_html(self):
        """Test fallback from failed PDF to successful HTML."""
        with patch('app.utils.citation_fetcher._fetch_pdf_content') as mock_pdf, \
             patch('app.utils.citation_fetcher._fetch_html_content') as mock_html:

            # PDF fails
            mock_pdf.side_effect = ContentFetchError("PDF not found")
            # HTML succeeds
            mock_html.return_value = "HTML fallback content successfully fetched."

            content, source = await fetch_citation_content(
                article_url="https://example.com/article",
                fulltext_url="https://example.com/paper.pdf"
            )

            assert source == "html"
            assert "HTML fallback" in content

    @pytest.mark.asyncio
    async def test_fallback_to_none(self):
        """Test fallback to empty content when both PDF and HTML fail."""
        with patch('app.utils.citation_fetcher._fetch_pdf_content') as mock_pdf, \
             patch('app.utils.citation_fetcher._fetch_html_content') as mock_html:

            # Both fail
            mock_pdf.side_effect = ContentFetchError("PDF error")
            mock_html.side_effect = ContentFetchError("HTML error")

            content, source = await fetch_citation_content(
                article_url="https://example.com/article",
                fulltext_url="https://example.com/paper.pdf"
            )

            # Should return empty content with "none" source (never block)
            assert source == "none"
            assert content == ""

    @pytest.mark.asyncio
    async def test_no_urls_provided(self):
        """Test behavior when no URLs are provided."""
        content, source = await fetch_citation_content(
            article_url=None,
            fulltext_url=None
        )

        # Should return empty content immediately
        assert source == "none"
        assert content == ""

    @pytest.mark.asyncio
    async def test_content_truncation(self):
        """Test content is truncated to max_chars."""
        long_content = "A" * 20000  # 20k characters

        with patch('app.utils.citation_fetcher._fetch_pdf_content') as mock_pdf:
            mock_pdf.return_value = long_content

            content, source = await fetch_citation_content(
                article_url=None,
                fulltext_url="https://example.com/paper.pdf",
                max_chars=5000
            )

            assert len(content) == 5000
            assert content == "A" * 5000

    @pytest.mark.asyncio
    async def test_retry_mechanism(self):
        """Test retry logic attempts fetching multiple times."""
        with patch('app.utils.citation_fetcher._fetch_pdf_content') as mock_pdf:
            # Fail first 2 attempts, succeed on 3rd
            mock_pdf.side_effect = [
                ContentFetchError("Timeout"),
                ContentFetchError("Timeout"),
                "Success after retries"
            ]

            content, source = await fetch_citation_content(
                article_url=None,
                fulltext_url="https://example.com/paper.pdf",
                max_retries=3
            )

            # Should succeed after retries
            assert source == "pdf"
            assert "Success" in content
            assert mock_pdf.call_count == 3

    @pytest.mark.asyncio
    async def test_short_content_triggers_fallback(self):
        """Test that content shorter than 100 chars triggers fallback."""
        with patch('app.utils.citation_fetcher._fetch_pdf_content') as mock_pdf, \
             patch('app.utils.citation_fetcher._fetch_html_content') as mock_html:

            # PDF returns very short content (< 100 chars)
            mock_pdf.return_value = "Short"
            # HTML has sufficient content
            mock_html.return_value = "A" * 150

            content, source = await fetch_citation_content(
                article_url="https://example.com/article",
                fulltext_url="https://example.com/paper.pdf"
            )

            # Should fallback to HTML due to short PDF content
            assert source == "html"
            assert len(content) == 150


class TestFetchPdfContent:
    """Test suite for PDF content fetching."""

    @pytest.mark.asyncio
    async def test_pdf_fetch_success(self):
        """Test successful PDF text extraction."""
        # Create a mock PDF document
        mock_pdf_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Extracted PDF text content."
        mock_pdf_doc.page_count = 1
        mock_pdf_doc.__getitem__.return_value = mock_page

        with patch('aiohttp.ClientSession.get') as mock_get, \
             patch('fitz.open') as mock_fitz_open:

            # Mock HTTP response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.read = AsyncMock(return_value=MOCK_PDF_BYTES)
            mock_get.return_value.__aenter__.return_value = mock_response

            # Mock fitz.open
            mock_fitz_open.return_value = mock_pdf_doc

            content = await _fetch_pdf_content("https://example.com/paper.pdf")

            assert "PDF text content" in content

    @pytest.mark.asyncio
    async def test_pdf_fetch_http_error(self):
        """Test PDF fetch handles HTTP errors."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock 404 response
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_get.return_value.__aenter__.return_value = mock_response

            with pytest.raises(ContentFetchError) as exc_info:
                await _fetch_pdf_content("https://example.com/missing.pdf")

            assert "HTTP 404" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_pdf_fetch_invalid_pdf(self):
        """Test PDF fetch handles invalid PDF files."""
        with patch('aiohttp.ClientSession.get') as mock_get, \
             patch('fitz.open') as mock_fitz_open:

            # Mock HTTP response with invalid PDF
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.read = AsyncMock(return_value=b'Not a PDF')
            mock_get.return_value.__aenter__.return_value = mock_response

            # Mock fitz error
            mock_fitz_open.side_effect = Exception("Invalid PDF format")

            with pytest.raises(ContentFetchError):
                await _fetch_pdf_content("https://example.com/invalid.pdf")


class TestFetchHtmlContent:
    """Test suite for HTML content fetching."""

    @pytest.mark.asyncio
    async def test_html_fetch_success(self):
        """Test successful HTML text extraction."""
        html_content = """
        <html>
            <body>
                <article>
                    <h1>Article Title</h1>
                    <p>This is the main article content.</p>
                </article>
            </body>
        </html>
        """

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value=html_content)
            mock_get.return_value.__aenter__.return_value = mock_response

            content = await _fetch_html_content("https://example.com/article")

            assert len(content) > 0
            assert "Article Title" in content or "article content" in content.lower()

    @pytest.mark.asyncio
    async def test_html_fetch_http_error(self):
        """Test HTML fetch handles HTTP errors."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 403
            mock_get.return_value.__aenter__.return_value = mock_response

            with pytest.raises(ContentFetchError) as exc_info:
                await _fetch_html_content("https://example.com/forbidden")

            assert "HTTP 403" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_html_removes_unwanted_elements(self):
        """Test HTML extraction removes scripts, styles, etc."""
        html_content = """
        <html>
            <head>
                <script>alert('test');</script>
                <style>body { color: red; }</style>
            </head>
            <body>
                <nav>Navigation</nav>
                <article>
                    <p>Main content here.</p>
                </article>
                <footer>Footer</footer>
            </body>
        </html>
        """

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value=html_content)
            mock_get.return_value.__aenter__.return_value = mock_response

            content = await _fetch_html_content("https://example.com/article")

            # Main content should be present
            assert "Main content" in content
            # Unwanted elements should be removed
            assert "alert" not in content
            assert "color: red" not in content


class TestCleanText:
    """Test suite for text cleaning function."""

    def test_clean_removes_control_characters(self):
        """Test cleaning removes control characters."""
        dirty = "Text\x00with\x1fcontrol\x7fchars"
        clean = _clean_text(dirty)
        assert "\x00" not in clean
        assert "\x1f" not in clean
        assert "\x7f" not in clean

    def test_clean_normalizes_whitespace(self):
        """Test cleaning normalizes whitespace."""
        dirty = "Too    many     spaces\t\t\ttabs"
        clean = _clean_text(dirty)
        assert "  " not in clean
        assert "\t" not in clean

    def test_clean_normalizes_newlines(self):
        """Test cleaning normalizes excessive newlines."""
        dirty = "Line1\n\n\n\n\nLine2"
        clean = _clean_text(dirty)
        # Should reduce to max 2 consecutive newlines
        assert "\n\n\n" not in clean

    def test_clean_strips_line_whitespace(self):
        """Test cleaning strips whitespace per line."""
        dirty = "  Line 1  \n  Line 2  \n"
        clean = _clean_text(dirty)
        assert clean == "Line 1\nLine 2"


class TestFetchMultipleCitations:
    """Test suite for concurrent multiple citation fetching."""

    @pytest.mark.asyncio
    async def test_fetch_multiple_success(self):
        """Test fetching multiple citations concurrently."""
        citations = [
            {"article_url": "https://example.com/1", "fulltext_url": None},
            {"article_url": None, "fulltext_url": "https://example.com/2.pdf"},
            {"article_url": "https://example.com/3", "fulltext_url": None}
        ]

        with patch('app.utils.citation_fetcher.fetch_citation_content') as mock_fetch:
            # Mock different results for each
            mock_fetch.side_effect = [
                ("Content 1", "html"),
                ("Content 2", "pdf"),
                ("Content 3", "html")
            ]

            results = await fetch_multiple_citations(citations)

            assert len(results) == 3
            assert results[0] == ("Content 1", "html")
            assert results[1] == ("Content 2", "pdf")
            assert results[2] == ("Content 3", "html")

    @pytest.mark.asyncio
    async def test_fetch_multiple_handles_exceptions(self):
        """Test handling of exceptions in concurrent fetching."""
        citations = [
            {"article_url": "https://example.com/1", "fulltext_url": None},
            {"article_url": "https://example.com/2", "fulltext_url": None}
        ]

        with patch('app.utils.citation_fetcher.fetch_citation_content') as mock_fetch:
            # First succeeds, second raises exception
            mock_fetch.side_effect = [
                ("Content 1", "html"),
                Exception("Network error")
            ]

            results = await fetch_multiple_citations(citations)

            # Should handle exception gracefully
            assert len(results) == 2
            assert results[0] == ("Content 1", "html")
            assert results[1] == ("", "none")  # Exception replaced with empty


class TestGetFetchStatistics:
    """Test suite for fetch statistics function."""

    def test_statistics_empty_results(self):
        """Test statistics for empty results."""
        stats = get_fetch_statistics([])
        assert stats['total'] == 0
        assert stats['from_pdf'] == 0
        assert stats['from_html'] == 0
        assert stats['failed'] == 0
        assert stats['avg_length'] == 0

    def test_statistics_mixed_results(self):
        """Test statistics for mixed success/failure results."""
        results = [
            ("Content from PDF", "pdf"),
            ("Content from HTML", "html"),
            ("", "none"),
            ("Another PDF content", "pdf"),
            ("", "none")
        ]

        stats = get_fetch_statistics(results)

        assert stats['total'] == 5
        assert stats['from_pdf'] == 2
        assert stats['from_html'] == 1
        assert stats['failed'] == 2

    def test_statistics_average_length(self):
        """Test average length calculation."""
        results = [
            ("A" * 100, "pdf"),   # 100 chars
            ("B" * 200, "html"),  # 200 chars
            ("", "none")          # 0 chars
        ]

        stats = get_fetch_statistics(results)

        assert stats['avg_length'] == pytest.approx(100.0)  # (100+200+0)/3
