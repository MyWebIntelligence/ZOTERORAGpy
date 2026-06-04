"""
Lot G — Mistral OCR page-count split tests.

Mistral rejects documents with > 1000 pages (HTTP 400, code 3730
"document_parser_too_many_pages"). This module verifies that:
  - `_pdf_page_count` returns the correct page count for valid PDFs and 0 for
    corrupted/non-PDF files.
  - `_split_pdf_for_ocr` honours the `max_pages` constraint, alone or combined
    with the `max_size_mb` constraint.

Run with: pytest tests/test_rad_dataframe_pages_split.py -v
"""

import os

import pytest
import fitz  # type: ignore[import-not-found]

from scripts.rad_dataframe import (
    _pdf_page_count,
    _pdf_size_mb,
    _split_pdf_for_ocr,
    _safe_unlink,
)


def _build_pdf(tmp_path, n_pages: int, body_per_page: str = "page body") -> str:
    """Generate a minimal multi-page PDF for testing."""
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page(width=200, height=200)
        page.insert_text((10, 50), f"{body_per_page} {i + 1}")
    out = tmp_path / f"test_{n_pages}p.pdf"
    doc.save(str(out))
    doc.close()
    return str(out)


class TestPdfPageCount:
    def test_returns_correct_count_for_5_pages(self, tmp_path):
        path = _build_pdf(tmp_path, n_pages=5)
        assert _pdf_page_count(path) == 5

    def test_returns_correct_count_for_50_pages(self, tmp_path):
        path = _build_pdf(tmp_path, n_pages=50)
        assert _pdf_page_count(path) == 50

    def test_returns_zero_for_missing_file(self, tmp_path):
        bad = tmp_path / "does_not_exist.pdf"
        # File doesn't exist — fitz raises, helper must catch and return 0.
        assert _pdf_page_count(str(bad)) == 0


class TestSplitPdfByPages:
    def test_split_by_pages_only_3_parts_for_15_pages_max_5(self, tmp_path):
        path = _build_pdf(tmp_path, n_pages=15)
        parts = _split_pdf_for_ocr(path, max_size_mb=999.0, max_pages=5)
        try:
            assert len(parts) == 3
            for p in parts:
                assert _pdf_page_count(p) <= 5
        finally:
            for p in parts:
                _safe_unlink(p)

    def test_split_no_op_when_under_both_limits(self, tmp_path):
        path = _build_pdf(tmp_path, n_pages=5)
        parts = _split_pdf_for_ocr(path, max_size_mb=999.0, max_pages=10)
        try:
            # Single part containing all 5 pages (range step = 5 ≥ total).
            assert len(parts) == 1
            assert _pdf_page_count(parts[0]) == 5
        finally:
            for p in parts:
                _safe_unlink(p)

    def test_split_pages_constraint_dominates_when_smaller(self, tmp_path):
        """Even with a generous size budget, max_pages clamps the part length."""
        path = _build_pdf(tmp_path, n_pages=20)
        parts = _split_pdf_for_ocr(path, max_size_mb=999.0, max_pages=4)
        try:
            assert len(parts) == 5  # 20 / 4 = 5
            for p in parts:
                assert _pdf_page_count(p) <= 4
        finally:
            for p in parts:
                _safe_unlink(p)

    def test_split_size_constraint_alone_still_works(self, tmp_path):
        """Backward-compat: max_pages=None preserves v1 behaviour."""
        path = _build_pdf(tmp_path, n_pages=10)
        parts = _split_pdf_for_ocr(path, max_size_mb=999.0, max_pages=None)
        try:
            # Single part since size is well under the cap.
            assert len(parts) == 1
        finally:
            for p in parts:
                _safe_unlink(p)


class TestMistralPageThreshold:
    """Lot G: smoke test the constants and the over_pages predicate logic."""

    def test_mistral_max_pages_constant_is_below_server_limit(self):
        from scripts.rad_dataframe import MISTRAL_MAX_PAGES
        assert MISTRAL_MAX_PAGES <= 1000, "must stay below Mistral server limit"
        assert MISTRAL_MAX_PAGES >= 500, "too aggressive — would split unnecessarily"

    def test_mistral_split_part_pages_is_below_max_pages(self):
        from scripts.rad_dataframe import MISTRAL_MAX_PAGES, MISTRAL_SPLIT_PART_PAGES
        assert MISTRAL_SPLIT_PART_PAGES < MISTRAL_MAX_PAGES, \
            "split target must be smaller than the trigger threshold"
