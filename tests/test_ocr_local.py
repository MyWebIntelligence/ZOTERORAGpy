"""
Tests du moteur OCR LOCAL (Lot 4) — scripts/ocr_local.py.

Vérifie la logique pure de pagination `<!-- Page N -->` (duck-typée, sans
Docling installé) et la détection de moteur. Le rendu réel Docling est validé
hors CI (POC 4-A) car les deps sont lourdes/optionnelles.

Run: pytest tests/test_ocr_local.py -v
"""

import pytest

from scripts.ocr_local import render_markdown_with_page_markers, check_engine


class _FakeDoc:
    """DoclingDocument minimal : pages + export_to_markdown(page_no=...)."""
    def __init__(self, pages, per_page=True):
        self.pages = {p: object() for p in pages}
        self._per_page = per_page

    def export_to_markdown(self, page_no=None):
        if page_no is None:
            return "DOCUMENT ENTIER"
        if not self._per_page:
            raise TypeError("export_to_markdown() got an unexpected keyword 'page_no'")
        return f"contenu page {page_no}"


class TestPageMarkers:
    def test_per_page_markers_in_order(self):
        out = render_markdown_with_page_markers(_FakeDoc([1, 2, 3]))
        assert out == (
            "<!-- Page 1 -->\ncontenu page 1\n\n"
            "<!-- Page 2 -->\ncontenu page 2\n\n"
            "<!-- Page 3 -->\ncontenu page 3"
        )

    def test_pages_sorted(self):
        out = render_markdown_with_page_markers(_FakeDoc([3, 1, 2]))
        assert out.index("Page 1") < out.index("Page 2") < out.index("Page 3")

    def test_max_pages_truncates(self):
        out = render_markdown_with_page_markers(_FakeDoc([1, 2, 3, 4, 5]), max_pages=2)
        assert "<!-- Page 2 -->" in out
        assert "<!-- Page 3 -->" not in out

    def test_fallback_when_page_no_unsupported(self):
        # Version Docling sans paramètre page_no → un seul marqueur global.
        out = render_markdown_with_page_markers(_FakeDoc([1, 2], per_page=False))
        assert out == "<!-- Page 1 -->\nDOCUMENT ENTIER"

    def test_no_pages_attr_uses_whole_export(self):
        class NoPages:
            pages = {}
            def export_to_markdown(self, page_no=None):
                return "WHOLE"
        assert render_markdown_with_page_markers(NoPages()) == "<!-- Page 1 -->\nWHOLE"

    def test_empty_document_returns_empty(self):
        class Empty:
            pages = {}
            def export_to_markdown(self, page_no=None):
                return ""
        assert render_markdown_with_page_markers(Empty()) == ""


class TestCheckEngine:
    def test_missing_engine_is_unavailable(self):
        assert check_engine("definitely_not_a_real_pkg_xyz") is False

    def test_returns_bool_for_docling(self):
        # True si Docling installé, False sinon — toujours un bool, jamais d'erreur.
        assert isinstance(check_engine("docling"), bool)
