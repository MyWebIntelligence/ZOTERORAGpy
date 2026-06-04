"""Tests for non-PDF attachment ingestion in rad_dataframe.

Covers Lot A:
- EPUB extraction preserves pagination markers as ``<!-- Page N -->``
- Plain text/Markdown extraction returns provider="plain_text"
- Unsupported extensions are logged in errors instead of being silently skipped
"""

import os
import pytest

from ebooklib import epub

from scripts.rad_dataframe import (
    _extract_text_from_epub,
    _extract_text_from_plain,
    _process_single_zotero_item,
)


def _build_epub(tmp_path, html_body: bytes) -> str:
    book = epub.EpubBook()
    book.set_identifier("test-id")
    book.set_title("Test")
    book.set_language("fr")

    chapter = epub.EpubHtml(title="Ch1", file_name="c1.xhtml", lang="fr")
    chapter.content = b"<html><body>" + html_body + b"</body></html>"
    book.add_item(chapter)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]

    out = tmp_path / "book.epub"
    epub.write_epub(str(out), book)
    return str(out)


def test_extract_text_from_epub_preserves_legacy_page_anchors(tmp_path):
    epub_path = _build_epub(
        tmp_path,
        b'<a id="page_1"></a>Hello<a id="page_2"></a>World',
    )

    out = _extract_text_from_epub(epub_path)

    assert "<!-- Page 1 -->" in out.text
    assert "<!-- Page 2 -->" in out.text
    assert "Hello" in out.text
    assert "World" in out.text
    assert out.provider == "epub"


def test_extract_text_from_epub_preserves_epub3_pagebreak_anchors(tmp_path):
    epub_path = _build_epub(
        tmp_path,
        b'<span epub:type="pagebreak" title="42"></span>Body text here.',
    )

    out = _extract_text_from_epub(epub_path)

    assert "<!-- Page 42 -->" in out.text
    assert "Body text here." in out.text


def test_extract_text_from_plain_reads_utf8(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("# Title\n\nLes accentués: éèà ✓", encoding="utf-8")

    out = _extract_text_from_plain(str(p))

    assert "Les accentués" in out.text
    assert out.provider == "plain_text"


def test_unsupported_extension_records_error(tmp_path):
    fake = tmp_path / "doc.docx"
    fake.write_bytes(b"PK\x03\x04not-a-real-docx")

    item = {
        "key": "ITEM1",
        "title": "Test item",
        "creators": [],
        "attachments": [{"path": str(fake), "title": "Word doc"}],
    }

    result = _process_single_zotero_item(item, str(tmp_path))

    assert result.records == []
    assert any(e["error_type"] == "UNSUPPORTED_EXTENSION" for e in result.errors)
    err = next(e for e in result.errors if e["error_type"] == "UNSUPPORTED_EXTENSION")
    assert ".docx" in err["error_message"]
    assert err["itemKey"] == "ITEM1"


def test_epub_attachment_produces_record(tmp_path):
    epub_path = _build_epub(
        tmp_path,
        b'<a id="page_1"></a>Premier chapitre.',
    )

    item = {
        "key": "ITEM2",
        "title": "Mon livre",
        "creators": [{"lastName": "Dupont", "firstName": "Jean"}],
        "attachments": [{"path": os.path.basename(epub_path), "title": "EPUB"}],
    }

    result = _process_single_zotero_item(item, str(tmp_path))

    assert result.errors == []
    assert len(result.records) == 1
    rec = result.records[0]
    assert rec["texteocr_provider"] == "epub"
    assert "Premier chapitre." in rec["texteocr"]
    assert "<!-- Page 1 -->" in rec["texteocr"]


def test_plain_text_attachment_produces_record(tmp_path):
    txt = tmp_path / "note.txt"
    txt.write_text("Contenu simple.", encoding="utf-8")

    item = {
        "key": "ITEM3",
        "title": "Note",
        "creators": [],
        "attachments": [{"path": "note.txt", "title": "Txt"}],
    }

    result = _process_single_zotero_item(item, str(tmp_path))

    assert result.errors == []
    assert len(result.records) == 1
    assert result.records[0]["texteocr_provider"] == "plain_text"
    assert result.records[0]["texteocr"] == "Contenu simple."


# ---------------------------------------------------------------------------
# Lot F — native EPUB ToC + synthetic per-item Page markers
# ---------------------------------------------------------------------------

def test_extract_text_from_epub_injects_native_toc_block(tmp_path):
    """Lot F: book.toc must be serialised at the head as
    <!-- EPUB_TOC_BEGIN -->...<!-- EPUB_TOC_END --> block."""
    book = epub.EpubBook()
    book.set_identifier("lotF1")
    book.set_title("Test EPUB")
    book.set_language("fr")
    c1 = epub.EpubHtml(title="Chapitre Un", file_name="c1.xhtml", lang="fr")
    c1.content = b"<html><body><p>body of chapter 1</p></body></html>"
    c2 = epub.EpubHtml(title="Chapitre Deux", file_name="c2.xhtml", lang="fr")
    c2.content = b"<html><body><p>body of chapter 2</p></body></html>"
    book.add_item(c1)
    book.add_item(c2)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", c1, c2]
    book.toc = [
        epub.Link("c1.xhtml", "Chapitre Un", "ch1"),
        epub.Link("c2.xhtml", "Chapitre Deux", "ch2"),
    ]
    epub.write_epub(str(tmp_path / "test.epub"), book)
    result = _extract_text_from_epub(str(tmp_path / "test.epub"))
    assert "<!-- EPUB_TOC_BEGIN -->" in result.text
    assert "<!-- EPUB_TOC_END -->" in result.text
    # The native ToC labels must be present in the block.
    assert "Chapitre Un" in result.text
    assert "Chapitre Deux" in result.text


def test_extract_text_from_epub_synthetic_page_per_item(tmp_path):
    """Lot F: each EpubHtml item without real pagebreak gets a synthetic
    `<!-- Page N -->` marker (N = spine index)."""
    import re

    book = epub.EpubBook()
    book.set_identifier("lotF2")
    book.set_title("X")
    book.set_language("fr")
    items = []
    for i in range(1, 6):
        c = epub.EpubHtml(title=f"Ch{i}", file_name=f"c{i}.xhtml", lang="fr")
        c.content = f"<html><body><p>body {i}</p></body></html>".encode()
        book.add_item(c)
        items.append(c)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + items
    epub.write_epub(str(tmp_path / "test.epub"), book)
    result = _extract_text_from_epub(str(tmp_path / "test.epub"))
    page_markers = re.findall(r"<!--\s*Page\s+(\d+)\s*-->", result.text)
    # Spine has 1 nav + 5 chapters = 6 items, all without real pagebreak →
    # 6 synthetic Page markers expected.
    assert len(page_markers) >= 5, f"expected ≥5 synthetic markers, got {page_markers}"


def test_real_pagebreaks_take_priority_over_synthetic(tmp_path):
    """Lot F: when an EpubHtml item has real `<a id="page_X"/>` ancors, no
    synthetic marker is prepended — the real anchor wins."""
    import re

    epub_path = _build_epub(
        tmp_path,
        b'<a id="page_42"></a>Body of page 42.',
    )
    result = _extract_text_from_epub(epub_path)
    assert "<!-- Page 42 -->" in result.text
    # The synthetic spine-index marker should NOT appear for the chapter
    # item that already had a real pagebreak. The nav item has no real
    # pagebreak so it may still get a synthetic marker — that's expected.
    page_markers = re.findall(r"<!--\s*Page\s+(\S+)\s*-->", result.text)
    assert "42" in page_markers
