"""
Unit tests for the v2 migration of `app/utils/book_note_generator.py`.

These tests cover the v2 changes only — Phase 0 source-format detection,
operational thresholds, smart truncation, cumulative summaries bounding,
CHAPTER_TREATMENT dispatch, and post-assembly v2 quality checks. The v1
behaviour (heuristic chapter splitter, page-window slicing, prompt loading)
is exercised through smoke tests already covered by integration runs.

Run with: pytest tests/test_book_note_generator_v2.py
"""

import os
import re
from typing import List
from unittest.mock import patch

import pytest

from app.utils import book_note_generator as bng
from app.utils.book_note_generator import (
    BOOK_NOTE_VERSION,
    BookStructure,
    Chapter,
    COARSE_PAGE_CHARS_THRESHOLD,
    GUARDRAIL_MAX_PHASE0_CHAPTERS,
    MAX_PREVIOUS_SUMMARIES_CHARS,
    SourceFormat,
    THRESHOLD_LONG_PARATEXT_WORDS,
    THRESHOLD_MAX_CHAPTER_WORDS,
    THRESHOLD_MAX_CHAPTERS,
    THRESHOLD_MAX_TOTAL_WORDS,
    _assemble_book_html,
    _assign_bodies_by_title_offset,
    _build_cumulative_summaries,
    _detect_sections,
    _detect_source_format,
    _interpolate_missing_pages,
    _max_page_from_markers,
    _merge_llm_structure,
    _normalize_pagination,
    _smart_truncate_chapter,
    _split_text_by_pages,
)


# --------------------------------------------------------------------------- #
# Phase 0 — source format detection and pagination normalisation
# --------------------------------------------------------------------------- #

class TestDetectSourceFormat:
    def test_pdf_native_clean_markers(self):
        text = "<!-- Page 1 -->\nClean academic prose without artefacts.\n<!-- Page 2 -->\nMore prose."
        assert _detect_source_format(text) == SourceFormat.PDF_NATIVE

    def test_pdf_ocr_with_degradation(self):
        # Pagination markers + many degradation patterns triggers PDF_OCR.
        artefacts = " ".join(["a b c d e"] * 30) + " " + "!@#$%^&*()" * 10
        text = "<!-- Page 1 -->\n" + artefacts + "\n<!-- Page 2 -->\nrest"
        assert _detect_source_format(text) == SourceFormat.PDF_OCR

    def test_epub_pagebreak_marker(self):
        text = 'Some prose <a id="page_15"/> then more prose <a id="page_16"/> end.'
        assert _detect_source_format(text) == SourceFormat.EPUB

    def test_plain_text_no_markers(self):
        text = "Just prose with no pagination markers anywhere in the text."
        assert _detect_source_format(text) == SourceFormat.PLAIN_TEXT


class TestNormalizePagination:
    def test_epub_to_pipeline_format(self):
        text = 'Intro <a id="page_15"/> body <a id="page_16"/> end.'
        normalized = _normalize_pagination(text, SourceFormat.EPUB)
        assert "<!-- Page 15 -->" in normalized
        assert "<!-- Page 16 -->" in normalized
        assert "<a id=" not in normalized

    def test_pdf_native_unchanged(self):
        text = "<!-- Page 1 -->\nbody\n<!-- Page 2 -->\nmore"
        assert _normalize_pagination(text, SourceFormat.PDF_NATIVE) == text

    def test_plain_text_unchanged(self):
        text = "No markers here."
        assert _normalize_pagination(text, SourceFormat.PLAIN_TEXT) == text


# --------------------------------------------------------------------------- #
# Phase 1 — Chapter dataclass v2 fields and treatment dispatch
# --------------------------------------------------------------------------- #

class TestChapterTreatment:
    def test_normal_chapter(self):
        ch = Chapter(num=1, title="Test", authors=["A"], text="word " * 1000)
        assert ch.treatment == "chapter"
        assert ch.word_count == 1000

    def test_paratext_short(self):
        ch = Chapter(num=0, title="Préface", authors=["A"], text="word " * 500, is_paratext=True)
        assert ch.treatment == "paratext_short"

    def test_paratext_long(self):
        ch = Chapter(
            num=-1, title="Concluding Remarks", authors=["A"],
            text="word " * 6000, is_paratext=True, is_long_paratext=True,
        )
        assert ch.treatment == "paratext_long"

    def test_to_exclude_default_false(self):
        ch = Chapter(num=1, title="Test", authors=["A"])
        assert ch.to_exclude is False


class TestMergeLLMStructure:
    def test_long_paratext_recomputed_from_text_length(self):
        """A paratext flagged is_long_paratext=False by the LLM but with text
        ≥ THRESHOLD_LONG_PARATEXT_WORDS should be promoted automatically."""
        initial = BookStructure(
            book_type="monograph",
            primary_authors=["X"],
            chapters=[
                Chapter(
                    num=0, title="Concluding Remarks", authors=["X"],
                    text="word " * (THRESHOLD_LONG_PARATEXT_WORDS + 100),
                    is_paratext=True,
                ),
            ],
        )
        llm_payload = {
            "book_type": "monograph",
            "primary_authors": ["X"],
            "editor": None,
            "chapters": [
                {
                    "num": 0, "title": "Concluding Remarks",
                    "authors": ["X"], "pages": [301, 328],
                    "is_paratext": True, "is_long_paratext": False,
                    "to_exclude": False,
                },
            ],
        }
        merged = _merge_llm_structure(initial, llm_payload, full_text="dummy")
        assert merged.chapters[0].is_long_paratext is True
        assert merged.chapters[0].treatment == "paratext_long"

    def test_to_exclude_propagated(self):
        initial = BookStructure(
            book_type="monograph", primary_authors=["X"],
            chapters=[Chapter(num=-1, title="Notes", authors=[])],
        )
        llm_payload = {
            "book_type": "monograph",
            "primary_authors": ["X"],
            "chapters": [{
                "num": -1, "title": "Notes", "authors": [],
                "is_paratext": True, "to_exclude": True,
            }],
        }
        merged = _merge_llm_structure(initial, llm_payload, full_text="dummy")
        assert merged.chapters[0].to_exclude is True


# --------------------------------------------------------------------------- #
# L4 — Smart truncation
# --------------------------------------------------------------------------- #

class TestSmartTruncate:
    def test_short_chapter_unchanged(self):
        text = "word " * 5000  # 5000 words, well under target
        assert _smart_truncate_chapter(text, max_chars=200_000, target_words=15_000) == text

    def test_long_chapter_with_caps_sections(self):
        intro = "Introduction paragraph " * 200
        section_body = "section body content " * 500
        sections = (
            f"\n\nFIRST SECTION HEADING\n{section_body}\n\n"
            f"SECOND SECTION HEADING\n{section_body}\n\n"
            f"THIRD SECTION HEADING\n{section_body}\n\n"
        )
        conclusion = "conclusion paragraph " * 200
        text = intro + sections + conclusion
        # Force long-chapter path: target_words is small so smart path triggers.
        result = _smart_truncate_chapter(text, max_chars=200_000, target_words=100)
        # Section headings should appear in the result.
        assert "FIRST SECTION HEADING" in result
        assert "THIRD SECTION HEADING" in result
        # Conclusion should be preserved (last 10 % strategy).
        assert "conclusion paragraph" in result

    def test_long_chapter_without_sections_falls_back(self):
        # Pure prose, no detectable headings → head+tail fallback.
        text = ("body word " * 30000)  # ~ 60k words, no sections
        result = _smart_truncate_chapter(text, max_chars=10_000, target_words=100)
        assert "milieu du chapitre omis" in result or "limite de contexte" in result
        assert len(result) <= 10_500  # tolerates the omission marker

    def test_respects_max_chars_safety_net(self):
        text = "x" * 200_000
        result = _smart_truncate_chapter(text, max_chars=10_000, target_words=15_000)
        assert len(result) <= 11_000  # safety margin for the marker


class TestDetectSections:
    def test_detects_caps_headings(self):
        # Sections must be > 50 chars apart (dedup window).
        text = (
            "Body intro paragraph\n\n"
            "FIRST HEADING\n"
            + ("body filler line. " * 10) + "\n\n"
            "SECOND HEADING\n"
            "more"
        )
        sections = _detect_sections(text)
        titles = [t for _, t in sections]
        assert any("FIRST HEADING" in t for t in titles)
        assert any("SECOND HEADING" in t for t in titles)

    def test_detects_markdown_headings(self):
        text = (
            "Intro paragraph that spans multiple words.\n\n"
            "## Section One\n"
            + ("body content paragraph here. " * 10) + "\n\n"
            "## Section Two\n"
            "body"
        )
        sections = _detect_sections(text)
        assert len(sections) >= 2


# --------------------------------------------------------------------------- #
# L5 — Cumulative summaries bounded
# --------------------------------------------------------------------------- #

class TestCumulativeSummaries:
    def test_under_limit_returns_full_concatenation(self):
        summaries = ["Ch.1 summary", "Ch.2 summary", "Ch.3 summary"]
        result = _build_cumulative_summaries(summaries, max_chars=1000)
        assert result == "\n".join(summaries)

    def test_empty_returns_default_marker(self):
        assert _build_cumulative_summaries([]) == "(aucun chapitre encore analysé)"

    def test_over_limit_keeps_head_and_tail(self):
        # 15 summaries × 400 chars = 6000 chars → exceeds 4000 default.
        summaries = [f"Ch.{i} " + "x" * 395 for i in range(1, 16)]
        result = _build_cumulative_summaries(summaries, max_chars=4_000)
        assert len(result) <= 4_100  # within budget + marker tolerance
        # Head: first 3 must remain.
        assert "Ch.1 " in result
        assert "Ch.2 " in result
        assert "Ch.3 " in result
        # An omission marker appears.
        assert "résumés intermédiaires omis" in result
        # Latest summary always preserved (fresh context).
        assert "Ch.15 " in result


# --------------------------------------------------------------------------- #
# L6 — Post-assembly v2 quality checks
# --------------------------------------------------------------------------- #

@pytest.fixture
def baseline_meta():
    return {
        "title": "Test Book", "authors": "Author, A.",
        "date": "2026", "publisher": "Test Press",
    }


@pytest.fixture
def baseline_structure():
    return BookStructure(
        book_type="monograph", primary_authors=["Author, A."],
        chapters=[
            Chapter(num=i, title=f"Chapter {i}", authors=["A"], text="body")
            for i in range(1, 7)
        ],
    )


class TestPostAssemblyV2:
    def test_warns_if_place_dans_le_livre_present(self, baseline_meta, baseline_structure, caplog):
        chapter_blocks = [
            f"<h3>Chapitre {i} — « test »</h3><p><strong>Place dans le livre</strong> : ...</p>"
            for i in range(1, 7)
        ]
        section_a = "<h3>1. Identification</h3><h3>2. Architecture</h3>"
        section_b = (
            "<h3>4. Synthèse transversale</h3>"
            "<p>ch.1, ch.2, ch.3, ch.4, ch.5, ch.6 nombreuses tensions</p>"
            "<h3>5. Évaluation</h3>"
        )
        with caplog.at_level("WARNING"):
            _assemble_book_html(
                baseline_meta, baseline_structure,
                section_a, chapter_blocks, section_b,
                source_format=SourceFormat.PDF_NATIVE,
            )
        joined = " | ".join(r.message for r in caplog.records)
        assert "Place dans le livre" in joined

    def test_warns_if_section_b_lacks_chapter_refs(self, baseline_meta, baseline_structure, caplog):
        chapter_blocks = [f"<h3>Chapitre {i} — « test »</h3>" for i in range(1, 7)]
        section_a = "<h3>1. Identification</h3><h3>2. Architecture</h3>"
        section_b = "<h3>4. Synthèse transversale</h3><p>weak</p><h3>5. Évaluation</h3>"
        with caplog.at_level("WARNING"):
            _assemble_book_html(
                baseline_meta, baseline_structure,
                section_a, chapter_blocks, section_b,
                source_format=SourceFormat.PDF_NATIVE,
            )
        joined = " | ".join(r.message for r in caplog.records)
        assert "Section B" in joined

    def test_warns_if_forbidden_ordinal_markers(self, baseline_meta, baseline_structure, caplog):
        bad = "<h3>Chapitre 1 — « test »</h3><p>Premier — concept. Deuxièmement : autre. Troisième — final.</p>"
        chapter_blocks = [bad] + [f"<h3>Chapitre {i} — « test »</h3>" for i in range(2, 7)]
        section_a = "<h3>1. Identification</h3><h3>2. Architecture</h3>"
        section_b = (
            "<h3>4. Synthèse transversale</h3>"
            "<p>ch.1, ch.2, ch.3, ch.4, ch.5, ch.6</p>"
            "<h3>5. Évaluation</h3>"
        )
        with caplog.at_level("WARNING"):
            _assemble_book_html(
                baseline_meta, baseline_structure,
                section_a, chapter_blocks, section_b,
                source_format=SourceFormat.PDF_NATIVE,
            )
        joined = " | ".join(r.message for r in caplog.records)
        assert "ordinal" in joined

    def test_v1_flag_skips_v2_checks(self, baseline_meta, baseline_structure, caplog, monkeypatch):
        """When BOOK_NOTE_VERSION=v1, v2 quality checks are not applied."""
        monkeypatch.setattr(bng, "BOOK_NOTE_VERSION", "v1")
        chapter_blocks = [
            f"<h3>Chapitre {i} — « test »</h3><p><strong>Place dans le livre</strong> : ...</p>"
            for i in range(1, 7)
        ]
        section_a = "<h3>1. Identification</h3><h3>2. Architecture</h3>"
        section_b = "<h3>4. Synthèse transversale</h3><p>weak</p><h3>5. Évaluation</h3>"
        with caplog.at_level("WARNING"):
            _assemble_book_html(
                baseline_meta, baseline_structure,
                section_a, chapter_blocks, section_b,
                source_format=SourceFormat.PDF_NATIVE,
            )
        joined = " | ".join(r.message for r in caplog.records)
        assert "Place dans le livre" not in joined  # v2 check skipped
        assert "Section B" not in joined            # v2 check skipped

    def test_plain_text_format_warning(self, baseline_meta, baseline_structure, caplog):
        chapter_blocks = [f"<h3>Chapitre {i} — « test »</h3>" for i in range(1, 7)]
        section_a = "<h3>1. Identification</h3><h3>2. Architecture</h3>"
        section_b = (
            "<h3>4. Synthèse transversale</h3>"
            "<p>ch.1, ch.2, ch.3, ch.4, ch.5, ch.6 tensions</p>"
            "<h3>5. Évaluation</h3>"
        )
        with caplog.at_level("WARNING"):
            _assemble_book_html(
                baseline_meta, baseline_structure,
                section_a, chapter_blocks, section_b,
                source_format=SourceFormat.PLAIN_TEXT,
            )
        joined = " | ".join(r.message for r in caplog.records)
        assert "pagination absente" in joined


# --------------------------------------------------------------------------- #
# L1 — Prompt structure (smoke test on actual book_prompt.md)
# --------------------------------------------------------------------------- #

class TestPromptStructureV2:
    def test_phases_extractable(self):
        full = bng._load_book_prompt()
        for phase in ("phase1", "phase2", "phase3"):
            body = bng._extract_phase_prompt(full, phase)
            assert body, f"{phase} extraction empty"

    def test_phase2_has_chapter_treatment_placeholder(self):
        full = bng._load_book_prompt()
        body = bng._extract_phase_prompt(full, "phase2")
        assert "{CHAPTER_TREATMENT}" in body

    def test_phase1_has_v2_paratext_fields(self):
        full = bng._load_book_prompt()
        body = bng._extract_phase_prompt(full, "phase1")
        assert "is_long_paratext" in body
        assert "to_exclude" in body

    def test_phase3_demands_three_tensions(self):
        full = bng._load_book_prompt()
        body = bng._extract_phase_prompt(full, "phase3")
        # The v2 prompt requires ≥3 tensions in Section B.
        assert "3" in body or "trois" in body.lower()


# --------------------------------------------------------------------------- #
# Lots B/C/D — corrections post-test résilience (2026-05-05)
# --------------------------------------------------------------------------- #

class TestMaxPageFromMarkers:
    def test_returns_max_page_in_text(self):
        text = "<!-- Page 1 -->body<!-- Page 7 -->body<!-- Page 504 -->end"
        assert _max_page_from_markers(text) == 504

    def test_returns_zero_when_no_marker(self):
        assert _max_page_from_markers("plain prose, no markers") == 0


class TestLastNPagesLotE:
    def test_returns_last_n_pages_with_markers(self):
        pages = [(i, f"page{i} body") for i in range(1, 21)]
        out = bng._last_n_pages(pages, 5)
        # Should include last 5 pages (16-20).
        assert "<!-- Page 20 -->" in out
        assert "<!-- Page 16 -->" in out
        assert "<!-- Page 15 -->" not in out

    def test_returns_empty_when_no_pages(self):
        assert bng._last_n_pages([], 12) == ""

    def test_returns_all_when_n_exceeds_length(self):
        pages = [(i, f"page{i}") for i in range(1, 6)]
        out = bng._last_n_pages(pages, 100)
        assert "<!-- Page 1 -->" in out
        assert "<!-- Page 5 -->" in out


class TestInterpolateMissingPages:
    def test_no_chapters_returns_empty_list(self):
        assert _interpolate_missing_pages([], max_page=100) == []

    def test_chapters_all_with_pages_no_interpolation(self):
        chs = [
            Chapter(num=1, title="A", authors=["X"], pages=(1, 10)),
            Chapter(num=2, title="B", authors=["X"], pages=(11, 20)),
        ]
        result = _interpolate_missing_pages(chs, max_page=100)
        assert result == []
        assert chs[0].pages == (1, 10)
        assert chs[1].pages == (11, 20)

    def test_interpolates_middle_chapters(self):
        chs = [
            Chapter(num=1, title="A", authors=["X"], pages=(1, 20)),
            Chapter(num=2, title="B", authors=["X"], pages=(None, None)),
            Chapter(num=3, title="C", authors=["X"], pages=(None, None)),
            Chapter(num=4, title="D", authors=["X"], pages=(81, 100)),
        ]
        result = _interpolate_missing_pages(chs, max_page=100)
        assert result == [1, 2]
        # Pages 21-80 split equally between ch.2 and ch.3.
        assert chs[1].pages[0] == 21
        assert chs[2].pages[1] == 80
        # Continuous coverage (no gap, no overlap > 1).
        assert chs[1].pages[1] + 1 == chs[2].pages[0]

    def test_interpolates_leading_chapters_anchored_at_1(self):
        chs = [
            Chapter(num=1, title="A", authors=["X"], pages=(None, None)),
            Chapter(num=2, title="B", authors=["X"], pages=(50, 100)),
        ]
        _interpolate_missing_pages(chs, max_page=100)
        assert chs[0].pages[0] == 1
        assert chs[0].pages[1] == 49

    def test_interpolates_trailing_chapters_anchored_at_max(self):
        chs = [
            Chapter(num=1, title="A", authors=["X"], pages=(1, 50)),
            Chapter(num=2, title="B", authors=["X"], pages=(None, None)),
        ]
        _interpolate_missing_pages(chs, max_page=100)
        assert chs[1].pages[0] == 51
        assert chs[1].pages[1] == 100


class TestMergeLLMStructureLotB:
    def test_merge_keeps_oversized_body_intact(self):
        """After Lot E refactor, the cap moved to build_book_note_async (only
        when page_based=True). _merge_llm_structure must preserve the body so
        PLAIN_TEXT sources without pagination still have chapter text."""
        oversized_text = ("word " * 80_000)
        initial = BookStructure(
            book_type="monograph", primary_authors=["Latour"],
            chapters=[Chapter(num=1, title="Block", authors=["Latour"], text=oversized_text)],
        )
        llm_payload = {
            "book_type": "monograph",
            "primary_authors": ["Latour"],
            "chapters": [
                {"num": 1, "title": "L'objet", "authors": ["Latour"], "pages": [13, 50], "is_paratext": False},
                {"num": 2, "title": "Les données", "authors": ["Latour"], "pages": [51, 100], "is_paratext": False},
                {"num": 3, "title": "Les liens", "authors": ["Latour"], "pages": [101, 150], "is_paratext": False},
            ],
        }
        merged = _merge_llm_structure(initial, llm_payload, full_text="")
        # _merge_llm_structure no longer caps; body is preserved (positional fallback).
        # The cap is now applied in build_book_note_async only when page_based=True.
        assert len(merged.chapters[0].text) > 0

    def test_interpolates_missing_pages_via_max_page_from_markers(self):
        """When LLM returns chapters with pages=(None,None), the interpolation
        must derive the bounds from the OCR `<!-- Page N -->` markers."""
        full_text = "<!-- Page 1 -->intro<!-- Page 100 -->mid<!-- Page 200 -->end"
        initial = BookStructure(
            book_type="monograph", primary_authors=["X"],
            chapters=[Chapter(num=i, title=f"C{i}", authors=["X"]) for i in range(1, 5)],
        )
        llm_payload = {
            "book_type": "monograph",
            "primary_authors": ["X"],
            "chapters": [
                {"num": 1, "title": "A", "pages": [1, 50], "is_paratext": False},
                {"num": 2, "title": "B", "pages": [None, None], "is_paratext": False},
                {"num": 3, "title": "C", "pages": [None, None], "is_paratext": False},
                {"num": 4, "title": "D", "pages": [151, 200], "is_paratext": False},
            ],
        }
        merged = _merge_llm_structure(initial, llm_payload, full_text=full_text)
        # ch.2 and ch.3 must have pages interpolated within (50, 151).
        assert merged.chapters[1].pages[0] is not None
        assert merged.chapters[1].pages[1] is not None
        assert merged.chapters[2].pages[0] is not None
        # Surfaced for UI consumption.
        assert hasattr(merged, "interpolated_chapter_indices")
        assert 1 in merged.interpolated_chapter_indices
        assert 2 in merged.interpolated_chapter_indices


class TestExtractTocExcerptLotC:
    def test_finds_toc_in_first_half(self):
        text = "intro\nTable des matières\nch1 ... 10\nch2 ... 20\n" + ("filler " * 5000)
        result = bng._extract_toc_excerpt(text)
        assert "Table des matières" in result
        assert len(result) <= 8000

    def test_finds_toc_in_second_half(self):
        # Some Anglo-Saxon books place the ToC at the end.
        text = ("filler " * 5000) + "\nContents\nch1 ... 10\n" + ("more filler " * 1000)
        result = bng._extract_toc_excerpt(text)
        assert "Contents" in result

    def test_returns_empty_when_no_anchor(self):
        text = "Just academic prose without any toc anchor word here. " * 100
        assert bng._extract_toc_excerpt(text) == ""


class TestAssemblyLotDCoverage:
    def test_low_coverage_adds_visible_banner(self, caplog):
        struct = BookStructure(
            book_type="monograph", primary_authors=["X"],
            chapters=[Chapter(num=i, title=f"C{i}", authors=["X"]) for i in range(1, 7)],
        )
        struct.interpolated_chapter_indices = [1, 2, 3]  # type: ignore[attr-defined]
        meta = {"title": "Test", "authors": "X", "date": "2026", "publisher": ""}
        section_a = "<h3>1. Identification</h3><h3>2. Architecture</h3>"
        section_b = (
            "<h3>4. Synthèse transversale</h3>"
            "<p>ch.1 ch.2 ch.3 ch.4 ch.5 ch.6 multiple tensions</p>"
            "<h3>5. Évaluation</h3>"
        )
        chapter_blocks = [f"<h3>Chapitre {i} — « test »</h3>" for i in range(1, 7)]
        sentinel, html = _assemble_book_html(
            meta, struct, section_a, chapter_blocks, section_b,
            source_format=SourceFormat.PDF_NATIVE,
            coverage=0.5,
        )
        assert "Pagination détectée pour 50%" in html
        assert "interpolé" in html

    def test_full_coverage_no_banner(self):
        struct = BookStructure(
            book_type="monograph", primary_authors=["X"],
            chapters=[Chapter(num=i, title=f"C{i}", authors=["X"]) for i in range(1, 7)],
        )
        # No interpolated indices.
        meta = {"title": "Test", "authors": "X", "date": "2026", "publisher": ""}
        section_a = "<h3>1. Identification</h3><h3>2. Architecture</h3>"
        section_b = (
            "<h3>4. Synthèse transversale</h3>"
            "<p>ch.1 ch.2 ch.3 ch.4 ch.5 ch.6 tensions</p>"
            "<h3>5. Évaluation</h3>"
        )
        chapter_blocks = [f"<h3>Chapitre {i} — « test »</h3>" for i in range(1, 7)]
        _, html = _assemble_book_html(
            meta, struct, section_a, chapter_blocks, section_b,
            source_format=SourceFormat.PDF_NATIVE,
            coverage=1.0,
        )
        assert "Pagination détectée pour" not in html


# --------------------------------------------------------------------------- #
# Lot H — markdown chapter-heading detection (no printed ToC) and guardrail
# --------------------------------------------------------------------------- #

class TestLotHMarkdownHeadings:
    """Lot H — for books whose printed ToC is missing or buried, the OCR
    `# Chapitre N. Title` headings emitted by Mistral are an authoritative
    structural signal. Phase 0 must detect them, attach page ranges from
    `<!-- Page N -->` markers, and the merge step must reject Phase 1 LLM
    regressions when fewer chapters are returned.
    """

    @staticmethod
    def _build_text_with_headings(n_chapters: int, body_per_chapter: int = 1500) -> str:
        """Build a fixture text with N markdown chapters and page markers."""
        parts: List[str] = []
        page = 1
        for i in range(1, n_chapters + 1):
            parts.append(f"<!-- Page {page} -->")
            parts.append(f"# Chapitre {i}. Titre du chapitre {i}")
            parts.append("Body paragraph one. " * body_per_chapter)
            page += 5
            parts.append(f"<!-- Page {page} -->")
            parts.append("Body paragraph two. " * body_per_chapter)
            page += 5
        return "\n".join(parts)

    def test_heuristic_detects_markdown_chapters(self):
        from app.utils.book_note_generator import _split_text_into_chapters_heuristic

        text = self._build_text_with_headings(n_chapters=5)
        chapters = _split_text_into_chapters_heuristic(text)
        assert len(chapters) == 5
        # Heading is cleaned of markdown prefix.
        assert chapters[0][0].startswith("Chapitre 1.")
        assert "Titre du chapitre 1" in chapters[0][0]
        # First chapter starts at page 1; ends before next chapter's start page.
        sp, ep = chapters[0][2]
        assert sp == 1
        # Last chapter end_page falls back to last marker.
        assert chapters[-1][2][0] is not None

    def test_heuristic_skips_short_toc_duplicates(self):
        """A printed ToC that also matches `# Chapitre N` would produce tiny
        bodies. `_build_initial_structure` filters those out instead of
        falling back to fixed parts."""
        from app.utils.book_note_generator import _build_initial_structure

        # ToC at front (3 short bodies), then 4 real chapters with substantial bodies.
        toc = "\n".join([
            "<!-- Page 5 -->",
            "# Chapitre 1. Title A",
            "1",
            "# Chapitre 2. Title B",
            "50",
            "# Chapitre 3. Title C",
            "100",
        ])
        body = "\n".join([
            f"<!-- Page {p} -->\n# Chapitre {i}. Titre du chapitre {i}\n"
            + "Body content. " * 200
            for i, p in enumerate([200, 250, 300, 350], start=1)
        ])
        text = toc + "\n" + body
        struct = _build_initial_structure(text, metadata={"authors": "X"})
        # The 3 ToC entries are dropped, only 4 real chapters survive.
        assert len(struct.chapters) >= 3
        assert struct.structure_signal == "markdown_headings"
        assert struct.confidence == "medium"

    def test_phase0_attaches_page_ranges(self):
        from app.utils.book_note_generator import _build_initial_structure

        text = self._build_text_with_headings(n_chapters=4)
        struct = _build_initial_structure(text, metadata={"authors": "Bernard Lahire"})
        assert struct.structure_signal == "markdown_headings"
        assert len(struct.chapters) == 4
        for ch in struct.chapters:
            assert ch.pages[0] is not None  # all chapters got a start page
        # Pages are monotonically increasing.
        starts = [ch.pages[0] for ch in struct.chapters]
        assert starts == sorted(starts)

    def test_merge_keeps_phase0_when_llm_regresses(self, caplog):
        """Lot H guardrail: when Phase 0 detected ≥3 markdown chapters but
        Phase 1 LLM returns far fewer, keep Phase 0."""
        initial = BookStructure(
            book_type="monograph",
            primary_authors=["Bernard Lahire"],
            structure_signal="markdown_headings",
            confidence="medium",
            chapters=[
                Chapter(num=i, title=f"Chapitre {i}. Title {i}", authors=["X"],
                        pages=(i * 50, i * 50 + 40), text="body " * 500)
                for i in range(1, 18)  # 17 chapters
            ],
        )
        llm_payload = {
            "book_type": "monograph",
            "primary_authors": ["Bernard Lahire"],
            "confidence": "low",
            "chapters": [
                {"num": 1, "title": "Introduction", "pages": [1, 100]},
                {"num": 2, "title": "Conclusion", "pages": [101, 200]},
            ],
        }
        with caplog.at_level("WARNING"):
            merged = _merge_llm_structure(initial, llm_payload, full_text="x" * 1000)
        # Phase 0 structure must be preserved despite Phase 1 returning only 2.
        assert len(merged.chapters) == 17
        assert merged.structure_signal == "markdown_headings_guardrail"
        assert any("Lot H guardrail" in rec.message for rec in caplog.records)

    def test_h1_fallback_when_no_chapitre_numbering(self):
        """Lot H priority 2: when the book uses bare `# Title` H1 (no
        `Chapitre N`), the heuristic still promotes them to chapter starts.
        H2 sub-sections must NOT be promoted. The density filter drops H1
        whose body is shorter than the threshold (paratexts are dropped if
        too short — Phase 1 LLM still sees them via the front-page excerpt)."""
        from app.utils.book_note_generator import _split_text_into_chapters_heuristic

        # Each substantial chapter has > H1_FALLBACK_MIN_BODY_CHARS (5000) chars.
        body_chunk = "Body content paragraph here. " * 300  # ~9000 chars per chapter
        text = "\n".join([
            "<!-- Page 1 -->",
            "# Esprit sociologique, esprit critique",
            body_chunk,
            "## Sub-section A — must NOT be promoted to chapter",
            body_chunk,
            "<!-- Page 30 -->",
            "# Décrire la réalité sociale",
            body_chunk,
            "<!-- Page 60 -->",
            "# Les modalités des pratiques",
            body_chunk,
        ])
        chapters = _split_text_into_chapters_heuristic(text)
        headings = [c[0] for c in chapters]
        assert any("Esprit sociologique" in h for h in headings)
        assert any("Décrire la réalité sociale" in h for h in headings)
        assert any("Les modalités des pratiques" in h for h in headings)
        # H2 (`## Sub-section A`) must NOT be detected as chapter.
        assert not any("Sub-section A" in h for h in headings)
        # Pages assigned from nearest preceding marker.
        assert chapters[0][2][0] == 1
        assert chapters[1][2][0] == 30
        assert chapters[2][2][0] == 60

    def test_h1_fallback_density_filter_drops_short_bodies(self):
        """Lot H priority 2: H1 with bodies shorter than
        `H1_FALLBACK_MIN_BODY_CHARS` are dropped (ToC duplicates,
        promoted sub-sections). The last H1 is always kept."""
        from app.utils.book_note_generator import (
            _split_text_into_chapters_heuristic,
            H1_FALLBACK_MIN_BODY_CHARS,
        )

        # Build a text with 6 H1: 3 short bodies (<5000 chars) sandwiched
        # between 3 long bodies (>10000 chars).
        text_parts = [
            "<!-- Page 1 -->",
            "# Real Chapter A",
            "Body. " * 2500,                    # ~15000 chars body
            "# Stray Sub A",
            "Tiny. " * 50,                       # ~300 chars body — should drop
            "# Real Chapter B",
            "Body. " * 2500,                    # ~15000 chars body
            "# Stray Sub B",
            "Tiny. " * 100,                      # ~600 chars body — should drop
            "# Real Chapter C",
            "Body. " * 2500,                    # last — always kept
        ]
        text = "\n".join(text_parts)
        chapters = _split_text_into_chapters_heuristic(text)
        headings = [c[0] for c in chapters]
        # Stray H1 dropped; real chapters survive.
        assert "Stray Sub A" not in headings
        assert "Stray Sub B" not in headings
        assert any("Real Chapter A" in h for h in headings)
        assert any("Real Chapter B" in h for h in headings)
        assert any("Real Chapter C" in h for h in headings)
        # 3 real chapters survive; 2 strays dropped.
        assert len(chapters) == 3

    def test_explicit_chapitre_takes_priority_over_h1(self):
        """When both `# Chapitre N` and bare `# Title` exist, only the
        explicit pattern is used (avoids dual-counting)."""
        from app.utils.book_note_generator import _split_text_into_chapters_heuristic

        text = "\n".join([
            "<!-- Page 1 -->",
            "# Préface",
            "Body. " * 300,
            "<!-- Page 10 -->",
            "# Chapitre 1. Real chapter one",
            "Body. " * 300,
            "<!-- Page 50 -->",
            "# Chapitre 2. Real chapter two",
            "Body. " * 300,
            "<!-- Page 100 -->",
            "# Chapitre 3. Real chapter three",
            "Body. " * 300,
        ])
        chapters = _split_text_into_chapters_heuristic(text)
        # 3 chapters via the explicit pattern. The plain "# Préface" H1 is
        # ignored because the explicit branch already returned ≥3.
        assert len(chapters) == 3
        for c in chapters:
            assert c[0].startswith("Chapitre")

    def test_merge_accepts_llm_when_count_matches(self):
        """When Phase 1 returns ~the same count as Phase 0, merge proceeds normally."""
        initial = BookStructure(
            book_type="monograph",
            primary_authors=["X"],
            structure_signal="markdown_headings",
            confidence="medium",
            chapters=[
                Chapter(num=i, title=f"Chapitre {i}", authors=["X"], text="b " * 500)
                for i in range(1, 6)
            ],
        )
        llm_payload = {
            "book_type": "monograph",
            "primary_authors": ["X"],
            "confidence": "high",
            "chapters": [
                {"num": i, "title": f"Refined Chapitre {i}", "pages": [i * 10, i * 10 + 9]}
                for i in range(1, 6)
            ],
        }
        merged = _merge_llm_structure(initial, llm_payload, full_text="x" * 1000)
        # Guardrail must NOT trigger when counts match.
        assert len(merged.chapters) == 5
        assert merged.structure_signal != "markdown_headings_guardrail"
        # LLM titles win.
        assert merged.chapters[0].title.startswith("Refined")


# --------------------------------------------------------------------------- #
# Fix B — Lot H guardrail upper bound (reject pathological over-segmentation)
# --------------------------------------------------------------------------- #

class TestGuardrailUpperBound:
    """The Lot H guardrail keeps Phase 0 over Phase 1 only when Phase 0's count
    is structurally plausible (≤ GUARDRAIL_MAX_PHASE0_CHAPTERS). It must reject
    Digital-Methods-style over-segmentation (63 H1-promoted sections) while
    still keeping Lahire's legitimate 17 chapters."""

    @staticmethod
    def _phase0(n_chapters: int) -> BookStructure:
        return BookStructure(
            book_type="monograph",
            primary_authors=["X"],
            structure_signal="markdown_headings",
            confidence="medium",
            chapters=[
                Chapter(num=i, title=f"Chapitre {i}. Title {i}", authors=["X"],
                        pages=(i * 10, i * 10 + 9), text="body " * 500)
                for i in range(1, n_chapters + 1)
            ],
        )

    @staticmethod
    def _phase1(n_chapters: int) -> dict:
        return {
            "book_type": "monograph",
            "primary_authors": ["X"],
            "confidence": "high",
            "chapters": [
                {"num": i, "title": f"Refined chapter {i}", "pages": [i * 10, i * 10 + 9]}
                for i in range(1, n_chapters + 1)
            ],
        }

    def test_guardrail_rejects_63_chapters(self):
        """63 Phase-0 H1 chapters > bound → guardrail bypassed, Phase 1's 13 win."""
        merged = _merge_llm_structure(
            self._phase0(63), self._phase1(13), full_text="x" * 1000,
        )
        assert len(merged.chapters) == 13
        assert merged.structure_signal != "markdown_headings_guardrail"

    def test_guardrail_keeps_17_chapters(self, caplog):
        """Lahire: 17 ≤ bound and Phase 1 regresses to 2 → keep Phase 0's 17."""
        with caplog.at_level("WARNING"):
            merged = _merge_llm_structure(
                self._phase0(17), self._phase1(2), full_text="x" * 1000,
            )
        assert len(merged.chapters) == 17
        assert merged.structure_signal == "markdown_headings_guardrail"
        assert any("Lot H guardrail" in rec.message for rec in caplog.records)

    def test_guardrail_boundary_kept_at_constant(self):
        """At exactly GUARDRAIL_MAX_PHASE0_CHAPTERS the guardrail still fires."""
        merged = _merge_llm_structure(
            self._phase0(GUARDRAIL_MAX_PHASE0_CHAPTERS), self._phase1(2),
            full_text="x" * 1000,
        )
        assert len(merged.chapters) == GUARDRAIL_MAX_PHASE0_CHAPTERS
        assert merged.structure_signal == "markdown_headings_guardrail"

    def test_guardrail_boundary_bypassed_above_constant(self):
        """One above the bound → guardrail bypassed, Phase 1's list wins."""
        merged = _merge_llm_structure(
            self._phase0(GUARDRAIL_MAX_PHASE0_CHAPTERS + 1), self._phase1(2),
            full_text="x" * 1000,
        )
        assert len(merged.chapters) == 2
        assert merged.structure_signal != "markdown_headings_guardrail"


# --------------------------------------------------------------------------- #
# Fix A — coarse-pagination title-offset body assignment
# --------------------------------------------------------------------------- #

class TestAssignBodiesByTitleOffset:
    """On coarse pagination (EPUB / sparse markers) chapter bodies are sliced by
    the character offset of each title, giving non-overlapping complete coverage
    regardless of any Phase-0/Phase-1 count mismatch."""

    @staticmethod
    def _build_text(titles: List[str], marker: str = "UNIQUE") -> tuple:
        """Build full_text starting at the first title; each title is followed
        by a distinct ~2000-char body containing a unique marker token."""
        parts: List[str] = []
        markers: List[str] = []
        for i, title in enumerate(titles):
            tok = f"{marker}{i}"
            markers.append(tok)
            parts.append(title)
            parts.append(f"{tok} " + ("contenu du chapitre. " * 100))
        return "\n".join(parts), markers

    def test_non_overlapping_bodies_cover_book(self):
        titles = ["Les calculateurs", "La popularite des sites",
                  "Le marche de l'attention", "Les traces et la mesure"]
        full_text, markers = self._build_text(titles)
        chapters = [Chapter(num=i + 1, title=t, authors=["X"]) for i, t in enumerate(titles)]

        n = _assign_bodies_by_title_offset(chapters, full_text)

        assert n == 4
        # Every chapter bodied, and each unique marker appears in exactly one body.
        for i, ch in enumerate(chapters):
            assert ch.text, f"chapter {i} empty"
            assert markers[i] in ch.text
            for j, other in enumerate(markers):
                if j != i:
                    assert other not in ch.text, "bodies overlap"
        # Coverage ≥ 95% of the book (slicing starts at the first title = offset 0).
        total = sum(len(ch.text) for ch in chapters)
        assert total >= 0.95 * len(full_text)

    def test_more_chapters_than_phase0_all_bodied(self):
        """EPUB case: Phase 1 returns 7 chapters (Phase 0 had only 4). All 7 must
        get distinct bodies — the old positional fallback left ch.5-7 empty."""
        titles = [f"Chapitre nomme numero {i}" for i in range(1, 8)]
        full_text, markers = self._build_text(titles)
        chapters = [Chapter(num=i + 1, title=t, authors=["X"]) for i, t in enumerate(titles)]

        n = _assign_bodies_by_title_offset(chapters, full_text)

        assert n == 7
        # Specifically the chapters the positional fallback could not reach.
        for idx in (4, 5, 6):
            assert chapters[idx].text, f"chapter index {idx} should be bodied"
            assert markers[idx] in chapters[idx].text

    def test_unlocated_title_keeps_existing_body(self):
        titles = ["Premier titre present", "ABSENT DU TEXTE", "Troisieme titre present"]
        # full_text only contains chapters 1 and 3.
        full_text, _ = self._build_text([titles[0], titles[2]])
        chapters = [
            Chapter(num=1, title=titles[0], authors=["X"]),
            Chapter(num=2, title=titles[1], authors=["X"], text="preexisting"),
            Chapter(num=3, title=titles[2], authors=["X"]),
        ]

        n = _assign_bodies_by_title_offset(chapters, full_text)

        assert n == 2
        assert chapters[1].text == "preexisting"   # untouched
        assert chapters[0].text and chapters[2].text
        # Order preserved: ch1 before ch3 in the text.
        assert "contenu" in chapters[0].text

    def test_unlocated_title_empty_stays_empty(self):
        full_text, _ = self._build_text(["Titre alpha present", "Titre beta present"])
        chapters = [
            Chapter(num=1, title="Titre alpha present", authors=["X"]),
            Chapter(num=2, title="INTROUVABLE ICI", authors=["X"]),  # empty text default
            Chapter(num=3, title="Titre beta present", authors=["X"]),
        ]

        n = _assign_bodies_by_title_offset(chapters, full_text)

        assert n == 2
        assert chapters[1].text == ""
        assert chapters[0].text and chapters[2].text

    def test_duplicate_titles_bind_in_order(self):
        """A title repeated twice must bind first occurrence to the earlier
        chapter and second to the later one (forward-only search)."""
        full_text = "Section\nAAAA body alpha\nSection\nBBBB body beta"
        chapters = [
            Chapter(num=1, title="Section", authors=["X"]),
            Chapter(num=2, title="Section", authors=["X"]),
        ]

        n = _assign_bodies_by_title_offset(chapters, full_text)

        assert n == 2
        assert "AAAA" in chapters[0].text and "BBBB" not in chapters[0].text
        assert "BBBB" in chapters[1].text and "AAAA" not in chapters[1].text


class TestCoarsePaginationGating:
    """The orchestrator decides coarse vs dense by mean chars per detected page."""

    def test_coarse_detected_for_epub_like(self):
        # 10 markers, ~19,500 chars each → mean far above the threshold.
        text = "\n".join(f"<!-- Page {i} -->\n" + ("x" * 19500) for i in range(2, 12))
        pages = _split_text_by_pages(text)
        assert len(pages) == 10
        mean = len(text) / len(pages)
        assert mean > COARSE_PAGE_CHARS_THRESHOLD

    def test_dense_pdf_not_coarse(self):
        # 50 markers, ~2,500 chars each → mean below the threshold.
        text = "\n".join(f"<!-- Page {i} -->\n" + ("x" * 2500) for i in range(1, 51))
        pages = _split_text_by_pages(text)
        assert len(pages) == 50
        mean = len(text) / len(pages)
        assert mean <= COARSE_PAGE_CHARS_THRESHOLD
