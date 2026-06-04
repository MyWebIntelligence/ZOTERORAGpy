"""
Book Reading Note Generator — multi-phase orchestrator.

Books cannot be processed as a single LLM shot like articles: they are too long
for context windows, their structure is heterogeneous (monograph vs. edited
volume), and a useful reading note must analyse each chapter individually
while replacing it in the dynamic of the whole book.

This module implements the 4-phase pipeline described in
.claude/tasks/book_reading_notes.md and uses the 3 sub-prompts in
app/utils/book_prompt.md:

    Phase 0 (no LLM)  — split the OCR text into chapters using ToC heuristics
    Phase 1 (1 call)  — confirm typology and refine the chapter list (LLM)
    Phase 2 (N calls) — analyse each chapter, with incremental short-summary
                        memory passed to subsequent calls
    Phase 3 (1 call)  — generate the global Identification / Architecture /
                        Synthesis / Evaluation / Exploitation sections
    Phase 4 (no LLM)  — assemble: prefix + Section A + chapters + Section B

Concurrency: all LLM calls go through the global `get_llm_semaphore()` so that
a multi-chapter book respects MAX_CONCURRENT_LLM_CALLS just like article calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from app.utils.llm_note_generator import (
    SENTINEL_PREFIX,
    _generate_with_llm,
    get_llm_semaphore,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

BOOK_PROMPT_FILENAME = "book_prompt.md"
PHASE_MARKERS = {
    "phase1": "=== PHASE 1 — DÉTECTION DE LA STRUCTURE ===",
    "phase2": "=== PHASE 2 — ANALYSE D'UN CHAPITRE ===",
    "phase3": "=== PHASE 3 — SYNTHÈSE TRANSVERSALE ET ÉVALUATION ===",
}

# Per-phase token budget guidance (informative — actual cap comes from
# NOTE_MODE_MAX_TOKENS["book"] in llm_note_generator.py, currently 8000).
# 8000 covers Phase 2 chapter notes (900-1300 words ≈ 1500-2400 tokens out)
# with comfortable headroom for thinking/iteration.
PHASE_MAX_TOKENS = {
    "phase1": 4000,    # JSON structure detection (under the 8000 cap)
    "phase2": 7000,    # one chapter dense analysis (capped at 8000 effective)
    "phase3": 6000,    # global sections (Section A + Section B)
}

# Number of book pages fed to Phase 1 for ToC / typology detection.
# Most academic books place the table of contents within the first ~10 pages
# (front matter), but Latour-style monographs and edited volumes with long
# prefaces can push it past 15. Lot C bumped from 12 → 20 to absorb these.
PHASE1_TOC_PAGES = 20

# Many academic books (especially French translations of Anglo-Saxon work,
# such as Latour's *Enquête sur les modes d'existence*) place the ToC at the
# END of the volume rather than at the front. We additionally feed the last
# N pages to Phase 1 so it can find a back-of-book ToC when one exists.
PHASE1_LAST_TOC_PAGES = 12

# How much OCR to feed Phase 1 if no page markers are detected (legacy
# fallback). Char-based.
PHASE1_FIRST_PAGES_CHARS = 20000

# Maximum characters per chapter sent to LLM (Phase 2). Beyond this, the
# chapter is truncated with an explicit marker. This keeps cost predictable
# even for very long chapters while preserving the most informative part.
# Raised to 80k so dense conceptual content is fully captured in chapter notes.
PHASE2_CHAPTER_MAX_CHARS = 80000

# When slicing a chapter by page range, expand the window by this many pages
# on each side. ToC pagination is often imprecise (ToC says ch.5 starts at
# p.87 but the actual chapter heading is on p.89 of the OCR), so we widen
# the slice and let Phase 2 LLM isolate the chapter content.
PHASE2_PAGE_MARGIN = 3

# A "real" printed/OCR page is ~2,000-3,000 chars. EPUB extractors (and some
# coarse OCR) emit very few canonical `<!-- Page N -->` markers for the whole
# book, so each "page" can average 15,000-20,000 chars. When the mean chars
# per detected page exceeds this floor, the ±PHASE2_PAGE_MARGIN page window
# grabs most of the book per chapter (overlap), so we slice chapter bodies by
# title offset instead. 8000 is >3x a dense PDF page and <0.5x a coarse EPUB
# page on the confirmed cases (EPUB "À quoi rêvent les algorithmes"
# ≈ 19,500 chars/page; Latour ≈ 1,900; Digital Methods ≈ 2,500) — a wide,
# safe separation.
COARSE_PAGE_CHARS_THRESHOLD = 8000

# Pattern used to split flat OCR text on page-boundary markers (injected by
# both Mistral and OpenAI vision OCR providers).
PAGE_MARKER_RE = re.compile(r"<!--\s*Page\s+(\d+)\s*-->")

# Heuristic detection thresholds.
MIN_CHAPTER_CHARS = 800           # below this, merge with the next chapter
MAX_FALLBACK_PARTS = 12           # if no ToC at all, split into N parts

# --------------------------------------------------------------------------- #
# v2 — Operational thresholds, smart truncation, version flag
# --------------------------------------------------------------------------- #

# Operational thresholds (v2). Crossing them logs a WARNING and surfaces a
# `threshold_warning` progress event. Beyond these, the quality of the last
# chapters and of the cross-chapter synthesis can degrade.
THRESHOLD_MAX_CHAPTERS = 12
THRESHOLD_MAX_TOTAL_WORDS = 80_000
THRESHOLD_MAX_CHAPTER_WORDS = 30_000
THRESHOLD_LONG_PARATEXT_WORDS = 5_000

# Lot H guardrail upper bound. The guardrail (in `_merge_llm_structure`) keeps
# Phase 0's markdown-heading chapter list over Phase 1's only when Phase 0's
# count is structurally PLAUSIBLE. Real monographs occasionally run above the
# operational seuil (Lahire's "L'esprit sociologique" legitimately has 17),
# but pathological over-segmentation (Mistral promoting ~63 section titles to
# `# H1`) must be rejected so Phase 1's consolidated list wins. 20 keeps 17
# and rejects 63 (= THRESHOLD_MAX_CHAPTERS + headroom).
GUARDRAIL_MAX_PHASE0_CHAPTERS = 20

# Cumulative chapter summaries fed to Phase 2 of the next chapter. Bounded
# so context does not blow up on long books (≈12-15 chapter summaries fit).
MAX_PREVIOUS_SUMMARIES_CHARS = 4_000

# Smart truncation target — beyond `target_words`, switch to section-aware
# sampling instead of brute truncation. Tuned so a typical theory chapter
# (15-20k words) still fits in one Phase 2 call.
SMART_TRUNCATE_TARGET_WORDS = 15_000

# Version flag for v1 → v2 migration. Set BOOK_NOTE_VERSION=v1 to disable
# all v2 post-assembly checks. Default is v2 (new standard).
BOOK_NOTE_VERSION = os.getenv("BOOK_NOTE_VERSION", "v2").lower()


class SourceFormat(str, Enum):
    """
    Detected source format. Drives the pagination normalisation strategy.

    PDF_NATIVE  — OCR or native-text PDF with `<!-- Page N -->` markers (clean).
    PDF_OCR     — same markers, but the surrounding text shows OCR-degradation
                  artefacts (citations may be unreliable).
    EPUB        — EPUB extraction with `<a id="page_X"/>` pagebreak markers;
                  needs normalisation before the rest of the pipeline.
    PLAIN_TEXT  — no pagination available; the pipeline runs but page
                  references will be missing and a warning is surfaced.
    """
    PDF_NATIVE = "pdf_native"
    PDF_OCR = "pdf_ocr"
    EPUB = "epub"
    PLAIN_TEXT = "plain_text"


# EPUB pagebreak marker emitted by most EPUB → text extractors.
_EPUB_PAGE_MARKER_RE = re.compile(r'<a\s+id="page_([0-9ivxlcdm]+)"\s*/?>', re.IGNORECASE)

# Heuristic patterns for OCR-degradation detection. Hits are counted on a
# 5k-char sample; > 50 hits flips PDF_NATIVE → PDF_OCR.
_OCR_DEGRADATION_PATTERNS = [
    re.compile(r"[^\w\s]{5,}"),                # runs of isolated symbols
    re.compile(r"\b[a-z]\s[a-z]\s[a-z]\b"),    # isolated chars separated by spaces
]

# Patterns for in-chapter section heading detection (used by smart truncation).
# Conservative — false positives drop sections from the truncation, which is
# safer than losing the conclusion.
_SECTION_PATTERNS = [
    re.compile(r"^[A-Z][A-Z\s\-:]{4,80}$", re.MULTILINE),                 # ALL CAPS line
    re.compile(r"^\d+(?:\.\d+)?\s+[A-Z][^\n]{3,80}$", re.MULTILINE),       # "1. Title" / "2.3 Subtitle"
    re.compile(r"^#{2,4}\s+.+$", re.MULTILINE),                           # Markdown headings
]


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class Chapter:
    """One book chapter with its OCR text and analysis state."""
    num: int
    title: str
    authors: List[str]
    pages: Tuple[Optional[int], Optional[int]] = (None, None)
    text: str = ""
    is_paratext: bool = False
    is_long_paratext: bool = False     # v2: paratext >= THRESHOLD_LONG_PARATEXT_WORDS → hybrid template
    to_exclude: bool = False           # v2: notes/index/bibliographie → skipped before Phase 2
    short_summary: Optional[str] = None
    html_block: Optional[str] = None

    @property
    def word_count(self) -> int:
        """Estimate of chapter length used for treatment dispatch and seuil checks."""
        return len(self.text.split()) if self.text else 0

    @property
    def treatment(self) -> str:
        """Phase 2 template branch: chapter | paratext_short | paratext_long."""
        if self.is_long_paratext:
            return "paratext_long"
        if self.is_paratext:
            return "paratext_short"
        return "chapter"


@dataclass
class BookStructure:
    """Result of Phase 0 + Phase 1 — typology and chapter list."""
    book_type: str = "monograph"          # monograph | edited_volume | coauthored | handbook
    primary_authors: List[str] = field(default_factory=list)
    editor: Optional[str] = None
    confidence: str = "medium"            # high | medium | low
    structure_signal: str = "heuristic_ocr"
    chapters: List[Chapter] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# v2 — Phase 0 source-format detection and pagination normalisation
# --------------------------------------------------------------------------- #

def _detect_source_format(text: str) -> SourceFormat:
    """
    Detect the source format from the raw text content.

    Decision tree:
      - If `<!-- Page N -->` markers are present → PDF_NATIVE or PDF_OCR
        (degradation score on the first 5k chars decides).
      - Else if EPUB pagebreak markers are present → EPUB.
      - Else → PLAIN_TEXT (pagination unavailable).
    """
    if PAGE_MARKER_RE.search(text):
        sample = text[:5000]
        degradation_score = sum(len(p.findall(sample)) for p in _OCR_DEGRADATION_PATTERNS)
        return SourceFormat.PDF_OCR if degradation_score > 50 else SourceFormat.PDF_NATIVE
    if _EPUB_PAGE_MARKER_RE.search(text):
        return SourceFormat.EPUB
    return SourceFormat.PLAIN_TEXT


def _normalize_pagination(text: str, source_format: SourceFormat) -> str:
    """
    Convert format-specific pagination markers into the canonical
    `<!-- Page N -->` form used by the rest of the pipeline.

    PDF_NATIVE / PDF_OCR  — already canonical, returned unchanged.
    EPUB                  — `<a id="page_X"/>` becomes `<!-- Page X -->`.
    PLAIN_TEXT            — no pagination available, returned unchanged.
    """
    if source_format == SourceFormat.EPUB:
        def replace_marker(m: "re.Match[str]") -> str:
            page_id = m.group(1)
            return f"\n<!-- Page {page_id} -->\n"
        return _EPUB_PAGE_MARKER_RE.sub(replace_marker, text)
    return text


# --------------------------------------------------------------------------- #
# Prompt parsing
# --------------------------------------------------------------------------- #

def _load_book_prompt() -> str:
    """Read app/utils/book_prompt.md."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, BOOK_PROMPT_FILENAME)
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _extract_phase_prompt(full: str, phase_key: str) -> str:
    """
    Extract the prompt body of one phase from book_prompt.md.

    Each phase prompt is enclosed between its `=== PHASE N ===` marker and the
    next `=== PHASE N+1 ===` marker (or the assembly section). Within the phase
    block, the actual instruction template is the content of the LAST fenced
    code block (```text ... ```) before the next marker, which mirrors how the
    file is structured.
    """
    start_marker = PHASE_MARKERS[phase_key]
    next_keys = {"phase1": "phase2", "phase2": "phase3", "phase3": None}
    next_key = next_keys[phase_key]
    end_marker = PHASE_MARKERS[next_key] if next_key else "## Assemblage final"

    start = full.find(start_marker)
    if start == -1:
        raise RuntimeError(f"book_prompt.md: marker '{start_marker}' missing")
    end = full.find(end_marker, start + len(start_marker))
    if end == -1:
        end = len(full)

    block = full[start:end]
    # Last fenced code block of the phase = the prompt template.
    code_blocks = re.findall(r"```(?:text)?\n(.*?)\n```", block, flags=re.DOTALL)
    if not code_blocks:
        raise RuntimeError(f"book_prompt.md: no code block found in {phase_key}")
    return code_blocks[-1].strip()


def _fill_placeholders(template: str, values: Dict[str, str]) -> str:
    """Replace {KEY} placeholders. Missing keys are left in place."""
    out = template
    for key, val in values.items():
        out = out.replace("{" + key + "}", val if val is not None else "")
    return out


# --------------------------------------------------------------------------- #
# Phase 0 — heuristic chapter detection (no LLM)
# --------------------------------------------------------------------------- #

# Patterns for chapter heading detection (line-level, after stripping).
# Lot H: priority 1 patterns — explicit `Chapitre N` / `Chapter N` / `IV. Title`,
# accepting an optional markdown header prefix `#{1,6}\s+` so Mistral-OCR text
# is recognised. Priority 2 (fallback): bare H1 markdown `# Title` lines, used
# when no explicit chapter numbering exists (typical for monographs whose
# chapters are named by their title only). Mistral reliably distinguishes H1
# (chapter level) from H2-H3 (in-chapter sections), so H1 is a trustworthy
# signal even without a printed ToC. Phase 1 LLM downstream tags front-matter
# H1 (DU MÊME AUTEUR, Remerciements, Bibliographie) as `is_paratext=True`.
_CHAPTER_PATTERNS_EXPLICIT = [
    re.compile(r"^(?:\s*#{1,6}\s+|\s*)Chapitre\s+(\d+|[IVXLC]+)\b[\s.:\-—–]*(.{0,200})", re.IGNORECASE),
    re.compile(r"^(?:\s*#{1,6}\s+|\s*)Chapter\s+(\d+|[IVXLC]+)\b[\s.:\-—–]*(.{0,200})", re.IGNORECASE),
    # Roman numeral chapter title — requires ≥2 chars to avoid matching
    # author initials in OCR'd bibliographies (`C. BOUGLÉ`, `L. PIRANDELLO`,
    # etc.). Real Roman-numbered chapters use `II.`, `III.`, `IV.`, ...
    re.compile(r"^(?:\s*#{1,6}\s+|\s*)([IVXLC]{2,})\.\s+(.{3,200})$"),
]
_CHAPTER_PATTERN_H1_FALLBACK = re.compile(r"^#\s+(.{3,200})$")
# Kept for backward compatibility with any external import.
_CHAPTER_PATTERNS = _CHAPTER_PATTERNS_EXPLICIT

# Lot H priority 2 — body-length filter applied ONLY to the H1 fallback.
# Mistral OCR sometimes promotes Table-of-Contents entries (front-matter list)
# and in-chapter sub-sections to bare `# H1`. Their body (distance to the next
# H1) is small (a few hundred to ~5k chars). Real chapters are typically
# ≥10k+ words = ≥30k chars, so a 5000-char floor drops the obvious offenders
# without sacrificing legitimate paratexts at the head/tail of the book.
# Phase 1 LLM consolidates the remaining candidates further.
H1_FALLBACK_MIN_BODY_CHARS = 5_000


def _page_marker_positions(full_text: str) -> List[Tuple[int, int]]:
    """Return list of `(offset, page_num)` for every `<!-- Page N -->` marker."""
    positions: List[Tuple[int, int]] = []
    for m in PAGE_MARKER_RE.finditer(full_text):
        try:
            positions.append((m.start(), int(m.group(1))))
        except (TypeError, ValueError):
            continue
    return positions


def _page_at_offset(positions: List[Tuple[int, int]], offset: int) -> Optional[int]:
    """Return the page number active at `offset`, or None when no marker precedes it."""
    last: Optional[int] = None
    for off, p in positions:
        if off > offset:
            return last
        last = p
    return last


def _norm_title(s: str) -> str:
    """Normalise a title for matching: collapse whitespace, lowercase, strip."""
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _split_text_into_chapters_heuristic(
    full_text: str,
) -> List[Tuple[str, str, Tuple[Optional[int], Optional[int]]]]:
    """
    Heuristic chapter splitter.

    Returns a list of `(heading, body, (start_page, end_page))` tuples. The page
    range is filled from the nearest `<!-- Page N -->` markers when available
    (Lot H — needed for books without a printed ToC where Mistral OCR is the
    only structure signal). If no chapter heading is found, returns a single
    `("Document", full_text, (None, None))` tuple — the caller then falls back
    to fixed-size splitting.
    """
    lines = full_text.splitlines(keepends=True)
    line_offsets: List[int] = [0]
    cursor = 0
    for line in lines:
        cursor += len(line)
        line_offsets.append(cursor)

    page_positions = _page_marker_positions(full_text)

    def _scan(patterns: List[re.Pattern]) -> List[Tuple[int, str, int]]:
        out: List[Tuple[int, str, int]] = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or len(stripped) > 250:
                continue
            for pattern in patterns:
                m = pattern.match(stripped)
                if m:
                    heading_clean = re.sub(r"^#+\s*", "", stripped).strip()
                    out.append((i, heading_clean, line_offsets[i]))
                    break
        return out

    # Priority 1: explicit `Chapitre N`/`Chapter N`/Roman numeral patterns.
    indices = _scan(_CHAPTER_PATTERNS_EXPLICIT)

    # Priority 2 (Lot H fallback): when no explicit chapter numbering exists,
    # promote bare H1 markdown lines (`# Title`) to chapter starts. Mistral OCR
    # uses H1 only for chapter-level breaks; H2-H3 are sub-sections handled by
    # `_detect_sections` inside chapters. Front-matter H1 (DU MÊME AUTEUR,
    # Remerciements, ...) is later flagged as `is_paratext=True` by Phase 1.
    if len(indices) < 3:
        indices = _scan([_CHAPTER_PATTERN_H1_FALLBACK])
        if indices:
            # Density filter — drop H1 whose body length (distance to the next
            # H1) is below H1_FALLBACK_MIN_BODY_CHARS. Always keep the last
            # match (its body extends to end-of-text). This eliminates ToC
            # entries duplicated at the front and stray in-chapter H1 that
            # Mistral promoted by mistake, without affecting the explicit
            # `Chapitre N` pattern (priority 1).
            filtered: List[Tuple[int, str, int]] = []
            for k in range(len(indices)):
                start_off = indices[k][2]
                end_off = indices[k + 1][2] if k + 1 < len(indices) else len(full_text)
                if end_off - start_off >= H1_FALLBACK_MIN_BODY_CHARS or k + 1 == len(indices):
                    filtered.append(indices[k])
            indices = filtered

    if not indices:
        return [("Document", full_text, (None, None))]

    chapters: List[Tuple[str, str, Tuple[Optional[int], Optional[int]]]] = []
    for k, (start_line, heading, start_off) in enumerate(indices):
        end_line = indices[k + 1][0] if k + 1 < len(indices) else len(lines)
        body = "".join(lines[start_line + 1:end_line]).strip()
        sp = _page_at_offset(page_positions, start_off)
        if k + 1 < len(indices):
            ep = _page_at_offset(page_positions, indices[k + 1][2] - 1)
        else:
            ep = page_positions[-1][1] if page_positions else None
        chapters.append((heading, body, (sp, ep)))
    return chapters


def _max_page_from_markers(text: str) -> int:
    """
    Return the highest page number found in `<!-- Page N -->` markers.

    Returns 0 when no marker is present (caller should fall back to length-based
    estimation). Used by the v2 interpolation logic in `_merge_llm_structure`
    to bound the page range when the LLM emits chapters with `pages=(None,None)`.
    """
    matches = PAGE_MARKER_RE.findall(text)
    if not matches:
        return 0
    try:
        return max(int(m) for m in matches)
    except (TypeError, ValueError):
        return 0


def _interpolate_missing_pages(
    chapters: List["Chapter"],
    max_page: int,
) -> List[int]:
    """
    Fill `pages=(None, None)` chapters by interpolation between known neighbours.

    Strategy: walk the chapter list in order. For each contiguous run of
    chapters without pages, locate the previous chapter with a known end page
    (`prev_end`) and the next chapter with a known start page (`next_start`).
    Distribute the page range `[prev_end+1, next_start-1]` equally between
    them. Edge cases:
      - Run at the very start: anchor at page 1.
      - Run at the very end: anchor at `max_page` (or last known + 1 if 0).
      - All chapters missing pages: split [1, max_page] equally.

    Returns the list of indices (0-based) whose pages were interpolated, so
    the caller can surface a structure_warning to the UI.

    The function mutates `chapters` in place.
    """
    if not chapters:
        return []

    interpolated: List[int] = []
    # Build a list of (idx, start, end) where None means missing.
    n = len(chapters)
    starts: List[Optional[int]] = []
    ends: List[Optional[int]] = []
    for ch in chapters:
        sp = ch.pages[0] if ch.pages else None
        ep = ch.pages[1] if ch.pages else None
        starts.append(sp)
        ends.append(ep)

    # Walk and find runs of (None, None) chapters.
    i = 0
    while i < n:
        if starts[i] is not None or ends[i] is not None:
            i += 1
            continue
        # Run starts at i. Find run end.
        j = i
        while j < n and starts[j] is None and ends[j] is None:
            j += 1
        # Run is [i, j-1] inclusive. Find anchors.
        prev_end: Optional[int] = None
        for k in range(i - 1, -1, -1):
            if ends[k] is not None:
                prev_end = ends[k]
                break
            if starts[k] is not None:
                prev_end = starts[k]
                break
        next_start: Optional[int] = None
        for k in range(j, n):
            if starts[k] is not None:
                next_start = starts[k]
                break
            if ends[k] is not None:
                next_start = ends[k]
                break

        lo = (prev_end + 1) if prev_end is not None else 1
        hi = (next_start - 1) if next_start is not None else (max_page or lo + (j - i) * 20)
        if hi < lo:
            hi = lo + (j - i) * 20  # degenerate, fall back to ~20 pages per chapter

        run_len = j - i
        span = max(1, hi - lo + 1)
        per_chapter = max(1, span // run_len)
        for offset, idx in enumerate(range(i, j)):
            ch_lo = lo + offset * per_chapter
            ch_hi = (lo + (offset + 1) * per_chapter - 1) if offset < run_len - 1 else hi
            chapters[idx].pages = (ch_lo, ch_hi)
            interpolated.append(idx)
        i = j

    return interpolated


def _split_text_by_pages(text: str) -> List[Tuple[int, str]]:
    """
    Split flat OCR text on `<!-- Page N -->` markers.

    Both Mistral OCR (since the page-marker patch) and the OpenAI vision
    fallback inject these markers, so this is the canonical way to recover
    per-page text from a flattened `texteocr` field.

    Args:
        text: full OCR text.

    Returns:
        List of (page_num, page_text) tuples in document order. Empty list if
        no markers are found, in which case the caller falls back to the
        legacy heuristic chapter splitter.
    """
    matches = list(PAGE_MARKER_RE.finditer(text))
    if not matches:
        return []
    pages: List[Tuple[int, str]] = []
    for i, m in enumerate(matches):
        try:
            page_num = int(m.group(1))
        except (TypeError, ValueError):
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        pages.append((page_num, body))
    return pages


def _slice_pages_window(
    pages: List[Tuple[int, str]],
    start_page: Optional[int],
    end_page: Optional[int],
    margin: int = PHASE2_PAGE_MARGIN,
) -> str:
    """
    Concatenate a window of pages around [start_page, end_page] with margin.

    The margin is applied symmetrically on both sides because OCR pagination
    rarely matches the printed page numbering exactly (off by 1-3 in edited
    volumes with front matter, plates, or unnumbered pages).

    Args:
        pages: output of `_split_text_by_pages`.
        start_page: declared chapter start (1-indexed in book pages). If None,
            the slice starts at the first available page in `pages`.
        end_page: declared chapter end. If None, the slice ends at the last
            available page.
        margin: extra pages on each side.

    Returns:
        Concatenated page texts with `<!-- Page N -->` markers preserved so
        the LLM can locate the chapter body within the slice.
    """
    if not pages:
        return ""
    # Available range in our OCR.
    avail_lo = pages[0][0]
    avail_hi = pages[-1][0]
    lo = (start_page if start_page is not None else avail_lo) - margin
    hi = (end_page if end_page is not None else avail_hi) + margin
    selected = [(n, t) for n, t in pages if lo <= n <= hi and t]
    if not selected:
        return ""
    return "\n\n".join(f"<!-- Page {n} -->\n{t}" for n, t in selected)


def _assign_bodies_by_title_offset(chapters: List["Chapter"], full_text: str) -> int:
    """
    Assign each chapter a NON-OVERLAPPING body by locating its title in
    `full_text` and slicing between consecutive heading offsets.

    Used for coarse pagination (EPUB / sparse `<!-- Page N -->` markers) where
    the page-window re-slice (`_slice_pages_window`) would grab most of the book
    per chapter. Unlike the positional fallback in `_merge_llm_structure`, this
    does not depend on the Phase-0 chapter count, so it correctly bodies the
    extra chapters when Phase 1 returns more than Phase 0 (e.g. 4 → 7).

    Each chapter title is matched as whitespace-flexible, case-insensitive tokens
    and searched forward-only (each search starts after the previous located
    title) so chapter order is preserved and a short title repeated later in the
    book cannot bind an earlier chapter. Chapters whose title cannot be located
    keep their existing body (never blanked, never reordered).

    Mutates `chapters[*].text` in place. Returns the number of chapters whose
    body was assigned from a located title offset.
    """
    if not full_text or not chapters:
        return 0

    located: List[Tuple[int, int]] = []  # (chapter_index, start_offset)
    search_start = 0
    for idx, ch in enumerate(chapters):
        # Strip a leading markdown heading prefix, then tokenise so internal
        # whitespace (and the optional `#`) matches flexibly in the raw text.
        cleaned = re.sub(r"^\s*#+\s*", "", (ch.title or "")).strip()
        tokens = cleaned.split()
        if not tokens or len("".join(tokens)) < 4:
            continue  # too short / empty → unlocatable, keep existing body
        pattern = re.compile(r"\s+".join(re.escape(t) for t in tokens), re.IGNORECASE)
        m = pattern.search(full_text, search_start)
        if not m:
            continue
        located.append((idx, m.start()))
        search_start = m.end()

    if not located:
        return 0

    for k, (idx, start) in enumerate(located):
        end = located[k + 1][1] if k + 1 < len(located) else len(full_text)
        chapters[idx].text = full_text[start:end].strip()

    unlocated = [chapters[i].num for i in range(len(chapters))
                 if i not in {idx for idx, _ in located}]
    if unlocated:
        logger.warning(
            "Coarse pagination: %d/%d chapter title(s) not located in text "
            "(kept existing body): ch.%s",
            len(unlocated), len(chapters), ", ch.".join(str(n) for n in unlocated),
        )
    return len(located)


def _last_n_pages(pages: List[Tuple[int, str]], n: int) -> str:
    """
    Concatenate the last N pages with markers (Lot C extension).

    Many academic books — Latour-style French translations from English,
    or 19th-century structures — place the table of contents at the end of
    the volume rather than at the front. We feed both ends to Phase 1 so the
    LLM can find the ToC wherever it lives.
    """
    tail = pages[-n:] if n > 0 else []
    if not tail:
        return ""
    return "\n\n".join(f"<!-- Page {num} -->\n{txt}" for num, txt in tail if txt)


def _first_n_pages(pages: List[Tuple[int, str]], n: int) -> str:
    """Concatenate the first N pages with markers (used to feed Phase 1)."""
    head = pages[:n]
    if not head:
        return ""
    return "\n\n".join(f"<!-- Page {num} -->\n{txt}" for num, txt in head if txt)


def _split_text_into_fixed_parts(full_text: str, n_parts: int) -> List[Tuple[str, str]]:
    """Fallback splitter when no chapter heading is found."""
    n_parts = max(1, min(n_parts, MAX_FALLBACK_PARTS))
    if n_parts == 1:
        return [("Partie 1", full_text)]
    size = len(full_text) // n_parts
    out: List[Tuple[str, str]] = []
    for i in range(n_parts):
        start = i * size
        end = (i + 1) * size if i < n_parts - 1 else len(full_text)
        out.append((f"Partie {i + 1}", full_text[start:end]))
    return out


def _extract_toc_excerpt(full_text: str) -> str:
    """
    Pull a "table of contents" / "sommaire" excerpt if present.

    Lot C: search the entire first half AND the last half of the book
    (Anglo-Saxon books sometimes place the ToC at the end), with a wider
    return window (8000 chars vs 6000 v1) to capture verbose ToCs.

    Lot F: an `<!-- EPUB_TOC_BEGIN -->...<!-- EPUB_TOC_END -->` block
    injected at the head of un-paginated EPUBs (see
    `scripts/rad_dataframe._extract_text_from_epub`) is recognised as the
    most authoritative source — the EPUB's native NCX/NAV ToC is always
    structurally correct, even when no pagination is available.
    """
    n = len(full_text)
    if n == 0:
        return ""

    # Lot F — Highest-priority source: the EPUB's native NCX/NAV ToC, if any.
    epub_toc_begin = full_text.find("<!-- EPUB_TOC_BEGIN -->")
    epub_toc_end = full_text.find("<!-- EPUB_TOC_END -->")
    if epub_toc_begin != -1 and epub_toc_end != -1 and epub_toc_end > epub_toc_begin:
        return full_text[epub_toc_begin:epub_toc_end + len("<!-- EPUB_TOC_END -->")]

    anchors = [
        r"Table\s+des\s+mati[èe]res",
        r"Sommaire",
        r"Contents",
        r"Table\s+of\s+Contents",
    ]
    mid = n // 2
    # Try first half (most common location).
    first_half = full_text[:mid]
    for anchor in anchors:
        m = re.search(anchor, first_half, flags=re.IGNORECASE)
        if m:
            start = m.start()
            return first_half[start:start + 8000]
    # Fall back to second half (back-of-book ToC, rare but happens).
    second_half = full_text[mid:]
    for anchor in anchors:
        m = re.search(anchor, second_half, flags=re.IGNORECASE)
        if m:
            start = m.start()
            return second_half[start:start + 8000]
    return ""


def _build_initial_structure(full_text: str, metadata: Dict) -> BookStructure:
    """
    Phase 0: build a tentative chapter list from heuristics.

    The output is later refined by Phase 1 (LLM). If the LLM call fails, or if
    Phase 1 returns a regression compared to the markdown signal (Lot H
    guardrail in `_merge_llm_structure`), this initial structure is used.
    """
    raw_chapters = _split_text_into_chapters_heuristic(full_text)

    # Lot H: when several markdown chapter headings are detected but a few have
    # tiny bodies (typical when the front-matter ToC also matches the regex),
    # drop just those entries instead of falling back to fixed parts wholesale.
    if len(raw_chapters) >= 3:
        substantial = [(h, b, p) for (h, b, p) in raw_chapters if len(b) >= MIN_CHAPTER_CHARS]
        if len(substantial) >= 3:
            raw_chapters = substantial

    if len(raw_chapters) >= 3 and all(len(b) >= MIN_CHAPTER_CHARS for _, b, _ in raw_chapters):
        # Lot H — markdown / `Chapitre N` headings present in the OCR text are
        # an authoritative structural signal even when no printed ToC exists.
        # We mark the structure_signal so Phase 1 LLM can refine titles but a
        # severe regression in chapter count is rejected by the merge guardrail.
        structure_signal = "markdown_headings"
        confidence = "medium"
    elif len(raw_chapters) <= 1 or any(len(b) < MIN_CHAPTER_CHARS for _, b, _ in raw_chapters):
        raw_chapters = [(h, b, (None, None)) for h, b in _split_text_into_fixed_parts(full_text, n_parts=8)]
        structure_signal = "fixed_parts"
        confidence = "low"
    else:
        structure_signal = "heuristic_ocr"
        confidence = "low"

    primary = [a.strip() for a in str(metadata.get("authors", "")).split(";") if a.strip()]
    structure = BookStructure(
        book_type="monograph",
        primary_authors=primary,
        editor=None,
        confidence=confidence,
        structure_signal=structure_signal,
    )
    for idx, (heading, body, pages) in enumerate(raw_chapters, start=1):
        structure.chapters.append(Chapter(
            num=idx,
            title=heading[:200],
            authors=primary or ["Auteur inconnu"],
            pages=pages,
            text=body,
        ))
    return structure


# --------------------------------------------------------------------------- #
# Phase 1 — LLM-driven structure detection (refine Phase 0)
# --------------------------------------------------------------------------- #

async def _phase1_detect_structure(
    full_text: str,
    metadata: Dict,
    phase1_template: str,
    *,
    pages: Optional[List[Tuple[int, str]]] = None,
    model: Optional[str],
    openai_api_key: Optional[str],
    openrouter_api_key: Optional[str],
) -> Optional[Dict]:
    """
    Run Phase 1 LLM call. Returns parsed JSON or None on failure.

    When `pages` is provided (preferred path), the LLM receives the first
    PHASE1_TOC_PAGES book pages AND the last PHASE1_LAST_TOC_PAGES with their
    `<!-- Page N -->` markers, which lets it return precise start/end pages
    per chapter — including for back-of-book ToCs (Lot E). Otherwise, falls
    back to a char-based excerpt of the front of `full_text`.
    """
    if pages:
        first_pages = _first_n_pages(pages, PHASE1_TOC_PAGES)
        last_pages = _last_n_pages(pages, PHASE1_LAST_TOC_PAGES)
        toc_raw = _extract_toc_excerpt(full_text) or "(détection à inférer des pages ci-dessous)"
    else:
        toc_raw = _extract_toc_excerpt(full_text) or "(aucune table des matières détectée)"
        first_pages = full_text[:PHASE1_FIRST_PAGES_CHARS]
        last_pages = full_text[-PHASE1_FIRST_PAGES_CHARS:] if len(full_text) > PHASE1_FIRST_PAGES_CHARS else ""

    prompt = _fill_placeholders(phase1_template, {
        "TITLE": str(metadata.get("title", "")),
        "AUTHORS": str(metadata.get("authors", "")),
        "DATE": str(metadata.get("date", "")),
        "PUBLISHER": str(metadata.get("publisher", "")),
        "DOI": str(metadata.get("doi", "")),
        "URL": str(metadata.get("url", "")),
        "LANGUAGE": str(metadata.get("language_label", "français")),
        "PROBLEMATIQUE": str(metadata.get("problematique", "Non spécifiée")),
        "ZOTERO_ITEMTYPE": str(metadata.get("itemType", "book")),
        "NUM_PAGES": str(metadata.get("numPages", "0") or "0"),
        "TOC_RAW": toc_raw,
        "FIRST_PAGES": first_pages,
        "LAST_PAGES": last_pages,
    })

    semaphore = get_llm_semaphore()
    async with semaphore:
        loop = asyncio.get_event_loop()
        try:
            raw = await loop.run_in_executor(
                None,
                lambda: _generate_with_llm(
                    prompt,
                    model=model,
                    temperature=0.0,
                    mode="book",
                    openai_api_key=openai_api_key,
                    openrouter_api_key=openrouter_api_key,
                ),
            )
        except Exception as exc:
            logger.warning("Phase 1 (book structure) LLM call failed: %s", exc)
            return None

    # Strip code fences if the model wrapped the JSON.
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n", "", raw)
        raw = re.sub(r"\n```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Phase 1 LLM output is not valid JSON: %s — payload starts with: %s",
                       exc, raw[:200])
        return None


def _merge_llm_structure(
    initial: BookStructure,
    llm_payload: Dict,
    full_text: str,
) -> BookStructure:
    """
    Merge LLM-detected chapter list with Phase-0 chapter texts.

    Each LLM-declared chapter is matched against the closest Phase-0 chapter by
    title similarity. If matching fails, we re-slice the full text by the
    LLM-provided page hints when available.
    """
    structure = BookStructure(
        book_type=str(llm_payload.get("book_type", initial.book_type)),
        primary_authors=list(llm_payload.get("primary_authors", initial.primary_authors)) or initial.primary_authors,
        editor=llm_payload.get("editor") or None,
        confidence=str(llm_payload.get("confidence", "medium")),
        structure_signal=str(llm_payload.get("structure_signal", "llm_inference")),
    )

    llm_chapters = llm_payload.get("chapters") or []
    if not llm_chapters:
        # Keep Phase-0 chapters but adopt the typology metadata.
        structure.chapters = initial.chapters
        return structure

    # Lot H — Guardrail: when Phase 0 detected a reliable markdown structure
    # (≥3 chapters via `# Chapitre N.` headings) but Phase 1 returned a
    # severely regressed list, the LLM probably saw only the front-matter or
    # bibliography (no ToC in the printed book) and hallucinated a 1-2 chapter
    # structure. Trust Phase 0 in that case — BUT only when Phase 0's count is
    # structurally plausible (≤ GUARDRAIL_MAX_PHASE0_CHAPTERS). When Phase 0
    # over-segmented (e.g. Mistral promoted ~63 section titles to `# H1`), the
    # guardrail is bypassed and the normal merge path runs, trusting Phase 1's
    # consolidated chapter list instead.
    if (
        initial.structure_signal == "markdown_headings"
        and len(initial.chapters) >= 3
        and len(initial.chapters) <= GUARDRAIL_MAX_PHASE0_CHAPTERS
        and len(llm_chapters) < max(3, len(initial.chapters) // 2)
    ):
        logger.warning(
            "Lot H guardrail: Phase 1 LLM returned %d chapters, Phase 0 detected %d markdown chapters — keeping Phase 0 structure.",
            len(llm_chapters), len(initial.chapters),
        )
        structure.chapters = initial.chapters
        structure.structure_signal = "markdown_headings_guardrail"
        return structure

    # Index initial chapters by lowercase normalised title.
    norm = _norm_title
    initial_by_title = {norm(c.title): c for c in initial.chapters}

    for raw in llm_chapters:
        try:
            num = int(raw.get("num", 0))
        except (TypeError, ValueError):
            num = len(structure.chapters) + 1
        title = str(raw.get("title", f"Chapitre {num}"))
        authors = list(raw.get("authors") or structure.primary_authors)
        pages = raw.get("pages") or [None, None]
        is_paratext = bool(raw.get("is_paratext", False))
        is_long_paratext = bool(raw.get("is_long_paratext", False))
        to_exclude = bool(raw.get("to_exclude", False))

        # Try to find body text by title match; fall back to whole text slice.
        body = ""
        match = initial_by_title.get(norm(title))
        if match:
            body = match.text
        else:
            # Crude fallback: pick a slice of full_text proportional to pages.
            body = ""

        structure.chapters.append(Chapter(
            num=num,
            title=title[:200],
            authors=[a for a in authors if a],
            pages=tuple(pages) if isinstance(pages, list) and len(pages) == 2 else (None, None),
            text=body,
            is_paratext=is_paratext,
            is_long_paratext=is_long_paratext,
            to_exclude=to_exclude,
        ))

    # If LLM lost text bodies but Phase 0 had them, fall back to Phase 0 bodies
    # by positional index when titles did not match.
    for i, ch in enumerate(structure.chapters):
        if not ch.text and i < len(initial.chapters):
            ch.text = initial.chapters[i].text

    # v2 (Lot B) — Interpolate missing chapter pages between known neighbours.
    # The LLM may emit `pages=(None, None)` for chapters when it cannot match
    # the absolute pagination from the front pages. Without this fix the
    # downstream `_slice_pages_window` cannot rebuild the chapter body, and
    # the ch.1 ends up with the entire fallback block. We track which
    # chapters were interpolated so the caller can surface a UI warning.
    max_page = _max_page_from_markers(full_text)
    interpolated_indices = _interpolate_missing_pages(structure.chapters, max_page)
    structure.interpolated_chapter_indices = interpolated_indices  # type: ignore[attr-defined]
    if interpolated_indices:
        logger.warning(
            "Lot B interpolation: %d chapter(s) had missing pages, interpolated from neighbours: %s",
            len(interpolated_indices),
            ", ".join(f"ch.{structure.chapters[i].num}" for i in interpolated_indices),
        )

    # v2 — recompute is_long_paratext from the actual text length: the LLM
    # may not see enough page count to decide reliably, but we now have the
    # body and can apply the THRESHOLD_LONG_PARATEXT_WORDS rule directly.
    for ch in structure.chapters:
        if ch.is_paratext and not ch.is_long_paratext:
            if ch.word_count >= THRESHOLD_LONG_PARATEXT_WORDS:
                ch.is_long_paratext = True

    return structure


# --------------------------------------------------------------------------- #
# Phase 2 — chapter analysis with incremental memory
# --------------------------------------------------------------------------- #

def _build_toc_short(structure: BookStructure) -> str:
    """Compact TOC string used in Phase 2 / Phase 3 prompts."""
    lines = []
    for ch in structure.chapters:
        authors = ", ".join(ch.authors) if ch.authors else "?"
        lines.append(f"{ch.num}. {ch.title} — {authors}")
    return " | ".join(lines)


def _truncate_chapter_legacy(text: str, max_chars: int) -> str:
    """
    Legacy brute truncation kept as defensive fallback for `_smart_truncate_chapter`.

    Preserved on purpose so a regression in the smart truncation path can
    fall back to a deterministic behaviour rather than crashing.
    """
    if len(text) <= max_chars:
        return text
    head = text[: max_chars - 200]
    return head + "\n\n[... troncature : suite du chapitre omise pour respecter la limite de contexte ...]"


def _detect_sections(text: str) -> List[Tuple[int, str]]:
    """
    Detect named sections inside a chapter via heading regex matching.

    Returns a list of (offset, heading) tuples sorted by position. Matches
    closer than 50 chars are deduplicated (keeps the first), which suppresses
    bursts of false positives caused by overlapping patterns.
    """
    matches: List[Tuple[int, str]] = []
    for pattern in _SECTION_PATTERNS:
        for m in pattern.finditer(text):
            matches.append((m.start(), m.group().strip()))
    matches.sort(key=lambda x: x[0])

    deduped: List[Tuple[int, str]] = []
    for offset, title in matches:
        if not deduped or offset - deduped[-1][0] > 50:
            deduped.append((offset, title))
    return deduped


def _smart_truncate_chapter(
    text: str,
    max_chars: int = PHASE2_CHAPTER_MAX_CHARS,
    target_words: int = SMART_TRUNCATE_TARGET_WORDS,
) -> str:
    """
    Truncate a long chapter by sampling sections instead of cutting the tail.

    Strategy:
      1. Short chapter (≤ target_words) → return the text (with a brute-truncate
         safety net at max_chars in pathological cases).
      2. Long chapter with ≥ 2 detected sections → keep the introduction,
         the head of each named section (~1.5k chars), and the conclusion.
         Insert explicit `[... section « X » développement intermédiaire omis ...]`
         markers so the LLM is aware of the omissions.
      3. Long chapter with no detectable sections → fall back to head+tail
         (70 % head, 20 % tail) with an explicit middle-omission marker.

    Resilience: if section-aware code raises, fall back to the legacy brute
    truncation rather than failing the whole chapter analysis.
    """
    try:
        word_count = len(text.split())

        if word_count <= target_words:
            if len(text) <= max_chars:
                return text
            return text[: max_chars - 200] + (
                "\n\n[... troncature : fin du chapitre omise pour respecter la limite ...]"
            )

        sections = _detect_sections(text)

        if len(sections) < 2:
            head_chars = int(max_chars * 0.7)
            tail_chars = int(max_chars * 0.2)
            head = text[:head_chars]
            tail = text[-tail_chars:]
            return (
                head
                + "\n\n[... troncature : milieu du chapitre omis pour respecter la limite "
                f"({word_count} mots, sections non détectées) ...]\n\n"
                + tail
            )

        intro_end = sections[0][0]
        conclusion_start = sections[-1][0]
        parts: List[str] = [text[:intro_end]]

        for i, (offset, title) in enumerate(sections[:-1]):
            next_offset = sections[i + 1][0]
            section_text = text[offset:next_offset]
            if len(section_text) > 2000:
                parts.append(
                    section_text[:1500]
                    + f"\n\n[... section « {title[:60]} » développement intermédiaire omis ...]\n\n"
                )
            else:
                parts.append(section_text)

        parts.append(text[conclusion_start:])
        result = "".join(parts)

        if len(result) > max_chars:
            result = result[: max_chars - 200] + (
                "\n\n[... troncature finale : limite de contexte atteinte ...]"
            )

        logger.info(
            "Smart truncate: %d → %d chars (%d sections, %d mots originaux)",
            len(text), len(result), len(sections), word_count,
        )
        return result

    except Exception as exc:  # noqa: BLE001 — defensive fallback
        logger.warning("Smart truncate failed (%s), falling back to legacy.", exc)
        return _truncate_chapter_legacy(text, max_chars)


def _build_cumulative_summaries(
    summaries: List[str],
    max_chars: int = MAX_PREVIOUS_SUMMARIES_CHARS,
) -> str:
    """
    Build the PREVIOUS_SUMMARIES block fed to Phase 2, bounded in size.

    Strategy: when the concatenation exceeds `max_chars`, keep the first 3
    summaries (paratextes initiaux + first theory chapter — the cumulative
    framing of the book) and as many of the latest summaries as fit in the
    remaining budget, joined by an omission marker.

    Pathological case (even 0 tail does not fit): truncate the head with a
    final `[...]` marker.
    """
    if not summaries:
        return "(aucun chapitre encore analysé)"

    full = "\n".join(summaries)
    if len(full) <= max_chars:
        return full

    head = summaries[:3]
    head_text = "\n".join(head)
    head_len = len(head_text)

    placeholder_marker = (
        "\n\n[... 0 résumés intermédiaires omis pour respecter la limite de contexte ...]\n\n"
    )
    available_for_tail = max_chars - head_len - len(placeholder_marker)

    tail: List[str] = []
    tail_len = 0
    for s in reversed(summaries[3:]):
        if tail_len + len(s) + 1 > available_for_tail:
            break
        tail.insert(0, s)
        tail_len += len(s) + 1

    if not tail:
        return head_text[: max_chars - 50] + "\n[...]"

    n_omitted = len(summaries) - len(head) - len(tail)
    omission_marker = (
        f"\n\n[... {n_omitted} résumés intermédiaires omis pour respecter la limite de contexte ...]\n\n"
        if n_omitted > 0
        else "\n"
    )
    return head_text + omission_marker + "\n".join(tail)


async def _phase2_analyse_chapter(
    chapter: Chapter,
    structure: BookStructure,
    book_meta: Dict,
    previous_summaries: List[str],
    phase2_template: str,
    *,
    model: Optional[str],
    openai_api_key: Optional[str],
    openrouter_api_key: Optional[str],
) -> Tuple[str, str]:
    """
    Run one Phase 2 LLM call. Returns (html_block, short_summary).
    """
    if not chapter.text or len(chapter.text.strip()) < 50:
        # Empty chapter — produce a minimal HTML block.
        authors = ", ".join(chapter.authors) or "?"
        pages = "—" if chapter.pages == (None, None) else f"pp. {chapter.pages[0]}-{chapter.pages[1]}"
        block = (
            f"<h3>Chapitre {chapter.num} — « {chapter.title} » — {authors} — {pages}</h3>"
            f"<p><em>Texte du chapitre indisponible dans l'OCR.</em></p>"
        )
        return block, f"Ch.{chapter.num} ({authors}) : texte OCR indisponible."

    prompt = _fill_placeholders(phase2_template, {
        "TITLE": str(book_meta.get("title", "")),
        "BOOK_TYPE": structure.book_type,
        "BOOK_PROJECT": book_meta.get("book_project", "Projet éditorial à inférer du texte."),
        "TOC_SHORT": _build_toc_short(structure),
        "CHAPTER_NUM": str(chapter.num),
        "CHAPTER_TITLE": chapter.title,
        "CHAPTER_AUTHORS": ", ".join(chapter.authors) or "Auteur inconnu",
        "CHAPTER_PAGES": (
            f"pp. {chapter.pages[0]}-{chapter.pages[1]}"
            if chapter.pages and chapter.pages[0] is not None
            else "(pages non détectées)"
        ),
        "CHAPTER_IS_PARATEXT": "true" if chapter.is_paratext else "false",
        "CHAPTER_TREATMENT": chapter.treatment,
        "CHAPTER_TEXT": _smart_truncate_chapter(chapter.text, PHASE2_CHAPTER_MAX_CHARS),
        "PREVIOUS_SUMMARIES": _build_cumulative_summaries(previous_summaries),
        "LANGUAGE": str(book_meta.get("language_label", "français")),
    })

    semaphore = get_llm_semaphore()
    async with semaphore:
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None,
            lambda: _generate_with_llm(
                prompt,
                model=model,
                temperature=0.2,
                mode="book",
                openai_api_key=openai_api_key,
                openrouter_api_key=openrouter_api_key,
            ),
        )

    # Split HTML block from "SUMMARY:" line.
    summary = ""
    if "SUMMARY:" in raw:
        html_part, _, summary_part = raw.partition("SUMMARY:")
        block = html_part.strip()
        summary = summary_part.strip()
    else:
        # Model forgot the SUMMARY line — derive a minimal one.
        block = raw.strip()
        summary = f"Ch.{chapter.num} ({', '.join(chapter.authors) or '?'}) : analyse fournie."

    # Defensive cleanup: ensure block opens with <h3>.
    if not block.lstrip().startswith("<h3"):
        block = (
            f"<h3>Chapitre {chapter.num} — « {chapter.title} » — "
            f"{', '.join(chapter.authors) or '?'}</h3>\n{block}"
        )

    # v2 — flag chapters whose generated volume drifts beyond the upper
    # tolerance for the chosen treatment. Logged only (no rejection): a
    # 1900-word block is still useful, but a 3000-word block is a sign the
    # LLM is repeating itself or hallucinating sections.
    block_word_count = len(re.sub(r"<[^>]+>", " ", block).split())
    expected_max = {
        "chapter": 1800,
        "paratext_short": 900,
        "paratext_long": 1500,
    }.get(chapter.treatment, 1300)
    if block_word_count > expected_max + 200:
        logger.warning(
            "Phase 2: chapter %d block exceeds target (%d > %d for treatment %s)",
            chapter.num, block_word_count, expected_max, chapter.treatment,
        )

    return block, summary


# --------------------------------------------------------------------------- #
# Phase 3 — global synthesis
# --------------------------------------------------------------------------- #

async def _phase3_synthesise(
    structure: BookStructure,
    summaries: List[str],
    book_meta: Dict,
    phase3_template: str,
    *,
    model: Optional[str],
    openai_api_key: Optional[str],
    openrouter_api_key: Optional[str],
) -> Tuple[str, str]:
    """Run Phase 3 LLM call. Returns (section_a_html, section_b_html)."""
    prompt = _fill_placeholders(phase3_template, {
        "TITLE": str(book_meta.get("title", "")),
        "PRIMARY_AUTHORS": ", ".join(structure.primary_authors) or "Auteur(s) inconnu(s)",
        "EDITOR": structure.editor or "null",
        "BOOK_TYPE": structure.book_type,
        "DATE": str(book_meta.get("date", "")),
        "PUBLISHER": str(book_meta.get("publisher", "")),
        "CHAPTERS_COUNT": str(len(structure.chapters)),
        "TOC_SHORT": _build_toc_short(structure),
        "BOOK_PROJECT": book_meta.get("book_project", "Projet éditorial à inférer du texte."),
        "ALL_SUMMARIES": "\n".join(summaries),
        "PROBLEMATIQUE": str(book_meta.get("problematique", "Non spécifiée")),
        "LANGUAGE": str(book_meta.get("language_label", "français")),
    })

    semaphore = get_llm_semaphore()
    async with semaphore:
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None,
            lambda: _generate_with_llm(
                prompt,
                model=model,
                temperature=0.2,
                mode="book",
                openai_api_key=openai_api_key,
                openrouter_api_key=openrouter_api_key,
            ),
        )

    # Expected delimiters: ===SECTION_A=== ... ===SECTION_B=== ... ===END===
    section_a = ""
    section_b = ""
    m_a = re.search(r"===SECTION_A===\s*(.*?)\s*===SECTION_B===", raw, flags=re.DOTALL)
    m_b = re.search(r"===SECTION_B===\s*(.*?)\s*===END===", raw, flags=re.DOTALL)
    if m_a:
        section_a = m_a.group(1).strip()
    if m_b:
        section_b = m_b.group(1).strip()

    if not section_a:
        section_a = (
            "<h3>1. Identification de l'ouvrage</h3>"
            f"<p>{book_meta.get('title', '')} — {', '.join(structure.primary_authors) or '?'}</p>"
        )
    if not section_b:
        section_b = "<h3>4. Synthèse transversale</h3><p><em>Synthèse non générée.</em></p>"
    return section_a, section_b


# --------------------------------------------------------------------------- #
# Phase 4 — assembly
# --------------------------------------------------------------------------- #

def _assemble_book_html(
    book_meta: Dict,
    structure: BookStructure,
    section_a: str,
    chapter_blocks: List[str],
    section_b: str,
    source_format: SourceFormat = SourceFormat.PDF_NATIVE,
    coverage: float = 1.0,
) -> Tuple[str, str]:
    """
    Assemble the final HTML note. Returns (sentinel, html).

    Performs post-assembly resilience checks:
      - sentinel + [LIVRE] prefix present
      - one <h3> chapter heading per declared chapter
      - obligatory global sections present
      - (Lot D) low pagination coverage warning visible to the user

    Mismatches are logged as warnings (we still return a usable note).
    """
    sentinel = f"{SENTINEL_PREFIX}{uuid.uuid4()}"
    title = book_meta.get("title", "Sans titre")
    authors = book_meta.get("authors", ", ".join(structure.primary_authors) or "")
    date = book_meta.get("date", "")
    publisher = book_meta.get("publisher", "")

    header = f"<h2>[LIVRE] {authors} ({date}). {title}." + (f" {publisher}." if publisher else "") + "</h2>"

    chapter_section_title = "<h3>3. Analyse chapitre par chapitre</h3>"
    # v2 (Lot D) — surface a visible banner when pagination coverage is low,
    # so the reader knows the chapter-by-chapter analysis was reconstructed
    # from interpolated page ranges rather than confirmed ToC pagination.
    coverage_banner = ""
    interpolated = getattr(structure, "interpolated_chapter_indices", None) or []
    if coverage < 0.8 or interpolated:
        coverage_pct = int(round(coverage * 100))
        n_interp = len(interpolated)
        coverage_banner = (
            f"<p><em>⚠️ Pagination détectée pour {coverage_pct}% des chapitres "
            f"({n_interp} chapitre{'s' if n_interp != 1 else ''} interpolé{'s' if n_interp != 1 else ''}). "
            f"Les fiches-chapitres concernées peuvent inclure du texte chevauchant.</em></p>"
        )

    chapters_html = "\n".join(chapter_blocks)

    body = "\n".join([header, section_a, chapter_section_title, coverage_banner, chapters_html, section_b])
    note_html = f"<!-- {sentinel} -->\n{body}"

    # --- Resilience checks -------------------------------------------------
    warnings_list: List[str] = []
    if "[LIVRE]" not in note_html[:500]:
        warnings_list.append("préfixe [LIVRE] manquant dans l'en-tête")

    found_chapters = chapters_html.count("<h3>Chapitre ")
    expected = len(structure.chapters)
    if found_chapters < expected:
        warnings_list.append(
            f"{expected - found_chapters} bloc(s) chapitre absent(s) (attendu {expected}, trouvé {found_chapters})"
        )

    for required in ("Identification", "Architecture", "Synthèse", "Évaluation"):
        if required not in note_html:
            warnings_list.append(f"section globale '{required}' absente")

    # v2 — quality checks (skipped if BOOK_NOTE_VERSION=v1).
    if BOOK_NOTE_VERSION == "v2":
        # Source-format hints surfaced as info, not errors.
        if source_format == SourceFormat.PLAIN_TEXT:
            warnings_list.append(
                "pagination absente dans le texte source — références par section au lieu de pages"
            )
        elif source_format == SourceFormat.PDF_OCR:
            warnings_list.append(
                "OCR potentiellement dégradé (artefacts détectés) — citations à vérifier"
            )

        # The « Place dans le livre » rubric was removed in v2; if the LLM
        # re-introduced it, flag for prompt regression.
        if "Place dans le livre" in chapters_html:
            warnings_list.append(
                "rubrique « Place dans le livre » présente dans une fiche-chapitre "
                "(devrait être absente en v2 — vérifier book_prompt.md)"
            )

        # The Section B must carry ≥ 3 cross-chapter tensions/dialogues. We
        # approximate by counting references of the form "ch.X" / "chapitre X".
        # Tolerance: only enforce on books with ≥ 6 chapters (smaller books
        # cannot mathematically reach 3 distinct tensions × 2 references).
        section_b_str = section_b or ""
        total_chapter_refs = len(re.findall(r"\b(?:ch\.|chapitre)\s*\d+", section_b_str, re.IGNORECASE))
        if expected >= 6 and total_chapter_refs < 6:
            warnings_list.append(
                f"Section B contient {total_chapter_refs} références à des chapitres "
                "(seuil v2: ≥6 pour 3 tensions/dialogues distincts)"
            )

        # Forbidden stylistic patterns. Tolerance: 2 occurrences (verbatim
        # quotes from the source author may legitimately use these formulas).
        forbidden_patterns_v2 = [
            (r"\b(Premier|Premièrement|Deuxième|Deuxièmement|Troisième|Troisièmement)\s*[—:.]",
             "marqueur ordinal"),
            (r"\bLe concept articule\b",  "formule « Le concept articule »"),
            (r"\bLe concept fournit\b",   "formule « Le concept fournit »"),
            (r"\bLe concept opère\b",     "formule « Le concept opère »"),
        ]
        for pattern, label in forbidden_patterns_v2:
            n = len(re.findall(pattern, chapters_html))
            if n > 2:
                warnings_list.append(f"{n} occurrences de {label} (interdit v2)")

    if warnings_list:
        logger.warning("[LIVRE] assembly warnings: %s", " | ".join(warnings_list))
    else:
        logger.info("[LIVRE] assembly OK: %d chapters, %d chars.", expected, len(note_html))

    return sentinel, note_html


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

async def build_book_note_async(
    metadata: Dict,
    text_content: str,
    *,
    model: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    progress_cb: Optional[Callable[[str, int, int, str], Awaitable[None]]] = None,
) -> Tuple[str, str]:
    """
    Generate a complete [LIVRE] reading note for a book.

    Args:
        metadata: dict with at least title, authors, date, language, problematique.
            Optional: publisher, doi, url, itemType, numPages.
        text_content: full OCR text of the book (texteocr column).
        model: LLM model identifier. If None, uses the env default.
        openai_api_key: per-user OpenAI key (security model — see core/credentials.py).
        openrouter_api_key: per-user OpenRouter key.
        progress_cb: async callable(stage, current, total, label) for SSE updates.

    Returns:
        (sentinel, note_html) tuple matching the build_note_html_async contract.

    Raises:
        ValueError: if no usable LLM credentials.
        RuntimeError: if book_prompt.md is malformed.
    """
    if not text_content or not text_content.strip():
        raise ValueError("Empty text_content for book note generation.")

    full_prompt = _load_book_prompt()
    phase1_tpl = _extract_phase_prompt(full_prompt, "phase1")
    phase2_tpl = _extract_phase_prompt(full_prompt, "phase2")
    phase3_tpl = _extract_phase_prompt(full_prompt, "phase3")

    book_meta = dict(metadata)
    # Normalise language: "fr" → "français" for prompt readability.
    lang = (book_meta.get("language") or "fr").lower().split("-")[0]
    book_meta["language_label"] = {
        "fr": "français", "en": "English", "es": "español",
        "de": "Deutsch", "it": "italiano", "pt": "português",
    }.get(lang, "français")

    # v2 — Phase 0a: detect source format and normalise pagination markers
    # before any further processing. EPUB inputs are converted to the
    # canonical `<!-- Page N -->` form; PLAIN_TEXT triggers a warning at
    # assembly time.
    source_format = _detect_source_format(text_content)
    logger.info("Phase 0: source format detected = %s", source_format.value)
    text_content = _normalize_pagination(text_content, source_format)
    if progress_cb:
        await progress_cb("format_detection", 1, 1, f"Format : {source_format.value}")

    # Phase 0 — try page-based slicing first (relies on `<!-- Page N -->`
    # markers injected by the OCR pipeline), fall back to heuristic regex
    # split if no markers are found (legacy PyMuPDF flow, articles, etc.).
    if progress_cb:
        await progress_cb("structure_detection", 0, 1, "Découpage initial")
    pages = _split_text_by_pages(text_content)
    page_based = bool(pages)
    if page_based:
        logger.info("Phase 0: page-based mode — %d OCR pages detected (range %d→%d).",
                    len(pages), pages[0][0], pages[-1][0])
    else:
        logger.info("Phase 0: no page markers found, falling back to heuristic split.")

    initial_structure = _build_initial_structure(text_content, book_meta)
    logger.info("Phase 0: heuristic structure with %d chapters.", len(initial_structure.chapters))

    # Phase 1 — refine via LLM (best-effort). When page-based, we feed the
    # first PHASE1_TOC_PAGES book pages so the LLM returns precise pagination.
    llm_payload = await _phase1_detect_structure(
        text_content, book_meta, phase1_tpl,
        pages=pages if page_based else None,
        model=model,
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key,
    )
    if llm_payload:
        structure = _merge_llm_structure(initial_structure, llm_payload, text_content)
        logger.info("Phase 1: typology=%s, %d chapters, confidence=%s",
                    structure.book_type, len(structure.chapters), structure.confidence)
    else:
        structure = initial_structure
        logger.info("Phase 1: skipped, using Phase 0 heuristic structure.")

    # Re-slice chapter texts by page range (with margin) when we have page
    # markers AND the LLM returned page hints. This overrides the title-match
    # body assignment from `_merge_llm_structure`, which was unreliable.
    coverage = 0.0
    n_resliced = 0
    # Fix A — detect coarse pagination (EPUB / sparse markers). When the mean
    # chars per detected page is large, the ±PHASE2_PAGE_MARGIN page window
    # grabs most of the book per chapter, so we slice by title offset instead.
    coarse_pagination = False
    if page_based:
        mean_chars_per_page = len(text_content) / max(1, len(pages))
        coarse_pagination = mean_chars_per_page > COARSE_PAGE_CHARS_THRESHOLD
        if coarse_pagination:
            logger.info(
                "Phase 0: coarse pagination (%d pages, ~%.0f chars/page > %d) — "
                "slicing chapters by title offset instead of page window.",
                len(pages), mean_chars_per_page, COARSE_PAGE_CHARS_THRESHOLD,
            )
    if page_based and not coarse_pagination:
        # v2 (Lot B/E refactor) — cap toxic positional fallback BEFORE re-slice,
        # only when page_based=True so we have something to rebuild from. On
        # PLAIN_TEXT sources we keep the body even if oversized — there is no
        # alternative source of chapter text without pagination markers.
        cap_words = 2 * THRESHOLD_MAX_CHAPTER_WORDS
        for ch in structure.chapters:
            if ch.word_count > cap_words:
                logger.warning(
                    "Lot B cap: chapter %d (%s) had %d words from positional fallback "
                    "(> %d), clearing body for re-slice by pages.",
                    ch.num, ch.title[:50], ch.word_count, cap_words,
                )
                ch.text = ""

        for ch in structure.chapters:
            sp, ep = ch.pages if ch.pages else (None, None)
            if sp is not None or ep is not None:
                sliced = _slice_pages_window(pages, sp, ep, margin=PHASE2_PAGE_MARGIN)
                if sliced:
                    ch.text = sliced
                    n_resliced += 1
        coverage = n_resliced / max(1, len(structure.chapters))
        logger.info("Phase 0: %d/%d chapters re-sliced by page range (±%d margin) — coverage %.0f%%.",
                    n_resliced, len(structure.chapters), PHASE2_PAGE_MARGIN, coverage * 100)
        # v2 (Lot D) — Surface a structure_warning to the UI when coverage is
        # below 80 %. This typically signals that Phase 1 returned chapters
        # without page hints (Latour test case: 4/12 → 33 %), and even after
        # Lot B interpolation the resulting fiche may be partially incorrect.
        if coverage < 0.8:
            logger.warning(
                "Lot D: low pagination coverage (%d/%d = %.0f%%) — fiche-chapitres possibly incomplete.",
                n_resliced, len(structure.chapters), coverage * 100,
            )
            if progress_cb:
                await progress_cb(
                    "structure_warning", 0, 1,
                    f"Pagination partielle ({n_resliced}/{len(structure.chapters)} chapitres) — interpolation appliquée",
                )
    elif page_based and coarse_pagination:
        # Fix A — coarse pagination: assign non-overlapping chapter bodies by
        # title offset (the page window would otherwise grab ~the whole book per
        # chapter). The helper is the source of truth here, so no Lot B cap.
        n_resliced = _assign_bodies_by_title_offset(structure.chapters, text_content)
        coverage = n_resliced / max(1, len(structure.chapters))
        logger.info(
            "Phase 0: %d/%d chapters bodied by title offset (coarse pagination) — coverage %.0f%%.",
            n_resliced, len(structure.chapters), coverage * 100,
        )
        if coverage < 0.8 and progress_cb:
            await progress_cb(
                "structure_warning", 0, 1,
                f"Pagination grossière — découpage par titre ({n_resliced}/{len(structure.chapters)} chapitres)",
            )

    # v2 — Filter excluded paratexts (Notes / Index / Bibliographie / etc.)
    # before running Phase 2. The LLM signals these via `to_exclude=true`
    # in Phase 1; if it missed any, the user-facing fiche will still be clean
    # because the exclusion is opt-in (default False).
    excluded = [ch for ch in structure.chapters if ch.to_exclude]
    if excluded:
        logger.info(
            "Phase 0: %d paratextes exclus de l'analyse: %s",
            len(excluded),
            ", ".join(f"ch.{c.num} ({c.title[:40]})" for c in excluded),
        )
    structure.chapters = [ch for ch in structure.chapters if not ch.to_exclude]

    # v2 — Operational threshold check: if the book exceeds any of the validated
    # operational seuils, log a warning + emit a `threshold_warning` progress
    # event. The pipeline still runs (the user may want to push through), but
    # quality of the last chapters and synthesis can degrade.
    total_words = sum(ch.word_count for ch in structure.chapters)
    max_chapter_words = max((ch.word_count for ch in structure.chapters), default=0)
    thresholds_exceeded: List[str] = []
    if len(structure.chapters) > THRESHOLD_MAX_CHAPTERS:
        thresholds_exceeded.append(
            f"{len(structure.chapters)} chapitres (seuil: {THRESHOLD_MAX_CHAPTERS})"
        )
    if total_words > THRESHOLD_MAX_TOTAL_WORDS:
        thresholds_exceeded.append(
            f"{total_words} mots utiles (seuil: {THRESHOLD_MAX_TOTAL_WORDS})"
        )
    if max_chapter_words > THRESHOLD_MAX_CHAPTER_WORDS:
        thresholds_exceeded.append(
            f"chapitre max {max_chapter_words} mots (seuil: {THRESHOLD_MAX_CHAPTER_WORDS})"
        )
    if thresholds_exceeded:
        logger.warning(
            "Seuils opérationnels franchis: %s. Qualité dégradée possible sur les derniers chapitres et la synthèse.",
            " | ".join(thresholds_exceeded),
        )
        if progress_cb:
            await progress_cb(
                "threshold_warning", 0, 1,
                "Seuils franchis : " + ", ".join(thresholds_exceeded),
            )

    if progress_cb:
        await progress_cb("structure_detection", 1, 1, f"{len(structure.chapters)} chapitres détectés")

    # Phase 2 — analyse each chapter sequentially. Sequential (not parallel)
    # so each call can read prior summaries; the global semaphore still gates
    # across other concurrent users.
    chapter_blocks: List[str] = []
    summaries: List[str] = []
    total = len(structure.chapters)
    for idx, chapter in enumerate(structure.chapters, start=1):
        if progress_cb:
            await progress_cb("chapter_analysis", idx, total,
                              f"Ch.{chapter.num} — {chapter.title[:60]}")
        try:
            block, summary = await _phase2_analyse_chapter(
                chapter, structure, book_meta, summaries, phase2_tpl,
                model=model,
                openai_api_key=openai_api_key,
                openrouter_api_key=openrouter_api_key,
            )
        except Exception as exc:
            logger.error("Phase 2: chapter %d failed: %s", chapter.num, exc)
            block = (
                f"<h3>Chapitre {chapter.num} — « {chapter.title} »</h3>"
                f"<p><em>⚠️ Analyse de ce chapitre indisponible (erreur LLM).</em></p>"
            )
            summary = f"Ch.{chapter.num} : non analysé."
        chapter.html_block = block
        chapter.short_summary = summary
        chapter_blocks.append(block)
        summaries.append(f"Ch.{chapter.num} ({', '.join(chapter.authors) or '?'}) : {summary}")

    # Phase 3 — global synthesis.
    if progress_cb:
        await progress_cb("synthesis", 0, 1, "Synthèse transversale")
    try:
        section_a, section_b = await _phase3_synthesise(
            structure, summaries, book_meta, phase3_tpl,
            model=model,
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
        )
    except Exception as exc:
        logger.error("Phase 3 failed, using minimal sections: %s", exc)
        section_a = (
            "<h3>1. Identification de l'ouvrage</h3>"
            f"<p>{book_meta.get('title', '')} — {book_meta.get('authors', '')}</p>"
        )
        section_b = "<h3>4. Synthèse transversale</h3><p><em>Synthèse globale indisponible.</em></p>"
    if progress_cb:
        await progress_cb("synthesis", 1, 1, "Synthèse terminée")

    # Phase 4 — assemble.
    sentinel, note_html = _assemble_book_html(
        book_meta, structure, section_a, chapter_blocks, section_b,
        source_format=source_format,
        coverage=coverage if page_based else 1.0,
    )
    logger.info("Book note assembled: %d chapters, sentinel=%s", len(structure.chapters), sentinel)
    return sentinel, note_html
