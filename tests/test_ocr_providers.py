"""
OCR resilience tests — SPRINT_ocr_resilience Lot 1 (Mistral retry/diagnostic)
and Lot 2 (anti-truncation guard).

Lot 1 verifies that:
  - `_classify_mistral_http_error` maps 401 to an actionable permanent error
    (spend-cap diagnostic), and 404/408/429/5xx to a retryable transient error.
  - `_mistral_upload_and_ocr` retries transient failures (404 "Could not get
    file", 429, 5xx, network timeouts) up to MISTRAL_OCR_RETRIES, but never
    retries a permanent failure (401).

Lot 2 verifies that:
  - `OCRResult` carries the new partial/pages metadata with safe defaults.
  - `_ocr_density_warning` flags suspiciously sparse extractions.
  - `extract_text_with_ocr` never returns a capped/sparse OCR as a silent
    success: a 10/284-page OpenAI result is flagged `partial=True` (unless a
    non-capped engine does better), and a sparse legacy result is flagged too.

Run with: pytest tests/test_ocr_providers.py -v
"""

import os
import subprocess
import types
from unittest import mock

import pytest
import requests
import fitz  # type: ignore[import-not-found]

from scripts import rad_dataframe as rad
from scripts.rad_dataframe import (
    OCRResult,
    _PagedOcr,
    _MistralOcr,
    _MistralTransientError,
    _MistralAuthError,
    OCRExtractionError,
    _classify_mistral_http_error,
    _ocr_density_warning,
    _finalize_ocr_result,
    _parse_retry_after,
    _mistral_retry_wait,
    _extract_text_with_mistral,
    extract_text_with_ocr,
)


def _build_pdf(tmp_path, n_pages: int, name: str = "doc") -> str:
    """Generate a minimal multi-page PDF for split-path tests."""
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page(width=200, height=200)
        page.insert_text((10, 50), f"page body {i + 1}")
    out = tmp_path / f"{name}_{n_pages}p.pdf"
    doc.save(str(out))
    doc.close()
    return str(out)


def _http_error_with_headers(status: int, headers: dict, text: str = "boom") -> requests.HTTPError:
    """Like _http_error but with response headers (for Retry-After tests)."""
    resp = mock.Mock()
    resp.status_code = status
    resp.text = text
    resp.headers = headers
    err = requests.HTTPError(f"HTTP {status}")
    err.response = resp
    return err


def _http_error(status: int, text: str = "boom") -> requests.HTTPError:
    """Build a requests.HTTPError carrying a response of the given status."""
    resp = mock.Mock()
    resp.status_code = status
    resp.text = text
    err = requests.HTTPError(f"HTTP {status}")
    err.response = resp
    return err


# ===========================================================================
# Lot 1 — error classification
# ===========================================================================

class TestClassifyMistralHttpError:
    def test_401_is_permanent_with_spend_cap_message(self):
        result = _classify_mistral_http_error(_http_error(401), "book.pdf")
        assert isinstance(result, OCRExtractionError)
        assert not isinstance(result, _MistralTransientError)
        msg = str(result).lower()
        assert "401" in msg
        assert "plafond" in msg  # spend-cap diagnostic surfaced

    def test_404_could_not_get_file_is_transient(self):
        result = _classify_mistral_http_error(
            _http_error(404, "Could not get file"), "book.pdf"
        )
        assert isinstance(result, _MistralTransientError)

    @pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
    def test_transient_statuses(self, status):
        result = _classify_mistral_http_error(_http_error(status), "book.pdf")
        assert isinstance(result, _MistralTransientError)

    @pytest.mark.parametrize("status", [400, 403, 413, 422])
    def test_other_4xx_is_permanent(self, status):
        result = _classify_mistral_http_error(_http_error(status), "book.pdf")
        assert isinstance(result, OCRExtractionError)
        assert not isinstance(result, _MistralTransientError)

    def test_missing_response_is_permanent(self):
        err = requests.HTTPError("no response attached")
        err.response = None
        result = _classify_mistral_http_error(err, "book.pdf")
        assert isinstance(result, OCRExtractionError)


# ===========================================================================
# Lot 1 — retry wrapper
# ===========================================================================

class TestMistralRetryWrapper:
    def test_transient_then_success_retries_and_returns(self, monkeypatch):
        monkeypatch.setattr(rad, "MISTRAL_OCR_RETRIES", 2)
        sleep = mock.Mock()
        monkeypatch.setattr(rad.time, "sleep", sleep)

        once = mock.Mock(side_effect=[
            _MistralTransientError("404 transient"),
            "# markdown OK",
        ])
        monkeypatch.setattr(rad, "_mistral_upload_and_ocr_once", once)

        result = rad._mistral_upload_and_ocr("book.pdf")
        assert result == "# markdown OK"
        assert once.call_count == 2          # one retry
        assert sleep.call_count == 1         # slept once between attempts

    def test_permanent_error_is_not_retried(self, monkeypatch):
        monkeypatch.setattr(rad, "MISTRAL_OCR_RETRIES", 3)
        sleep = mock.Mock()
        monkeypatch.setattr(rad.time, "sleep", sleep)

        once = mock.Mock(side_effect=OCRExtractionError("401 spend cap"))
        monkeypatch.setattr(rad, "_mistral_upload_and_ocr_once", once)

        with pytest.raises(OCRExtractionError, match="401"):
            rad._mistral_upload_and_ocr("book.pdf")
        assert once.call_count == 1          # no retry on permanent failure
        assert sleep.call_count == 0

    def test_all_transient_raises_after_exhausting_retries(self, monkeypatch):
        monkeypatch.setattr(rad, "MISTRAL_OCR_RETRIES", 2)
        monkeypatch.setattr(rad.time, "sleep", mock.Mock())

        once = mock.Mock(side_effect=_MistralTransientError("still 503"))
        monkeypatch.setattr(rad, "_mistral_upload_and_ocr_once", once)

        with pytest.raises(OCRExtractionError, match="après 3 tentatives"):
            rad._mistral_upload_and_ocr("book.pdf")
        assert once.call_count == 3          # 1 + 2 retries

    def test_network_timeout_is_retried(self, monkeypatch):
        monkeypatch.setattr(rad, "MISTRAL_OCR_RETRIES", 1)
        monkeypatch.setattr(rad.time, "sleep", mock.Mock())

        once = mock.Mock(side_effect=[requests.Timeout("slow"), "# ok"])
        monkeypatch.setattr(rad, "_mistral_upload_and_ocr_once", once)

        assert rad._mistral_upload_and_ocr("book.pdf") == "# ok"
        assert once.call_count == 2


# ===========================================================================
# Lot 2 — OCRResult metadata & density guard
# ===========================================================================

class TestOcrResultMetadata:
    def test_defaults_are_safe(self):
        r = OCRResult(text="hello", provider="mistral")
        assert r.partial is False
        assert r.pages_done == 0
        assert r.pages_total == 0
        assert r.error is None

    def test_finalize_propagates_partial_fields(self):
        r = _finalize_ocr_result(
            "txt", "openai", True,
            partial=True, pages_done=10, pages_total=284, error="capped",
        )
        assert isinstance(r, OCRResult)
        assert r.partial is True
        assert (r.pages_done, r.pages_total) == (10, 284)
        assert r.error == "capped"

    def test_finalize_returns_plain_text_when_not_detailed(self):
        assert _finalize_ocr_result("txt", "openai", False, partial=True) == "txt"


class TestDensityWarning:
    def test_sparse_text_is_flagged(self):
        msg = _ocr_density_warning("x" * 100, pages_total=50, pdf_path="b.pdf", provider="legacy")
        assert msg is not None
        assert "suspect" in msg.lower()

    def test_healthy_density_is_not_flagged(self):
        text = "x" * (rad.OCR_MIN_CHARS_PER_PAGE * 10 + 100)
        assert _ocr_density_warning(text, pages_total=10, pdf_path="b.pdf", provider="legacy") is None

    def test_unknown_page_count_is_not_flagged(self):
        assert _ocr_density_warning("x" * 10, pages_total=0, pdf_path="b.pdf", provider="legacy") is None

    def test_empty_text_is_not_flagged(self):
        assert _ocr_density_warning("", pages_total=50, pdf_path="b.pdf", provider="legacy") is None


# ===========================================================================
# Lot 2 — fallback chain never returns a silent partial success
# ===========================================================================

@pytest.fixture
def no_mistral_with_openai(monkeypatch):
    """Force the chain to skip Mistral and reach the (opt-in) OpenAI branch.

    OpenAI is OFF by default now, so these tests explicitly opt in to keep
    validating the OpenAI partial/truncation guards (Lot 2).
    """
    monkeypatch.setattr(rad, "MISTRAL_API_KEY", None)
    monkeypatch.setattr(rad, "OCR_ENABLE_OPENAI_FALLBACK", True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


class TestCappedOpenAiIsFlaggedPartial:
    def test_capped_openai_with_empty_legacy_returns_flagged_partial(
        self, no_mistral_with_openai, monkeypatch
    ):
        monkeypatch.setattr(
            rad, "_extract_text_with_openai",
            mock.Mock(return_value=_PagedOcr(text="ten pages only", pages_done=10, pages_total=284)),
        )
        # Scanned book → legacy text layer is empty.
        monkeypatch.setattr(rad, "_extract_text_with_legacy_pdf", mock.Mock(return_value=""))

        result = extract_text_with_ocr("book.pdf", return_details=True)
        assert result.provider == "openai"
        assert result.partial is True
        assert (result.pages_done, result.pages_total) == (10, 284)
        assert result.error and "284" in result.error

    def test_capped_openai_prefers_more_complete_legacy(
        self, no_mistral_with_openai, monkeypatch
    ):
        monkeypatch.setattr(
            rad, "_extract_text_with_openai",
            mock.Mock(return_value=_PagedOcr(text="short excerpt", pages_done=10, pages_total=284)),
        )
        # Born-digital PDF → legacy extracts the full (much longer) text.
        full_text = "full book text " * 500
        monkeypatch.setattr(rad, "_extract_text_with_legacy_pdf", mock.Mock(return_value=full_text))

        result = extract_text_with_ocr("book.pdf", return_details=True)
        assert result.provider == "legacy"
        assert result.partial is False
        assert result.text == full_text

    def test_full_openai_healthy_is_not_partial(self, no_mistral_with_openai, monkeypatch):
        text = "x" * (rad.OCR_MIN_CHARS_PER_PAGE * 5 + 200)
        monkeypatch.setattr(
            rad, "_extract_text_with_openai",
            mock.Mock(return_value=_PagedOcr(text=text, pages_done=5, pages_total=5)),
        )
        result = extract_text_with_ocr("article.pdf", return_details=True)
        assert result.provider == "openai"
        assert result.partial is False
        assert result.error is None

    def test_full_openai_but_sparse_is_flagged_by_density(self, no_mistral_with_openai, monkeypatch):
        monkeypatch.setattr(
            rad, "_extract_text_with_openai",
            mock.Mock(return_value=_PagedOcr(text="tiny", pages_done=5, pages_total=5)),
        )
        result = extract_text_with_ocr("article.pdf", return_details=True)
        assert result.provider == "openai"
        assert result.partial is True
        assert result.error and "suspect" in result.error.lower()


class TestSparseLegacyIsFlagged:
    def test_sparse_legacy_fallback_is_flagged_partial(self, monkeypatch):
        # No remote OCR providers available → legacy is the only path.
        monkeypatch.setattr(rad, "MISTRAL_API_KEY", None)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(rad, "_extract_text_with_legacy_pdf", mock.Mock(return_value="x" * 100))
        monkeypatch.setattr(rad, "_pdf_page_count", mock.Mock(return_value=50))

        result = extract_text_with_ocr("scanned.pdf", return_details=True)
        assert result.provider == "legacy"
        assert result.partial is True
        assert result.error and "suspect" in result.error.lower()

    def test_dense_legacy_fallback_is_not_flagged(self, monkeypatch):
        monkeypatch.setattr(rad, "MISTRAL_API_KEY", None)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        dense = "word " * 5000
        monkeypatch.setattr(rad, "_extract_text_with_legacy_pdf", mock.Mock(return_value=dense))
        monkeypatch.setattr(rad, "_pdf_page_count", mock.Mock(return_value=10))

        result = extract_text_with_ocr("digital.pdf", return_details=True)
        assert result.provider == "legacy"
        assert result.partial is False
        assert result.error is None


# ===========================================================================
# Lot 1+ — hardened Mistral retry: Retry-After, exponential backoff
# ===========================================================================

class TestParseRetryAfter:
    def test_delta_seconds(self):
        assert _parse_retry_after("30") == 30.0

    def test_float_seconds(self):
        assert _parse_retry_after("2.5") == 2.5

    def test_http_date_past_is_zero(self):
        # Past HTTP-date → retry now (0 seconds), not None.
        assert _parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0

    def test_http_date_future_is_positive(self):
        assert _parse_retry_after("Wed, 21 Oct 2099 07:28:00 GMT") > 0

    def test_unparseable_string_is_none(self):
        assert _parse_retry_after("not a date at all") is None

    def test_none_and_empty(self):
        assert _parse_retry_after(None) is None
        assert _parse_retry_after("") is None

    def test_negative_is_rejected(self):
        assert _parse_retry_after("-5") is None


class TestRetryWaitComputation:
    def test_exponential_growth(self, monkeypatch):
        monkeypatch.setattr(rad, "MISTRAL_OCR_RETRY_BACKOFF", 3.0)
        monkeypatch.setattr(rad, "MISTRAL_OCR_RETRY_MAX_BACKOFF", 1000.0)
        assert _mistral_retry_wait(0) == 3.0
        assert _mistral_retry_wait(1) == 6.0
        assert _mistral_retry_wait(2) == 12.0
        assert _mistral_retry_wait(3) == 24.0

    def test_capped_at_max(self, monkeypatch):
        monkeypatch.setattr(rad, "MISTRAL_OCR_RETRY_BACKOFF", 3.0)
        monkeypatch.setattr(rad, "MISTRAL_OCR_RETRY_MAX_BACKOFF", 60.0)
        assert _mistral_retry_wait(10) == 60.0

    def test_retry_after_overrides_backoff(self, monkeypatch):
        monkeypatch.setattr(rad, "MISTRAL_OCR_RETRY_MAX_BACKOFF", 60.0)
        assert _mistral_retry_wait(5, retry_after=2.5) == 2.5

    def test_retry_after_is_also_capped(self, monkeypatch):
        monkeypatch.setattr(rad, "MISTRAL_OCR_RETRY_MAX_BACKOFF", 60.0)
        assert _mistral_retry_wait(0, retry_after=999.0) == 60.0


class TestClassifierAttachesRetryAfter:
    def test_429_with_retry_after_header(self):
        err = _http_error_with_headers(429, {"Retry-After": "12"})
        result = _classify_mistral_http_error(err, "book.pdf")
        assert isinstance(result, _MistralTransientError)
        assert result.retry_after == 12.0

    def test_429_without_header_has_no_retry_after(self):
        err = _http_error_with_headers(429, {})
        result = _classify_mistral_http_error(err, "book.pdf")
        assert isinstance(result, _MistralTransientError)
        assert result.retry_after is None


class TestRetryLoopHonorsRetryAfter:
    def test_sleep_uses_retry_after_exactly(self, monkeypatch):
        monkeypatch.setattr(rad, "MISTRAL_OCR_RETRIES", 1)
        sleep = mock.Mock()
        monkeypatch.setattr(rad.time, "sleep", sleep)

        transient = _MistralTransientError("429 rate limited")
        transient.retry_after = 7.0
        once = mock.Mock(side_effect=[transient, "# ok"])
        monkeypatch.setattr(rad, "_mistral_upload_and_ocr_once", once)

        assert rad._mistral_upload_and_ocr("book.pdf") == "# ok"
        # Retry-After is respected exactly (no jitter on that path).
        sleep.assert_called_once_with(7.0)

    def test_exponential_path_adds_jitter_within_bounds(self, monkeypatch):
        monkeypatch.setattr(rad, "MISTRAL_OCR_RETRIES", 1)
        monkeypatch.setattr(rad, "MISTRAL_OCR_RETRY_BACKOFF", 4.0)
        monkeypatch.setattr(rad, "MISTRAL_OCR_RETRY_MAX_BACKOFF", 1000.0)
        sleep = mock.Mock()
        monkeypatch.setattr(rad.time, "sleep", sleep)

        once = mock.Mock(side_effect=[_MistralTransientError("503"), "# ok"])
        monkeypatch.setattr(rad, "_mistral_upload_and_ocr_once", once)

        assert rad._mistral_upload_and_ocr("book.pdf") == "# ok"
        waited = sleep.call_args[0][0]
        # base=4.0 (attempt 0), jitter in [0, min(1.0, 5.0)] → [4.0, 5.0]
        assert 4.0 <= waited <= 5.0


# ===========================================================================
# OpenAI fallback is OFF by default (operational requirement)
# ===========================================================================

class TestOpenAiDisabledByDefault:
    def test_default_flag_is_false(self):
        # In a clean test environment OCR_ENABLE_OPENAI_FALLBACK must default off.
        assert rad.OCR_ENABLE_OPENAI_FALLBACK is False

    def test_openai_not_called_when_disabled_mistral_fails(self, monkeypatch):
        """Mistral fails, OpenAI key present but fallback OFF → legacy, OpenAI untouched."""
        monkeypatch.setattr(rad, "MISTRAL_API_KEY", "key")
        monkeypatch.setattr(rad, "OCR_ENABLE_OPENAI_FALLBACK", False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(
            rad, "_extract_text_with_mistral",
            mock.Mock(side_effect=OCRExtractionError("mistral down")),
        )
        openai_spy = mock.Mock(side_effect=AssertionError("OpenAI must NOT be called"))
        monkeypatch.setattr(rad, "_extract_text_with_openai", openai_spy)
        dense = "word " * 5000
        monkeypatch.setattr(rad, "_extract_text_with_legacy_pdf", mock.Mock(return_value=dense))
        monkeypatch.setattr(rad, "_pdf_page_count", mock.Mock(return_value=10))

        result = extract_text_with_ocr("book.pdf", return_details=True)
        assert result.provider == "legacy"
        openai_spy.assert_not_called()

    def test_openai_used_when_explicitly_enabled(self, monkeypatch):
        """Re-enabling the flag restores the opt-in OpenAI path."""
        monkeypatch.setattr(rad, "MISTRAL_API_KEY", None)
        monkeypatch.setattr(rad, "OCR_ENABLE_OPENAI_FALLBACK", True)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        text = "x" * (rad.OCR_MIN_CHARS_PER_PAGE * 5 + 200)
        monkeypatch.setattr(
            rad, "_extract_text_with_openai",
            mock.Mock(return_value=_PagedOcr(text=text, pages_done=5, pages_total=5)),
        )
        result = extract_text_with_ocr("article.pdf", return_details=True)
        assert result.provider == "openai"


# ===========================================================================
# 401 = account-level auth error (aborts a multi-part book)
# ===========================================================================

class TestMistralAuthError:
    def test_401_classified_as_auth_error_subclass(self):
        result = _classify_mistral_http_error(_http_error(401), "book.pdf")
        assert isinstance(result, _MistralAuthError)
        assert isinstance(result, OCRExtractionError)        # still a permanent OCR error
        assert not isinstance(result, _MistralTransientError)

    def test_auth_error_is_not_retried(self, monkeypatch):
        monkeypatch.setattr(rad, "MISTRAL_OCR_RETRIES", 3)
        sleep = mock.Mock()
        monkeypatch.setattr(rad.time, "sleep", sleep)
        once = mock.Mock(side_effect=_MistralAuthError("401 spend cap"))
        monkeypatch.setattr(rad, "_mistral_upload_and_ocr_once", once)
        with pytest.raises(_MistralAuthError):
            rad._mistral_upload_and_ocr("book.pdf")
        assert once.call_count == 1
        assert sleep.call_count == 0


# ===========================================================================
# Retry wrapper — extra coverage (review-driven)
# ===========================================================================

class TestRetryWrapperExtra:
    def test_connection_error_is_retried(self, monkeypatch):
        monkeypatch.setattr(rad, "MISTRAL_OCR_RETRIES", 1)
        monkeypatch.setattr(rad.time, "sleep", mock.Mock())
        once = mock.Mock(side_effect=[requests.ConnectionError("host unreachable"), "# ok"])
        monkeypatch.setattr(rad, "_mistral_upload_and_ocr_once", once)
        assert rad._mistral_upload_and_ocr("book.pdf") == "# ok"
        assert once.call_count == 2

    def test_max_pages_passed_through_to_once(self, monkeypatch):
        monkeypatch.setattr(rad, "MISTRAL_OCR_RETRIES", 2)
        once = mock.Mock(return_value="# ok")
        monkeypatch.setattr(rad, "_mistral_upload_and_ocr_once", once)
        rad._mistral_upload_and_ocr("book.pdf", max_pages=123)
        once.assert_called_once_with("book.pdf", max_pages=123)


# ===========================================================================
# Split-path partial-book salvage (review HIGH finding)
# ===========================================================================

@pytest.fixture
def force_split(monkeypatch):
    """Force _extract_text_with_mistral down the split path: tiny page cap."""
    monkeypatch.setattr(rad, "MISTRAL_API_KEY", "key")
    monkeypatch.setattr(rad, "MISTRAL_MAX_PAGES", 2)
    monkeypatch.setattr(rad, "MISTRAL_SPLIT_PART_PAGES", 1)
    monkeypatch.setattr(rad, "MISTRAL_AUTO_SPLIT", True)
    monkeypatch.setattr(rad, "MISTRAL_AUTO_COMPRESS", False)


class TestSplitPartialSalvage:
    def test_fast_path_returns_non_partial_struct(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rad, "MISTRAL_API_KEY", "key")
        monkeypatch.setattr(rad, "_mistral_upload_and_ocr", mock.Mock(return_value="<!-- Page 1 -->\nhi"))
        pdf = _build_pdf(tmp_path, 1, "small")
        out = _extract_text_with_mistral(pdf)
        assert isinstance(out, _MistralOcr)
        assert out.partial is False
        assert out.pages_total == 1

    def test_some_parts_fail_salvages_and_flags_partial(self, tmp_path, force_split, monkeypatch):
        # 3 pages → 3 one-page parts; part 2 fails (part-local), 1 and 3 succeed.
        calls = [
            "<!-- Page 1 -->\nAAA",
            OCRExtractionError("part 2 OCR échoué après 5 tentatives"),
            "<!-- Page 1 -->\nCCC",
        ]
        monkeypatch.setattr(rad, "_mistral_upload_and_ocr", mock.Mock(side_effect=calls))
        pdf = _build_pdf(tmp_path, 3, "book")
        out = _extract_text_with_mistral(pdf)
        assert out.partial is True
        assert out.pages_done == 2 and out.pages_total == 3
        assert "OCR ÉCHOUÉ" in out.text          # explicit gap marker for the failed part
        assert "AAA" in out.text and "CCC" in out.text
        assert out.error and "1/3 parts" in out.error
        # Page renumbering preserved: part 3's page becomes global page 3.
        assert "<!-- Page 3 -->" in out.text

    def test_all_parts_fail_raises(self, tmp_path, force_split, monkeypatch):
        monkeypatch.setattr(
            rad, "_mistral_upload_and_ocr",
            mock.Mock(side_effect=OCRExtractionError("down")),
        )
        pdf = _build_pdf(tmp_path, 3, "book")
        with pytest.raises(OCRExtractionError, match="Aucune part"):
            _extract_text_with_mistral(pdf)

    def test_auth_error_mid_split_aborts_whole_book(self, tmp_path, force_split, monkeypatch):
        # Part 1 ok, part 2 → 401: must abort (not salvage) so caller surfaces it.
        calls = ["<!-- Page 1 -->\nAAA", _MistralAuthError("401 spend cap")]
        monkeypatch.setattr(rad, "_mistral_upload_and_ocr", mock.Mock(side_effect=calls))
        pdf = _build_pdf(tmp_path, 3, "book")
        with pytest.raises(_MistralAuthError):
            _extract_text_with_mistral(pdf)


class TestMistralPartialFlowsToOcrResult:
    def test_partial_mistral_surfaces_in_ocrresult(self, monkeypatch):
        monkeypatch.setattr(rad, "MISTRAL_API_KEY", "key")
        monkeypatch.setattr(
            rad, "_extract_text_with_mistral",
            mock.Mock(return_value=_MistralOcr(
                text="salvaged", partial=True, pages_done=2, pages_total=3,
                error="OCR Mistral partiel : 1/3 parts en échec",
            )),
        )
        result = extract_text_with_ocr("book.pdf", return_details=True)
        assert result.provider == "mistral"
        assert result.partial is True
        assert (result.pages_done, result.pages_total) == (2, 3)
        assert result.error and "partiel" in result.error

    def test_full_mistral_is_not_partial(self, monkeypatch):
        monkeypatch.setattr(rad, "MISTRAL_API_KEY", "key")
        monkeypatch.setattr(
            rad, "_extract_text_with_mistral",
            mock.Mock(return_value=_MistralOcr(text="full", partial=False, pages_done=10, pages_total=10)),
        )
        result = extract_text_with_ocr("book.pdf", return_details=True)
        assert result.provider == "mistral"
        assert result.partial is False
        assert result.error is None


# ===========================================================================
# Lot 4 — OCR LOCAL (Docling) provider
# ===========================================================================

@pytest.fixture
def mistral_fails_no_openai(monkeypatch):
    """Mistral configuré mais échoue ; OpenAI off (défaut) → on atteint le local."""
    monkeypatch.setattr(rad, "MISTRAL_API_KEY", "key")
    monkeypatch.setattr(rad, "OCR_ENABLE_OPENAI_FALLBACK", False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        rad, "_extract_text_with_mistral",
        mock.Mock(side_effect=OCRExtractionError("mistral down")),
    )


class TestLocalOcrChain:
    def test_local_used_before_legacy_when_available(self, mistral_fails_no_openai, monkeypatch):
        monkeypatch.setattr(rad, "OCR_ENABLE_LOCAL_FALLBACK", True)
        monkeypatch.setattr(rad, "_local_ocr_available", lambda: True)
        dense = "word " * 5000  # 25 000 car. → 2 500 car./page sur 10 pages (> seuil 500)
        monkeypatch.setattr(rad, "_extract_text_with_local", mock.Mock(return_value=dense))
        monkeypatch.setattr(rad, "_pdf_page_count", mock.Mock(return_value=10))
        legacy_spy = mock.Mock(side_effect=AssertionError("legacy must NOT run when local succeeds"))
        monkeypatch.setattr(rad, "_extract_text_with_legacy_pdf", legacy_spy)

        result = extract_text_with_ocr("book.pdf", return_details=True)
        assert result.provider == rad.LOCAL_OCR_ENGINE  # "docling"
        assert result.partial is False
        legacy_spy.assert_not_called()

    def test_local_sparse_flagged_partial(self, mistral_fails_no_openai, monkeypatch):
        monkeypatch.setattr(rad, "OCR_ENABLE_LOCAL_FALLBACK", True)
        monkeypatch.setattr(rad, "_local_ocr_available", lambda: True)
        monkeypatch.setattr(rad, "_extract_text_with_local", mock.Mock(return_value="x" * 50))
        monkeypatch.setattr(rad, "_pdf_page_count", mock.Mock(return_value=200))
        monkeypatch.setattr(rad, "_extract_text_with_legacy_pdf", mock.Mock(return_value=""))

        result = extract_text_with_ocr("scan.pdf", return_details=True)
        assert result.provider == rad.LOCAL_OCR_ENGINE
        assert result.partial is True
        assert result.error and "suspect" in result.error.lower()

    def test_local_skipped_when_unavailable(self, mistral_fails_no_openai, monkeypatch):
        monkeypatch.setattr(rad, "OCR_ENABLE_LOCAL_FALLBACK", True)
        monkeypatch.setattr(rad, "_local_ocr_available", lambda: False)
        local_spy = mock.Mock(side_effect=AssertionError("local must NOT run when unavailable"))
        monkeypatch.setattr(rad, "_extract_text_with_local", local_spy)
        monkeypatch.setattr(rad, "_extract_text_with_legacy_pdf", mock.Mock(return_value="word " * 5000))
        monkeypatch.setattr(rad, "_pdf_page_count", mock.Mock(return_value=10))

        result = extract_text_with_ocr("book.pdf", return_details=True)
        assert result.provider == "legacy"
        local_spy.assert_not_called()

    def test_local_disabled_by_flag(self, mistral_fails_no_openai, monkeypatch):
        monkeypatch.setattr(rad, "OCR_ENABLE_LOCAL_FALLBACK", False)
        monkeypatch.setattr(rad, "_local_ocr_available", lambda: True)
        local_spy = mock.Mock(side_effect=AssertionError("local must NOT run when flag off"))
        monkeypatch.setattr(rad, "_extract_text_with_local", local_spy)
        monkeypatch.setattr(rad, "_extract_text_with_legacy_pdf", mock.Mock(return_value="word " * 5000))
        monkeypatch.setattr(rad, "_pdf_page_count", mock.Mock(return_value=10))

        result = extract_text_with_ocr("book.pdf", return_details=True)
        assert result.provider == "legacy"
        local_spy.assert_not_called()

    def test_local_failure_falls_through_to_legacy(self, mistral_fails_no_openai, monkeypatch):
        monkeypatch.setattr(rad, "OCR_ENABLE_LOCAL_FALLBACK", True)
        monkeypatch.setattr(rad, "_local_ocr_available", lambda: True)
        monkeypatch.setattr(
            rad, "_extract_text_with_local",
            mock.Mock(side_effect=OCRExtractionError("docling crashed")),
        )
        monkeypatch.setattr(rad, "_extract_text_with_legacy_pdf", mock.Mock(return_value="word " * 5000))
        monkeypatch.setattr(rad, "_pdf_page_count", mock.Mock(return_value=10))

        result = extract_text_with_ocr("book.pdf", return_details=True)
        assert result.provider == "legacy"


class TestLocalOcrSubprocessWrapper:
    def test_success_reads_output_markdown(self, monkeypatch):
        def fake_run(cmd, capture_output, text, timeout):
            out = cmd[cmd.index("--output") + 1]
            with open(out, "w", encoding="utf-8") as f:
                f.write("<!-- Page 1 -->\nhello")
            return types.SimpleNamespace(returncode=0, stderr="", stdout="")
        monkeypatch.setattr(rad.subprocess, "run", fake_run)
        assert rad._extract_text_with_local("book.pdf") == "<!-- Page 1 -->\nhello"

    def test_nonzero_returncode_raises(self, monkeypatch):
        def fake_run(cmd, capture_output, text, timeout):
            return types.SimpleNamespace(returncode=3, stderr="boom", stdout="")
        monkeypatch.setattr(rad.subprocess, "run", fake_run)
        with pytest.raises(OCRExtractionError, match="rc=3"):
            rad._extract_text_with_local("book.pdf")

    def test_empty_output_raises(self, monkeypatch):
        def fake_run(cmd, capture_output, text, timeout):
            out = cmd[cmd.index("--output") + 1]
            open(out, "w").close()
            return types.SimpleNamespace(returncode=0, stderr="", stdout="")
        monkeypatch.setattr(rad.subprocess, "run", fake_run)
        with pytest.raises(OCRExtractionError, match="vide"):
            rad._extract_text_with_local("book.pdf")

    def test_timeout_raises(self, monkeypatch):
        def fake_run(cmd, capture_output, text, timeout):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)
        monkeypatch.setattr(rad.subprocess, "run", fake_run)
        with pytest.raises(OCRExtractionError, match="timeout"):
            rad._extract_text_with_local("book.pdf")

    def test_max_pages_forwarded(self, monkeypatch):
        captured = {}
        def fake_run(cmd, capture_output, text, timeout):
            captured["cmd"] = cmd
            out = cmd[cmd.index("--output") + 1]
            with open(out, "w", encoding="utf-8") as f:
                f.write("ok")
            return types.SimpleNamespace(returncode=0, stderr="", stdout="")
        monkeypatch.setattr(rad.subprocess, "run", fake_run)
        rad._extract_text_with_local("book.pdf", max_pages=42)
        assert "--max-pages" in captured["cmd"]
        assert captured["cmd"][captured["cmd"].index("--max-pages") + 1] == "42"


class TestLocalOcrAvailability:
    def test_available_true_when_check_returns_zero(self, monkeypatch):
        monkeypatch.setattr(rad, "_LOCAL_OCR_AVAILABLE", None)
        monkeypatch.setattr(
            rad.subprocess, "run",
            lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
        )
        assert rad._local_ocr_available() is True

    def test_available_false_when_check_nonzero(self, monkeypatch):
        monkeypatch.setattr(rad, "_LOCAL_OCR_AVAILABLE", None)
        monkeypatch.setattr(
            rad.subprocess, "run",
            lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="", stderr=""),
        )
        assert rad._local_ocr_available() is False

    def test_venv_python_preferred_when_present(self, monkeypatch):
        monkeypatch.setattr(rad, "LOCAL_OCR_PYTHON", "")
        monkeypatch.setattr(rad.os.path, "exists", lambda p: p == rad._LOCAL_OCR_VENV_PYTHON)
        assert rad._local_ocr_python() == rad._LOCAL_OCR_VENV_PYTHON

    def test_explicit_python_overrides(self, monkeypatch):
        monkeypatch.setattr(rad, "LOCAL_OCR_PYTHON", "/custom/py")
        assert rad._local_ocr_python() == "/custom/py"


class TestRadChunkSkipsLocalProviders:
    def test_recode_skip_set_includes_local_engines(self):
        try:
            from scripts.rad_chunk import RECODE_SKIP_PROVIDERS
        except Exception:
            pytest.skip("rad_chunk import indisponible dans cet environnement")
        assert "docling" in RECODE_SKIP_PROVIDERS
        assert "mistral" in RECODE_SKIP_PROVIDERS  # non-régression
        assert "csv" in RECODE_SKIP_PROVIDERS
