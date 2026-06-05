"""
rad_dataframe.py - Extraction de données Zotero et OCR de documents PDF.

Ce module fournit des fonctionnalités pour :
- Charger des exports JSON Zotero et extraire les métadonnées
- Effectuer l'OCR sur les PDFs associés (Mistral, OpenAI, ou legacy)
- Générer un DataFrame pandas avec texte extrait et métadonnées
- Support du traitement parallèle avec ThreadPoolExecutor (Phase 2)

Usage en ligne de commande:
    python rad_dataframe.py --json export.json --dir ./pdfs --output result.csv

Example:
    >>> from scripts.rad_dataframe import load_zotero_to_dataframe_incremental
    >>> df = load_zotero_to_dataframe_incremental(
    ...     "zotero_export.json",
    ...     "/path/to/pdfs",
    ...     "output.csv"
    ... )
"""

import os
import sys
import json
import re
import unicodedata
import base64
import time
import csv
import random
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import pandas as pd
from typing import Optional, List, NamedTuple, Dict, Any, Tuple, Set

# Constants for retry logic
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # Exponential backoff: 2^attempt seconds

# Thread-safe lock for CSV writing and progress saving
_CSV_LOCK = threading.Lock()
_PROGRESS_LOCK = threading.Lock()

# Semaphore for API rate limiting (Phase 2 optimization)
# Limits concurrent OCR API calls to prevent rate limit errors
MISTRAL_CONCURRENT_CALLS = int(os.getenv('MISTRAL_CONCURRENT_CALLS', 3))
MISTRAL_SEMAPHORE = threading.Semaphore(MISTRAL_CONCURRENT_CALLS)

def strip_accents(s: str) -> str:
    """
    Remove accents from a string using Unicode normalization.

    Converts accented characters to their base form by decomposing
    them (NFD) and removing combining diacritical marks.

    Args:
        s: Input string potentially containing accented characters

    Returns:
        String with all accents removed

    Example:
        >>> strip_accents("café résumé")
        'cafe resume'
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def ascii_flat(s: str) -> str:
    """
    Convert a string to lowercase ASCII, removing non-ASCII characters.

    Uses NFKD normalization to decompose characters and then encodes
    to ASCII, ignoring characters that cannot be represented.

    Args:
        s: Input string to convert

    Returns:
        Lowercase ASCII string

    Example:
        >>> ascii_flat("Café Résumé 日本語")
        'cafe resume '
    """
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii').lower()


def alphanum_only(s: str) -> str:
    """
    Extract only alphanumeric characters from a string.

    First converts to ASCII lowercase, then keeps only letters and digits.
    Useful for fuzzy filename matching.

    Args:
        s: Input string to filter

    Returns:
        String containing only alphanumeric characters

    Example:
        >>> alphanum_only("Document (v2) - Final.pdf")
        'documentv2finalpdf'
    """
    return ''.join(c for c in ascii_flat(s) if c.isalnum())


def levenshtein(a: str, b: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.

    The Levenshtein distance is the minimum number of single-character
    edits (insertions, deletions, or substitutions) required to
    transform one string into another.

    Uses a simple O(n*m) dynamic programming approach, suitable for
    short strings like filenames.

    Args:
        a: First string
        b: Second string

    Returns:
        Integer distance between the two strings

    Example:
        >>> levenshtein("kitten", "sitting")
        3
    """
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    previous_row = range(len(b) + 1)
    for i, ca in enumerate(a):
        current_row = [i + 1]
        for j, cb in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (ca != cb)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]
import fitz  # PyMuPDF
from tqdm import tqdm
import logging
import argparse
import requests
from dotenv import load_dotenv

# Metrics tracking (optional - works with or without prometheus)
try:
    from scripts.metrics_helper import track_pdf_extraction, track_error, log_metrics_summary
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    track_pdf_extraction = None
    track_error = None
    log_metrics_summary = None

# Load environment variables early
load_dotenv()

# ----------------------------------------------------------------------
# Environment variable helper with validation
# ----------------------------------------------------------------------
def get_env_int(key: str, default: int, min_val: int = 1) -> int:
    """Get integer from environment with validation and fallback."""
    try:
        value = int(os.getenv(key, default))
        return max(min_val, value)
    except (ValueError, TypeError):
        return default

# Concurrency configuration (pour Phase 2: parallélisation OCR)
PDF_EXTRACTION_WORKERS = get_env_int('PDF_EXTRACTION_WORKERS', 1)

# --- Path and Logging Setup ---
SCRIPT_FILE_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_FILE_PATH)  # Should be /.../__RAG/ragpy/scripts
RAGPY_DIR_SCRIPT = os.path.dirname(SCRIPT_DIR)    # Should be /.../__RAG/ragpy
LOG_DIR_SCRIPT = os.path.join(RAGPY_DIR_SCRIPT, "logs")

os.makedirs(LOG_DIR_SCRIPT, exist_ok=True)
pdf_processing_log_file = os.path.join(LOG_DIR_SCRIPT, 'pdf_processing.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(pdf_processing_log_file),
        logging.StreamHandler() # Keep console output for the script as well
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Script SCRIPT_DIR: {SCRIPT_DIR}")
logger.info(f"Script RAGPY_DIR_SCRIPT: {RAGPY_DIR_SCRIPT}")
logger.info(f"Script LOG_DIR_SCRIPT: {LOG_DIR_SCRIPT}")
logger.info(f"Script log file: {pdf_processing_log_file}")
# --- End Path and Logging Setup ---

load_dotenv()

def _truthy_env(value: Optional[str], default: bool) -> bool:
    """
    Parse a string as a boolean value.

    Interprets "1", "true", "yes", "on" (case-insensitive) as True.
    Returns the default value if input is None.

    Args:
        value: String value to parse, or None
        default: Default value if input is None

    Returns:
        Boolean interpretation of the string value
    """
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """
    Get an integer value from an environment variable.

    Returns the default value if the variable is not set or cannot
    be parsed as an integer.

    Args:
        name: Name of the environment variable
        default: Default value if not set or invalid

    Returns:
        Integer value from environment or default
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"Valeur invalide pour {name}='{raw}', fallback sur {default}.")
        return default


def _env_float(name: str, default: float) -> float:
    """
    Get a float value from an environment variable.

    Returns the default value if the variable is not set or cannot
    be parsed as a float.

    Args:
        name: Name of the environment variable
        default: Default value if not set or invalid

    Returns:
        Float value from environment or default
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"Valeur invalide pour {name}='{raw}', fallback sur {default}.")
        return default


MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_BASE_URL = os.getenv("MISTRAL_API_BASE_URL", "https://api.mistral.ai")
MISTRAL_OCR_MODEL = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest")
MISTRAL_OCR_TIMEOUT = _env_int("MISTRAL_OCR_TIMEOUT", 300)
MISTRAL_DELETE_UPLOADED_FILE = _truthy_env(os.getenv("MISTRAL_DELETE_UPLOADED_FILE"), True)

# Lot 1 — transient-error resilience. Mistral occasionally returns a 404
# "Could not get file" right after a successful upload (upload→OCR race) or a
# 5xx/429; those are worth retrying before falling back to a degraded provider.
# A 401 is NOT retried (key refused or monthly spend cap).
# Backoff is exponential (MISTRAL_OCR_RETRY_BACKOFF * 2**attempt), capped at
# MISTRAL_OCR_RETRY_MAX_BACKOFF, with jitter added at the sleep site to avoid a
# thundering herd when many parallel workers hit a 429 in the same window. A
# server `Retry-After` header (429) overrides the computed wait.
MISTRAL_OCR_RETRIES = _env_int("MISTRAL_OCR_RETRIES", 4)
MISTRAL_OCR_RETRY_BACKOFF = _env_float("MISTRAL_OCR_RETRY_BACKOFF", 3.0)
MISTRAL_OCR_RETRY_MAX_BACKOFF = _env_float("MISTRAL_OCR_RETRY_MAX_BACKOFF", 60.0)

# Provider policy — the OpenAI Vision fallback is OFF by default (operational
# requirement). Its page-by-page transcription is lower fidelity AND hard-capped
# at OPENAI_OCR_MAX_PAGES, silently truncating books. The OCR chain is therefore
# Mistral → legacy (PyMuPDF). Set OCR_ENABLE_OPENAI_FALLBACK=1 to re-enable the
# opt-in OpenAI branch (kept intact, with the Lot 2 partial/truncation guards).
OCR_ENABLE_OPENAI_FALLBACK = _truthy_env(os.getenv("OCR_ENABLE_OPENAI_FALLBACK"), False)

# Large-file handling for Mistral OCR uploads (server-side limit ≈ 50 MB).
# When a PDF exceeds MISTRAL_MAX_UPLOAD_MB, the pipeline first attempts to
# recompress it (PyMuPDF garbage collection + deflate). If the file is still
# too large, it is split into sub-PDFs of MISTRAL_SPLIT_PART_MB and each part
# is OCR'd separately, then concatenated.
MISTRAL_MAX_UPLOAD_MB = _env_float("MISTRAL_MAX_UPLOAD_MB", 45.0)
MISTRAL_AUTO_COMPRESS = _truthy_env(os.getenv("MISTRAL_AUTO_COMPRESS"), True)
MISTRAL_AUTO_SPLIT = _truthy_env(os.getenv("MISTRAL_AUTO_SPLIT"), True)
MISTRAL_SPLIT_PART_MB = _env_float("MISTRAL_SPLIT_PART_MB", 30.0)

# Mistral OCR also rejects documents whose page count exceeds 1000 (HTTP 400
# code 3730: "document_parser_too_many_pages"). The size-MB split alone does
# not catch this because some PDFs are large in pages but light in bytes
# (e.g. text-only academic books > 1000 pages but < 15 MB). Lot G:
MISTRAL_MAX_PAGES = _env_int("MISTRAL_MAX_PAGES", 950)
MISTRAL_SPLIT_PART_PAGES = _env_int("MISTRAL_SPLIT_PART_PAGES", 500)

OPENAI_OCR_MODEL = os.getenv("OPENAI_OCR_MODEL", "gpt-4o-mini")
OPENAI_OCR_PROMPT = os.getenv(
    "OPENAI_OCR_PROMPT",
    "Transcris cette page PDF en Markdown lisible sans résumer ni modifier le contenu."
)
OPENAI_OCR_MAX_PAGES = _env_int("OPENAI_OCR_MAX_PAGES", 10)
OPENAI_OCR_MAX_TOKENS = _env_int("OPENAI_OCR_MAX_TOKENS", 2048)
OPENAI_OCR_RENDER_SCALE = _env_float("OPENAI_OCR_RENDER_SCALE", 2.0)

# Lot 2 — anti-truncation guard. Below this average character density a text
# document is almost certainly under-extracted (e.g. a scanned book run
# through the PyMuPDF legacy text extractor). Such results are flagged
# `partial` instead of being returned as a silent success.
OCR_MIN_CHARS_PER_PAGE = _env_int("OCR_MIN_CHARS_PER_PAGE", 500)

# Lot 4 — OCR LOCAL (Docling). Voie d'OCR sans clé/cap/réseau, exécutée dans un
# subprocess isolé (scripts/ocr_local.py) pour garder torch/easyocr HORS du
# process FastAPI. Insérée dans la chaîne avant le dernier recours `legacy`.
# `available()` se désactive proprement si le moteur n'est pas installé.
OCR_ENABLE_LOCAL_FALLBACK = _truthy_env(os.getenv("OCR_ENABLE_LOCAL_FALLBACK"), True)
LOCAL_OCR_ENGINE = os.getenv("LOCAL_OCR_ENGINE", "docling")
LOCAL_OCR_DEVICE = os.getenv("LOCAL_OCR_DEVICE", "cpu")
LOCAL_OCR_MAX_PAGES = _env_int("LOCAL_OCR_MAX_PAGES", 0)        # 0 = pas de cap
LOCAL_OCR_TIMEOUT = _env_int("LOCAL_OCR_TIMEOUT", 1800)        # par document (s)
LOCAL_OCR_CONCURRENCY = _env_int("LOCAL_OCR_CONCURRENCY", 1)   # CPU/RAM-bound → bas
LOCAL_OCR_SEMAPHORE = threading.Semaphore(LOCAL_OCR_CONCURRENCY)
_LOCAL_OCR_SCRIPT = os.path.join(SCRIPT_DIR, "ocr_local.py")
# Interpréteur Python pour le subprocess OCR local. Docling vit dans un venv
# DÉDIÉ (/opt/ocr-venv) pour ne pas casser numpy/httpx du pipeline principal ;
# on utilise donc son python. Surchargeable via LOCAL_OCR_PYTHON ; repli sur
# l'interpréteur courant (cas install bare-metal dans le même venv, déconseillé).
_LOCAL_OCR_VENV_PYTHON = "/opt/ocr-venv/bin/python"
LOCAL_OCR_PYTHON = os.getenv("LOCAL_OCR_PYTHON", "")
_LOCAL_OCR_AVAILABLE: Optional[bool] = None  # cache résolu une fois


def _local_ocr_python() -> str:
    """Interpréteur à utiliser pour `ocr_local.py` (venv Docling si présent)."""
    if LOCAL_OCR_PYTHON:
        return LOCAL_OCR_PYTHON
    if os.path.exists(_LOCAL_OCR_VENV_PYTHON):
        return _LOCAL_OCR_VENV_PYTHON
    return sys.executable


# ============================================================================
# PROGRESS TRACKING & INCREMENTAL SAVE UTILITIES
# ============================================================================

def get_progress_file_path(output_csv: str) -> str:
    """Get the path to the progress tracking file."""
    base = os.path.splitext(output_csv)[0]
    return f"{base}.progress.json"


def get_errors_file_path(output_csv: str) -> str:
    """Get the path to the errors tracking file."""
    base = os.path.splitext(output_csv)[0]
    return f"{base}_errors.json"


def load_progress(output_csv: str) -> set:
    """
    Load the set of already processed itemKeys from progress file.

    Args:
        output_csv: Path to the output CSV file

    Returns:
        Set of itemKeys that have been successfully processed
    """
    progress_file = get_progress_file_path(output_csv)
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                processed = set(data.get("processed_keys", []))
                logger.info(f"Loaded progress: {len(processed)} items already processed")
                return processed
        except Exception as e:
            logger.warning(f"Failed to load progress file: {e}")
    return set()


def save_progress(output_csv: str, processed_keys: set):
    """
    Save the set of processed itemKeys to progress file in a thread-safe manner.

    Uses _PROGRESS_LOCK to ensure safe concurrent writes from multiple threads.

    Args:
        output_csv: Path to the output CSV file
        processed_keys: Set of successfully processed itemKeys
    """
    with _PROGRESS_LOCK:
        progress_file = get_progress_file_path(output_csv)
        try:
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "processed_keys": list(processed_keys),
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save progress file: {e}")


def append_record_to_csv(output_csv: str, record: Dict[str, Any], fieldnames: List[str]):
    """
    Append a single record to the CSV file in a thread-safe manner.

    Creates the file with headers if it doesn't exist. Uses _CSV_LOCK
    to ensure safe concurrent writes from multiple threads.

    Args:
        output_csv: Path to the output CSV file
        record: Dictionary containing the record data
        fieldnames: List of column names

    Raises:
        Exception: If writing fails after acquiring the lock
    """
    with _CSV_LOCK:
        file_exists = os.path.exists(output_csv)

        try:
            with open(output_csv, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, escapechar='\\', quoting=csv.QUOTE_MINIMAL)

                if not file_exists:
                    writer.writeheader()

                writer.writerow(record)
        except Exception as e:
            logger.error(f"Failed to append record to CSV: {e}")
            raise


def save_errors(output_csv: str, errors: List[Dict[str, Any]]):
    """
    Save the list of errors to an errors file.

    Args:
        output_csv: Path to the output CSV file
        errors: List of error dictionaries
    """
    errors_file = get_errors_file_path(output_csv)
    try:
        with open(errors_file, 'w', encoding='utf-8') as f:
            json.dump({
                "total_errors": len(errors),
                "errors": errors,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2, ensure_ascii=False)
        if errors:
            logger.info(f"Saved {len(errors)} errors to {errors_file}")
    except Exception as e:
        logger.warning(f"Failed to save errors file: {e}")


def extract_text_with_ocr_retry(
    pdf_path: str,
    max_pages: Optional[int] = None,
    *,
    return_details: bool = False,
    max_retries: int = MAX_RETRIES,
):
    """
    Attempt OCR extraction with retry logic for transient failures.

    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to process
        return_details: Whether to return OCRResult with provider info
        max_retries: Maximum number of retry attempts

    Returns:
        OCR result (text or OCRResult depending on return_details)

    Raises:
        OCRExtractionError: If all retries fail
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            return extract_text_with_ocr(
                pdf_path,
                max_pages=max_pages,
                return_details=return_details
            )
        except OCRExtractionError:
            # Don't retry for permanent failures (like missing API keys)
            raise
        except (requests.RequestException, requests.Timeout, ConnectionError) as e:
            # Retry for network-related errors
            last_error = e
            if attempt < max_retries - 1:
                wait_time = RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    f"OCR attempt {attempt + 1}/{max_retries} failed for {pdf_path}: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} OCR attempts failed for {pdf_path}")
        except Exception as e:
            # For other errors, log and retry
            last_error = e
            if attempt < max_retries - 1:
                wait_time = RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    f"OCR attempt {attempt + 1}/{max_retries} failed for {pdf_path}: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} OCR attempts failed for {pdf_path}")

    raise OCRExtractionError(f"OCR extraction failed after {max_retries} attempts: {last_error}")


class OCRExtractionError(Exception):
    """Raised when OCR extraction fails for all providers."""


class _MistralTransientError(Exception):
    """Internal: a transient Mistral OCR failure worth retrying (Lot 1).

    Examples: HTTP 404 "Could not get file" right after upload, 429 rate
    limiting, 5xx server errors. Distinguished from permanent failures (401
    auth/spend-cap, other 4xx, empty response) which must NOT be retried and
    propagate as OCRExtractionError instead.

    `retry_after` carries the server-directed wait (seconds) parsed from a 429
    `Retry-After` header when present; the retry loop honours it over its own
    computed backoff.
    """
    retry_after: Optional[float] = None


class _MistralAuthError(OCRExtractionError):
    """Permanent, account-level Mistral failure (HTTP 401).

    Either the key is invalid or the workspace monthly spend cap is reached
    (see memory `reference-mistral-401-spend-cap`). It is account-global, so
    when a multi-part book is being OCR'd part-by-part there is no point trying
    the remaining parts — they would all 401 too. The split loop re-raises this
    immediately (aborting the book) instead of salvaging, whereas a part-local
    failure salvages the successful parts.
    """


class OCRResult(NamedTuple):
    """OCR outcome for one document.

    `partial` (Lot 2) is True when the text is known to be incomplete — a
    capped provider (OpenAI vision, OPENAI_OCR_MAX_PAGES) that processed fewer
    pages than the document holds, or a result whose character density is
    suspiciously low. `pages_done`/`pages_total` quantify the coverage and
    `error` carries a human-readable explanation for errors.json / the UI.
    """
    text: str
    provider: str
    partial: bool = False
    pages_done: int = 0
    pages_total: int = 0
    error: Optional[str] = None


class _PagedOcr(NamedTuple):
    """Internal return type carrying page-coverage info from a page-based
    provider (currently OpenAI vision) up to `extract_text_with_ocr`."""
    text: str
    pages_done: int
    pages_total: int


class _MistralOcr(NamedTuple):
    """Internal return type of `_extract_text_with_mistral`.

    Beyond the markdown `text`, it reports whether the document was only
    PARTIALLY OCR'd — used when a split book had some parts fail but others
    succeed (salvage instead of discarding everything). `pages_done` /
    `pages_total` quantify coverage; `error` summarises the failed parts.
    """
    text: str
    partial: bool = False
    pages_done: int = 0
    pages_total: int = 0
    error: Optional[str] = None


def _extract_text_with_legacy_pdf(pdf_path: str, max_pages: Optional[int] = None) -> str:
    """
    Extract text from a PDF using PyMuPDF (fitz) legacy extraction.

    Falls back to OCR if text extraction yields sparse results
    (less than 50 words per page).

    Args:
        pdf_path: Path to the PDF file
        max_pages: Optional maximum number of pages to process

    Returns:
        Extracted text content, with pages separated by double newlines
    """
    full_text: List[str] = []
    try:
        with fitz.open(pdf_path) as doc:
            num_pages = min(max_pages, len(doc)) if max_pages else len(doc)
            for page_num in tqdm(
                range(num_pages),
                desc=f"Extracting {os.path.basename(pdf_path)} (legacy)",
            ):
                try:
                    page = doc.load_page(page_num)
                    text = page.get_text("text").strip()
                    if len(text.split()) < 50:
                        text = page.get_text("ocr").strip()
                    full_text.append(text)
                except Exception as page_error:
                    logger.warning(f"Page {page_num} error in {pdf_path}: {page_error}")
                    continue
    except Exception as e:
        logger.error(f"Failed to process {pdf_path}: {e}")
        return ""
    return "\n\n".join(filter(None, full_text))


def _pdf_size_mb(pdf_path: str) -> float:
    """Return the size of a PDF file in megabytes."""
    return os.path.getsize(pdf_path) / (1024 * 1024)


def _pdf_page_count(pdf_path: str) -> int:
    """
    Return the page count of a PDF using PyMuPDF.

    Returns 0 on any error (corrupted PDF, missing dep). Callers must treat
    a 0 return as "unknown" — Lot G uses this defensively to avoid splitting
    a PDF whose page count cannot be determined.
    """
    try:
        with fitz.open(pdf_path) as doc:
            return doc.page_count
    except Exception as exc:  # noqa: BLE001 — defensive
        logger.warning("Page count unavailable for %s: %s", pdf_path, exc)
        return 0


def _compress_pdf_for_ocr(pdf_path: str) -> str:
    """
    Recompress a PDF using PyMuPDF to reduce file size before Mistral upload.

    Applies garbage collection (level 4 — most aggressive) and deflate
    compression on streams, images and fonts. The visual content is preserved;
    only object overhead and uncompressed streams are reduced.

    Args:
        pdf_path: Path to the source PDF file.

    Returns:
        Path to a temporary recompressed PDF file. Caller is responsible for
        deleting the temporary file.

    Raises:
        OCRExtractionError: If PyMuPDF fails to open or save the document.
    """
    import tempfile

    try:
        doc = fitz.open(pdf_path)
        tmp = tempfile.NamedTemporaryFile(
            prefix="ragpy_compressed_",
            suffix=".pdf",
            delete=False,
        )
        tmp.close()
        doc.save(
            tmp.name,
            garbage=4,
            deflate=True,
            deflate_images=True,
            deflate_fonts=True,
        )
        doc.close()
        return tmp.name
    except Exception as exc:
        raise OCRExtractionError(
            f"Échec de la compression PDF pour {pdf_path}: {exc}"
        ) from exc


def _split_pdf_for_ocr(
    pdf_path: str,
    max_size_mb: float,
    max_pages: Optional[int] = None,
) -> List[str]:
    """
    Split a large PDF into sub-PDFs each respecting both constraints.

    The number of pages per part is the minimum that satisfies BOTH:
      - the file-size constraint (`max_size_mb`), estimated by linear ratio
        from the source size;
      - the page-count constraint (`max_pages`), enforced as a hard cap so
        we stay under the Mistral OCR server limit (1000 pages, code 3730).

    Each part is recompressed (garbage=4, deflate) to maximise the chance of
    staying under the size cap.

    Args:
        pdf_path: Path to the source PDF file.
        max_size_mb: Target maximum size per part, in megabytes.
        max_pages: Optional hard cap on pages per part (None = no cap).

    Returns:
        Ordered list of temporary file paths for each part. Caller is
        responsible for deleting them.

    Raises:
        OCRExtractionError: If splitting fails or the resulting parts are empty.
    """
    import tempfile

    try:
        size_mb = _pdf_size_mb(pdf_path)
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            raise OCRExtractionError(f"PDF vide: {pdf_path}")

        # Pages per part — start from file-size ratio, then clamp by page cap.
        ratio = max_size_mb / max(size_mb, 0.001)
        pages_per_part = max(1, int(total_pages * ratio))
        if max_pages is not None and max_pages > 0:
            pages_per_part = min(pages_per_part, max_pages)
        # Always ≤ total_pages so range() loops at least once.
        pages_per_part = min(pages_per_part, total_pages)

        parts: List[str] = []
        for start in range(0, total_pages, pages_per_part):
            end = min(start + pages_per_part - 1, total_pages - 1)
            sub = fitz.open()
            sub.insert_pdf(doc, from_page=start, to_page=end)

            tmp = tempfile.NamedTemporaryFile(
                prefix=f"ragpy_part_p{start + 1}-{end + 1}_",
                suffix=".pdf",
                delete=False,
            )
            tmp.close()
            sub.save(tmp.name, garbage=4, deflate=True)
            sub.close()
            parts.append(tmp.name)

        doc.close()

        if not parts:
            raise OCRExtractionError(
                f"Échec du découpage du PDF: aucune part générée pour {pdf_path}"
            )
        return parts
    except OCRExtractionError:
        raise
    except Exception as exc:
        raise OCRExtractionError(
            f"Échec du découpage PDF pour {pdf_path}: {exc}"
        ) from exc


def _extract_text_with_mistral(pdf_path: str, max_pages: Optional[int] = None) -> "_MistralOcr":
    """
    Extract text from PDF using Mistral OCR with size-aware preprocessing.

    Mistral's `/v1/files` endpoint rejects uploads above ~50 MB. This wrapper
    implements a 3-step strategy before delegating to the actual upload call:

    1. If the file is below MISTRAL_MAX_UPLOAD_MB, upload it directly.
    2. If MISTRAL_AUTO_COMPRESS is enabled, attempt PyMuPDF recompression.
       If the result fits, OCR it as a single document.
    3. If MISTRAL_AUTO_SPLIT is enabled, split the (compressed) PDF into
       parts of MISTRAL_SPLIT_PART_MB, OCR each part sequentially, and
       concatenate the markdown outputs with explicit page-range markers.

    Resilience (review-driven): in the split path, a part that fails after its
    own retries no longer discards the whole book. The successful parts are
    kept, the failed part leaves an explicit `<!-- … OCR ÉCHOUÉ … -->` marker
    (preserving global page numbering), and the result is flagged `partial`.
    Only an account-level 401 (`_MistralAuthError`) or an all-parts-failed run
    raises — so the caller can fall back.

    Args:
        pdf_path: Path to the PDF file to process.
        max_pages: Optional cap on the number of pages to process. When
            splitting, this cap is honoured globally (parts beyond the cap
            are skipped).

    Returns:
        `_MistralOcr(text, partial, pages_done, pages_total, error)`. `text` is
        Markdown; split documents prefix each part with `<!-- Part N … -->`.

    Raises:
        _MistralAuthError: On HTTP 401 (account-level — aborts the book).
        OCRExtractionError: If MISTRAL_API_KEY is missing, every part failed,
            or the file remains over-size with auto-handling disabled.
    """
    if not MISTRAL_API_KEY:
        raise OCRExtractionError("MISTRAL_API_KEY manquante.")

    size_mb = _pdf_size_mb(pdf_path)
    page_count = _pdf_page_count(pdf_path)
    over_size = size_mb > MISTRAL_MAX_UPLOAD_MB
    over_pages = page_count > MISTRAL_MAX_PAGES > 0

    # Fast path: file under both limits.
    if not over_size and not over_pages:
        text = _mistral_upload_and_ocr(pdf_path, max_pages=max_pages)
        return _MistralOcr(
            text=text, partial=False,
            pages_done=page_count, pages_total=page_count,
        )

    logger.info(
        "PDF %s = %.1f MB / %d pages > seuils Mistral (%.1f MB / %d pages) — "
        "bascule sur le mode large-file.",
        pdf_path, size_mb, page_count, MISTRAL_MAX_UPLOAD_MB, MISTRAL_MAX_PAGES,
    )

    # Step 1: try compression. Only useful when the size is the blocker;
    # compression cannot reduce the page count, so a 1188-page PDF will still
    # need to be split even after recompression.
    skip_compression = over_pages and not over_size
    compressed_path: Optional[str] = None
    if MISTRAL_AUTO_COMPRESS and not skip_compression:
        try:
            compressed_path = _compress_pdf_for_ocr(pdf_path)
            compressed_size = _pdf_size_mb(compressed_path)
            logger.info(
                "Compression PDF: %.1f MB → %.1f MB (%s)",
                size_mb, compressed_size, os.path.basename(compressed_path),
            )
            # Only return early on compression if BOTH limits are now satisfied
            # (compression cannot reduce the page count).
            if compressed_size <= MISTRAL_MAX_UPLOAD_MB and not over_pages:
                try:
                    text = _mistral_upload_and_ocr(compressed_path, max_pages=max_pages)
                    return _MistralOcr(
                        text=text, partial=False,
                        pages_done=page_count, pages_total=page_count,
                    )
                finally:
                    _safe_unlink(compressed_path)
        except OCRExtractionError as comp_err:
            logger.warning("Compression échouée pour %s: %s", pdf_path, comp_err)
            _safe_unlink(compressed_path)
            compressed_path = None

    # Step 2: split (use compressed file if available, otherwise the original).
    if not MISTRAL_AUTO_SPLIT:
        _safe_unlink(compressed_path)
        reason = (
            f"PDF trop volumineux ({size_mb:.1f} MB)"
            if over_size
            else f"PDF trop long ({page_count} pages > {MISTRAL_MAX_PAGES})"
        )
        raise OCRExtractionError(
            f"{reason} pour Mistral et MISTRAL_AUTO_SPLIT désactivé. "
            f"Activez MISTRAL_AUTO_SPLIT=true ou réduisez le document en amont."
        )

    source_for_split = compressed_path or pdf_path
    parts: List[str] = []
    try:
        parts = _split_pdf_for_ocr(
            source_for_split,
            max_size_mb=MISTRAL_SPLIT_PART_MB,
            max_pages=MISTRAL_SPLIT_PART_PAGES,
        )
        logger.info(
            "Découpage PDF %s en %d parts (cible ≤ %.1f MB / ≤ %d pages chacune).",
            os.path.basename(pdf_path),
            len(parts),
            MISTRAL_SPLIT_PART_MB,
            MISTRAL_SPLIT_PART_PAGES,
        )

        markdown_blocks: List[str] = []
        remaining_pages = max_pages
        page_offset = 0  # Lot G — global page renumbering across parts.
        succeeded_parts = 0
        failed_parts: List[str] = []   # human-readable "i/N (pages A-B)"
        pages_done = 0
        for idx, part_path in enumerate(parts, start=1):
            # Compute per-part max_pages so the global cap is honoured.
            part_max_pages = remaining_pages if remaining_pages is not None else None
            if part_max_pages is not None and part_max_pages <= 0:
                logger.info("Cap max_pages atteint, parts restantes ignorées.")
                break

            # Page count is read up-front so global numbering advances even for
            # a failed part (the gap stays honest and book_note sees it).
            with fitz.open(part_path) as part_doc:
                part_page_count = len(part_doc)
            part_range = f"{page_offset + 1}-{page_offset + part_page_count}"

            try:
                part_text = _mistral_upload_and_ocr(part_path, max_pages=part_max_pages)
            except _MistralAuthError:
                # Account-level 401 (spend cap / bad key): the remaining parts
                # would all fail too — abort the whole book so the caller can
                # surface the actionable error rather than salvage a fragment.
                logger.error(
                    "Échec OCR Mistral 401 (compte) sur la part %d/%d — abandon du livre.",
                    idx, len(parts),
                )
                raise
            except Exception as part_err:
                # Part-local failure (transient exhausted, oversize, empty …).
                # Salvage: keep going, leave an explicit gap marker.
                logger.error(
                    "Échec OCR Mistral sur la part %d/%d (pages %s): %s — "
                    "part ignorée, OCR partiel.",
                    idx, len(parts), part_range, part_err,
                )
                failed_parts.append(f"{idx}/{len(parts)} (pages {part_range})")
                markdown_blocks.append(
                    f"<!-- Part {idx}/{len(parts)} (pages {part_range}) — "
                    f"OCR ÉCHOUÉ: {part_err} -->"
                )
                page_offset += part_page_count
                if remaining_pages is not None:
                    remaining_pages -= part_page_count
                continue

            # Lot G — Mistral indexes page markers from 1 within each part.
            # When we concatenate N parts, raw output would have duplicate
            # `<!-- Page 1 -->`, `<!-- Page 2 -->`, ... per part, breaking
            # downstream chapter slicing (book_note_generator). Renumber by
            # adding a cumulative offset = sum of pages in previous parts.
            if page_offset > 0:
                def _shift_page(m: "re.Match[str]", offset: int = page_offset) -> str:
                    try:
                        return f"<!-- Page {int(m.group(1)) + offset} -->"
                    except (TypeError, ValueError):
                        return m.group(0)
                part_text = re.sub(
                    r"<!--\s*Page\s+(\d+)\s*-->",
                    _shift_page,
                    part_text,
                )

            markdown_blocks.append(
                f"<!-- Part {idx}/{len(parts)} (pages {part_range}) -->\n{part_text}"
            )
            succeeded_parts += 1
            pages_done += part_page_count
            page_offset += part_page_count

            if remaining_pages is not None:
                remaining_pages -= part_page_count

        if succeeded_parts == 0:
            # Every part failed — let the caller fall back (legacy).
            raise OCRExtractionError(
                f"Aucune part OCRisée avec succès pour {pdf_path} "
                f"({len(failed_parts)} parts en échec)."
            )

        partial = bool(failed_parts)
        error_summary = None
        if partial:
            error_summary = (
                f"OCR Mistral partiel : {len(failed_parts)}/{len(parts)} parts "
                f"en échec ({', '.join(failed_parts)})."
            )
            logger.warning("%s pour %s", error_summary, pdf_path)
        return _MistralOcr(
            text="\n\n".join(markdown_blocks),
            partial=partial,
            pages_done=pages_done,
            pages_total=page_offset,
            error=error_summary,
        )
    finally:
        for p in parts:
            _safe_unlink(p)
        _safe_unlink(compressed_path)


def _safe_unlink(path: Optional[str]) -> None:
    """Best-effort deletion of a temporary file, never raises."""
    if not path:
        return
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError as exc:
        logger.debug("Nettoyage de %s impossible: %s", path, exc)


def _classify_mistral_http_error(exc: requests.HTTPError, pdf_path: str) -> Exception:
    """Map a Mistral HTTP error to a transient or a permanent OCR exception (Lot 1).

    Retry policy:
      - 401 → permanent: key refused OR workspace monthly spend cap reached
        (see memory `reference-mistral-401-spend-cap`). Surfaces an actionable
        message; NOT retried.
      - 404 / 408 / 429 / 5xx → transient (`_MistralTransientError`): retried by
        the caller before any fallback. 404 covers the "Could not get file"
        upload→OCR race.
      - any other 4xx → permanent (`OCRExtractionError`).

    Args:
        exc: The raised `requests.HTTPError`.
        pdf_path: Path of the PDF being processed (for the message).

    Returns:
        An exception instance to raise (transient or permanent).
    """
    resp = getattr(exc, "response", None)
    status = resp.status_code if resp is not None else None
    body = ""
    if resp is not None:
        try:
            body = resp.text or ""
        except Exception:  # noqa: BLE001 — body is best-effort only
            body = ""
    snippet = body[:300]
    name = os.path.basename(pdf_path)

    if status == 401:
        return _MistralAuthError(
            "OCR Mistral refusé (401) : clé API invalide OU plafond de dépense "
            "mensuel du workspace atteint (console.mistral.ai → Limits/Usage). "
            "Vérifiez le plafond AVANT de régénérer une clé — la même clé "
            "refonctionne une fois le plafond relevé."
        )
    if status in (404, 408, 429) or (status is not None and 500 <= status < 600):
        label = (
            "fichier introuvable (transitoire, upload→OCR)"
            if status == 404
            else f"erreur transitoire {status}"
        )
        transient = _MistralTransientError(f"Mistral OCR {label} pour {name}: {snippet}")
        if status == 429 and resp is not None:
            ra = _parse_retry_after(getattr(resp, "headers", {}).get("Retry-After"))
            if ra is not None:
                transient.retry_after = ra
        return transient
    return OCRExtractionError(f"Erreur Mistral ({status}) pour {name}: {snippet}")


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Parse a `Retry-After` header into seconds (Lot 1).

    Supports both RFC 7231 forms:
      - delta-seconds (e.g. "30") → returned as-is;
      - HTTP-date (e.g. "Wed, 21 Oct 2015 07:28:00 GMT") → converted to the
        number of seconds from now (clamped to >= 0; past dates → 0).
    Callers fall back to exponential backoff when this returns None.

    Args:
        value: Raw header value or None.

    Returns:
        Non-negative float seconds, or None if absent/unparseable.
    """
    if not value:
        return None
    text = str(value).strip()
    try:
        seconds = float(text)
        return seconds if seconds >= 0 else None
    except (TypeError, ValueError):
        pass
    # HTTP-date form.
    try:
        from email.utils import parsedate_to_datetime
        from datetime import datetime, timezone

        target = parsedate_to_datetime(text)
        if target is None:
            return None
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        delta = (target - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, delta)
    except (TypeError, ValueError, OverflowError):
        return None


def _mistral_retry_wait(attempt: int, retry_after: Optional[float] = None) -> float:
    """Compute the backoff (seconds) before Mistral retry `attempt` (0-indexed).

    Honours a server-directed `Retry-After` when present (capped at
    MISTRAL_OCR_RETRY_MAX_BACKOFF); otherwise applies exponential backoff
    (MISTRAL_OCR_RETRY_BACKOFF * 2**attempt) capped at the same ceiling. Jitter
    is intentionally NOT added here — the caller adds it only on the
    exponential path, so this helper stays deterministic and testable.

    Args:
        attempt: 0-indexed attempt number that just failed.
        retry_after: Optional server-directed wait in seconds.

    Returns:
        Wait in seconds (>= 0), capped at MISTRAL_OCR_RETRY_MAX_BACKOFF.
    """
    if retry_after is not None and retry_after > 0:
        return min(float(retry_after), MISTRAL_OCR_RETRY_MAX_BACKOFF)
    base = MISTRAL_OCR_RETRY_BACKOFF * (2 ** attempt)
    return min(base, MISTRAL_OCR_RETRY_MAX_BACKOFF)


def _mistral_upload_and_ocr(pdf_path: str, max_pages: Optional[int] = None) -> str:
    """
    Upload a PDF to Mistral and request OCR, retrying transient failures (Lot 1).

    Wraps `_mistral_upload_and_ocr_once` with a retry loop (MISTRAL_OCR_RETRIES)
    on transient errors only — 404 "Could not get file" (upload→OCR race), 429,
    5xx and network timeouts. Backoff is exponential (capped) with jitter, and a
    429 `Retry-After` header overrides the computed wait. The backoff sleep
    happens between attempts, outside the per-attempt MISTRAL_SEMAPHORE slot held
    inside `_once`, so a sleeping retry does not block other workers. Permanent
    failures (401 spend-cap, other 4xx, empty response) are raised immediately
    without retry.

    Args:
        pdf_path: Path to the PDF file (already verified to fit upload limit).
        max_pages: Optional `pages` list end for the OCR request.

    Returns:
        Extracted markdown text.

    Raises:
        OCRExtractionError: On permanent failure or once retries are exhausted.
    """
    last_error: Optional[Exception] = None
    total_attempts = MISTRAL_OCR_RETRIES + 1
    for attempt in range(total_attempts):
        try:
            return _mistral_upload_and_ocr_once(pdf_path, max_pages=max_pages)
        except (_MistralTransientError, requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            if attempt < MISTRAL_OCR_RETRIES:
                retry_after = getattr(exc, "retry_after", None)
                wait = _mistral_retry_wait(attempt, retry_after)
                # Add jitter only on the exponential path; respect an explicit
                # server Retry-After exactly (capped).
                if retry_after is None:
                    wait += random.uniform(0, min(wait * 0.25, 5.0))
                logger.warning(
                    "OCR Mistral transitoire (tentative %d/%d) pour %s: %s — "
                    "nouvel essai dans %.1fs%s",
                    attempt + 1, total_attempts, pdf_path, exc, wait,
                    " (Retry-After serveur)" if retry_after is not None else "",
                )
                time.sleep(wait)
            else:
                logger.error(
                    "OCR Mistral: échec après %d tentatives pour %s: %s",
                    total_attempts, pdf_path, exc,
                )
    raise OCRExtractionError(
        f"OCR Mistral échoué après {total_attempts} tentatives pour "
        f"{os.path.basename(pdf_path)}: {last_error}"
    )


def _mistral_upload_and_ocr_once(pdf_path: str, max_pages: Optional[int] = None) -> str:
    """
    Single Mistral upload+OCR attempt (no retry). Internal helper used by
    `_mistral_upload_and_ocr` (which adds retries) for both the fast path and
    each split part.

    Uses the global MISTRAL_SEMAPHORE to enforce concurrency limits.

    Args:
        pdf_path: Path to the PDF file (already verified to fit upload limit).
        max_pages: Optional `page_ranges.end` for the OCR request.

    Returns:
        Extracted markdown text.

    Raises:
        OCRExtractionError: Permanent failure (401, other 4xx, missing file_id,
            empty response).
        _MistralTransientError: Transient failure worth retrying (404/429/5xx).
    """
    base_url = MISTRAL_API_BASE_URL.rstrip("/")
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}

    # Acquire semaphore to limit concurrent API calls (rate limiting)
    logger.debug(f"Acquiring Mistral semaphore for {pdf_path} (limit: {MISTRAL_CONCURRENT_CALLS})")
    with MISTRAL_SEMAPHORE:
        logger.debug(f"Semaphore acquired for {pdf_path}")
        with requests.Session() as session:
            with open(pdf_path, "rb") as pdf_file:
                files = {
                    "file": (os.path.basename(pdf_path), pdf_file, "application/pdf")
                }
                data = {"purpose": "ocr"}
                upload_resp = session.post(
                    f"{base_url}/v1/files",
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=MISTRAL_OCR_TIMEOUT,
                )

            try:
                upload_resp.raise_for_status()
            except requests.HTTPError as http_err:
                raise _classify_mistral_http_error(http_err, pdf_path) from http_err
            upload_payload = upload_resp.json()
            file_id = (
                upload_payload.get("id")
                or upload_payload.get("file_id")
                or upload_payload.get("data", {}).get("id")
            )
            if not file_id:
                raise OCRExtractionError(
                    "La réponse Mistral ne contient pas d'identifiant de fichier pour l'OCR."
                )

            payload = {
                "model": MISTRAL_OCR_MODEL,
                "document": {
                    "type": "file",
                    "file_id": file_id,
                },
                "include_image_base64": False,
            }
            if max_pages:
                # The Mistral OCR API expects an explicit list of 1-indexed
                # page numbers under "pages" (the previous "page_ranges" form
                # is rejected with HTTP 422 "extra_forbidden").
                payload["pages"] = list(range(1, max_pages + 1))

            response = session.post(
                f"{base_url}/v1/ocr",
                headers={**headers, "Content-Type": "application/json"},
                json=payload,
                timeout=MISTRAL_OCR_TIMEOUT,
            )
            try:
                response.raise_for_status()
            except requests.HTTPError as http_err:
                logger.warning(
                    "Appel Mistral OCR échoué (%s) pour %s: %s",
                    response.status_code,
                    pdf_path,
                    response.text,
                )
                raise _classify_mistral_http_error(http_err, pdf_path) from http_err

            response_payload = response.json()

            # Build markdown with explicit `<!-- Page N -->` markers so
            # downstream consumers (book mode, chunker) can split by page.
            # Prefer the API-provided `pages` array, since it carries indices.
            markdown_text = ""
            if isinstance(response_payload, dict):
                pages = response_payload.get("pages")
                if isinstance(pages, list) and pages:
                    page_blocks: List[str] = []
                    for fallback_idx, page in enumerate(pages, start=1):
                        if not isinstance(page, dict):
                            continue
                        # Mistral returns a 0-indexed `index`; normalise to
                        # 1-indexed page numbers. Fallback to enumerate order.
                        raw_idx = page.get("index")
                        if isinstance(raw_idx, int):
                            page_num = raw_idx + 1
                        else:
                            page_num = fallback_idx
                        page_md = ""
                        for key in ("markdown", "text"):
                            value = page.get(key)
                            if isinstance(value, str) and value.strip():
                                page_md = value.strip()
                                break
                        if page_md:
                            page_blocks.append(f"<!-- Page {page_num} -->\n{page_md}")
                    if page_blocks:
                        markdown_text = "\n\n".join(page_blocks)

                if not markdown_text:
                    # Top-level markdown (older API shape) — no page boundaries.
                    for key in ("markdown", "text"):
                        candidate = response_payload.get(key)
                        if isinstance(candidate, str) and candidate.strip():
                            markdown_text = candidate.strip()
                            break

                if not markdown_text:
                    # Final fallback: outputs[] array (rare).
                    outputs = response_payload.get("output")
                    if isinstance(outputs, list):
                        blocks: List[str] = []
                        for block in outputs:
                            if isinstance(block, dict):
                                for key in ("markdown", "text", "content"):
                                    value = block.get(key)
                                    if isinstance(value, str) and value.strip():
                                        blocks.append(value.strip())
                                        break
                        markdown_text = "\n\n".join(blocks)

            markdown_text = markdown_text.strip()
            if not markdown_text:
                logger.warning(
                    "Réponse OCR Mistral vide pour %s (keys=%s)",
                    pdf_path,
                    list(response_payload.keys()) if isinstance(response_payload, dict) else type(response_payload),
                )
                raise OCRExtractionError("La réponse Mistral est vide.")

            if MISTRAL_DELETE_UPLOADED_FILE:
                try:
                    session.delete(
                        f"{base_url}/v1/files/{file_id}",
                        headers=headers,
                        timeout=15,
                    )
                except requests.RequestException as cleanup_error:
                    logger.debug(
                        "Échec du nettoyage du fichier OCR Mistral %s: %s",
                        file_id,
                        cleanup_error,
                    )

            logger.debug(f"Semaphore released for {pdf_path}")
            return markdown_text


def _extract_text_with_openai(
    pdf_path: str,
    api_key: str,
    max_pages: Optional[int] = None,
) -> _PagedOcr:
    """
    Transcribe a PDF page-by-page with the OpenAI vision model.

    This provider is HARD-CAPPED at OPENAI_OCR_MAX_PAGES pages (one vision API
    call per page is expensive). It therefore returns coverage information
    (`pages_done`, `pages_total`) so the caller can detect and flag silent
    truncation (Lot 2) rather than presenting a 10-page excerpt of a 284-page
    book as a success.

    Returns:
        `_PagedOcr(text, pages_done, pages_total)`. `pages_done < pages_total`
        signals the cap truncated the document.

    Raises:
        OCRExtractionError: If no page produced any text.
    """
    from openai import OpenAI

    base_limit = max_pages if max_pages is not None else float("inf")
    max_allowed = OPENAI_OCR_MAX_PAGES if OPENAI_OCR_MAX_PAGES > 0 else float("inf")

    outputs: List[str] = []
    client = OpenAI(api_key=api_key)
    total_pages = 0
    limit = 0

    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        limit = int(min(base_limit, max_allowed, total_pages))

        for page_index in tqdm(
            range(limit),
            desc=f"OpenAI OCR {os.path.basename(pdf_path)}",
        ):
            page = doc.load_page(page_index)
            matrix = fitz.Matrix(OPENAI_OCR_RENDER_SCALE, OPENAI_OCR_RENDER_SCALE)
            pix = page.get_pixmap(matrix=matrix)
            image_bytes = pix.tobytes("png")
            image_b64 = base64.b64encode(image_bytes).decode("ascii")

            user_content = [
                {
                    "type": "text",
                    "text": f"{OPENAI_OCR_PROMPT}\nPage {page_index + 1}.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ]

            response = client.chat.completions.create(
                model=OPENAI_OCR_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a meticulous OCR engine that outputs Markdown without omitting any content.",
                    },
                    {"role": "user", "content": user_content},
                ],
                max_tokens=OPENAI_OCR_MAX_TOKENS,
            )

            choice = response.choices[0] if response.choices else None
            page_text = ""
            if choice and getattr(choice, "message", None):
                page_text = (choice.message.content or "").strip()

            if page_text:
                outputs.append(f"<!-- Page {page_index + 1} -->\n{page_text}")

    if not outputs:
        raise OCRExtractionError("La réponse OpenAI est vide.")

    return _PagedOcr(
        text="\n\n".join(outputs),
        pages_done=limit,
        pages_total=total_pages,
    )


def _local_ocr_available() -> bool:
    """
    True si le moteur OCR local (Docling par défaut) est utilisable (Lot 4).

    Docling étant isolé dans un venv dédié, on délègue la détection à
    `ocr_local.py --check`, exécuté avec l'interpréteur du venv. Ce `--check`
    fait un simple `find_spec` (n'importe PAS torch). Résultat mis en cache.
    Le subprocess garantit qu'on teste le bon environnement, pas le process
    FastAPI courant.
    """
    global _LOCAL_OCR_AVAILABLE
    if _LOCAL_OCR_AVAILABLE is not None:
        return _LOCAL_OCR_AVAILABLE
    if not os.path.exists(_LOCAL_OCR_SCRIPT):
        _LOCAL_OCR_AVAILABLE = False
        return False
    try:
        proc = subprocess.run(
            [_local_ocr_python(), _LOCAL_OCR_SCRIPT, "--check", "--engine", LOCAL_OCR_ENGINE],
            capture_output=True, text=True, timeout=60,
        )
        _LOCAL_OCR_AVAILABLE = (proc.returncode == 0)
    except Exception as exc:  # noqa: BLE001 — détection best-effort
        logger.debug("Détection OCR local indisponible: %s", exc)
        _LOCAL_OCR_AVAILABLE = False
    return _LOCAL_OCR_AVAILABLE


def _extract_text_with_local(pdf_path: str, max_pages: Optional[int] = None) -> str:
    """
    OCR via le moteur LOCAL, exécuté dans le subprocess isolé `ocr_local.py`.

    Garde les dépendances lourdes (Docling → torch/easyocr) hors du process
    FastAPI. Renvoie le markdown (+ `<!-- Page N -->`). Concurrence limitée par
    `LOCAL_OCR_SEMAPHORE` (CPU/RAM-bound, ≠ rate-limit réseau).

    Args:
        pdf_path: Chemin du PDF.
        max_pages: Cap de pages (sinon `LOCAL_OCR_MAX_PAGES`, 0 = aucun).

    Returns:
        Markdown extrait.

    Raises:
        OCRExtractionError: rc≠0, timeout, ou sortie vide.
    """
    import tempfile

    tmp = tempfile.NamedTemporaryFile(prefix="ragpy_localocr_", suffix=".md", delete=False)
    tmp.close()
    cmd = [
        _local_ocr_python(), _LOCAL_OCR_SCRIPT,
        "--input", pdf_path,
        "--output", tmp.name,
        "--engine", LOCAL_OCR_ENGINE,
        "--device", LOCAL_OCR_DEVICE,
    ]
    eff_max = max_pages if (max_pages and max_pages > 0) else (LOCAL_OCR_MAX_PAGES or 0)
    if eff_max and eff_max > 0:
        cmd += ["--max-pages", str(eff_max)]

    try:
        with LOCAL_OCR_SEMAPHORE:
            logger.debug("OCR local: %s", " ".join(cmd))
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=LOCAL_OCR_TIMEOUT)
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()[-300:]
            raise OCRExtractionError(
                f"OCR local ({LOCAL_OCR_ENGINE}) échec rc={proc.returncode} pour "
                f"{os.path.basename(pdf_path)}: {stderr}"
            )
        with open(tmp.name, encoding="utf-8") as fh:
            text = fh.read().strip()
        if not text:
            raise OCRExtractionError(
                f"OCR local ({LOCAL_OCR_ENGINE}) : sortie vide pour {os.path.basename(pdf_path)}"
            )
        return text
    except subprocess.TimeoutExpired as exc:
        raise OCRExtractionError(
            f"OCR local ({LOCAL_OCR_ENGINE}) timeout ({LOCAL_OCR_TIMEOUT}s) pour "
            f"{os.path.basename(pdf_path)}"
        ) from exc
    finally:
        _safe_unlink(tmp.name)


def _finalize_ocr_result(
    text: str,
    provider: str,
    return_details: bool,
    *,
    partial: bool = False,
    pages_done: int = 0,
    pages_total: int = 0,
    error: Optional[str] = None,
):
    """
    Format the OCR result based on the return_details flag.

    Args:
        text: Extracted text content
        provider: Name of the OCR provider used ('mistral', 'openai', 'legacy')
        return_details: If True, return OCRResult; otherwise return just text
        partial: Lot 2 — True when the text is known to be incomplete.
        pages_done: Number of pages actually transcribed.
        pages_total: Total page count of the source document.
        error: Human-readable explanation when partial/suspect.

    Returns:
        OCRResult namedtuple if return_details is True, else string
    """
    if return_details:
        return OCRResult(
            text=text,
            provider=provider,
            partial=partial,
            pages_done=pages_done,
            pages_total=pages_total,
            error=error,
        )
    return text


def _ocr_density_warning(
    text: str,
    pages_total: int,
    pdf_path: str,
    provider: str,
) -> Optional[str]:
    """
    Generic sanity guard (Lot 2): flag suspiciously sparse OCR output.

    When the average character density falls below OCR_MIN_CHARS_PER_PAGE,
    the extraction is almost certainly incomplete (e.g. a scanned book run
    through the PyMuPDF legacy text extractor returning a few headers). Logs a
    WARNING and returns an explanatory message; returns None when the density
    is healthy or the page count is unknown.

    Args:
        text: Extracted text.
        pages_total: Source document page count (0 = unknown → no check).
        pdf_path: Path of the document (for the log line).
        provider: Provider name (for the message).

    Returns:
        Explanatory string when suspect, else None.
    """
    if pages_total <= 0 or not text:
        return None
    density = len(text) / pages_total
    if density < OCR_MIN_CHARS_PER_PAGE:
        msg = (
            f"OCR suspect ({provider}) : {density:.0f} car./page sur "
            f"{pages_total} pages (< {OCR_MIN_CHARS_PER_PAGE}) — extraction "
            f"probablement incomplète."
        )
        logger.warning("%s (%s)", msg, pdf_path)
        return msg
    return None


def extract_text_with_ocr(
    pdf_path: str,
    max_pages: Optional[int] = None,
    *,
    return_details: bool = False,
):
    """
    Extract text from a PDF using the best available OCR provider.

    Provider order (default):
    1. Mistral OCR API (if MISTRAL_API_KEY is set) — with transient-error retry.
    2. PyMuPDF legacy extraction (always available) — local last resort.

    The OpenAI Vision fallback is DISABLED by default (OCR_ENABLE_OPENAI_FALLBACK)
    because it is hard-capped at OPENAI_OCR_MAX_PAGES and silently truncates
    books. When explicitly re-enabled, it is tried between Mistral and legacy
    with the Lot 2 partial/truncation guards.

    Args:
        pdf_path: Path to the PDF file to process
        max_pages: Optional maximum number of pages to process
        return_details: If True, return OCRResult with provider info

    Returns:
        Extracted text (str) or OCRResult if return_details is True

    Raises:
        OCRExtractionError: If all extraction methods fail
    """
    # Track extraction time for metrics
    start_time = time.time()
    provider_used = "unknown"
    extraction_success = False

    last_error: Optional[Exception] = None

    try:
        if MISTRAL_API_KEY:
            try:
                logger.debug("Tentative d'OCR Mistral pour %s", pdf_path)
                mistral_outcome = _extract_text_with_mistral(pdf_path, max_pages=max_pages)
                provider_used = "mistral"
                extraction_success = True
                # A split book with some failed parts is salvaged but flagged
                # partial (never a silent success).
                return _finalize_ocr_result(
                    mistral_outcome.text, "mistral", return_details,
                    partial=mistral_outcome.partial,
                    pages_done=mistral_outcome.pages_done,
                    pages_total=mistral_outcome.pages_total,
                    error=mistral_outcome.error,
                )
            except Exception as mistral_error:
                last_error = mistral_error
                logger.warning(
                    "Échec Mistral OCR pour %s: %s",
                    pdf_path,
                    mistral_error,
                )
                # Track error
                if METRICS_AVAILABLE and track_error:
                    track_error('pdf_extraction', type(mistral_error).__name__)
        else:
            logger.debug("MISTRAL_API_KEY absente, OCR Mistral ignoré pour %s", pdf_path)

        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and not OCR_ENABLE_OPENAI_FALLBACK:
            # Default policy — OpenAI Vision OCR is disabled (truncates books).
            logger.info(
                "OCR OpenAI désactivé par défaut (OCR_ENABLE_OPENAI_FALLBACK=0) — "
                "chaîne Mistral → legacy uniquement pour %s.", pdf_path,
            )
        if openai_key and OCR_ENABLE_OPENAI_FALLBACK:
            try:
                logger.debug("Fallback OpenAI OCR (opt-in) pour %s", pdf_path)
                openai_outcome = _extract_text_with_openai(pdf_path, openai_key, max_pages=max_pages)
                openai_text = openai_outcome.text
                pages_done = openai_outcome.pages_done
                pages_total = openai_outcome.pages_total
                is_capped = pages_total > 0 and pages_done < pages_total

                if is_capped:
                    # Lot 2 — the OpenAI cap truncated the document. Never return
                    # this as a silent success: try the non-capped legacy engine
                    # first, and if it is no better, return the OpenAI text FLAGGED
                    # partial so it surfaces in the UI / errors.json.
                    logger.warning(
                        "OCR PARTIEL OpenAI pour %s : %d/%d pages transcrites "
                        "(cap OPENAI_OCR_MAX_PAGES=%s). Tentative d'un moteur non "
                        "plafonné avant d'accepter le résultat partiel.",
                        pdf_path, pages_done, pages_total, OPENAI_OCR_MAX_PAGES,
                    )
                    legacy_text = _extract_text_with_legacy_pdf(pdf_path, max_pages=max_pages)
                    if legacy_text.strip() and len(legacy_text) > len(openai_text):
                        logger.info(
                            "Moteur legacy plus complet que l'OpenAI plafonné pour "
                            "%s (%d > %d caractères) — adoption de legacy.",
                            pdf_path, len(legacy_text), len(openai_text),
                        )
                        provider_used = "legacy"
                        extraction_success = True
                        return _finalize_ocr_result(
                            legacy_text, "legacy", return_details,
                            pages_done=pages_total, pages_total=pages_total,
                        )
                    provider_used = "openai"
                    extraction_success = True
                    return _finalize_ocr_result(
                        openai_text, "openai", return_details,
                        partial=True, pages_done=pages_done, pages_total=pages_total,
                        error=f"OCR plafonné OpenAI : {pages_done}/{pages_total} pages",
                    )

                # Full OpenAI result — generic density sanity check (Lot 2).
                density_error = _ocr_density_warning(openai_text, pages_total, pdf_path, "openai")
                provider_used = "openai"
                extraction_success = True
                return _finalize_ocr_result(
                    openai_text, "openai", return_details,
                    partial=bool(density_error),
                    pages_done=pages_done, pages_total=pages_total,
                    error=density_error,
                )
            except Exception as openai_error:
                last_error = openai_error
                logger.warning(
                    "Échec OpenAI OCR pour %s: %s",
                    pdf_path,
                    openai_error,
                )
                # Track error
                if METRICS_AVAILABLE and track_error:
                    track_error('pdf_extraction', type(openai_error).__name__)
        elif not openai_key:
            logger.debug("OPENAI_API_KEY absente, OCR OpenAI ignoré pour %s", pdf_path)

        # Lot 4 — OCR LOCAL (Docling), sans clé/cap, AVANT le dernier recours
        # `legacy`. C'est la voie qui sauve les PDF scannés que legacy ne sait
        # pas lire. Sauté proprement si le moteur n'est pas installé.
        if OCR_ENABLE_LOCAL_FALLBACK and _local_ocr_available():
            try:
                logger.info("Tentative OCR local (%s) pour %s", LOCAL_OCR_ENGINE, pdf_path)
                local_text = _extract_text_with_local(pdf_path, max_pages=max_pages)
                local_pages = _pdf_page_count(pdf_path)
                density_error = _ocr_density_warning(local_text, local_pages, pdf_path, LOCAL_OCR_ENGINE)
                provider_used = LOCAL_OCR_ENGINE
                extraction_success = True
                return _finalize_ocr_result(
                    local_text, LOCAL_OCR_ENGINE, return_details,
                    partial=bool(density_error),
                    pages_done=local_pages, pages_total=local_pages,
                    error=density_error,
                )
            except Exception as local_error:
                last_error = local_error
                logger.warning(
                    "Échec OCR local (%s) pour %s: %s",
                    LOCAL_OCR_ENGINE, pdf_path, local_error,
                )
                if METRICS_AVAILABLE and track_error:
                    track_error('pdf_extraction', type(local_error).__name__)

        if not MISTRAL_API_KEY and not openai_key:
            logger.error(
                "Aucune clé API OCR configurée (MISTRAL_API_KEY ou OPENAI_API_KEY). "
                "Basculer sur l'extraction locale peut dégrader la qualité du texte."
            )

        if last_error:
            logger.info("Retour au processus OCR historique pour %s", pdf_path)

        legacy_text = _extract_text_with_legacy_pdf(pdf_path, max_pages=max_pages)
        if legacy_text.strip():
            # Lot 2 — legacy is the last resort; flag suspiciously sparse output
            # (typical of a scanned book with no real text layer) as partial.
            legacy_pages = _pdf_page_count(pdf_path)
            density_error = _ocr_density_warning(legacy_text, legacy_pages, pdf_path, "legacy")
            provider_used = "legacy"
            extraction_success = True
            return _finalize_ocr_result(
                legacy_text, "legacy", return_details,
                partial=bool(density_error),
                pages_done=legacy_pages, pages_total=legacy_pages,
                error=density_error,
            )

    finally:
        # Record metrics
        duration = time.time() - start_time
        if METRICS_AVAILABLE and track_pdf_extraction:
            # Use context manager pattern for metrics
            try:
                from scripts.metrics_helper import _collector
                _collector.increment_counter(
                    'pdf_processed',
                    provider=provider_used,
                    status='success' if extraction_success else 'failure'
                )
                _collector.observe_histogram('pdf_duration', duration, provider=provider_used)
            except Exception:
                pass  # Don't fail on metrics errors

    error_message = (
        "Impossible d'extraire le texte du document. Configurez MISTRAL_API_KEY "
        "pour activer l'OCR Markdown (le fallback OpenAI est désactivé par défaut ; "
        "activez OCR_ENABLE_OPENAI_FALLBACK=1 pour l'autoriser)."
    )
    raise OCRExtractionError(error_message)


# ============================================================================
# Non-PDF extractors (EPUB, plain text)
# ============================================================================

def _flatten_epub_toc(toc_entry: Any, depth: int = 0) -> List[Tuple[int, str, str]]:
    """
    Flatten the nested EPUB native ToC into a list of (depth, label, href) tuples.

    `book.toc` from ebooklib is a tree of `epub.Link` objects and tuples
    ``(Section, [children])``. We walk it depth-first and yield every Link.
    Used by Lot F to inject the EPUB's native ToC at the head of the
    extracted text so Phase 1 can rely on a high-quality structural index
    for un-paginated EPUBs.
    """
    from ebooklib import epub  # local import: ebooklib is optional dep

    out: List[Tuple[int, str, str]] = []
    if toc_entry is None:
        return out
    if isinstance(toc_entry, list):
        for child in toc_entry:
            out.extend(_flatten_epub_toc(child, depth))
        return out
    if isinstance(toc_entry, tuple):
        # (Section, [children]) form
        sec, children = toc_entry[0], toc_entry[1] if len(toc_entry) > 1 else []
        sec_label = getattr(sec, "title", None) or str(sec)
        out.append((depth, sec_label, ""))
        out.extend(_flatten_epub_toc(children, depth + 1))
        return out
    if isinstance(toc_entry, epub.Link):
        out.append((depth, toc_entry.title or "", toc_entry.href or ""))
        return out
    if isinstance(toc_entry, epub.Section):
        out.append((depth, toc_entry.title or "", ""))
        return out
    return out


def _extract_text_from_epub(epub_path: str) -> OCRResult:
    """Extract text from an EPUB, promoting pagination anchors to ``<!-- Page N -->`` markers.

    Lot F enhancements:
    1. Native EPUB ToC (``book.toc``) is serialised as a leading
       ``<!-- EPUB_TOC_BEGIN -->...<!-- EPUB_TOC_END -->`` block, which gives
       Phase 1 a high-quality structural index even when no `<a id="page_X"/>`
       pagebreak anchors are present.
    2. Each ``EpubHtml`` item gets a synthetic ``<!-- Page N -->`` marker
       prepended (where N is the spine index). This guarantees that
       SourceFormat detection sees pagination markers, that
       ``_split_text_by_pages`` can produce a per-item map, and that
       the chapter slicer in Lot B can rebuild chapter bodies after the
       positional fallback cap fires.
    3. Real EPUB 3 ``epub:type="pagebreak"`` anchors and legacy
       ``<a id="page_N"/>`` ancors are still promoted to ``<!-- Page N -->``
       markers when present (preferred over the synthetic ones).
    """
    from ebooklib import epub, ITEM_DOCUMENT
    from bs4 import BeautifulSoup

    book = epub.read_epub(epub_path)

    # 1. Serialise native ToC as a leading block. We list label + href + depth
    #    so Phase 1 can match TOC entries to chapter files even without page
    #    numbers. Limit to first 200 entries to bound prompt size.
    toc_entries = _flatten_epub_toc(book.toc)
    toc_lines: List[str] = []
    for depth, label, href in toc_entries[:200]:
        if not label:
            continue
        indent = "  " * min(depth, 4)
        toc_lines.append(f"{indent}- {label}" + (f" → {href}" if href else ""))
    toc_block = ""
    if toc_lines:
        toc_block = (
            "<!-- EPUB_TOC_BEGIN -->\n"
            "Table of Contents (extracted from EPUB native NCX/NAV):\n"
            + "\n".join(toc_lines)
            + "\n<!-- EPUB_TOC_END -->\n\n"
        )

    # 2. Walk the spine items, prepending a synthetic Page marker per item.
    #    Real pagebreak/page anchors override the synthetic marker by being
    #    inserted inline at their natural position in the text.
    parts: List[str] = []
    has_real_pagebreaks = False
    spine_index = 0
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        spine_index += 1
        soup = BeautifulSoup(item.get_content(), "lxml")

        item_has_pagebreak = False
        for anchor in soup.find_all(attrs={"epub:type": "pagebreak"}):
            label = anchor.get("title") or anchor.get("id") or "?"
            anchor.replace_with(f"\n<!-- Page {label} -->\n")
            item_has_pagebreak = True

        for anchor in soup.find_all("a", id=lambda v: bool(v) and v.startswith("page")):
            label = anchor["id"].split("_", 1)[-1] if "_" in anchor["id"] else anchor["id"][4:] or "?"
            anchor.replace_with(f"\n<!-- Page {label} -->\n")
            item_has_pagebreak = True

        if item_has_pagebreak:
            has_real_pagebreaks = True

        item_text = soup.get_text("\n", strip=True)
        if not item_text:
            continue

        # Title heuristic for the synthetic boundary: first <h1>/<h2>/<title>
        # found in the item, used as a debug hint only — Phase 1 reads the
        # text directly to extract chapter titles.
        title_hint = ""
        for tag_name in ("title", "h1", "h2", "h3"):
            tag = soup.find(tag_name)
            if tag and tag.get_text(strip=True):
                title_hint = tag.get_text(strip=True)[:120]
                break

        # Prepend a synthetic marker only when the item itself did not emit
        # real pagebreak ancors. This avoids pseudo-pages collisions with
        # real ones in EPUB 3 paginated books.
        if not item_has_pagebreak:
            marker = f"<!-- Page {spine_index} -->\n"
            if title_hint:
                marker += f"<!-- EpubItemTitle: {title_hint} -->\n"
            parts.append(marker + item_text)
        else:
            parts.append(item_text)

    text = toc_block + "\n\n".join(p for p in parts if p)
    return OCRResult(text=text, provider="epub")


def _extract_text_from_plain(path: str) -> OCRResult:
    """Read a ``.txt`` or ``.md`` file as UTF-8 (errors='replace')."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return OCRResult(text=f.read(), provider="plain_text")


# ============================================================================
# PARALLEL PROCESSING SUPPORT (Phase 2 optimization)
# ============================================================================

class ItemProcessingResult(NamedTuple):
    """
    Result of processing a single Zotero item.

    Attributes:
        item_key: Unique identifier for the Zotero item
        records: List of successfully extracted records (PDF data)
        errors: List of errors encountered during processing
        success: Whether processing completed without critical errors
    """
    item_key: str
    records: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    success: bool


def _process_single_zotero_item(
    item: Dict[str, Any],
    pdf_base_dir: str,
    item_index: int = 0
) -> ItemProcessingResult:
    """
    Process a single Zotero item with its PDF attachments.

    This function is designed to be thread-safe and can be called from
    multiple threads in a ThreadPoolExecutor. It extracts metadata from
    the Zotero item and performs OCR on associated PDF files.

    Args:
        item: Zotero item dictionary containing metadata and attachments
        pdf_base_dir: Base directory for resolving relative PDF paths
        item_index: Index of the item in the processing queue (for logging)

    Returns:
        ItemProcessingResult containing extracted records and any errors

    Note:
        This function does not write to files directly. The caller is
        responsible for persisting results using thread-safe methods.
    """
    item_key = item.get("key") or item.get("itemKey", "")
    records = []
    errors = []

    try:
        # Extract base metadata
        metadata = {
            "itemKey": item_key,
            "type": item.get("itemType", ""),
            "title": item.get("title", ""),
            "abstract": item.get("abstractNote", ""),
            "date": item.get("date", ""),
            "url": item.get("url", ""),
            "doi": item.get("DOI", ""),
            "authors": ", ".join([
                f"{c.get('lastName', '').strip()} {c.get('firstName', '').strip()}"
                for c in item.get("creators", [])
                if c.get('lastName') or c.get('firstName')
            ])
        }

        # Process attachments (PDF, EPUB, plain text)
        for attachment in item.get("attachments", []):
            path_from_json = attachment.get("path", "").strip()
            if not path_from_json:
                continue

            ext = os.path.splitext(path_from_json)[1].lower()
            if ext not in (".pdf", ".epub", ".txt", ".md"):
                logger.warning(f"[{item_index}] Unsupported extension {ext or '(none)'} for {path_from_json}")
                errors.append({
                    "itemKey": item_key,
                    "title": metadata.get("title", ""),
                    "error_type": "UNSUPPORTED_EXTENSION",
                    "error_message": f"Extension non supportée: {ext or '(none)'}",
                    "path": path_from_json,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                continue

            # Resolve attachment path
            if os.path.isabs(path_from_json):
                actual_path = path_from_json
            else:
                actual_path = os.path.join(pdf_base_dir, path_from_json)

            # Fuzzy search (PDF-only — non-PDF attachments rarely have renamed files)
            if not os.path.exists(actual_path):
                if ext == ".pdf":
                    actual_path = _find_pdf_fuzzy(actual_path, path_from_json, pdf_base_dir)
                else:
                    actual_path = None

                if actual_path is None:
                    logger.warning(f"[{item_index}] File not found: {path_from_json}")
                    errors.append({
                        "itemKey": item_key,
                        "title": metadata.get("title", ""),
                        "error_type": "PDF_NOT_FOUND" if ext == ".pdf" else "FILE_NOT_FOUND",
                        "error_message": f"File not found: {path_from_json}",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    continue

            logger.info(f"[{item_index}] Processing {ext} file: {os.path.basename(actual_path)}")

            try:
                if ext == ".pdf":
                    ocr_payload = extract_text_with_ocr_retry(actual_path, return_details=True)
                elif ext == ".epub":
                    ocr_payload = _extract_text_from_epub(actual_path)
                else:  # .txt, .md
                    ocr_payload = _extract_text_from_plain(actual_path)

                # Build record
                ocr_partial = bool(getattr(ocr_payload, "partial", False))
                ocr_pages_done = int(getattr(ocr_payload, "pages_done", 0) or 0)
                ocr_pages_total = int(getattr(ocr_payload, "pages_total", 0) or 0)
                record = {
                    **metadata,
                    "filename": os.path.basename(path_from_json),
                    "path": actual_path,
                    "attachment_title": attachment.get("title", ""),
                    "texteocr": ocr_payload.text,
                    "texteocr_provider": ocr_payload.provider,
                    "texteocr_partial": ocr_partial,
                    "texteocr_pages_done": ocr_pages_done,
                    "texteocr_pages_total": ocr_pages_total,
                }
                records.append(record)

                if ocr_partial:
                    # Lot 2 — partial OCR is still saved (better than nothing) but
                    # surfaced as a WARNING and recorded in errors.json so it is
                    # never a silent success.
                    partial_msg = getattr(ocr_payload, "error", None) or (
                        f"OCR partiel : {ocr_pages_done}/{ocr_pages_total} pages"
                    )
                    logger.warning(
                        f"[{item_index}] ⚠ OCR PARTIEL pour {item_key} "
                        f"({ocr_payload.provider}) : {partial_msg}"
                    )
                    errors.append({
                        "itemKey": item_key,
                        "title": metadata.get("title", ""),
                        "error_type": "OCR_PARTIAL",
                        "error_message": partial_msg,
                        "provider": ocr_payload.provider,
                        "pages_done": ocr_pages_done,
                        "pages_total": ocr_pages_total,
                        "pdf_path": actual_path,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                else:
                    logger.info(f"[{item_index}] ✓ Extraction success for {item_key} ({ocr_payload.provider})")

            except OCRExtractionError as ocr_error:
                logger.error(f"[{item_index}] OCR failed for {actual_path}: {ocr_error}")
                errors.append({
                    "itemKey": item_key,
                    "title": metadata.get("title", ""),
                    "error_type": "OCR_FAILED",
                    "error_message": str(ocr_error),
                    "pdf_path": actual_path,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
            except Exception as extract_error:
                logger.error(f"[{item_index}] Extraction failed for {actual_path}: {extract_error}")
                errors.append({
                    "itemKey": item_key,
                    "title": metadata.get("title", ""),
                    "error_type": "EXTRACTION_FAILED",
                    "error_message": str(extract_error),
                    "path": actual_path,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })

        return ItemProcessingResult(
            item_key=item_key,
            records=records,
            errors=errors,
            success=True
        )

    except Exception as e:
        logger.error(f"[{item_index}] Error processing item {item_key}: {e}")
        errors.append({
            "itemKey": item_key,
            "title": item.get("title", ""),
            "error_type": "PROCESSING_ERROR",
            "error_message": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        return ItemProcessingResult(
            item_key=item_key,
            records=[],
            errors=errors,
            success=False
        )


def load_zotero_to_dataframe_incremental(json_path: str, pdf_base_dir: str, output_csv: str) -> pd.DataFrame:
    """
    Charge les métadonnées Zotero depuis un JSON vers un DataFrame
    avec extraction OCR du texte complet pour chaque PDF.

    VERSION INCRÉMENTALE avec support parallélisation (Phase 2):
    - Sauvegarde chaque item immédiatement après traitement
    - Supporte la reprise après interruption (checkpoint)
    - Retry avec backoff exponentiel pour les erreurs réseau
    - Fichier d'erreurs séparé pour diagnostic
    - Traitement parallèle configurable via PDF_EXTRACTION_WORKERS

    Modes de traitement:
    - PDF_EXTRACTION_WORKERS=1: Mode séquentiel (comportement original)
    - PDF_EXTRACTION_WORKERS>1: Mode parallèle avec ThreadPoolExecutor

    Args:
        json_path: Chemin vers le fichier JSON Zotero
        pdf_base_dir: Répertoire de base pour résoudre les chemins PDF relatifs
        output_csv: Chemin vers le fichier CSV de sortie

    Returns:
        DataFrame pandas avec les enregistrements traités
    """
    # CSV column order (consistent with original)
    CSV_FIELDNAMES = [
        "itemKey", "type", "title", "abstract", "date", "url", "doi",
        "authors", "filename", "path", "attachment_title", "texteocr", "texteocr_provider",
        "texteocr_partial", "texteocr_pages_done", "texteocr_pages_total"
    ]

    # Load progress (items already processed)
    progress_file = get_progress_file_path(output_csv)
    processed_keys = load_progress(output_csv)

    # CRITICAL FIX: If no progress file but CSV exists, we have stale data
    # This happens when user runs new extraction without cleaning up old files
    if not os.path.exists(progress_file) and os.path.exists(output_csv):
        logger.warning(f"No progress file found but CSV exists. Clearing stale data to start fresh.")
        try:
            os.remove(output_csv)
            logger.info(f"Removed stale CSV: {output_csv}")
        except Exception as e:
            logger.error(f"Failed to remove stale CSV: {e}")

    # Use thread-safe set for parallel mode
    processed_keys_lock = threading.Lock()
    all_errors = []
    errors_lock = threading.Lock()
    records_count = [0]  # Use list for mutable reference in nested function
    records_count_lock = threading.Lock()

    try:
        logger.info(f"Chargement du fichier JSON Zotero depuis : {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Support both Zotero export formats
        if isinstance(data, list):
            items = data
            logger.info(f"Detected Zotero JSON format: direct array with {len(items)} items")
        elif isinstance(data, dict) and "items" in data:
            items = data["items"]
            logger.info(f"Detected Zotero JSON format: object with 'items' key, {len(items)} items")
        else:
            logger.error(f"Invalid Zotero JSON format: expected array or object with 'items' key")
            return pd.DataFrame()

        # Calculate how many are already done
        total_items = len(items)
        already_done = len(processed_keys)
        if already_done > 0:
            logger.info(f"Resuming: {already_done}/{total_items} items already processed")

        # Filter items that need processing
        items_to_process = []
        for idx, item in enumerate(items):
            item_key = item.get("key") or item.get("itemKey", "")
            if not item_key or item_key not in processed_keys:
                items_to_process.append((idx, item))

        items_remaining = len(items_to_process)
        logger.info(f"Items to process: {items_remaining} (skipping {total_items - items_remaining} already done)")

        # Emit init event for SSE progress tracking
        print(f"PROGRESS|init|{total_items}|Found {total_items} Zotero items to process", flush=True)

        # Choose processing mode based on worker count
        if PDF_EXTRACTION_WORKERS > 1 and items_remaining > 1:
            # ============================================================
            # PARALLEL MODE (Phase 2 optimization)
            # ============================================================
            logger.info(f"Using PARALLEL mode with {PDF_EXTRACTION_WORKERS} workers")
            logger.info(f"Mistral API concurrent calls limited to {MISTRAL_CONCURRENT_CALLS}")

            completed_count = [0]

            def process_and_save(item_data: Tuple[int, Dict[str, Any]]) -> None:
                """Process item and save results (thread-safe)."""
                idx, item = item_data
                item_key = item.get("key") or item.get("itemKey", "")

                # Double-check not already processed (race condition protection)
                with processed_keys_lock:
                    if item_key and item_key in processed_keys:
                        return

                # Process item
                result = _process_single_zotero_item(item, pdf_base_dir, idx)

                # Save records (thread-safe)
                for record in result.records:
                    append_record_to_csv(output_csv, record, CSV_FIELDNAMES)
                    with records_count_lock:
                        records_count[0] += 1

                # Collect errors (thread-safe)
                if result.errors:
                    with errors_lock:
                        all_errors.extend(result.errors)

                # Mark as processed (thread-safe)
                if result.item_key:
                    with processed_keys_lock:
                        processed_keys.add(result.item_key)
                        save_progress(output_csv, processed_keys)

                # Update progress
                with records_count_lock:
                    completed_count[0] += 1
                    current = already_done + completed_count[0]
                    title_short = item.get("title", f"Item {item_key}")[:50]
                    print(f"PROGRESS|row|{current}/{total_items}|{title_short}", flush=True)

            # Execute with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=PDF_EXTRACTION_WORKERS) as executor:
                # Submit all items
                futures = {
                    executor.submit(process_and_save, item_data): item_data[0]
                    for item_data in items_to_process
                }

                # Wait for completion with progress bar
                for future in tqdm(
                    as_completed(futures),
                    total=len(items_to_process),
                    desc=f"Processing PDFs ({PDF_EXTRACTION_WORKERS} workers)"
                ):
                    try:
                        future.result()  # Raise any exceptions
                    except Exception as e:
                        idx = futures[future]
                        logger.error(f"[{idx}] Unexpected error in thread: {e}")

            logger.info(f"Parallel processing complete: {records_count[0]} records saved")

        else:
            # ============================================================
            # SEQUENTIAL MODE (original behavior, PDF_EXTRACTION_WORKERS=1)
            # ============================================================
            logger.info(f"Using SEQUENTIAL mode (workers={PDF_EXTRACTION_WORKERS})")

            item_count = already_done
            for idx, item in tqdm(items_to_process, desc="Processing Zotero items"):
                item_count += 1
                item_key = item.get("key") or item.get("itemKey", "")

                # Process using the shared function
                result = _process_single_zotero_item(item, pdf_base_dir, idx)

                # Save records
                for record in result.records:
                    append_record_to_csv(output_csv, record, CSV_FIELDNAMES)
                    records_count[0] += 1
                    logger.info(f"✓ Item {item_key} saved ({records_count[0]} total)")

                # Collect errors
                all_errors.extend(result.errors)

                # Mark as processed
                if result.item_key:
                    processed_keys.add(result.item_key)
                    save_progress(output_csv, processed_keys)

                # Emit progress for SSE
                title_short = item.get("title", f"Item {item_key}")[:50]
                print(f"PROGRESS|row|{item_count}/{total_items}|{title_short}", flush=True)

    except Exception as e:
        logger.error(f"Failed to load Zotero JSON: {e}")

    # Save errors to file
    save_errors(output_csv, all_errors)

    # Return DataFrame from CSV (for compatibility)
    if os.path.exists(output_csv):
        try:
            return pd.read_csv(output_csv, encoding='utf-8-sig', escapechar='\\')
        except Exception as e:
            logger.error(f"Failed to read output CSV: {e}")

    return pd.DataFrame()


def _find_pdf_fuzzy(actual_pdf_path: str, path_from_json: str, pdf_base_dir: str) -> Optional[str]:
    """
    Recherche fuzzy pour trouver un PDF avec différentes normalisations.

    Returns:
        Le chemin du PDF trouvé, ou None si non trouvé
    """
    base_dir = os.path.dirname(actual_pdf_path)
    candidates = []

    if os.path.exists(base_dir):
        for root, dirs, files in os.walk(base_dir):
            for f in files:
                candidates.append(f)

    if not candidates:
        return None

    target_names = [
        unicodedata.normalize('NFC', os.path.basename(path_from_json)).lower(),
        unicodedata.normalize('NFD', os.path.basename(path_from_json)).lower(),
        strip_accents(unicodedata.normalize('NFC', os.path.basename(path_from_json))).lower(),
        strip_accents(unicodedata.normalize('NFD', os.path.basename(path_from_json))).lower(),
        ascii_flat(os.path.basename(path_from_json)),
        alphanum_only(os.path.basename(path_from_json))
    ]

    for f in candidates:
        f_forms = [
            unicodedata.normalize('NFC', f).lower(),
            unicodedata.normalize('NFD', f).lower(),
            strip_accents(unicodedata.normalize('NFC', f)).lower(),
            strip_accents(unicodedata.normalize('NFD', f)).lower(),
            ascii_flat(f),
            alphanum_only(f)
        ]

        # Fuzzy match (Levenshtein)
        t_alpha = alphanum_only(os.path.basename(path_from_json))
        f_alpha = alphanum_only(f)
        lev = levenshtein(t_alpha, f_alpha)
        fuzzy_match = lev <= 2 and min(len(t_alpha), len(f_alpha)) > 0

        if any(t == ff for t in target_names for ff in f_forms) or fuzzy_match:
            found_path = os.path.join(base_dir, f)
            logger.info(f"Correspondance fuzzy trouvée: {path_from_json} -> {found_path}")
            return found_path

    return None


# Keep the old function for backward compatibility (non-incremental)
def load_zotero_to_dataframe(json_path: str, pdf_base_dir: str) -> pd.DataFrame:
    """
    DEPRECATED: Use load_zotero_to_dataframe_incremental instead.

    Cette version batch est conservée pour compatibilité mais n'est plus recommandée
    car elle ne supporte pas la reprise après interruption.
    """
    logger.warning("Using deprecated batch mode. Consider using incremental mode for better resilience.")
    records = []
    try:
        logger.info(f"Chargement du fichier JSON Zotero depuis : {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Support both Zotero export formats
        if isinstance(data, list):
            items = data
            logger.info(f"Detected Zotero JSON format: direct array with {len(items)} items")
        elif isinstance(data, dict) and "items" in data:
            items = data["items"]
            logger.info(f"Detected Zotero JSON format: object with 'items' key, {len(items)} items")
        else:
            logger.error(f"Invalid Zotero JSON format: expected array or object with 'items' key")
            return pd.DataFrame()

        for item in tqdm(items, desc="Processing Zotero items"):
            try:
                # Extraction des métadonnées de base
                metadata = {
                    "itemKey": item.get("key") or item.get("itemKey", ""),
                    "type": item.get("itemType", ""),
                    "title": item.get("title", ""),
                    "abstract": item.get("abstractNote", ""),
                    "date": item.get("date", ""),
                    "url": item.get("url", ""),
                    "doi": item.get("DOI", ""),
                    "authors": ", ".join([
                        f"{c.get('lastName', '').strip()} {c.get('firstName', '').strip()}"
                        for c in item.get("creators", [])
                        if c.get('lastName') or c.get('firstName')
                    ])
                }

                # Traitement des attachments (PDF, EPUB, texte brut)
                for attachment in item.get("attachments", []):
                    path_from_json = attachment.get("path", "").strip()
                    if not path_from_json:
                        continue

                    ext = os.path.splitext(path_from_json)[1].lower()
                    if ext not in (".pdf", ".epub", ".txt", ".md"):
                        logger.warning(f"Extension non supportée: {ext or '(none)'} pour {path_from_json}")
                        continue

                    # Résoudre le chemin de la pièce jointe
                    if os.path.isabs(path_from_json):
                        actual_path = path_from_json
                    else:
                        actual_path = os.path.join(pdf_base_dir, path_from_json)

                    if not os.path.exists(actual_path):
                        if ext == ".pdf":
                            actual_path = _find_pdf_fuzzy(actual_path, path_from_json, pdf_base_dir)
                        else:
                            actual_path = None
                        if actual_path is None:
                            logger.warning(f"Fichier non trouvé: {path_from_json}")
                            continue

                    logger.info(f"Traitement du fichier {ext} : {actual_path}")
                    try:
                        if ext == ".pdf":
                            ocr_payload = extract_text_with_ocr(actual_path, return_details=True)
                        elif ext == ".epub":
                            ocr_payload = _extract_text_from_epub(actual_path)
                        else:  # .txt, .md
                            ocr_payload = _extract_text_from_plain(actual_path)
                    except OCRExtractionError as ocr_error:
                        logger.error(
                            "Échec OCR pour %s (%s): %s",
                            actual_path,
                            path_from_json,
                            ocr_error,
                        )
                        continue
                    except Exception as extract_error:
                        logger.error(
                            "Échec extraction pour %s (%s): %s",
                            actual_path,
                            path_from_json,
                            extract_error,
                        )
                        continue

                    records.append({
                        **metadata,
                        "filename": os.path.basename(path_from_json),
                        "path": actual_path,
                        "attachment_title": attachment.get("title", ""),
                        "texteocr": ocr_payload.text,
                        "texteocr_provider": ocr_payload.provider,
                        "texteocr_partial": bool(getattr(ocr_payload, "partial", False)),
                        "texteocr_pages_done": int(getattr(ocr_payload, "pages_done", 0) or 0),
                        "texteocr_pages_total": int(getattr(ocr_payload, "pages_total", 0) or 0),
                    })
            except Exception as item_error:
                logger.error(f"Error processing item: {item_error}")
                continue

    except Exception as e:
        logger.error(f"Failed to load Zotero JSON: {e}")

    return pd.DataFrame(records)

def extract_pdf_metadata_to_dataframe(pdf_directory: str) -> pd.DataFrame:
    """
    Extrait métadonnées + texte OCR des PDF d'un répertoire.
    
    Args:
        pdf_directory: Chemin du répertoire contenant les PDF
        
    Returns:
        DataFrame pandas avec métadonnées et texte extrait
    """
    if not os.path.exists(pdf_directory):
        logger.error(f"Directory not found: {pdf_directory}")
        return pd.DataFrame()
    
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_directory}")
        return pd.DataFrame()
        
    records = []
    for filename in tqdm(pdf_files, desc="Processing PDF files"):
        full_path = os.path.join(pdf_directory, filename)
        try:
            with fitz.open(full_path) as doc:
                try:
                    ocr_payload = extract_text_with_ocr(
                        full_path,
                        return_details=True,
                    )
                except OCRExtractionError as ocr_error:
                    logger.error(
                        "Échec OCR pour %s: %s",
                        full_path,
                        ocr_error,
                    )
                    continue

                records.append({
                    "type": "article",
                    "authors": doc.metadata.get('author', ''),
                    "title": doc.metadata.get('title', ''),
                    "date": format_pdf_date(doc.metadata.get('creationDate', '')),
                    "url": "",
                    "doi": extract_doi_from_pdf(doc),
                    "filename": filename,
                    "path": full_path,
                    "attachment_title": os.path.splitext(filename)[0],
                    "texteocr": ocr_payload.text,
                    "texteocr_provider": ocr_payload.provider,
                    "texteocr_partial": bool(getattr(ocr_payload, "partial", False)),
                    "texteocr_pages_done": int(getattr(ocr_payload, "pages_done", 0) or 0),
                    "texteocr_pages_total": int(getattr(ocr_payload, "pages_total", 0) or 0),
                })
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            continue
            
    return pd.DataFrame(records)

def format_pdf_date(date_string: str) -> str:
    """Formate une date PDF en texte lisible"""
    if not date_string:
        return ""
    try:
        if isinstance(date_string, str) and date_string.startswith('D:'):
            date_str = date_string[2:14]  # Extraire YYYYMMDDHHmm
            if len(date_str) >= 8:
                return f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return str(date_string)
    except:
        return str(date_string)

def extract_doi_from_pdf(doc: fitz.Document) -> str:
    """Tente d'extraire un DOI du document PDF"""
    doi_pattern = r'(10\.\d{4,}(?:\.\d+)*\/\S+[^;,.\s])'
    if doc.metadata.get('doi'):
        return doc.metadata.get('doi')
    for page_num in range(min(3, doc.page_count)):
        text = doc[page_num].get_text()
        if match := re.search(doi_pattern, text):
            return match.group(0)
    return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Zotero JSON and associated PDFs to create a CSV with extracted text.")
    parser.add_argument("--json", required=True, help="Path to the Zotero JSON file.")
    parser.add_argument("--dir", required=True, help="Base directory for resolving relative PDF paths from the JSON.")
    parser.add_argument("--output", required=True, help="Path to save the output CSV file.")
    parser.add_argument("--batch", action="store_true", help="Use deprecated batch mode (not recommended)")

    args = parser.parse_args()

    logger.info(f"Starting Zotero data processing for JSON: {args.json} with PDF base directory: {args.dir}")

    # S'assurer que le répertoire de sortie existe
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            print(f"Error creating output directory {output_dir}: {e}")
            exit(1)

    if args.batch:
        # Use deprecated batch mode
        logger.warning("Using deprecated batch mode. Use incremental mode for better resilience.")
        df_zotero = load_zotero_to_dataframe(args.json, args.dir)

        if not df_zotero.empty:
            try:
                df_zotero.to_csv(args.output, index=False, encoding='utf-8-sig', escapechar='\\')
                logger.info(f"DataFrame successfully saved to {args.output}")
                print(f"Output CSV saved to: {args.output}")
            except Exception as e:
                logger.error(f"Failed to save DataFrame to CSV: {e}")
                print(f"Error saving CSV: {e}")
                exit(1)
        else:
            logger.warning("No data processed from Zotero JSON. Output CSV will not be created.")
            print("No data processed. Output CSV not created.")
    else:
        # Use new incremental mode (default)
        logger.info("Using incremental mode with checkpoint support")
        df_zotero = load_zotero_to_dataframe_incremental(args.json, args.dir, args.output)

        if os.path.exists(args.output):
            logger.info(f"Processing complete. Output CSV: {args.output}")
            print(f"Output CSV saved to: {args.output}")

            # Show summary
            progress_file = get_progress_file_path(args.output)
            errors_file = get_errors_file_path(args.output)

            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    print(f"  - Items processed: {len(progress.get('processed_keys', []))}")

            if os.path.exists(errors_file):
                with open(errors_file, 'r') as f:
                    errors = json.load(f)
                    error_count = errors.get('total_errors', 0)
                    if error_count > 0:
                        print(f"  - Errors: {error_count} (see {errors_file})")
        else:
            logger.warning("No data processed from Zotero JSON. Output CSV will not be created.")
            print("No data processed. Output CSV not created.")
