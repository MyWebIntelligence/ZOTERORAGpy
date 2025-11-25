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
import json
import re
import unicodedata
import base64
import time
import csv
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

OPENAI_OCR_MODEL = os.getenv("OPENAI_OCR_MODEL", "gpt-4o-mini")
OPENAI_OCR_PROMPT = os.getenv(
    "OPENAI_OCR_PROMPT",
    "Transcris cette page PDF en Markdown lisible sans résumer ni modifier le contenu."
)
OPENAI_OCR_MAX_PAGES = _env_int("OPENAI_OCR_MAX_PAGES", 10)
OPENAI_OCR_MAX_TOKENS = _env_int("OPENAI_OCR_MAX_TOKENS", 2048)
OPENAI_OCR_RENDER_SCALE = _env_float("OPENAI_OCR_RENDER_SCALE", 2.0)


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


class OCRResult(NamedTuple):
    text: str
    provider: str


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


def _extract_text_with_mistral(pdf_path: str, max_pages: Optional[int] = None) -> str:
    """
    Extract text from PDF using Mistral OCR API with rate limiting.

    Uses a semaphore to limit concurrent API calls and prevent rate limit errors.
    The semaphore value is controlled by MISTRAL_CONCURRENT_CALLS env variable.

    Args:
        pdf_path: Path to the PDF file to process
        max_pages: Optional maximum number of pages to process

    Returns:
        Extracted text in Markdown format

    Raises:
        OCRExtractionError: If API key is missing or extraction fails
    """
    if not MISTRAL_API_KEY:
        raise OCRExtractionError("MISTRAL_API_KEY manquante.")

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

            upload_resp.raise_for_status()
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
                payload["page_ranges"] = [
                    {
                        "start": 1,
                        "end": max_pages,
                    }
                ]

            response = session.post(
                f"{base_url}/v1/ocr",
                headers={**headers, "Content-Type": "application/json"},
                json=payload,
                timeout=MISTRAL_OCR_TIMEOUT,
            )
            try:
                response.raise_for_status()
            except requests.HTTPError:
                logger.warning(
                    "Appel Mistral OCR échoué (%s) pour %s: %s",
                    response.status_code,
                    pdf_path,
                    response.text,
                )
                raise

            response_payload = response.json()

            text_fragments: List[str] = []
            if isinstance(response_payload, dict):
                possible_fields = [
                    response_payload.get("markdown"),
                    response_payload.get("text"),
                ]
                for candidate in possible_fields:
                    if isinstance(candidate, str) and candidate.strip():
                        text_fragments.append(candidate.strip())

                pages = response_payload.get("pages")
                if isinstance(pages, list):
                    for page in pages:
                        if isinstance(page, dict):
                            for key in ("markdown", "text"):
                                page_text = page.get(key)
                                if isinstance(page_text, str) and page_text.strip():
                                    text_fragments.append(page_text.strip())

                outputs = response_payload.get("output")
                if isinstance(outputs, list):
                    for block in outputs:
                        if isinstance(block, dict):
                            for key in ("markdown", "text", "content"):
                                value = block.get(key)
                                if isinstance(value, str) and value.strip():
                                    text_fragments.append(value.strip())

            markdown_text = "\n\n".join(dict.fromkeys(text_fragments))
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
) -> str:
    from openai import OpenAI

    base_limit = max_pages if max_pages is not None else float("inf")
    max_allowed = OPENAI_OCR_MAX_PAGES if OPENAI_OCR_MAX_PAGES > 0 else float("inf")

    outputs: List[str] = []
    client = OpenAI(api_key=api_key)

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

    return "\n\n".join(outputs)


def _finalize_ocr_result(text: str, provider: str, return_details: bool):
    """
    Format the OCR result based on the return_details flag.

    Args:
        text: Extracted text content
        provider: Name of the OCR provider used ('mistral', 'openai', 'legacy')
        return_details: If True, return OCRResult; otherwise return just text

    Returns:
        OCRResult namedtuple if return_details is True, else string
    """
    if return_details:
        return OCRResult(text=text, provider=provider)
    return text


def extract_text_with_ocr(
    pdf_path: str,
    max_pages: Optional[int] = None,
    *,
    return_details: bool = False,
):
    """
    Extract text from a PDF using the best available OCR provider.

    Attempts providers in order of preference:
    1. Mistral OCR API (if MISTRAL_API_KEY is set)
    2. OpenAI Vision API (if OPENAI_API_KEY is set)
    3. PyMuPDF legacy extraction (always available)

    Args:
        pdf_path: Path to the PDF file to process
        max_pages: Optional maximum number of pages to process
        return_details: If True, return OCRResult with provider info

    Returns:
        Extracted text (str) or OCRResult if return_details is True

    Raises:
        OCRExtractionError: If all extraction methods fail
    """

    last_error: Optional[Exception] = None

    if MISTRAL_API_KEY:
        try:
            logger.debug("Tentative d'OCR Mistral pour %s", pdf_path)
            mistral_text = _extract_text_with_mistral(pdf_path, max_pages=max_pages)
            return _finalize_ocr_result(mistral_text, "mistral", return_details)
        except Exception as mistral_error:
            last_error = mistral_error
            logger.warning(
                "Échec Mistral OCR pour %s: %s",
                pdf_path,
                mistral_error,
            )
    else:
        logger.debug("MISTRAL_API_KEY absente, OCR Mistral ignoré pour %s", pdf_path)

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            logger.debug("Fallback OpenAI OCR pour %s", pdf_path)
            openai_text = _extract_text_with_openai(pdf_path, openai_key, max_pages=max_pages)
            return _finalize_ocr_result(openai_text, "openai", return_details)
        except Exception as openai_error:
            last_error = openai_error
            logger.warning(
                "Échec OpenAI OCR pour %s: %s",
                pdf_path,
                openai_error,
            )
    else:
        logger.debug("OPENAI_API_KEY absente, OCR OpenAI ignoré pour %s", pdf_path)

    if not MISTRAL_API_KEY and not openai_key:
        logger.error(
            "Aucune clé API OCR configurée (MISTRAL_API_KEY ou OPENAI_API_KEY). "
            "Basculer sur l'extraction locale peut dégrader la qualité du texte."
        )

    if last_error:
        logger.info("Retour au processus OCR historique pour %s", pdf_path)

    legacy_text = _extract_text_with_legacy_pdf(pdf_path, max_pages=max_pages)
    if legacy_text.strip():
        return _finalize_ocr_result(legacy_text, "legacy", return_details)

    error_message = (
        "Impossible d'extraire le texte du document. Configurez MISTRAL_API_KEY ou OPENAI_API_KEY "
        "pour activer l'OCR en Markdown."
    )
    raise OCRExtractionError(error_message)


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

        # Process PDF attachments
        for attachment in item.get("attachments", []):
            path_from_json = attachment.get("path", "").strip()
            if not path_from_json or not path_from_json.lower().endswith(".pdf"):
                continue

            # Resolve PDF path
            if os.path.isabs(path_from_json):
                actual_pdf_path = path_from_json
            else:
                actual_pdf_path = os.path.join(pdf_base_dir, path_from_json)

            # Fuzzy search if not found
            if not os.path.exists(actual_pdf_path):
                actual_pdf_path = _find_pdf_fuzzy(actual_pdf_path, path_from_json, pdf_base_dir)
                if actual_pdf_path is None:
                    logger.warning(f"[{item_index}] PDF not found: {path_from_json}")
                    errors.append({
                        "itemKey": item_key,
                        "title": metadata.get("title", ""),
                        "error_type": "PDF_NOT_FOUND",
                        "error_message": f"PDF not found: {path_from_json}",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    continue

            logger.info(f"[{item_index}] Processing PDF: {os.path.basename(actual_pdf_path)}")

            try:
                # Perform OCR with retry logic
                ocr_payload = extract_text_with_ocr_retry(
                    actual_pdf_path,
                    return_details=True,
                )

                # Build record
                record = {
                    **metadata,
                    "filename": os.path.basename(path_from_json),
                    "path": actual_pdf_path,
                    "attachment_title": attachment.get("title", ""),
                    "texteocr": ocr_payload.text,
                    "texteocr_provider": ocr_payload.provider,
                }
                records.append(record)
                logger.info(f"[{item_index}] ✓ OCR success for {item_key} ({ocr_payload.provider})")

            except OCRExtractionError as ocr_error:
                logger.error(f"[{item_index}] OCR failed for {actual_pdf_path}: {ocr_error}")
                errors.append({
                    "itemKey": item_key,
                    "title": metadata.get("title", ""),
                    "error_type": "OCR_FAILED",
                    "error_message": str(ocr_error),
                    "pdf_path": actual_pdf_path,
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
        "authors", "filename", "path", "attachment_title", "texteocr", "texteocr_provider"
    ]

    # Load progress (items already processed)
    processed_keys = load_progress(output_csv)
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

                # Traitement des attachments PDF
                for attachment in item.get("attachments", []):
                    path_from_json = attachment.get("path", "").strip()
                    if path_from_json and path_from_json.lower().endswith(".pdf"):

                        # Résoudre le chemin du PDF
                        if os.path.isabs(path_from_json):
                            actual_pdf_path = path_from_json
                        else:
                            actual_pdf_path = os.path.join(pdf_base_dir, path_from_json)

                        if not os.path.exists(actual_pdf_path):
                            actual_pdf_path = _find_pdf_fuzzy(actual_pdf_path, path_from_json, pdf_base_dir)
                            if actual_pdf_path is None:
                                logger.warning(f"PDF non trouvé: {path_from_json}")
                                continue

                        logger.info(f"Traitement du PDF : {actual_pdf_path}")
                        try:
                            ocr_payload = extract_text_with_ocr(
                                actual_pdf_path,
                                return_details=True,
                            )
                        except OCRExtractionError as ocr_error:
                            logger.error(
                                "Échec OCR pour %s (%s): %s",
                                actual_pdf_path,
                                path_from_json,
                                ocr_error,
                            )
                            continue

                        records.append({
                            **metadata,
                            "filename": os.path.basename(path_from_json),
                            "path": actual_pdf_path,
                            "attachment_title": attachment.get("title", ""),
                            "texteocr": ocr_payload.text,
                            "texteocr_provider": ocr_payload.provider,
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
