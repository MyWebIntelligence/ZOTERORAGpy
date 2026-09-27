"""
Citation LLM Filtering Module
==============================

This module handles LLM-based filtering of citations from Publish or Perish exports.
It evaluates citation relevance for research projects and generates Zotero metadata.

Key Features:
- LLM-based relevance scoring with justification
- Automatic Zotero itemType inference and validation
- Author name parsing (handling various formats)
- Integration with global LLM semaphore for concurrency control
- Robust JSON parsing with markdown extraction
- Fallback handling for LLM errors

Output Format:
{
  "relevance_score": int (0-100),
  "relevance_reason": str,
  "zotero_item": {
    "itemType": str,
    "title": str,
    "creators": [{firstName, lastName, creatorType}],
    ...
  }
}
OR "NA" for irrelevant citations

Albert (opt-in, ``albert/<id>`` model):
- ``_call_llm_api`` makes one single send (``single_attempt``), max_tokens 1500
  (+ the reasoning headroom and a ``low`` effort for gpt-oss); an empty answer
  raises ``AlbertTruncatedError``. OpenAI/OpenRouter are never called.
- ``pre_filter_citation`` and ``filter_citation_with_llm`` go through
  ``run_llm_slot`` (single retry layer, no sleep while holding a semaphore,
  no nested acquisition), outside the historical ``max_retries`` loop, with
  the role fallback on repeated 503 answers (``citation`` role chain). The
  limiter budget follows the model sent (gpt-oss: ``notes``; ministral and the
  other chain models: ``recode``).
- Pre-filter answers are read as an exact token (first word in {RELEVANT, NA},
  otherwise the citation is kept); account errors are raised, never fail-open.
- Usage ledger: ``_call_llm_api``, ``pre_filter_citation`` and
  ``filter_citation_with_llm`` accept a trailing keyword argument
  ``albert_usage_ledger`` (a ``UsageLedger`` created by the route for an Albert
  request only), handed to every Albert client so that each call is recorded
  into it; ``None`` or an OpenAI/OpenRouter model changes nothing.

Author: RAGpy Team
Date: 2025-12-05
"""

import os
import json
import re
import asyncio
import logging
import unicodedata
from typing import TYPE_CHECKING, Dict, Union, List, Optional, Tuple, Any
from pathlib import Path
from pydantic import BaseModel, field_validator
from openai import OpenAI
from dotenv import load_dotenv

from app.utils.llm_note_generator import (
    ALBERT_ACCOUNT_ERRORS,
    PROVIDER_ALBERT,
    PROVIDER_OPENROUTER,
    AlbertChatRequest,
    AlbertTruncatedError,
    _albert_slot_chat,
    _get_albert_client,
    _get_llm_clients,
    albert_wire_id,
    get_llm_semaphore,
    resolve_llm_route,
)

if TYPE_CHECKING:  # annotations only: the usage module is never imported here at runtime
    from scripts.rad_albert.usage import UsageLedger

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Default LLM model (uses OpenRouter format by default)
# Format "provider/model" → OpenRouter (e.g., google/gemini-2.5-flash)
# Format "model" → OpenAI direct (e.g., gpt-4o-mini)
# Format "albert/<id>" → Albert (DINUM), when ALBERT_ENABLED=1
DEFAULT_LLM_MODEL = os.getenv("OPENROUTER_DEFAULT_MODEL", "gpt-4o-mini")

# Albert branch: role of the model catalog, answer budget (the client adds the
# reasoning headroom for gpt-oss) and reasoning effort of the filter calls.
ALBERT_CITATION_ROLE = "citation"
ALBERT_CITATION_MAX_TOKENS = 1500
ALBERT_CITATION_REASONING_EFFORT = "low"

# Exact tokens accepted from the Albert pre-filter answer.
PREFILTER_TOKENS = ("RELEVANT", "NA")

# Valid Zotero item types
VALID_ZOTERO_TYPES = {
    "journalArticle", "conferencePaper", "preprint", "book", "bookSection",
    "report", "thesis", "webpage", "manuscript", "document", "patent",
    "presentation", "magazineArticle", "newspaperArticle"
}

# Mapping of common incorrect types to valid ones
ITEMTYPE_MAPPINGS = {
    "article": "journalArticle",
    "conference": "conferencePaper",
    "paper": "journalArticle",
    "web": "webpage",
    "techreport": "report",
    "phdthesis": "thesis",
    "mastersthesis": "thesis"
}

# Valid fields for each Zotero itemType (based on Zotero schema)
# Used to sanitize items before sending to Zotero API
ITEMTYPE_VALID_FIELDS = {
    "journalArticle": {
        "title", "creators", "date", "publicationTitle", "DOI", "url",
        "abstractNote", "volume", "issue", "pages", "ISSN", "language",
        "tags", "extra", "journalAbbreviation", "series", "seriesTitle",
        "seriesText", "accessDate", "archive", "archiveLocation", "libraryCatalog",
        "callNumber", "rights", "shortTitle"
    },
    "conferencePaper": {
        "title", "creators", "date", "conferenceName", "DOI", "url",
        "abstractNote", "pages", "publisher", "place", "language", "tags",
        "extra", "proceedingsTitle", "volume", "series", "ISBN", "accessDate",
        "archive", "archiveLocation", "libraryCatalog", "callNumber", "rights",
        "shortTitle"
    },
    "preprint": {
        "title", "creators", "date", "repository", "DOI", "url",
        "abstractNote", "archiveID", "language", "tags", "extra",
        "accessDate", "libraryCatalog", "rights", "shortTitle"
    },  # Note: publicationTitle is NOT valid for preprint - use repository instead
    "book": {
        "title", "creators", "date", "publisher", "place", "ISBN", "url",
        "abstractNote", "numPages", "edition", "language", "tags", "extra",
        "series", "seriesNumber", "volume", "numberOfVolumes", "accessDate",
        "archive", "archiveLocation", "libraryCatalog", "callNumber", "rights",
        "shortTitle"
    },  # Note: DOI and ISSN are NOT valid for book
    "bookSection": {
        "title", "creators", "date", "bookTitle", "publisher", "place",
        "ISBN", "url", "abstractNote", "pages", "edition", "language",
        "tags", "extra", "series", "seriesNumber", "volume", "numberOfVolumes",
        "accessDate", "archive", "archiveLocation", "libraryCatalog",
        "callNumber", "rights", "shortTitle"
    },  # Note: DOI is NOT valid for bookSection
    "thesis": {
        "title", "creators", "date", "university", "thesisType", "url",
        "abstractNote", "numPages", "place", "language", "tags", "extra",
        "accessDate", "archive", "archiveLocation", "libraryCatalog",
        "callNumber", "rights", "shortTitle"
    },
    "report": {
        "title", "creators", "date", "institution", "reportNumber", "reportType",
        "url", "abstractNote", "pages", "place", "language", "tags", "extra",
        "seriesTitle", "accessDate", "archive", "archiveLocation",
        "libraryCatalog", "callNumber", "rights", "shortTitle"
    },
    "webpage": {
        "title", "creators", "date", "websiteTitle", "websiteType", "url",
        "abstractNote", "accessDate", "language", "tags", "extra",
        "rights", "shortTitle"
    },  # Note: DOI, ISSN, volume, issue, pages are NOT valid for webpage
    "manuscript": {
        "title", "creators", "date", "manuscriptType", "place", "url",
        "abstractNote", "numPages", "language", "tags", "extra",
        "accessDate", "archive", "archiveLocation", "libraryCatalog",
        "callNumber", "rights", "shortTitle"
    },  # Note: DOI is NOT valid for manuscript
    "document": {
        "title", "creators", "date", "publisher", "url", "abstractNote",
        "language", "tags", "extra", "accessDate", "archive",
        "archiveLocation", "libraryCatalog", "callNumber", "rights", "shortTitle"
    },
    "presentation": {
        "title", "creators", "date", "meetingName", "place", "url",
        "abstractNote", "type", "language", "tags", "extra",
        "accessDate", "rights", "shortTitle"
    },  # Note: DOI is NOT valid for presentation
    "patent": {
        "title", "creators", "date", "place", "country", "assignee",
        "issuingAuthority", "patentNumber", "applicationNumber", "priorityNumbers",
        "issueDate", "references", "legalStatus", "url", "abstractNote",
        "language", "tags", "extra", "accessDate", "rights", "shortTitle"
    },
    "magazineArticle": {
        "title", "creators", "date", "publicationTitle", "url", "abstractNote",
        "volume", "issue", "pages", "ISSN", "language", "tags", "extra",
        "accessDate", "archive", "archiveLocation", "libraryCatalog",
        "callNumber", "rights", "shortTitle"
    },
    "newspaperArticle": {
        "title", "creators", "date", "publicationTitle", "url", "abstractNote",
        "place", "edition", "section", "pages", "ISSN", "language", "tags",
        "extra", "accessDate", "archive", "archiveLocation", "libraryCatalog",
        "callNumber", "rights", "shortTitle"
    },
}

# Universal fields valid for ALL item types
UNIVERSAL_FIELDS = {
    "title", "creators", "date", "url", "abstractNote", "language",
    "tags", "extra", "itemType", "accessDate", "rights", "shortTitle"
}


def sanitize_zotero_item(item_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove invalid fields for the given itemType before sending to Zotero API.

    This function ensures that only valid fields for the specific itemType are
    included in the item data, preventing Zotero API validation errors like:
    - "'DOI' is not a valid field for type 'bookSection'"
    - "'publicationTitle' is not a valid field for type 'preprint'"

    Args:
        item_data: Zotero item dictionary with itemType field

    Returns:
        Sanitized item dictionary with only valid fields for the itemType

    Examples:
        >>> item = {"itemType": "bookSection", "title": "Test", "DOI": "10.1234/test"}
        >>> sanitize_zotero_item(item)
        {"itemType": "bookSection", "title": "Test"}  # DOI removed

        >>> item = {"itemType": "preprint", "title": "Test", "publicationTitle": "arXiv"}
        >>> sanitize_zotero_item(item)
        {"itemType": "preprint", "title": "Test", "repository": "arXiv"}  # Converted
    """
    item_type = item_data.get("itemType", "document")

    # Get valid fields for this item type, fallback to universal fields
    valid_fields = ITEMTYPE_VALID_FIELDS.get(item_type, UNIVERSAL_FIELDS)
    all_valid = valid_fields | UNIVERSAL_FIELDS

    sanitized = {}
    removed_fields = []

    for key, value in item_data.items():
        if key in all_valid:
            sanitized[key] = value
        elif value is not None and value != "" and value != []:
            # Only track removal of non-empty fields
            removed_fields.append(key)

    if removed_fields:
        logger.info(f"Sanitized {item_type}: removed invalid fields {removed_fields}")

    # Special conversions for specific item types
    # 1. Convert publicationTitle -> repository for preprints
    if item_type == "preprint" and "publicationTitle" in item_data:
        pub_title = item_data.get("publicationTitle")
        if pub_title and "repository" not in sanitized:
            sanitized["repository"] = pub_title
            logger.info(f"Converted publicationTitle -> repository for preprint: {pub_title}")

    # 2. Convert publicationTitle -> bookTitle for bookSection
    if item_type == "bookSection" and "publicationTitle" in item_data:
        pub_title = item_data.get("publicationTitle")
        if pub_title and "bookTitle" not in sanitized:
            sanitized["bookTitle"] = pub_title
            logger.info(f"Converted publicationTitle -> bookTitle for bookSection: {pub_title}")

    # 3. Convert publicationTitle -> websiteTitle for webpage
    if item_type == "webpage" and "publicationTitle" in item_data:
        pub_title = item_data.get("publicationTitle")
        if pub_title and "websiteTitle" not in sanitized:
            sanitized["websiteTitle"] = pub_title
            logger.info(f"Converted publicationTitle -> websiteTitle for webpage: {pub_title}")

    # 4. Convert publicationTitle -> conferenceName for conferencePaper
    if item_type == "conferencePaper" and "publicationTitle" in item_data:
        pub_title = item_data.get("publicationTitle")
        if pub_title and "conferenceName" not in sanitized and "proceedingsTitle" not in sanitized:
            sanitized["conferenceName"] = pub_title
            logger.info(f"Converted publicationTitle -> conferenceName for conferencePaper: {pub_title}")

    return sanitized


class ZoteroCreator(BaseModel):
    """Pydantic model for Zotero creator (author, editor, etc.)."""
    creatorType: str = "author"
    firstName: str
    lastName: str

    @field_validator('firstName', 'lastName')
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Creator name parts cannot be empty")
        return v.strip()


class ZoteroItemData(BaseModel):
    """Pydantic model for Zotero item metadata."""
    itemType: str
    title: str
    creators: List[ZoteroCreator]
    date: Optional[str] = None
    publicationTitle: Optional[str] = None
    DOI: Optional[str] = None
    url: Optional[str] = None
    abstractNote: Optional[str] = None
    language: Optional[str] = None
    tags: Optional[List[Dict[str, str]]] = None
    extra: Optional[str] = None  # For citation count and other metadata

    @field_validator('itemType')
    @classmethod
    def validate_item_type(cls, v: str) -> str:
        """Validate and normalize itemType."""
        # Normalize to lowercase for mapping
        v_lower = v.lower().strip()

        # Check mappings first
        if v_lower in ITEMTYPE_MAPPINGS:
            mapped_type = ITEMTYPE_MAPPINGS[v_lower]
            logger.warning(f"Mapped invalid itemType '{v}' -> '{mapped_type}'")
            return mapped_type

        # Check if valid as-is
        if v in VALID_ZOTERO_TYPES:
            return v

        # Default to journalArticle if unknown
        logger.warning(f"Unknown itemType '{v}', defaulting to 'journalArticle'")
        return "journalArticle"


class CitationFilterResult(BaseModel):
    """Complete citation filter result with relevance scoring."""
    relevance_score: int
    relevance_reason: str
    zotero_item: ZoteroItemData

    @field_validator('relevance_score')
    @classmethod
    def validate_score(cls, v: int) -> int:
        if not 0 <= v <= 100:
            raise ValueError("Relevance score must be between 0 and 100")
        return v


def _load_filter_prompt_template() -> str:
    """
    Load the citation filter prompt template.

    Returns:
        Prompt template string with placeholders

    Raises:
        FileNotFoundError: If template file not found
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file = os.path.join(current_dir, "citation_filter_prompt.md")

    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Citation filter prompt template not found: {prompt_file}")

    with open(prompt_file, "r", encoding="utf-8") as f:
        template = f.read()

    logger.info(f"Loaded citation filter prompt template from {prompt_file}")
    return template


def _build_filter_prompt(
    citation: Dict,
    web_content: str,
    web_content_source: str,
    project_name: str,
    project_description: str,
    collection_name: str,
    collection_description: str
) -> str:
    """
    Build LLM prompt for citation filtering.

    Args:
        citation: Citation dictionary from PoP parser
        web_content: Extracted web content (PDF/HTML text)
        web_content_source: Source type ("pdf", "html", "none")
        project_name: Research project name
        project_description: Project description
        collection_name: Target Zotero collection name
        collection_description: Collection description

    Returns:
        Formatted prompt string
    """
    # Load template
    template = _load_filter_prompt_template()

    # Safely extract citation fields
    def safe_str(value, default="N/A"):
        if value is None or value == "":
            return default
        return str(value)

    # Replace placeholders
    prompt = template.replace("{PROJECT_NAME}", project_name)
    prompt = prompt.replace("{PROJECT_DESCRIPTION}", project_description or "Non spécifiée")
    prompt = prompt.replace("{COLLECTION_NAME}", collection_name)
    prompt = prompt.replace("{COLLECTION_DESCRIPTION}", collection_description or "Non spécifiée")

    prompt = prompt.replace("{CITATION_UID}", safe_str(citation.get("uid")))
    prompt = prompt.replace("{CITATION_TITLE}", safe_str(citation.get("title"), "Sans titre"))
    prompt = prompt.replace("{CITATION_AUTHORS}", safe_str(", ".join(citation.get("authors", [])), "Auteurs inconnus"))
    prompt = prompt.replace("{CITATION_YEAR}", safe_str(citation.get("year")))
    prompt = prompt.replace("{CITATION_SOURCE}", safe_str(citation.get("source")))
    prompt = prompt.replace("{CITATION_DOI}", safe_str(citation.get("doi")))
    prompt = prompt.replace("{CITATION_ABSTRACT}", safe_str(citation.get("abstract")))
    prompt = prompt.replace("{CITATION_CITES}", safe_str(citation.get("cites"), "0"))

    prompt = prompt.replace("{WEB_CONTENT_SOURCE}", web_content_source)
    prompt = prompt.replace("{WEB_CONTENT}", web_content[:10000] if web_content else "Aucun contenu récupéré")

    return prompt


def _parse_llm_response(response_text: str) -> Union[Dict, str]:
    """
    Parse LLM response, extracting JSON or "NA".

    This function handles various LLM output formats:
    - Clean JSON object
    - JSON wrapped in markdown code blocks
    - "NA" string (case-insensitive)
    - Text with embedded JSON

    Args:
        response_text: Raw LLM response

    Returns:
        Parsed dictionary if JSON found, "NA" string if not relevant

    Raises:
        ValueError: If response cannot be parsed
    """
    response_text = response_text.strip()

    # Check for "NA" (case-insensitive)
    if response_text.upper() == "NA":
        logger.info("LLM returned NA (not relevant)")
        return "NA"

    # Try extracting JSON from markdown code blocks
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(json_pattern, response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
        logger.debug("Extracted JSON from markdown code block")
    else:
        # Try finding JSON object directly
        json_obj_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
        match = re.search(json_obj_pattern, response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            logger.debug("Extracted raw JSON object")
        else:
            # No JSON found
            raise ValueError(f"No valid JSON found in LLM response: {response_text[:200]}...")

    # Parse JSON
    try:
        data = json.loads(json_str)
        logger.debug("Successfully parsed JSON response")
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in LLM response: {str(e)}\nJSON string: {json_str[:200]}...")


def _validate_filter_result(data: Dict) -> CitationFilterResult:
    """
    Validate and normalize filter result using Pydantic.

    Args:
        data: Parsed JSON dictionary from LLM

    Returns:
        Validated CitationFilterResult

    Raises:
        ValueError: If validation fails
    """
    try:
        result = CitationFilterResult(**data)
        logger.info(f"Validated filter result: score={result.relevance_score}, itemType={result.zotero_item.itemType}")
        return result
    except Exception as e:
        raise ValueError(f"Filter result validation failed: {str(e)}")


def _call_llm_api(
    prompt: str,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.2,
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    albert_api_key: Optional[str] = None,
    *,
    albert_usage_ledger: Optional["UsageLedger"] = None
) -> str:
    """
    Call LLM API (synchronous wrapper for OpenAI/OpenRouter/Albert).

    Args:
        prompt: Formatted prompt
        model: Model identifier ("albert/<id>" selects Albert)
        temperature: Sampling temperature (0-1)
        openai_api_key: Optional OpenAI API key (falls back to env for admin users)
        openrouter_api_key: Optional OpenRouter API key (falls back to env for admin users)
        albert_api_key: Optional Albert key, used only for an Albert model
            (single send: the retries belong to ``run_llm_slot``)
        albert_usage_ledger: Optional usage ledger of the request
            (``UsageLedger``): the Albert call is recorded into it; ignored on
            the OpenAI/OpenRouter path

    Returns:
        Raw LLM response text

    Raises:
        ValueError: If API call fails or no clients available
        AlbertDisabledError: ``albert/…`` model while Albert is disabled
        AlbertError: Albert failure (classified; never replaced by OpenAI)
    """
    # Single resolver, before any OpenAI/OpenRouter client is built
    resolution = resolve_llm_route(model)
    if resolution.provider == PROVIDER_ALBERT:
        return _call_albert_api(
            prompt, resolution.wire_model, temperature=temperature, albert_api_key=albert_api_key,
            albert_usage_ledger=albert_usage_ledger
        )

    openai_client, openrouter_client, default_model = _get_llm_clients(
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key
    )

    # Determine which client to use
    if resolution.provider == PROVIDER_OPENROUTER:  # OpenRouter format
        if not openrouter_client:
            raise ValueError("OpenRouter client not available (missing OPENROUTER_API_KEY)")
        client = openrouter_client
        logger.info(f"Using OpenRouter with model: {model}")
    else:  # OpenAI
        if not openai_client:
            raise ValueError("OpenAI client not available (missing OPENAI_API_KEY)")
        client = openai_client
        logger.info(f"Using OpenAI with model: {model}")

    # Make API call
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1500  # Sufficient for our JSON response
        )

        response_text = response.choices[0].message.content.strip()
        logger.debug(f"LLM response received ({len(response_text)} chars)")
        return response_text

    except Exception as e:
        logger.error(f"LLM API call failed: {str(e)}")
        raise ValueError(f"LLM API error: {str(e)}")


def _call_albert_api(
    prompt: str,
    wire_model: str,
    *,
    temperature: float,
    albert_api_key: Optional[str],
    albert_usage_ledger: Optional["UsageLedger"] = None
) -> str:
    """
    Albert branch of ``_call_llm_api``: one single send, no retry, no limiter.

    The retries, the limiter (acquired outside the semaphores) and the backoff
    (slept outside the semaphores) belong to ``run_llm_slot`` in the async
    callers. Budget: ``ALBERT_CITATION_MAX_TOKENS``; for a reasoning model
    (gpt-oss) the client adds the reasoning headroom and the ``low`` effort is
    sent. Only ``message.content`` is used (never the ``reasoning`` field).

    Args:
        prompt: Formatted prompt
        wire_model: Model name without the ``albert/`` prefix
        temperature: Sampling temperature
        albert_api_key: The caller's Albert key
        albert_usage_ledger: Usage ledger of the request, given to the client,
            or ``None``

    Returns:
        The stripped answer text

    Raises:
        AlbertTruncatedError: Empty answer (``content`` None or blank)
        AlbertAuthError: Account error, including a missing key
        AlbertError: Other classified errors (single attempt)
    """
    client = _get_albert_client(albert_api_key, use_limiter=False, ledger=albert_usage_ledger)
    try:
        result = client.chat(
            [{"role": "user", "content": prompt}],
            albert_wire_id(wire_model),
            role=ALBERT_CITATION_ROLE,
            max_tokens=ALBERT_CITATION_MAX_TOKENS,
            temperature=temperature,
            reasoning_effort=ALBERT_CITATION_REASONING_EFFORT,
            single_attempt=True,
        )
    finally:
        client.close()
    return _albert_citation_text(result)


def _albert_citation_text(result: Any) -> str:
    """
    Stripped text of an Albert citation-filter answer, never an empty answer.

    Only ``message.content`` is used (never the ``reasoning`` field of gpt-oss);
    ``content`` None or blank raises, whatever the finish reason.

    Args:
        result: ``ChatResult`` returned by ``AlbertClient.chat``

    Returns:
        The stripped answer text

    Raises:
        AlbertTruncatedError: Empty answer (``content`` None or blank)
    """
    content = getattr(result, "content", result)
    text = str(content).strip() if content is not None else ""
    if not text:
        raise AlbertTruncatedError(
            f"Réponse Albert vide pour le filtre de citations "
            f"(finish_reason={getattr(result, 'finish_reason', None)}, modèle {getattr(result, 'model', None)}).",
            status=200,
            endpoint="/v1/chat/completions",
        )
    logger.debug(f"Albert response received ({len(text)} chars)")
    return text


async def _albert_citation_call(
    prompt: str,
    model: str,
    *,
    temperature: float,
    albert_api_key: Optional[str],
    albert_usage_ledger: Optional["UsageLedger"] = None
) -> str:
    """
    One Albert citation-filter call from the event loop, with the role fallback on 503.

    Same request as ``_call_albert_api`` (``citation`` role, budget
    ``ALBERT_CITATION_MAX_TOKENS``, ``low`` effort for a reasoning model),
    sent through ``_albert_slot_chat`` like the notes: each model of the chain
    (explicit model first, then the ``citation`` role chain) gets single-attempt
    sends retried by ``run_llm_slot`` (limiter and backoff outside the
    semaphores); after ``ALBERT_BUSY_RETRIES`` answers « Model is too busy »,
    the next model of the chain is used (``ALBERT_MODEL_FALLBACK=0`` keeps the
    first model only). The limiter budget follows each model sent: an explicit
    gpt-oss consumes the ``notes`` budget, a ministral fallback the ``recode``
    budget. Fallbacks absent from the cached ``/v1/models`` listing of the
    account are skipped. A 404 never triggers a fallback, and OpenAI or
    OpenRouter are never called.

    Args:
        prompt: Formatted prompt
        model: ``albert/<id>`` model
        temperature: Sampling temperature
        albert_api_key: The caller's Albert key
        albert_usage_ledger: Usage ledger of the request, given to the client
            (every send, retries and chain fallbacks included, is recorded
            into it), or ``None``

    Returns:
        The stripped answer text

    Raises:
        AlbertAuthError: Account error, including a missing key
        AlbertModelBusy: Every model of the chain stayed busy
        AlbertTruncatedError: Empty answer
        AlbertError: Other classified errors, after the retries
    """
    resolution = resolve_llm_route(model)
    request = AlbertChatRequest(
        messages=[{"role": "user", "content": prompt}],
        model=albert_wire_id(resolution.wire_model),
        role=ALBERT_CITATION_ROLE,
        max_tokens=ALBERT_CITATION_MAX_TOKENS,
        temperature=temperature,
    )
    client = _get_albert_client(albert_api_key, use_limiter=False, ledger=albert_usage_ledger)
    try:
        result = await _albert_slot_chat(
            client,
            request,
            get_llm_semaphore(),
            max_tokens=ALBERT_CITATION_MAX_TOKENS,
            reasoning_effort=ALBERT_CITATION_REASONING_EFFORT,
        )
    finally:
        client.close()
    return _albert_citation_text(result)


def _prefilter_token(response_text: Optional[str]) -> Optional[str]:
    """
    Read the Albert pre-filter answer as an exact token.

    The first word is normalized (accents removed, upper case, non-letters
    dropped: ``"**NA**"``, ``"N/A"`` and ``"relevant."`` are recognized).

    Args:
        response_text: Raw LLM answer

    Returns:
        ``"RELEVANT"``, ``"NA"``, or ``None`` when the first word is neither
    """
    words = str(response_text or "").strip().split()
    if not words:
        return None
    first = unicodedata.normalize("NFKD", words[0]).upper()
    token = re.sub(r"[^A-Z]", "", first)
    return token if token in PREFILTER_TOKENS else None


async def pre_filter_citation(
    citation: Dict,
    project_name: str,
    project_description: str,
    collection_name: str,
    collection_description: str,
    model: str = DEFAULT_LLM_MODEL,
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    albert_api_key: Optional[str] = None,
    *,
    albert_usage_ledger: Optional["UsageLedger"] = None
) -> bool:
    """
    Pré-filtrage rapide basé uniquement sur titre/abstract/source.

    Cette fonction fait un test de pertinence AVANT le fetch web coûteux.
    Elle utilise un prompt léger et retourne simplement True/False.

    Args:
        citation: Citation dictionary avec title, abstract, source, authors
        project_name: Nom du projet de recherche
        project_description: Description du projet
        collection_name: Nom de la collection cible
        collection_description: Description de la collection
        model: Modèle LLM à utiliser (``albert/<id>`` : Albert)
        openai_api_key: Clé API OpenAI
        openrouter_api_key: Clé API OpenRouter
        albert_api_key: Clé API Albert (utilisée seulement pour un modèle Albert)
        albert_usage_ledger: Journal d'usage de la requête (``UsageLedger``),
            alimenté par l'appel Albert seulement ; ``None`` ou un modèle
            OpenAI/OpenRouter ne change rien

    Returns:
        True si la citation semble pertinente, False sinon

    Raises:
        AlbertDisabledError: modèle ``albert/…`` alors qu'Albert est désactivé
        AlbertAuthError: erreur de compte Albert (jamais convertie en « pertinent »)

    Example:
        >>> is_relevant = await pre_filter_citation(
        ...     {"title": "Deep Learning", "abstract": "..."},
        ...     "AI Research", "ML papers", "NLP", ""
        ... )
        >>> if is_relevant:
        ...     # Fetch web content et faire filtrage complet
    """
    # Construire un prompt léger pour le pré-filtrage
    title = citation.get("title", "")
    abstract = citation.get("abstract", "")
    source = citation.get("source", "")
    authors = citation.get("authors", [])
    if isinstance(authors, list):
        authors_str = ", ".join(authors[:3])
        if len(authors) > 3:
            authors_str += " et al."
    else:
        authors_str = str(authors)

    prompt = f"""Tu es un assistant de recherche. Détermine rapidement si cet article est pertinent pour le projet.

PROJET: {project_name}
DESCRIPTION: {project_description}
COLLECTION: {collection_name} - {collection_description}

ARTICLE:
- Titre: {title}
- Auteurs: {authors_str}
- Source: {source}
- Résumé: {abstract[:1000] if abstract else "Non disponible"}

INSTRUCTIONS:
- Réponds UNIQUEMENT par "RELEVANT" ou "NA"
- RELEVANT = l'article correspond au sujet du projet/collection
- NA = l'article n'est pas pertinent ou hors sujet

RÉPONSE:"""

    # Résolveur unique : un modèle albert/ passe par run_llm_slot (branche dédiée)
    if resolve_llm_route(model).provider == PROVIDER_ALBERT:
        return await _pre_filter_citation_albert(
            prompt,
            title,
            model=model,
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
            albert_api_key=albert_api_key,
            albert_usage_ledger=albert_usage_ledger
        )

    # Acquérir le semaphore global
    semaphore = get_llm_semaphore()

    async with semaphore:
        try:
            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(
                None,
                lambda: _call_llm_api(
                    prompt,
                    model=model,
                    temperature=0.1,  # Très déterministe
                    openai_api_key=openai_api_key,
                    openrouter_api_key=openrouter_api_key
                )
            )

            # Parser la réponse simple
            response_upper = response_text.strip().upper()
            is_relevant = "RELEVANT" in response_upper and "NA" not in response_upper

            logger.debug(
                f"Pre-filter result for '{title[:40]}...': "
                f"{'RELEVANT' if is_relevant else 'NA'}"
            )
            return is_relevant

        except Exception as e:
            logger.warning(f"Pre-filter failed for '{title[:30]}': {e}")
            # En cas d'erreur, considérer comme potentiellement pertinent
            # pour ne pas manquer d'articles
            return True


async def _pre_filter_citation_albert(
    prompt: str,
    title: str,
    *,
    model: str,
    openai_api_key: Optional[str],
    openrouter_api_key: Optional[str],
    albert_api_key: Optional[str],
    albert_usage_ledger: Optional["UsageLedger"] = None
) -> bool:
    """
    Branche Albert du pré-filtre : ``run_llm_slot`` et lecture par jeton exact.

    Un seul envoi par essai sous les sémaphores (Albert puis global), limiteur
    (budget du modèle envoyé : ``notes`` pour gpt-oss, ``recode`` sinon)
    acquis et attente de réessai hors sémaphores ; aucune boucle de réessai
    supplémentaire. Après des 503 répétés, repli sur le modèle suivant de la
    chaîne du rôle ``citation`` (``_albert_citation_call``). Réponse : premier
    mot normalisé ``NA`` → non pertinent, ``RELEVANT`` → pertinent, autre →
    citation gardée. Fail-open conservé pour les autres erreurs (y compris une
    chaîne entièrement surchargée), sauf les erreurs de compte, qui remontent.

    Args:
        prompt: Prompt de pré-filtrage
        title: Titre de la citation (journaux)
        model: Modèle ``albert/<id>``
        openai_api_key: Clé API OpenAI (jamais utilisée sur cette branche)
        openrouter_api_key: Clé API OpenRouter (jamais utilisée sur cette branche)
        albert_api_key: Clé API Albert
        albert_usage_ledger: Journal d'usage de la requête, ou ``None``

    Returns:
        True si la citation est gardée, False si la réponse est exactement NA

    Raises:
        AlbertAuthError: erreur de compte (clé, compte expiré, budget, quota)
    """
    try:
        response_text = await _albert_citation_call(
            prompt,
            model,
            temperature=0.1,  # Très déterministe
            albert_api_key=albert_api_key,
            albert_usage_ledger=albert_usage_ledger
        )
    except ALBERT_ACCOUNT_ERRORS:
        raise
    except Exception as e:
        logger.warning(f"Pre-filter failed for '{title[:30]}': {e}")
        # En cas d'erreur, considérer comme potentiellement pertinent
        return True

    token = _prefilter_token(response_text)
    is_relevant = token != "NA"
    logger.debug(
        f"Pre-filter result for '{title[:40]}...': "
        f"{'RELEVANT' if is_relevant else 'NA'} (Albert, token={token})"
    )
    return is_relevant


async def filter_citation_with_llm(
    citation: Dict,
    web_content: str,
    web_content_source: str,
    project_name: str,
    project_description: str,
    collection_name: str,
    collection_description: str,
    model: str = DEFAULT_LLM_MODEL,
    max_retries: int = 1,
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    albert_api_key: Optional[str] = None,
    *,
    albert_usage_ledger: Optional["UsageLedger"] = None
) -> Union[Dict, str]:
    """
    Filter citation using LLM with global concurrency control.

    This function:
    1. Builds a prompt from template with all context
    2. Acquires global LLM semaphore
    3. Calls LLM API (with retry on failure)
    4. Parses and validates response
    5. Returns structured result or "NA"

    With an ``albert/<id>`` model, steps 2-3 go through ``run_llm_slot`` (single
    retry layer, semaphores held during each send only); ``max_retries`` and
    the 2-second sleep of the historical loop do not apply.

    Args:
        citation: Citation dictionary (from PopCitation.dict())
        web_content: Extracted web content text
        web_content_source: Source type ("pdf", "html", "none")
        project_name: Research project name
        project_description: Project description
        collection_name: Target Zotero collection
        collection_description: Collection description
        model: LLM model identifier
        max_retries: Number of retries on failure (OpenAI/OpenRouter only)
        openai_api_key: Optional OpenAI API key (for non-admin users)
        openrouter_api_key: Optional OpenRouter API key (for non-admin users)
        albert_api_key: Optional Albert key, used only for an Albert model
        albert_usage_ledger: Optional usage ledger of the request
            (``UsageLedger``), filled by the Albert calls only; ``None`` or an
            OpenAI/OpenRouter model changes nothing

    Returns:
        If relevant: Dictionary with keys:
            - relevance_score: int (0-100)
            - relevance_reason: str
            - zotero_item: dict (complete Zotero metadata)
        If not relevant: String "NA"

    Raises:
        ValueError: If LLM call fails after retries or response invalid
        AlbertDisabledError: ``albert/…`` model while Albert is disabled
        AlbertAuthError: Albert account error (never converted to ValueError)

    Examples:
        >>> result = await filter_citation_with_llm(
        ...     citation={"title": "...", "authors": [...]},
        ...     web_content="Full text...",
        ...     web_content_source="pdf",
        ...     project_name="My Research",
        ...     project_description="...",
        ...     collection_name="Main Papers",
        ...     collection_description="...",
        ... )
        >>> if isinstance(result, dict):
        ...     print(f"Relevant (score={result['relevance_score']})")
        ... else:
        ...     print("Not relevant")
    """
    # Build prompt
    prompt = _build_filter_prompt(
        citation=citation,
        web_content=web_content,
        web_content_source=web_content_source,
        project_name=project_name,
        project_description=project_description,
        collection_name=collection_name,
        collection_description=collection_description
    )

    # Single resolver: an albert/ model goes through run_llm_slot, outside the
    # retry loop below
    if resolve_llm_route(model).provider == PROVIDER_ALBERT:
        return await _filter_citation_with_albert(
            citation,
            prompt,
            model=model,
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
            albert_api_key=albert_api_key,
            albert_usage_ledger=albert_usage_ledger
        )

    # Get global semaphore
    semaphore = get_llm_semaphore()

    async with semaphore:
        remaining = semaphore._value
        logger.debug(f"Acquired LLM slot for citation filtering ({remaining} slots remaining)")

        # Retry logic
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # Run LLM call in executor (blocking → async)
                loop = asyncio.get_event_loop()
                response_text = await loop.run_in_executor(
                    None,
                    lambda: _call_llm_api(
                        prompt,
                        model=model,
                        temperature=0.2,
                        openai_api_key=openai_api_key,
                        openrouter_api_key=openrouter_api_key
                    )
                )

                # Parse response
                parsed = _parse_llm_response(response_text)

                # If "NA", return immediately
                if parsed == "NA":
                    return "NA"

                # Validate, post-process and return structured result
                return _finalize_filter_result(parsed, citation)

            except Exception as e:
                last_error = e
                logger.warning(f"Filter attempt {attempt + 1}/{max_retries + 1} failed: {str(e)}")
                if attempt < max_retries:
                    await asyncio.sleep(2)  # Wait before retry

        # All retries failed
        logger.error(f"Citation filtering failed after {max_retries + 1} attempts: {str(last_error)}")
        raise ValueError(f"Citation filtering failed: {str(last_error)}")


def _finalize_filter_result(parsed: Dict, citation: Dict) -> Dict:
    """
    Validate a parsed filter answer and override its fields with the citation data.

    Shared by every provider: hallucinated URL/DOI/title are replaced by the
    original citation values, the citation count is added to ``extra``,
    "N/A"-like values are cleaned and the item is sanitized for its itemType.

    Args:
        parsed: JSON dictionary parsed from the LLM answer
        citation: Original citation dictionary

    Returns:
        The structured result (relevance_score, relevance_reason, zotero_item)

    Raises:
        ValueError: If validation fails
    """
    # Validate and return structured result
    validated = _validate_filter_result(parsed)
    result = validated.dict()

    # Post-process: Override LLM-generated fields with original citation data
    # This prevents hallucinated URLs/DOIs from being used
    zotero_item = result.get("zotero_item", {})

    # Use original URL (article_url or fulltext_url from citation)
    original_url = citation.get("article_url") or citation.get("fulltext_url")
    if original_url:
        zotero_item["url"] = str(original_url)
    elif not zotero_item.get("url"):
        # No original URL, remove any hallucinated one
        zotero_item["url"] = ""

    # Use original DOI if available
    original_doi = citation.get("doi")
    if original_doi:
        zotero_item["DOI"] = original_doi

    # Use original title if LLM truncated it
    original_title = citation.get("title")
    if original_title and len(original_title) > len(zotero_item.get("title", "")):
        zotero_item["title"] = original_title

    # Inject citation count into extra field (citation:n format)
    cites = citation.get("cites")
    if cites is not None and isinstance(cites, int) and cites >= 0:
        existing_extra = zotero_item.get("extra", "") or ""
        citation_entry = f"citation:{cites}"
        if existing_extra:
            zotero_item["extra"] = f"{existing_extra}\n{citation_entry}"
        else:
            zotero_item["extra"] = citation_entry
        logger.debug(f"Added citation count to extra field: {citation_entry}")

    # Clean up "N/A" values that LLM may generate
    # Zotero API rejects these as invalid values
    na_patterns = {"N/A", "n/a", "N.A.", "n.a.", "NA", "na", "None", "null", "undefined", "-"}
    fields_to_clean = ["DOI", "ISSN", "ISBN", "pages", "volume", "issue", "callNumber"]
    for field in fields_to_clean:
        if field in zotero_item and zotero_item[field] in na_patterns:
            logger.debug(f"Cleaning N/A value from field {field}")
            zotero_item[field] = ""

    # Sanitize item: remove invalid fields for this itemType
    # This prevents errors like "'DOI' is not a valid field for type 'bookSection'"
    zotero_item = sanitize_zotero_item(zotero_item)

    result["zotero_item"] = zotero_item
    return result


async def _filter_citation_with_albert(
    citation: Dict,
    prompt: str,
    *,
    model: str,
    openai_api_key: Optional[str],
    openrouter_api_key: Optional[str],
    albert_api_key: Optional[str],
    albert_usage_ledger: Optional["UsageLedger"] = None
) -> Union[Dict, str]:
    """
    Albert branch of ``filter_citation_with_llm`` (``run_llm_slot``, no outer loop).

    One send per attempt under the Albert then the global semaphore; the
    limiter (budget of the model sent: ``notes`` for gpt-oss, ``recode``
    otherwise) is acquired and the backoff slept outside them; the historical
    ``max_retries`` loop and its 2-second sleep are bypassed (single retry
    layer). After repeated 503 answers, the next model of the ``citation``
    role chain is used (``_albert_citation_call``). The global semaphore is
    never held by this function around the slot, so there is no nested
    acquisition.

    Args:
        citation: Citation dictionary
        prompt: Formatted filter prompt
        model: ``albert/<id>`` model
        openai_api_key: OpenAI key (never used on this branch)
        openrouter_api_key: OpenRouter key (never used on this branch)
        albert_api_key: The caller's Albert key
        albert_usage_ledger: Usage ledger of the request, or ``None``

    Returns:
        Structured result, or "NA"

    Raises:
        AlbertAuthError: Account error (aborts the citation job)
        ValueError: Any other failure ("Citation filtering failed: …")
    """
    try:
        response_text = await _albert_citation_call(
            prompt,
            model,
            temperature=0.2,
            albert_api_key=albert_api_key,
            albert_usage_ledger=albert_usage_ledger
        )
        parsed = _parse_llm_response(response_text)
        if parsed == "NA":
            return "NA"
        return _finalize_filter_result(parsed, citation)
    except ALBERT_ACCOUNT_ERRORS:
        raise
    except Exception as e:
        logger.error(f"Citation filtering failed (Albert): {str(e)}")
        raise ValueError(f"Citation filtering failed: {str(e)}") from e


def infer_item_type_from_source(source: Optional[str]) -> str:
    """
    Infer Zotero itemType from publication source string using heuristics.

    Args:
        source: Publication source (e.g., "Nature", "IEEE Conference", "arXiv")

    Returns:
        Inferred itemType (e.g., "journalArticle", "conferencePaper")

    Examples:
        >>> infer_item_type_from_source("Proceedings of ACL 2024")
        'conferencePaper'
        >>> infer_item_type_from_source("Journal of Machine Learning Research")
        'journalArticle'
        >>> infer_item_type_from_source("arXiv preprint")
        'preprint'
    """
    if not source:
        return "journalArticle"  # Default

    source_lower = source.lower()

    # Preprint indicators
    if any(x in source_lower for x in ["arxiv", "biorxiv", "medrxiv", "ssrn", "hal", "preprint"]):
        return "preprint"

    # Conference indicators
    if any(x in source_lower for x in ["proceedings", "conference", "symposium", "workshop", "acl", "nips", "icml", "cvpr"]):
        return "conferencePaper"

    # Book indicators
    if any(x in source_lower for x in ["springer", "book", "chapter"]):
        return "book"

    # Report indicators
    if any(x in source_lower for x in ["technical report", "tech report", "working paper"]):
        return "report"

    # Thesis indicators
    if any(x in source_lower for x in ["thesis", "dissertation"]):
        return "thesis"

    # Journal indicators (default for most academic sources)
    if any(x in source_lower for x in ["journal", "transactions", "letters", "review", "science", "nature"]):
        return "journalArticle"

    # Default fallback
    return "journalArticle"


def build_basic_zotero_item(citation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construit un item Zotero basique à partir des données PoP brutes (sans LLM).

    Utilise :
    - infer_item_type_from_source() pour deviner itemType
    - parse_authors_list() pour parser les auteurs
    - Mapping direct des champs PoP disponibles

    Args:
        citation: Dict contenant les champs PoP (title, authors, year, source, etc.)

    Returns:
        Dict Zotero item compatible avec create_or_update_item()

    Example:
        >>> citation = {"title": "Test", "authors": ["Smith J"], "year": 2024, "source": "Nature"}
        >>> zotero_item = build_basic_zotero_item(citation)
        >>> zotero_item["itemType"]
        'journalArticle'
    """
    # 1. Inférer itemType depuis le champ 'source'
    source = citation.get("source", "")
    item_type = infer_item_type_from_source(source)

    # 2. Parser les auteurs
    authors_raw = citation.get("authors", [])
    if isinstance(authors_raw, str):
        authors_raw = [authors_raw]
    creators = parse_authors_list(authors_raw)

    # 3. Construire l'item Zotero
    zotero_item = {
        "itemType": item_type,
        "title": citation.get("title", "Untitled"),
        "creators": creators,
        "date": str(citation.get("year", "")) if citation.get("year") else "",
        "url": citation.get("article_url") or citation.get("fulltext_url") or "",
        "DOI": citation.get("doi", ""),
        "abstractNote": citation.get("abstract", ""),
        "extra": f"citation:{citation.get('cites', 0)}",  # Préserver le nombre de citations
    }

    # 4. Ajouter publicationTitle si pertinent
    if source and item_type in ["journalArticle", "conferencePaper"]:
        zotero_item["publicationTitle"] = source

    # 5. Nettoyer champs vides
    zotero_item = {k: v for k, v in zotero_item.items() if v}

    # 6. Sanitize selon itemType (utiliser sanitize_zotero_item existant)
    zotero_item = sanitize_zotero_item(zotero_item)

    return zotero_item


def parse_author_name(author_str: str) -> ZoteroCreator:
    """
    Parse author name string into Zotero creator format.

    Handles various formats:
    - "Smith, J." → {"firstName": "J.", "lastName": "Smith"}
    - "H Lin" → {"firstName": "H", "lastName": "Lin"}
    - "Jean Dupont" → {"firstName": "Jean", "lastName": "Dupont"}

    Args:
        author_str: Author name string

    Returns:
        ZoteroCreator with parsed first/last names

    Examples:
        >>> parse_author_name("Smith, J.")
        ZoteroCreator(firstName="J.", lastName="Smith")
        >>> parse_author_name("H Lin")
        ZoteroCreator(firstName="H", "lastName"="Lin")
    """
    author_str = author_str.strip()

    # Handle "LastName, FirstName" format
    if "," in author_str:
        parts = author_str.split(",", 1)
        last_name = parts[0].strip()
        first_name = parts[1].strip()
        return ZoteroCreator(creatorType="author", firstName=first_name, lastName=last_name)

    # Handle "FirstName LastName" format
    parts = author_str.split()
    if len(parts) >= 2:
        first_name = " ".join(parts[:-1])
        last_name = parts[-1]
        return ZoteroCreator(creatorType="author", firstName=first_name, lastName=last_name)

    # Single name (use as lastName with placeholder firstName to satisfy validation)
    return ZoteroCreator(creatorType="author", firstName="-", lastName=author_str)


def parse_authors_list(authors: List[str]) -> List[Dict[str, str]]:
    """
    Parse list of author strings into Zotero creators format.

    Args:
        authors: List of author name strings

    Returns:
        List of creator dictionaries

    Examples:
        >>> parse_authors_list(["Smith, J.", "Doe, A."])
        [{"firstName": "J.", "lastName": "Smith", "creatorType": "author"},
         {"firstName": "A.", "lastName": "Doe", "creatorType": "author"}]
    """
    creators = []
    for author in authors:
        if author and author.strip():
            try:
                creator = parse_author_name(author)
                creators.append(creator.dict())
            except Exception as e:
                logger.warning(f"Failed to parse author '{author}': {e}")
                continue
    return creators if creators else [{"creatorType": "author", "firstName": "", "lastName": "Unknown"}]
