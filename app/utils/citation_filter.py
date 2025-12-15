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

Author: RAGpy Team
Date: 2025-12-05
"""

import os
import json
import re
import asyncio
import logging
from typing import Dict, Union, List, Optional, Tuple, Any
from pathlib import Path
from pydantic import BaseModel, field_validator
from openai import OpenAI
from dotenv import load_dotenv

from app.utils.llm_note_generator import get_llm_semaphore, _get_llm_clients

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Default LLM model (uses OpenRouter format by default)
# Format "provider/model" → OpenRouter (e.g., google/gemini-2.5-flash)
# Format "model" → OpenAI direct (e.g., gpt-4o-mini)
DEFAULT_LLM_MODEL = os.getenv("OPENROUTER_DEFAULT_MODEL", "gpt-4o-mini")

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
    openrouter_api_key: Optional[str] = None
) -> str:
    """
    Call LLM API (synchronous wrapper for OpenAI/OpenRouter).

    Args:
        prompt: Formatted prompt
        model: Model identifier
        temperature: Sampling temperature (0-1)
        openai_api_key: Optional OpenAI API key (falls back to env for admin users)
        openrouter_api_key: Optional OpenRouter API key (falls back to env for admin users)

    Returns:
        Raw LLM response text

    Raises:
        ValueError: If API call fails or no clients available
    """
    openai_client, openrouter_client, default_model = _get_llm_clients(
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key
    )

    # Determine which client to use
    if "/" in model:  # OpenRouter format
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


async def pre_filter_citation(
    citation: Dict,
    project_name: str,
    project_description: str,
    collection_name: str,
    collection_description: str,
    model: str = DEFAULT_LLM_MODEL,
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None
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
        model: Modèle LLM à utiliser
        openai_api_key: Clé API OpenAI
        openrouter_api_key: Clé API OpenRouter

    Returns:
        True si la citation semble pertinente, False sinon

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
    openrouter_api_key: Optional[str] = None
) -> Union[Dict, str]:
    """
    Filter citation using LLM with global concurrency control.

    This function:
    1. Builds a prompt from template with all context
    2. Acquires global LLM semaphore
    3. Calls LLM API (with retry on failure)
    4. Parses and validates response
    5. Returns structured result or "NA"

    Args:
        citation: Citation dictionary (from PopCitation.dict())
        web_content: Extracted web content text
        web_content_source: Source type ("pdf", "html", "none")
        project_name: Research project name
        project_description: Project description
        collection_name: Target Zotero collection
        collection_description: Collection description
        model: LLM model identifier
        max_retries: Number of retries on failure
        openai_api_key: Optional OpenAI API key (for non-admin users)
        openrouter_api_key: Optional OpenRouter API key (for non-admin users)

    Returns:
        If relevant: Dictionary with keys:
            - relevance_score: int (0-100)
            - relevance_reason: str
            - zotero_item: dict (complete Zotero metadata)
        If not relevant: String "NA"

    Raises:
        ValueError: If LLM call fails after retries or response invalid

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

            except Exception as e:
                last_error = e
                logger.warning(f"Filter attempt {attempt + 1}/{max_retries + 1} failed: {str(e)}")
                if attempt < max_retries:
                    await asyncio.sleep(2)  # Wait before retry

        # All retries failed
        logger.error(f"Citation filtering failed after {max_retries + 1} attempts: {str(last_error)}")
        raise ValueError(f"Citation filtering failed: {str(last_error)}")


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
