"""
LLM Note Generator
==================

This module handles the generation of structured reading notes using Large Language Models
(LLMs). It integrates with OpenAI and OpenRouter APIs to analyze document content and
produce academic-style reading notes.

Key Features:
- Prompt Engineering: Dynamically builds prompts based on document metadata and language.
- Multi-Provider Support: Supports both OpenAI and OpenRouter.
- Concurrency Control: Uses a global semaphore to limit concurrent API calls.
- Idempotence: Generates unique sentinels to track generated notes.
- Fallback Mechanism: Provides a template-based fallback if LLM generation fails.
"""

import os
import uuid
import logging
import asyncio
import html as html_module
from typing import Dict, Tuple, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Sentinel prefix for idempotence
SENTINEL_PREFIX = "ragpy-note-id:"

# =============================================================================
# Note Mode Configuration
# =============================================================================

# Mapping of note modes to template files
TEMPLATE_MAP = {
    "extended": "zotero_prompt.md",
    "short": "zotero_prompt_short.md",
    "pedagogique": "zotero_prompt_pedagogique.md",
    "evaluation": "zotero_prompt_evaluation.md"
}

# Mapping of note modes to display prefixes (for Zotero note identification)
NOTE_MODE_PREFIX = {
    "extended": "[FICHE]",
    "pedagogique": "[CLAIR]",
    "evaluation": "[EVAL]",
    "short": ""  # No prefix for short summaries
}

# Mapping of note modes to max_tokens
NOTE_MODE_MAX_TOKENS = {
    "extended": 16000,
    "short": 2000,
    "pedagogique": 10000,
    "evaluation": 10000
}

# Display names for UI
NOTE_MODE_DISPLAY = {
    "extended": "Fiche de lecture [FICHE]",
    "pedagogique": "Fiche pédagogique [CLAIR]",
    "evaluation": "Grille d'évaluation [EVAL]",
    "short": "Résumé court"
}

# =============================================================================
# Global LLM Semaphore for Concurrency Control
# =============================================================================
_llm_semaphore: Optional[asyncio.Semaphore] = None


def get_llm_semaphore() -> asyncio.Semaphore:
    """
    Get the global LLM semaphore (lazy initialization).

    This semaphore limits concurrent LLM API calls across ALL users
    to prevent API rate limiting and resource exhaustion.

    Returns:
        asyncio.Semaphore configured with MAX_CONCURRENT_LLM_CALLS
    """
    global _llm_semaphore
    if _llm_semaphore is None:
        max_concurrent = int(os.getenv('MAX_CONCURRENT_LLM_CALLS', '5'))
        _llm_semaphore = asyncio.Semaphore(max_concurrent)
        logger.info(f"LLM semaphore initialized: max {max_concurrent} concurrent calls")
    return _llm_semaphore


def _get_llm_clients(
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    openrouter_model: Optional[str] = None
) -> Tuple[Optional[OpenAI], Optional[OpenAI], str]:
    """
    Initializes and returns LLM clients.

    This function accepts API keys as parameters for secure credential handling.
    If no credentials are passed, falls back to environment variables for
    backward compatibility.

    Args:
        openai_api_key: OpenAI API key. If None, uses OPENAI_API_KEY from env.
        openrouter_api_key: OpenRouter API key. If None, uses OPENROUTER_API_KEY from env.
        openrouter_model: Default OpenRouter model. If None, uses OPENROUTER_DEFAULT_MODEL from env.

    Returns:
        A tuple containing:
        - openai_client (Optional[OpenAI]): An initialized OpenAI client if the
          API key is available, otherwise None.
        - openrouter_client (Optional[OpenAI]): An initialized client for OpenRouter
          if the API key is available, otherwise None.
        - default_model (str): The default model identifier.

    Security Note:
        When using this module from authenticated endpoints, always pass
        credentials explicitly from the user's credential store to prevent
        non-admin users from accessing .env credentials.
    """
    # Use provided credentials or fall back to environment
    if openai_api_key is None or openrouter_api_key is None:
        # Reload .env only if falling back to environment
        load_dotenv(override=True)

    _openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    _openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
    default_model = openrouter_model or os.getenv("OPENROUTER_DEFAULT_MODEL", "gpt-4o-mini")

    openai_client = None
    openrouter_client = None

    if _openai_api_key:
        openai_client = OpenAI(api_key=_openai_api_key)
        logger.debug("OpenAI client initialized for note generation")

    if _openrouter_api_key:
        openrouter_client = OpenAI(
            api_key=_openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        logger.debug("OpenRouter client initialized for note generation")

    return openai_client, openrouter_client, default_model


def _detect_language(metadata: Dict) -> str:
    """
    Detect the target language for the note based on metadata.

    Args:
        metadata: Dictionary with item metadata (should contain 'language' field)

    Returns:
        Language code (e.g., "fr", "en", "es")
    """
    # Check if language is explicitly specified in metadata
    lang = metadata.get("language", "").lower()

    if lang:
        # Extract language code (e.g., "en-US" -> "en")
        lang_code = lang.split("-")[0].split("_")[0]
        if lang_code in ("fr", "en", "es", "de", "it", "pt"):
            return lang_code

    # Default to French
    return "fr"


def _load_prompt_template(mode: str = "extended") -> str:
    """
    Load the prompt template based on the specified mode.

    Args:
        mode: Note generation mode. One of:
              - "extended": Full analysis template (zotero_prompt.md)
              - "short": Quick summary template (zotero_prompt_short.md)
              - "pedagogique": Pedagogical template for L3 students (zotero_prompt_pedagogique.md)
              - "evaluation": Peer review evaluation grid (zotero_prompt_evaluation.md)

    Returns:
        Prompt template string with placeholders

    Raises:
        FileNotFoundError: If the prompt file is not found
        ValueError: If the mode is not recognized
    """
    # Validate mode
    if mode not in TEMPLATE_MAP:
        logger.warning(f"Unknown mode '{mode}', falling back to 'extended'")
        mode = "extended"

    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Choose template based on mode
    template_filename = TEMPLATE_MAP[mode]
    prompt_file = os.path.join(current_dir, template_filename)

    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            template = f.read()
        logger.info(f"Loaded prompt template from {prompt_file} (mode: {mode})")
        return template
    except FileNotFoundError:
        logger.error(f"Prompt template not found at {prompt_file}")
        raise


def _add_note_prefix(html_content: str, mode: str) -> str:
    """
    Add a mode prefix to the first h2 heading for Zotero note identification.

    This allows users to quickly identify the type of note in Zotero:
    - [FICHE] for extended analysis
    - [CLAIR] for pedagogical notes
    - [EVAL] for evaluation grids
    - (no prefix for short summaries)

    Args:
        html_content: The HTML content of the generated note
        mode: The note generation mode

    Returns:
        HTML content with the prefix injected in the first h2 heading

    Example:
        >>> html = "<h2>Smith (2024). Title...</h2><p>Content...</p>"
        >>> _add_note_prefix(html, "pedagogique")
        '<h2>[CLAIR] Smith (2024). Title...</h2><p>Content...</p>'
    """
    import re

    prefix = NOTE_MODE_PREFIX.get(mode, "")
    if not prefix:
        return html_content

    # Find the first <h2> and inject the prefix
    pattern = r'(<h2>)(.*?)(</h2>)'
    replacement = rf'\1{prefix} \2\3'
    return re.sub(pattern, replacement, html_content, count=1)


def _build_prompt(metadata: Dict, text_content: str, language: str, mode: str = "extended") -> str:
    """
    Build the LLM prompt by loading template and replacing placeholders.

    Args:
        metadata: Dictionary with item metadata
        text_content: Full text content (texteocr)
        language: Target language code
        mode: Note generation mode. One of:
              - "extended": Full analysis (no text limit)
              - "short": Quick summary (8000 char limit)
              - "pedagogique": Pedagogical note for L3 students (no text limit)
              - "evaluation": Peer review evaluation grid (no text limit)

    Returns:
        Formatted prompt string
    """
    # Extract key metadata and convert to strings (handle pandas NaN/float values)
    def safe_str(value, default="N/A"):
        """Convert value to string, handling NaN and None."""
        if value is None or value == "":
            return default
        # Check for pandas NaN (float type)
        if isinstance(value, float):
            import math
            if math.isnan(value):
                return default
        return str(value)

    title = safe_str(metadata.get("title"), "Sans titre")
    authors = safe_str(metadata.get("authors"), "N/A")
    date = safe_str(metadata.get("date"), "N/A")
    abstract = safe_str(metadata.get("abstract"), "")
    doi = safe_str(metadata.get("doi"), "")
    url = safe_str(metadata.get("url"), "")
    problematique = safe_str(metadata.get("problematique"), "Non spécifiée")

    # Language-specific names
    lang_instructions = {
        "fr": "français",
        "en": "English",
        "es": "español",
        "de": "Deutsch",
        "it": "italiano",
        "pt": "português"
    }
    target_lang = lang_instructions.get(language, "français")

    # Limit text based on mode (only 'short' mode has text limit)
    if mode == "short":
        # Limit to 8000 characters for quick summary
        text_limited = safe_str(text_content[:8000] if text_content else None, "Non disponible")
    else:
        # Use full text for all other modes (extended, pedagogique, evaluation)
        text_limited = safe_str(text_content if text_content else None, "Non disponible")

    abstract_text = abstract if abstract else "Non disponible"

    try:
        # Load template from file based on mode
        template = _load_prompt_template(mode=mode)

        # Replace placeholders
        prompt = template.replace("{TITLE}", title)
        prompt = prompt.replace("{AUTHORS}", authors)
        prompt = prompt.replace("{DATE}", date)
        prompt = prompt.replace("{DOI}", doi)
        prompt = prompt.replace("{URL}", url)
        prompt = prompt.replace("{PROBLEMATIQUE}", problematique)
        prompt = prompt.replace("{ABSTRACT}", abstract_text)
        prompt = prompt.replace("{TEXT}", text_limited)
        prompt = prompt.replace("{LANGUAGE}", target_lang)

        logger.debug(f"Built prompt from template for: {title} (mode: {mode})")
        return prompt

    except FileNotFoundError:
        # Fallback to hardcoded prompt if file not found
        logger.warning("Prompt template file not found, using fallback hardcoded prompt")
        prompt = f"""Tu es un assistant spécialisé dans la rédaction de fiches de lecture académiques.

CONTEXTE :
Titre : {title}
Auteurs : {authors}
Date : {date}
DOI : {doi}
URL : {url}
Problématique de recherche : {problematique}

Résumé (si disponible) :
{abstract_text}

TEXTE COMPLET :
{text_limited}

CONSIGNE :
Rédige une fiche de lecture structurée en {target_lang}, au format HTML simplifié (balises : <p>, <strong>, <em>, <ul>, <li>).

STRUCTURE REQUISE :
1. **Référence bibliographique** : Titre, auteurs, date, lien si disponible
2. **Problématique** : Question(s) de recherche ou objectif principal
3. **Méthodologie** : Approche, données, méthodes utilisées
4. **Résultats clés** : Principales conclusions ou découvertes
5. **Limites et perspectives** : Points faibles, questions ouvertes

CONTRAINTES :
- Ton : Neutre, informatif, académique
- Format : HTML propre (pas de <html>, <head>, <body>)
- Concentre-toi sur les points essentiels

Commence directement par le contenu HTML, sans préambule."""

        return prompt


def _generate_with_llm(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    mode: str = "extended",
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None
) -> str:
    """
    Generate note content using LLM.

    Args:
        prompt: The prompt to send to the LLM
        model: Model name (e.g., "gpt-4o-mini" or "google/gemini-2.5-flash").
               If None, uses OPENROUTER_DEFAULT_MODEL from .env
        temperature: Sampling temperature (0.0 to 1.0)
        mode: Note generation mode. Determines max_tokens:
              - "extended": 16000 tokens
              - "short": 2000 tokens
              - "pedagogique": 10000 tokens
              - "evaluation": 10000 tokens
        openai_api_key: Optional OpenAI API key. If None, uses environment.
        openrouter_api_key: Optional OpenRouter API key. If None, uses environment.

    Returns:
        Generated HTML content

    Raises:
        ValueError: If no LLM client is available
        Exception: If the API call fails
    """
    # Get clients with provided credentials or from environment
    openai_client, openrouter_client, default_model = _get_llm_clients(
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key
    )

    # Use default model if no model specified
    if not model:
        model = default_model
        logger.info(f"No model specified, using default: {model}")

    # Detect which client to use based on model format
    use_openrouter = "/" in model  # OpenRouter models have format "provider/model"

    if use_openrouter:
        if not openrouter_client:
            logger.warning(f"OpenRouter model '{model}' requested but client not initialized. Falling back to OpenAI.")
            if not openai_client:
                raise ValueError("No LLM client available (neither OpenAI nor OpenRouter). Check your API keys in Settings.")
            active_client = openai_client
            model = "gpt-4o-mini"
        else:
            active_client = openrouter_client
            logger.info(f"Using OpenRouter with model: {model}")
    else:
        if not openai_client:
            raise ValueError("OpenAI client not initialized (OPENAI_API_KEY missing). Check your API keys in Settings.")
        active_client = openai_client
        logger.info(f"Using OpenAI with model: {model}")

    # Set max_tokens based on mode
    max_tokens = NOTE_MODE_MAX_TOKENS.get(mode, 16000)

    # Retry configuration: 1 retry with 2 second delay
    max_attempts = 2
    retry_delay = 2  # seconds
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"LLM API call attempt {attempt}/{max_attempts} for model: {model}")

            response = active_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un assistant spécialisé en rédaction de fiches de lecture académiques."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Validate response structure
            if not response or not response.choices:
                logger.error(f"LLM API returned empty response or no choices. Response: {response}")
                raise ValueError(f"LLM API returned invalid response (no choices). Model: {model}")

            if not response.choices[0].message or response.choices[0].message.content is None:
                logger.error(f"LLM API returned empty message content. Response: {response}")
                raise ValueError(f"LLM API returned empty content. Model: {model}")

            content = response.choices[0].message.content.strip()

            if not content:
                logger.warning(f"LLM returned empty string for model {model}")
                raise ValueError(f"LLM returned empty response. Model: {model}")

            logger.debug(f"Generated note content (length: {len(content)} chars)")
            return content

        except Exception as e:
            last_error = e
            logger.error(f"LLM API error (attempt {attempt}/{max_attempts}): {e}")

            if attempt < max_attempts:
                logger.info(f"Retrying in {retry_delay} seconds...")
                import time
                time.sleep(retry_delay)
            else:
                logger.error(f"All {max_attempts} attempts failed for model {model}")

    # If we get here, all retries failed
    if last_error:
        raise last_error
    raise RuntimeError(f"LLM API failed after {max_attempts} attempts")


def _fallback_template(metadata: Dict, language: str) -> str:
    """
    Generate a simple HTML template if LLM is unavailable.

    Args:
        metadata: Dictionary with item metadata
        language: Target language code

    Returns:
        HTML template string
    """
    # Helper to safely convert values to strings
    def safe_str(value, default="N/A"):
        """Convert value to string, handling NaN and None."""
        if value is None or value == "":
            return default
        # Check for pandas NaN (float type)
        if isinstance(value, float):
            import math
            if math.isnan(value):
                return default
        return str(value)

    title = html_module.escape(safe_str(metadata.get("title"), "Sans titre"))
    authors = html_module.escape(safe_str(metadata.get("authors"), "N/A"))
    date = html_module.escape(safe_str(metadata.get("date"), "N/A")[:10])
    abstract_raw = safe_str(metadata.get("abstract"), "")
    abstract = html_module.escape(abstract_raw[:1200] if abstract_raw else "")
    url = html_module.escape(safe_str(metadata.get("url") or metadata.get("doi"), ""))

    # Language-specific labels
    labels = {
        "fr": {
            "title": "Fiche de lecture",
            "ref": "Référence",
            "problem": "Problématique",
            "method": "Méthodologie",
            "results": "Résultats clés",
            "limits": "Limites",
            "abstract": "Résumé",
            "tbd": "à compléter"
        },
        "en": {
            "title": "Reading Note",
            "ref": "Reference",
            "problem": "Research Question",
            "method": "Methodology",
            "results": "Key Results",
            "limits": "Limitations",
            "abstract": "Abstract",
            "tbd": "to be completed"
        }
    }

    lang_labels = labels.get(language, labels["fr"])

    return f"""<p><em>Fiche générée automatiquement (template).</em></p>
<h3>{lang_labels["title"]}</h3>
<p><strong>{lang_labels["ref"]} :</strong> {title} — {authors} — {date} — {url}</p>
<ul>
  <li><strong>{lang_labels["problem"]} :</strong> {lang_labels["tbd"]}</li>
  <li><strong>{lang_labels["method"]} :</strong> {lang_labels["tbd"]}</li>
  <li><strong>{lang_labels["results"]} :</strong> {lang_labels["tbd"]}</li>
  <li><strong>{lang_labels["limits"]} :</strong> {lang_labels["tbd"]}</li>
</ul>
<p><strong>{lang_labels["abstract"]} :</strong> {abstract}</p>"""


def build_note_html(
    metadata: Dict,
    text_content: Optional[str] = None,
    model: Optional[str] = None,
    use_llm: bool = True,
    mode: str = "extended",
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None
) -> Tuple[str, str]:
    """
    Build a reading note in HTML format with a unique sentinel.

    This is the main entry point for generating reading notes.

    Args:
        metadata: Dictionary with item metadata (title, authors, abstract, etc.)
        text_content: Full text content (texteocr). If None, will use abstract only.
        model: LLM model to use. If None, uses OPENROUTER_DEFAULT_MODEL from .env.
               Examples: "gpt-4o-mini", "google/gemini-2.5-flash"
        use_llm: Whether to use LLM or fallback to template (default: True)
        mode: Note generation mode. One of:
              - "extended": Full analysis [FICHE] (2500-3000 words)
              - "short": Quick summary (400-600 words, no prefix)
              - "pedagogique": Pedagogical note [CLAIR] for L3 students (2400-2800 words)
              - "evaluation": Peer review evaluation grid [EVAL] (2200-2750 words)
        openai_api_key: Optional OpenAI API key for secure credential passing.
                        If None, falls back to environment variable.
        openrouter_api_key: Optional OpenRouter API key for secure credential passing.
                           If None, falls back to environment variable.

    Returns:
        Tuple of (sentinel, note_html):
        - sentinel: Unique identifier (e.g., "ragpy-note-id:uuid")
        - note_html: Complete HTML with sentinel comment and mode prefix

    Example:
        >>> metadata = {
        ...     "title": "Machine Learning for NLP",
        ...     "authors": "Smith, J.",
        ...     "date": "2024",
        ...     "abstract": "This paper presents...",
        ...     "language": "en"
        ... }
        >>> sentinel, html = build_note_html(metadata, text_content="Full text...", mode="pedagogique")
        >>> print(sentinel)
        ragpy-note-id:abc123...
        >>> "[CLAIR]" in html
        True
    """
    # Validate mode
    if mode not in TEMPLATE_MAP:
        logger.warning(f"Unknown mode '{mode}', falling back to 'extended'")
        mode = "extended"

    # Get clients with provided credentials or from environment
    openai_client, openrouter_client, default_model = _get_llm_clients(
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key
    )

    # Use default model if no model specified
    if not model:
        model = default_model
        logger.info(f"No model specified, using default: {model}")

    # Detect target language
    language = _detect_language(metadata)
    logger.info(f"Generating note in language: {language} (mode: {mode})")

    # Generate the note body
    if use_llm and (openai_client or openrouter_client):
        try:
            # Use text_content if available, otherwise use abstract
            content = text_content or metadata.get("abstract", "")

            if not content:
                logger.warning("No text content or abstract available, using template fallback")
                body_html = _fallback_template(metadata, language)
            else:
                # Build prompt and generate with LLM
                prompt = _build_prompt(metadata, content, language, mode=mode)
                body_html = _generate_with_llm(
                    prompt,
                    model=model,
                    mode=mode,
                    openai_api_key=openai_api_key,
                    openrouter_api_key=openrouter_api_key
                )
                # Add mode prefix to first h2 heading
                body_html = _add_note_prefix(body_html, mode)
        except Exception as e:
            logger.error(f"LLM generation failed, using template fallback: {e}")
            body_html = _fallback_template(metadata, language)
    else:
        logger.info("LLM not available or disabled, using template")
        body_html = _fallback_template(metadata, language)

    # Generate unique sentinel
    sentinel = f"{SENTINEL_PREFIX}{uuid.uuid4()}"

    # Build complete HTML with sentinel comment
    note_html = f"<!-- {sentinel} -->\n{body_html}"

    logger.info(f"Generated note with sentinel: {sentinel}")
    return sentinel, note_html


def build_abstract_text(
    metadata: Dict,
    text_content: Optional[str] = None,
    model: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None
) -> str:
    """
    Build an abstract/summary text to enrich Zotero's abstractNote field.

    This function generates a plain text summary (not HTML) that can be
    appended to the existing abstract in Zotero.

    Args:
        metadata: Dictionary with item metadata (title, authors, abstract, etc.)
        text_content: Full text content (texteocr). If None, will use abstract only.
        model: LLM model to use. If None, uses OPENROUTER_DEFAULT_MODEL from .env.
        openai_api_key: Optional OpenAI API key for secure credential passing.
                        If None, falls back to environment variable.
        openrouter_api_key: Optional OpenRouter API key for secure credential passing.
                           If None, falls back to environment variable.

    Returns:
        Plain text summary string (200-350 words)

    Example:
        >>> metadata = {
        ...     "title": "Machine Learning for NLP",
        ...     "authors": "Smith, J.",
        ...     "date": "2024",
        ...     "abstract": "This paper presents...",
        ...     "language": "en"
        ... }
        >>> summary = build_abstract_text(metadata, text_content="Full text...")
        >>> print(summary)
        This study investigates...
    """
    # Get clients with provided credentials or from environment
    openai_client, openrouter_client, default_model = _get_llm_clients(
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key
    )

    # Use default model if no model specified
    if not model:
        model = default_model
        logger.info(f"No model specified, using default: {model}")

    # Detect target language
    language = _detect_language(metadata)
    logger.info(f"Generating abstract summary in language: {language}")

    # Check if LLM is available
    if not (openai_client or openrouter_client):
        logger.error("No LLM client available for abstract generation")
        raise ValueError("No LLM client available (neither OpenAI nor OpenRouter). Check your API keys in Settings.")

    # Use text_content if available, otherwise use abstract
    content = text_content or metadata.get("abstract", "")

    if not content:
        logger.warning("No text content or abstract available for summary generation")
        raise ValueError("No text content available to generate summary")

    try:
        # Build prompt using the short template (mode="short")
        prompt = _build_prompt(metadata, content, language, mode="short")

        # Generate with LLM (use smaller max_tokens for plain text summary)
        summary = _generate_with_llm(
            prompt,
            model=model,
            mode="short",
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key
        )

        # Clean up the response - remove any HTML tags that might have slipped through
        import re
        summary = re.sub(r'<[^>]+>', '', summary)
        summary = summary.strip()

        logger.info(f"Generated abstract summary (length: {len(summary)} chars)")
        return summary

    except Exception as e:
        logger.error(f"Abstract generation failed: {e}")
        raise


def sentinel_in_html(html_text: str) -> bool:
    """
    Check if a sentinel is present in HTML text.

    Args:
        html_text: HTML text to check

    Returns:
        True if a ragpy sentinel is found, False otherwise
    """
    return SENTINEL_PREFIX in (html_text or "")


def extract_sentinel_from_html(html_text: str) -> Optional[str]:
    """
    Extract the sentinel ID from HTML text.

    Args:
        html_text: HTML text containing a sentinel

    Returns:
        The sentinel string if found, None otherwise
    """
    if not html_text:
        return None

    # Look for the sentinel pattern in HTML comments
    import re
    pattern = rf"<!--\s*({SENTINEL_PREFIX}[a-f0-9\-]+)\s*-->"
    match = re.search(pattern, html_text)

    if match:
        return match.group(1)

    return None


# =============================================================================
# Async Wrapper Functions (with global concurrency control)
# =============================================================================

async def build_note_html_async(
    metadata: Dict,
    text_content: Optional[str] = None,
    model: Optional[str] = None,
    use_llm: bool = True,
    mode: str = "extended",
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None
) -> Tuple[str, str]:
    """
    Async version of build_note_html with global concurrency control.

    Uses a semaphore to limit concurrent LLM calls across all users.
    See build_note_html for full documentation.

    Args:
        metadata: Dictionary with item metadata (title, authors, abstract, etc.)
        text_content: Full text content (texteocr). If None, will use abstract only.
        model: LLM model to use. If None, uses OPENROUTER_DEFAULT_MODEL from .env.
        use_llm: Whether to use LLM or fallback to template (default: True)
        mode: Note generation mode. One of:
              - "extended": Full analysis [FICHE]
              - "short": Quick summary (no prefix)
              - "pedagogique": Pedagogical note [CLAIR]
              - "evaluation": Peer review evaluation grid [EVAL]
        openai_api_key: Optional OpenAI API key for secure credential passing.
        openrouter_api_key: Optional OpenRouter API key for secure credential passing.
    """
    semaphore = get_llm_semaphore()

    async with semaphore:
        remaining = semaphore._value
        logger.debug(f"Acquired LLM slot ({remaining} slots remaining)")

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: build_note_html(
                    metadata=metadata,
                    text_content=text_content,
                    model=model,
                    use_llm=use_llm,
                    mode=mode,
                    openai_api_key=openai_api_key,
                    openrouter_api_key=openrouter_api_key
                )
            )
            return result
        finally:
            logger.debug("Released LLM slot")


async def build_abstract_text_async(
    metadata: Dict,
    text_content: Optional[str] = None,
    model: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None
) -> str:
    """
    Async version of build_abstract_text with global concurrency control.

    Uses a semaphore to limit concurrent LLM calls across all users.
    See build_abstract_text for full documentation.

    Args:
        openai_api_key: Optional OpenAI API key for secure credential passing.
        openrouter_api_key: Optional OpenRouter API key for secure credential passing.
    """
    semaphore = get_llm_semaphore()

    async with semaphore:
        remaining = semaphore._value
        logger.debug(f"Acquired LLM slot ({remaining} slots remaining)")

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: build_abstract_text(
                    metadata=metadata,
                    text_content=text_content,
                    model=model,
                    openai_api_key=openai_api_key,
                    openrouter_api_key=openrouter_api_key
                )
            )
            return result
        finally:
            logger.debug("Released LLM slot")
