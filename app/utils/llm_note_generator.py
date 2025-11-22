"""
LLM-based reading note generator for Zotero integration.

This module generates structured reading notes using LLM APIs (OpenAI or OpenRouter)
and includes unique sentinels for idempotence.
"""

import os
import uuid
import logging
import html as html_module
from typing import Dict, Tuple, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Sentinel prefix for idempotence
SENTINEL_PREFIX = "ragpy-note-id:"


def _get_llm_clients():
    """
    Get LLM clients by loading API keys fresh from environment.
    This ensures that keys added after server start are picked up.

    Returns:
        Tuple of (openai_client, openrouter_client, default_model)
    """
    # Reload .env to pick up any changes
    load_dotenv(override=True)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    default_model = os.getenv("OPENROUTER_DEFAULT_MODEL", "gpt-4o-mini")

    openai_client = None
    openrouter_client = None

    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)
        logger.debug("OpenAI client initialized for note generation")

    if openrouter_api_key:
        openrouter_client = OpenAI(
            api_key=openrouter_api_key,
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


def _load_prompt_template(extended_analysis: bool = True) -> str:
    """
    Load the prompt template from zotero_prompt.md or zotero_prompt_short.md file.

    Args:
        extended_analysis: If True, load exhaustive analysis template (zotero_prompt.md)
                          If False, load short summary template (zotero_prompt_short.md)

    Returns:
        Prompt template string with placeholders

    Raises:
        FileNotFoundError: If the prompt file is not found
    """
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Choose template based on analysis mode
    template_filename = "zotero_prompt.md" if extended_analysis else "zotero_prompt_short.md"
    prompt_file = os.path.join(current_dir, template_filename)

    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            template = f.read()
        logger.info(f"Loaded prompt template from {prompt_file} (extended: {extended_analysis})")
        return template
    except FileNotFoundError:
        logger.error(f"Prompt template not found at {prompt_file}")
        raise


def _build_prompt(metadata: Dict, text_content: str, language: str, extended_analysis: bool = True) -> str:
    """
    Build the LLM prompt by loading template and replacing placeholders.

    Args:
        metadata: Dictionary with item metadata
        text_content: Full text content (texteocr)
        language: Target language code
        extended_analysis: If True, use full text and exhaustive template. If False, limit text and use short template.

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

    # Limit text based on analysis mode
    if extended_analysis:
        # Use full text for exhaustive analysis
        text_limited = safe_str(text_content if text_content else None, "Non disponible")
    else:
        # Limit to 8000 characters for quick summary
        text_limited = safe_str(text_content[:8000] if text_content else None, "Non disponible")

    abstract_text = abstract if abstract else "Non disponible"

    try:
        # Load template from file
        template = _load_prompt_template(extended_analysis=extended_analysis)

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

        logger.debug(f"Built prompt from template for: {title}")
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


def _generate_with_llm(prompt: str, model: str = None, temperature: float = 0.2, extended_analysis: bool = True) -> str:
    """
    Generate note content using LLM.

    Args:
        prompt: The prompt to send to the LLM
        model: Model name (e.g., "gpt-4o-mini" or "openai/gemini-2.5-flash").
               If None, uses OPENROUTER_DEFAULT_MODEL from .env
        temperature: Sampling temperature (0.0 to 1.0)
        extended_analysis: If True, use max_tokens=16000. If False, use max_tokens=2000.

    Returns:
        Generated HTML content

    Raises:
        ValueError: If no LLM client is available
        Exception: If the API call fails
    """
    # Get fresh clients from environment
    openai_client, openrouter_client, default_model = _get_llm_clients()

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

    # Set max_tokens based on analysis mode
    max_tokens = 16000 if extended_analysis else 2000

    # Make the API call
    try:
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

        content = response.choices[0].message.content.strip()
        logger.debug(f"Generated note content (length: {len(content)} chars)")
        return content

    except Exception as e:
        logger.error(f"Error calling LLM API: {e}")
        raise


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
    model: str = None,
    use_llm: bool = True,
    extended_analysis: bool = True
) -> Tuple[str, str]:
    """
    Build a reading note in HTML format with a unique sentinel.

    This is the main entry point for generating reading notes.

    Args:
        metadata: Dictionary with item metadata (title, authors, abstract, etc.)
        text_content: Full text content (texteocr). If None, will use abstract only.
        model: LLM model to use. If None, uses OPENROUTER_DEFAULT_MODEL from .env.
               Examples: "gpt-4o-mini", "openai/gemini-2.5-flash"
        use_llm: Whether to use LLM or fallback to template (default: True)
        extended_analysis: If True, generate exhaustive analysis (8000-12000 words).
                          If False, generate quick summary (200-300 words).

    Returns:
        Tuple of (sentinel, note_html):
        - sentinel: Unique identifier (e.g., "ragpy-note-id:uuid")
        - note_html: Complete HTML with sentinel comment

    Example:
        >>> metadata = {
        ...     "title": "Machine Learning for NLP",
        ...     "authors": "Smith, J.",
        ...     "date": "2024",
        ...     "abstract": "This paper presents...",
        ...     "language": "en"
        ... }
        >>> sentinel, html = build_note_html(metadata, text_content="Full text...", extended_analysis=True)
        >>> print(sentinel)
        ragpy-note-id:abc123...
    """
    # Get fresh clients from environment
    openai_client, openrouter_client, default_model = _get_llm_clients()

    # Use default model if no model specified
    if not model:
        model = default_model
        logger.info(f"No model specified, using default: {model}")

    # Detect target language
    language = _detect_language(metadata)
    logger.info(f"Generating note in language: {language}")

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
                prompt = _build_prompt(metadata, content, language, extended_analysis=extended_analysis)
                body_html = _generate_with_llm(prompt, model=model, extended_analysis=extended_analysis)
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
