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

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_DEFAULT_MODEL = os.getenv("OPENROUTER_DEFAULT_MODEL", "gpt-4o-mini")

# Initialize clients
openai_client = None
openrouter_client = None

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client initialized for note generation")

if OPENROUTER_API_KEY:
    openrouter_client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
    logger.info("OpenRouter client initialized for note generation")


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


def _load_prompt_template() -> str:
    """
    Load the prompt template from zotero_prompt.md file.

    Returns:
        Prompt template string with placeholders

    Raises:
        FileNotFoundError: If the prompt file is not found
    """
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file = os.path.join(current_dir, "zotero_prompt.md")

    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            template = f.read()
        logger.info(f"Loaded prompt template from {prompt_file}")
        return template
    except FileNotFoundError:
        logger.error(f"Prompt template not found at {prompt_file}")
        raise


def _build_prompt(metadata: Dict, text_content: str, language: str) -> str:
    """
    Build the LLM prompt by loading template and replacing placeholders.

    Args:
        metadata: Dictionary with item metadata
        text_content: Full text content (texteocr)
        language: Target language code

    Returns:
        Formatted prompt string
    """
    # Extract key metadata
    title = metadata.get("title", "Sans titre")
    authors = metadata.get("authors", "N/A")
    date = metadata.get("date", "N/A")
    abstract = metadata.get("abstract", "")
    doi = metadata.get("doi", "")
    url = metadata.get("url", "")

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

    # Limit text content to avoid token limits
    text_limited = text_content[:8000] if text_content else "Non disponible"
    abstract_text = abstract if abstract else "Non disponible"

    try:
        # Load template from file
        template = _load_prompt_template()

        # Replace placeholders
        prompt = template.replace("{TITLE}", title)
        prompt = prompt.replace("{AUTHORS}", authors)
        prompt = prompt.replace("{DATE}", date)
        prompt = prompt.replace("{DOI}", doi)
        prompt = prompt.replace("{URL}", url)
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
- Longueur : 200-300 mots maximum
- Ton : Neutre, informatif, académique
- Format : HTML propre (pas de <html>, <head>, <body>)
- Pas de citations directes longues
- Concentre-toi sur les points essentiels

Commence directement par le contenu HTML, sans préambule."""

        return prompt


def _generate_with_llm(prompt: str, model: str = None, temperature: float = 0.2) -> str:
    """
    Generate note content using LLM.

    Args:
        prompt: The prompt to send to the LLM
        model: Model name (e.g., "gpt-4o-mini" or "openai/gemini-2.5-flash").
               If None, uses OPENROUTER_DEFAULT_MODEL from .env
        temperature: Sampling temperature (0.0 to 1.0)

    Returns:
        Generated HTML content

    Raises:
        ValueError: If no LLM client is available
        Exception: If the API call fails
    """
    # Use OPENROUTER_DEFAULT_MODEL if no model specified
    if not model:
        model = OPENROUTER_DEFAULT_MODEL
        logger.info(f"No model specified, using default: {model}")

    # Detect which client to use based on model format
    use_openrouter = "/" in model  # OpenRouter models have format "provider/model"

    if use_openrouter:
        if not openrouter_client:
            logger.warning(f"OpenRouter model '{model}' requested but client not initialized. Falling back to OpenAI.")
            if not openai_client:
                raise ValueError("No LLM client available (neither OpenAI nor OpenRouter)")
            active_client = openai_client
            model = "gpt-4o-mini"
        else:
            active_client = openrouter_client
            logger.info(f"Using OpenRouter with model: {model}")
    else:
        if not openai_client:
            raise ValueError("OpenAI client not initialized (OPENAI_API_KEY missing)")
        active_client = openai_client
        logger.info(f"Using OpenAI with model: {model}")

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
            max_tokens=2000
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
    title = html_module.escape(metadata.get("title", "Sans titre"))
    authors = html_module.escape(metadata.get("authors", "N/A"))
    date = html_module.escape(str(metadata.get("date", "N/A"))[:10])
    abstract = html_module.escape((metadata.get("abstract", ""))[:1200])
    url = html_module.escape(metadata.get("url", metadata.get("doi", "")))

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
    use_llm: bool = True
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
        >>> sentinel, html = build_note_html(metadata, text_content="Full text...")
        >>> print(sentinel)
        ragpy-note-id:abc123...
    """
    # Use OPENROUTER_DEFAULT_MODEL if no model specified
    if not model:
        model = OPENROUTER_DEFAULT_MODEL
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
                prompt = _build_prompt(metadata, content, language)
                body_html = _generate_with_llm(prompt, model=model)
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
