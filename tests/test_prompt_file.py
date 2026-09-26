"""
Test simple pour vérifier que le fichier zotero_prompt.md est valide.

Run with: pytest tests/test_prompt_file.py
"""

from pathlib import Path

import pytest

PROMPT_FILE = Path(__file__).parent.parent / "app" / "utils" / "zotero_prompt.md"

REQUIRED_PLACEHOLDERS = [
    "{TITLE}",
    "{AUTHORS}",
    "{DATE}",
    "{DOI}",
    "{URL}",
    "{ABSTRACT}",
    "{TEXT}",
    "{LANGUAGE}",
    "{PROBLEMATIQUE}"
]


@pytest.fixture
def content():
    """Contenu du gabarit ``zotero_prompt.md`` (UTF-8)."""
    return PROMPT_FILE.read_text(encoding="utf-8")


def test_prompt_file_exists():
    """Vérifie que le fichier zotero_prompt.md existe."""
    assert PROMPT_FILE.exists(), f"Prompt file not found at {PROMPT_FILE}"


def test_prompt_file_readable():
    """Vérifie que le fichier peut être lu et n'est pas vide."""
    text = PROMPT_FILE.read_text(encoding="utf-8")
    assert len(text) > 0, "Prompt file is empty"


def test_prompt_placeholders(content):
    """Vérifie que tous les placeholders requis sont présents."""
    missing = [p for p in REQUIRED_PLACEHOLDERS if p not in content]
    assert not missing, f"Missing placeholders: {', '.join(missing)}"


def test_prompt_structure(content):
    """Vérifie la structure basique du prompt (au moins 3 contrôles sur 4)."""
    checks = {
        "Has content": len(content) > 100,
        "Has title marker (#)": "#" in content,
        "Mentions HTML": "html" in content.lower(),
        "Has instructions": "consigne" in content.lower() or "instructions" in content.lower(),
    }
    failed = [name for name, ok in checks.items() if not ok]
    assert len(failed) <= 1, f"Structure checks failed: {', '.join(failed)}"


def test_prompt_replacement():
    """Teste le remplacement des placeholders."""
    template = PROMPT_FILE.read_text(encoding="utf-8")

    test_values = {
        "{TITLE}": "Test Article Title",
        "{AUTHORS}": "Smith, J.; Doe, M.",
        "{DATE}": "2024",
        "{DOI}": "10.1234/test",
        "{URL}": "https://example.com",
        "{ABSTRACT}": "Test abstract content",
        "{TEXT}": "Full test text content",
        "{LANGUAGE}": "français",
        "{PROBLEMATIQUE}": "Test problematique"
    }

    result = template
    for placeholder, value in test_values.items():
        result = result.replace(placeholder, value)

    remaining = [p for p in test_values if p in result]
    assert not remaining, f"Placeholders not replaced: {', '.join(remaining)}"

    absent = [v for v in test_values.values() if v not in result]
    assert not absent, f"Values not found in result: {', '.join(absent)}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
