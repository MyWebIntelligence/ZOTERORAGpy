"""
Test simple pour vérifier que le fichier zotero_prompt.md est valide.

Run with: python tests/test_prompt_file.py
"""

import os
import sys
from pathlib import Path


def test_prompt_file_exists():
    """Vérifie que le fichier zotero_prompt.md existe."""
    print("Testing prompt file existence...")

    # Construire le chemin vers le fichier
    repo_root = Path(__file__).parent.parent
    prompt_file = repo_root / "app" / "utils" / "zotero_prompt.md"

    assert prompt_file.exists(), f"❌ Prompt file not found at {prompt_file}"
    print(f"✅ Prompt file exists at: {prompt_file}")

    return True


def test_prompt_file_readable():
    """Vérifie que le fichier peut être lu."""
    print("\nTesting prompt file readability...")

    repo_root = Path(__file__).parent.parent
    prompt_file = repo_root / "app" / "utils" / "zotero_prompt.md"

    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert len(content) > 0, "❌ Prompt file is empty"
        print(f"✅ Prompt file readable, length: {len(content)} characters")

        return content

    except Exception as e:
        print(f"❌ Error reading prompt file: {e}")
        return None


def test_prompt_placeholders(content):
    """Vérifie que tous les placeholders requis sont présents."""
    print("\nTesting prompt placeholders...")

    required_placeholders = [
        "{TITLE}",
        "{AUTHORS}",
        "{DATE}",
        "{DOI}",
        "{URL}",
        "{ABSTRACT}",
        "{TEXT}",
        "{LANGUAGE}"
    ]

    missing = []
    for placeholder in required_placeholders:
        if placeholder not in content:
            missing.append(placeholder)

    if missing:
        print(f"❌ Missing placeholders: {', '.join(missing)}")
        return False

    print(f"✅ All {len(required_placeholders)} placeholders found:")
    for placeholder in required_placeholders:
        print(f"   • {placeholder}")

    return True


def test_prompt_structure(content):
    """Vérifie la structure basique du prompt."""
    print("\nTesting prompt structure...")

    checks = {
        "Has content": len(content) > 100,
        "Has title marker (#)": "#" in content,
        "Mentions HTML": "html" in content.lower() or "HTML" in content,
        "Has instructions": "consigne" in content.lower() or "instructions" in content.lower(),
    }

    passed = 0
    for check_name, result in checks.items():
        status = "✅" if result else "⚠️"
        print(f"   {status} {check_name}")
        if result:
            passed += 1

    print(f"\n   {passed}/{len(checks)} structure checks passed")

    return passed >= len(checks) - 1  # Au moins 3/4 checks doivent passer


def test_prompt_replacement():
    """Teste le remplacement des placeholders."""
    print("\nTesting placeholder replacement...")

    repo_root = Path(__file__).parent.parent
    prompt_file = repo_root / "app" / "utils" / "zotero_prompt.md"

    with open(prompt_file, "r", encoding="utf-8") as f:
        template = f.read()

    # Simuler le remplacement
    test_values = {
        "{TITLE}": "Test Article Title",
        "{AUTHORS}": "Smith, J.; Doe, M.",
        "{DATE}": "2024",
        "{DOI}": "10.1234/test",
        "{URL}": "https://example.com",
        "{ABSTRACT}": "Test abstract content",
        "{TEXT}": "Full test text content",
        "{LANGUAGE}": "français"
    }

    result = template
    for placeholder, value in test_values.items():
        result = result.replace(placeholder, value)

    # Vérifier qu'aucun placeholder ne reste
    remaining = []
    for placeholder in test_values.keys():
        if placeholder in result:
            remaining.append(placeholder)

    if remaining:
        print(f"❌ Placeholders not replaced: {', '.join(remaining)}")
        return False

    # Vérifier que les valeurs sont présentes
    for value in test_values.values():
        if value not in result:
            print(f"❌ Value not found in result: {value}")
            return False

    print("✅ All placeholders replaced successfully")
    print(f"   Template length: {len(template)} chars")
    print(f"   Result length: {len(result)} chars")
    print("\n--- Preview (first 400 chars) ---")
    print(result[:400] + "...")

    return True


if __name__ == "__main__":
    print("=" * 70)
    print("ZOTERO PROMPT FILE VALIDATION TESTS")
    print("=" * 70)
    print()

    results = []

    # Test 1: Existence
    try:
        results.append(test_prompt_file_exists())
    except AssertionError as e:
        print(str(e))
        results.append(False)

    # Test 2: Lisibilité
    content = test_prompt_file_readable()
    results.append(content is not None)

    if content:
        # Test 3: Placeholders
        results.append(test_prompt_placeholders(content))

        # Test 4: Structure
        results.append(test_prompt_structure(content))

        # Test 5: Remplacement
        results.append(test_prompt_replacement())

    print("\n" + "=" * 70)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 SUCCESS: All {total} tests passed!")
    else:
        print(f"⚠️  PARTIAL: {passed}/{total} tests passed")

    print("=" * 70)

    sys.exit(0 if passed == total else 1)
