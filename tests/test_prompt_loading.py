"""
Test pour vérifier le chargement du prompt depuis zotero_prompt.md

Run with: python tests/test_prompt_loading.py
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils import llm_note_generator


def test_load_prompt_template():
    """Test loading the prompt template from file."""
    print("Testing prompt template loading...")

    try:
        template = llm_note_generator._load_prompt_template()

        # Check that template was loaded
        assert template, "Template is empty"
        assert len(template) > 0, "Template has no content"

        # Check for expected placeholders
        required_placeholders = [
            "{TITLE}",
            "{AUTHORS}",
            "{DATE}",
            "{TEXT}",
            "{LANGUAGE}"
        ]

        for placeholder in required_placeholders:
            assert placeholder in template, f"Missing placeholder: {placeholder}"

        print("✅ Template loaded successfully!")
        print(f"   Length: {len(template)} characters")
        print(f"   All required placeholders found: {', '.join(required_placeholders)}")

        return True

    except Exception as e:
        print(f"❌ Error loading template: {e}")
        return False


def test_build_prompt():
    """Test building a complete prompt with metadata."""
    print("\nTesting prompt building with metadata...")

    metadata = {
        "title": "Test Article",
        "authors": "Smith, J.; Doe, M.",
        "date": "2024",
        "abstract": "This is a test abstract.",
        "doi": "10.1234/test",
        "url": "https://example.com",
        "language": "fr"
    }

    text_content = "This is the full text of the article for testing purposes."

    try:
        prompt = llm_note_generator._build_prompt(metadata, text_content, "fr")

        # Check that metadata was inserted
        assert "Test Article" in prompt, "Title not found in prompt"
        assert "Smith, J.; Doe, M." in prompt, "Authors not found in prompt"
        assert "2024" in prompt, "Date not found in prompt"
        assert "This is a test abstract" in prompt, "Abstract not found in prompt"
        assert "français" in prompt, "Language not found in prompt"
        assert "This is the full text" in prompt, "Text content not found in prompt"

        # Check that no placeholders remain
        assert "{TITLE}" not in prompt, "Placeholder {TITLE} not replaced"
        assert "{AUTHORS}" not in prompt, "Placeholder {AUTHORS} not replaced"
        assert "{DATE}" not in prompt, "Placeholder {DATE} not replaced"
        assert "{TEXT}" not in prompt, "Placeholder {TEXT} not replaced"
        assert "{LANGUAGE}" not in prompt, "Placeholder {LANGUAGE} not replaced"

        print("✅ Prompt built successfully!")
        print(f"   Length: {len(prompt)} characters")
        print("   All placeholders replaced")
        print("\n--- Preview (first 300 chars) ---")
        print(prompt[:300] + "...")

        return True

    except Exception as e:
        print(f"❌ Error building prompt: {e}")
        return False


def test_fallback_prompt():
    """Test fallback to hardcoded prompt if file is missing."""
    print("\nTesting fallback prompt mechanism...")

    # Temporarily rename the prompt file to test fallback
    import shutil
    from pathlib import Path

    utils_dir = Path(__file__).parent.parent / "app" / "utils"
    prompt_file = utils_dir / "zotero_prompt.md"
    backup_file = utils_dir / "zotero_prompt.md.backup"

    # Only test if the file exists
    if not prompt_file.exists():
        print("⚠️  Prompt file doesn't exist, skipping fallback test")
        return True

    try:
        # Backup the file
        shutil.move(str(prompt_file), str(backup_file))

        # Try to build prompt (should use fallback)
        metadata = {"title": "Test", "language": "en"}
        prompt = llm_note_generator._build_prompt(metadata, "Test text", "en")

        assert "Test" in prompt, "Fallback prompt failed"
        print("✅ Fallback prompt works!")

        return True

    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
        return False

    finally:
        # Restore the file
        if backup_file.exists():
            shutil.move(str(backup_file), str(prompt_file))
            print("   (Original file restored)")


if __name__ == "__main__":
    print("=" * 60)
    print("ZOTERO PROMPT LOADING TESTS")
    print("=" * 60)

    results = []

    results.append(test_load_prompt_template())
    results.append(test_build_prompt())
    results.append(test_fallback_prompt())

    print("\n" + "=" * 60)
    print(f"RESULTS: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    sys.exit(0 if all(results) else 1)
