"""
Test pour vérifier que le parser Zotero gère les deux formats de JSON.

Format 1: Tableau direct [{item1}, {item2}, ...]
Format 2: Objet avec clé "items" {"items": [{item1}, {item2}, ...]}

Run with: python tests/test_zotero_json_formats.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils import zotero_parser


def create_test_json_format1(path):
    """Create a test JSON in format 1 (direct array)."""
    data = [
        {
            "itemType": "journalArticle",
            "itemKey": "ABC123XY",
            "title": "Test Article 1",
            "uri": "http://zotero.org/users/12345/items/ABC123XY"
        },
        {
            "itemType": "book",
            "itemKey": "DEF456UV",
            "title": "Test Book 1",
            "uri": "http://zotero.org/users/12345/items/DEF456UV"
        }
    ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def create_test_json_format2(path):
    """Create a test JSON in format 2 (object with items key)."""
    data = {
        "items": [
            {
                "itemType": "journalArticle",
                "itemKey": "GHI789WX",
                "title": "Test Article 2",
                "uri": "http://zotero.org/users/67890/items/GHI789WX"
            },
            {
                "itemType": "conferencePaper",
                "itemKey": "JKL012YZ",
                "title": "Test Conference Paper",
                "uri": "http://zotero.org/users/67890/items/JKL012YZ"
            }
        ]
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def test_format1_direct_array():
    """Test parsing format 1 (direct array)."""
    print("Testing Format 1: Direct Array")
    print("-" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "zotero_export_format1.json"
        create_test_json_format1(json_path)

        # Test extract_library_info_from_session
        result = zotero_parser.extract_library_info_from_session(tmpdir)

        print(f"Success: {result.get('success')}")
        print(f"Library Type: {result.get('library_type')}")
        print(f"Library ID: {result.get('library_id')}")

        assert result["success"], "Should successfully parse format 1"
        assert result["library_type"] == "users", "Should extract library type"
        assert result["library_id"] == "12345", "Should extract library ID"

        # Test extract_item_keys_from_json
        items = zotero_parser.extract_item_keys_from_json(str(json_path))

        print(f"\nExtracted Items: {len(items)}")
        for item in items:
            print(f"  - {item['itemKey']}: {item['title']}")

        assert len(items) == 2, "Should extract 2 items"
        assert items[0]["itemKey"] == "ABC123XY", "Should extract correct itemKey"

        print("\n✅ Format 1 test PASSED\n")
        return True


def test_format2_items_key():
    """Test parsing format 2 (object with items key)."""
    print("Testing Format 2: Object with 'items' key")
    print("-" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "zotero_export_format2.json"
        create_test_json_format2(json_path)

        # Test extract_library_info_from_session
        result = zotero_parser.extract_library_info_from_session(tmpdir)

        print(f"Success: {result.get('success')}")
        print(f"Library Type: {result.get('library_type')}")
        print(f"Library ID: {result.get('library_id')}")

        assert result["success"], "Should successfully parse format 2"
        assert result["library_type"] == "users", "Should extract library type"
        assert result["library_id"] == "67890", "Should extract library ID"

        # Test extract_item_keys_from_json
        items = zotero_parser.extract_item_keys_from_json(str(json_path))

        print(f"\nExtracted Items: {len(items)}")
        for item in items:
            print(f"  - {item['itemKey']}: {item['title']}")

        assert len(items) == 2, "Should extract 2 items"
        assert items[0]["itemKey"] == "GHI789WX", "Should extract correct itemKey"

        print("\n✅ Format 2 test PASSED\n")
        return True


def test_invalid_format():
    """Test that invalid formats are rejected gracefully."""
    print("Testing Invalid Format Handling")
    print("-" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "invalid.json"

        # Create an invalid JSON structure (string instead of array/object)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump("This is not a valid Zotero export", f)

        result = zotero_parser.extract_library_info_from_session(tmpdir)

        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        assert not result["success"], "Should fail on invalid format"
        assert "Invalid Zotero JSON format" in result.get("error", ""), "Should provide helpful error message"

        print("\n✅ Invalid format test PASSED\n")
        return True


def test_empty_array():
    """Test handling of empty array."""
    print("Testing Empty Array Handling")
    print("-" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "empty.json"

        # Create an empty array
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([], f)

        result = zotero_parser.extract_library_info_from_session(tmpdir)

        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        assert not result["success"], "Should fail on empty array"
        assert "No items found" in result.get("error", ""), "Should provide helpful error message"

        print("\n✅ Empty array test PASSED\n")
        return True


if __name__ == "__main__":
    print("=" * 70)
    print("ZOTERO JSON FORMAT COMPATIBILITY TESTS")
    print("=" * 70)
    print()

    results = []

    try:
        results.append(test_format1_direct_array())
    except Exception as e:
        print(f"❌ Format 1 test failed: {e}\n")
        results.append(False)

    try:
        results.append(test_format2_items_key())
    except Exception as e:
        print(f"❌ Format 2 test failed: {e}\n")
        results.append(False)

    try:
        results.append(test_invalid_format())
    except Exception as e:
        print(f"❌ Invalid format test failed: {e}\n")
        results.append(False)

    try:
        results.append(test_empty_array())
    except Exception as e:
        print(f"❌ Empty array test failed: {e}\n")
        results.append(False)

    print("=" * 70)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 SUCCESS: All {total} tests passed!")
        print("\nThe parser now correctly handles both Zotero JSON formats:")
        print("  ✓ Format 1: Direct array [{...}, {...}]")
        print("  ✓ Format 2: Object with 'items' key {\"items\": [{...}]}")
    else:
        print(f"⚠️  PARTIAL: {passed}/{total} tests passed")

    print("=" * 70)

    sys.exit(0 if passed == total else 1)
