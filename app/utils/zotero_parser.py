"""
Zotero metadata parser for RAGpy.

This module extracts library information (library_type, library_id, itemKey)
from Zotero export JSON files.
"""

import os
import json
import re
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Regex to extract library info from Zotero URIs
# Format: http://zotero.org/users/15681/items/4G3PMW5F
# Format: http://zotero.org/groups/12345/items/ABC123XY
URI_PATTERN = re.compile(
    r"zotero\.org/(users|groups)/(\d+)/items/([A-Z0-9]{8})",
    re.IGNORECASE
)


def extract_library_info_from_uri(uri: str) -> Optional[Tuple[str, str, str]]:
    """
    Extract library information from a Zotero URI.

    Args:
        uri: Zotero URI (e.g., "http://zotero.org/users/15681/items/4G3PMW5F")

    Returns:
        Tuple of (library_type, library_id, item_key) if successful, None otherwise

    Example:
        >>> extract_library_info_from_uri("http://zotero.org/users/15681/items/4G3PMW5F")
        ("users", "15681", "4G3PMW5F")
    """
    match = URI_PATTERN.search(uri)
    if match:
        library_type = match.group(1).lower()  # "users" or "groups"
        library_id = match.group(2)  # "15681"
        item_key = match.group(3)  # "4G3PMW5F"
        return library_type, library_id, item_key

    return None


def find_zotero_json(session_dir: str) -> Optional[str]:
    """
    Find the Zotero JSON file in a session directory.

    Args:
        session_dir: Path to the session directory

    Returns:
        Path to the JSON file if found, None otherwise
    """
    session_path = Path(session_dir)

    if not session_path.exists() or not session_path.is_dir():
        logger.error(f"Session directory does not exist: {session_dir}")
        return None

    # Look for JSON files
    json_files = list(session_path.glob("**/*.json"))

    if not json_files:
        logger.warning(f"No JSON files found in {session_dir}")
        return None

    # If multiple JSON files, prefer the largest one (likely the main export)
    if len(json_files) > 1:
        json_files.sort(key=lambda p: p.stat().st_size, reverse=True)
        logger.info(f"Found {len(json_files)} JSON files, using largest: {json_files[0].name}")

    return str(json_files[0])


def extract_library_info_from_session(session_dir: str) -> Dict:
    """
    Extract library information from a Zotero export in a session directory.

    This function:
    1. Finds the Zotero JSON file in the session directory
    2. Parses the first item's URI to extract library_type and library_id
    3. Returns this information for use in API calls

    Args:
        session_dir: Path to the session directory (e.g., "uploads/<session>/")

    Returns:
        Dictionary with:
        - library_type: "users" or "groups"
        - library_id: The library ID (e.g., "15681")
        - json_path: Path to the JSON file
        - success: bool

    Example:
        >>> result = extract_library_info_from_session("uploads/abc123_MyLibrary/")
        >>> print(result)
        {
            "success": True,
            "library_type": "users",
            "library_id": "15681",
            "json_path": "uploads/abc123_MyLibrary/MyLibrary.json"
        }
    """
    # Find the JSON file
    json_path = find_zotero_json(session_dir)

    if not json_path:
        return {
            "success": False,
            "error": "No Zotero JSON file found in session directory"
        }

    # Parse the JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON file {json_path}: {e}")
        return {
            "success": False,
            "error": f"Failed to read JSON file: {str(e)}"
        }

    # Normalize the data structure
    # Zotero exports can be either:
    # 1. Direct array: [{item1}, {item2}, ...]
    # 2. Object with items key: {"items": [{item1}, {item2}, ...]}
    if isinstance(data, list):
        # Format 1: Direct array
        items = data
        logger.info(f"Detected Zotero JSON format: direct array with {len(items)} items")
    elif isinstance(data, dict) and "items" in data:
        # Format 2: Object with items key
        items = data["items"]
        logger.info(f"Detected Zotero JSON format: object with 'items' key, {len(items)} items")
    else:
        return {
            "success": False,
            "error": f"Invalid Zotero JSON format: expected array or object with 'items' key, got {type(data).__name__}"
        }

    # Validate items
    if not isinstance(items, list):
        return {
            "success": False,
            "error": "Invalid Zotero JSON format: 'items' is not an array"
        }

    if not items:
        return {
            "success": False,
            "error": "No items found in Zotero JSON"
        }

    # Extract library info from the first item with a URI
    for item in items:
        # Try to get URI from item
        uri = item.get("uri")

        if uri:
            lib_info = extract_library_info_from_uri(uri)
            if lib_info:
                library_type, library_id, _ = lib_info
                logger.info(f"Extracted library info: type={library_type}, id={library_id}")
                return {
                    "success": True,
                    "library_type": library_type,
                    "library_id": library_id,
                    "json_path": json_path
                }

    # If no URI found, return error
    return {
        "success": False,
        "error": "Could not extract library information from any item URI"
    }


def extract_item_keys_from_json(json_path: str) -> List[Dict]:
    """
    Extract all item keys and basic metadata from a Zotero JSON file.

    Args:
        json_path: Path to the Zotero JSON file

    Returns:
        List of dictionaries with item information:
        - itemKey: The item's key
        - title: Item title
        - itemType: Type of item (article, book, etc.)
        - uri: Zotero URI

    Example:
        >>> items = extract_item_keys_from_json("MyLibrary.json")
        >>> for item in items:
        ...     print(f"{item['itemKey']}: {item['title']}")
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON file {json_path}: {e}")
        return []

    # Normalize the data structure (same as extract_library_info_from_session)
    if isinstance(data, list):
        # Format 1: Direct array
        items = data
    elif isinstance(data, dict) and "items" in data:
        # Format 2: Object with items key
        items = data["items"]
    else:
        logger.error(f"Invalid JSON structure: expected array or object with 'items' key, got {type(data).__name__}")
        return []

    items_info = []

    for item in items:
        # Skip attachments and notes (we only want parent items)
        item_type = item.get("itemType", "")
        if item_type in ("attachment", "note"):
            continue

        # Extract itemKey
        item_key = item.get("itemKey")
        if not item_key:
            # Try to extract from URI
            uri = item.get("uri", "")
            lib_info = extract_library_info_from_uri(uri)
            if lib_info:
                item_key = lib_info[2]

        if item_key:
            items_info.append({
                "itemKey": item_key,
                "title": item.get("title", "Untitled"),
                "itemType": item_type,
                "uri": item.get("uri", "")
            })

    logger.info(f"Extracted {len(items_info)} item keys from {json_path}")
    return items_info


def is_zotero_export(session_dir: str) -> bool:
    """
    Check if a session directory contains a valid Zotero export.

    Args:
        session_dir: Path to the session directory

    Returns:
        True if valid Zotero export detected, False otherwise
    """
    result = extract_library_info_from_session(session_dir)
    return result.get("success", False)
