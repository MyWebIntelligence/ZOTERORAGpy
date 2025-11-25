"""
Zotero Parser
=============

This module parses Zotero JSON export files to extract library metadata and item
information. It supports both modern and legacy Zotero export formats.

Key Features:
- Library Detection: Extracts library type (users/groups) and library ID.
- Item Extraction: Retrieves item keys and basic metadata from JSON exports.
- Format Compatibility: Handles multiple Zotero export formats (direct array, object with items).
- Fallback Mechanism: Uses environment variables if metadata is missing from JSON.
"""

import os
import json
import re
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

    This function tries multiple extraction methods:
    1. Modern format: item["library"]["id"] and item["library"]["type"]
    2. Legacy format: Parse item["uri"] to extract library info
    3. Fallback: Use ZOTERO_USER_ID or ZOTERO_GROUP_ID from .env

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
            "error": "The file you uploaded is not a BibTeX JSON export. You cannot populate your Zotero references without a valid Zotero JSON file."
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

    # Extract library info from items (try multiple methods)
    for item in items:
        # Method 1: Modern format - check "library" field
        library_field = item.get("library")
        if library_field and isinstance(library_field, dict):
            lib_type = library_field.get("type")  # "user" or "group"
            lib_id = library_field.get("id")

            if lib_type and lib_id:
                # Convert "user" → "users", "group" → "groups" for API compatibility
                library_type = "users" if lib_type == "user" else "groups" if lib_type == "group" else lib_type
                library_id = str(lib_id)
                logger.info(f"Extracted library info from 'library' field: type={library_type}, id={library_id}")
                return {
                    "success": True,
                    "library_type": library_type,
                    "library_id": library_id,
                    "json_path": json_path
                }

        # Method 2: Legacy format - extract from URI
        uri = item.get("uri")
        if uri:
            lib_info = extract_library_info_from_uri(uri)
            if lib_info:
                library_type, library_id, _ = lib_info
                logger.info(f"Extracted library info from URI: type={library_type}, id={library_id}")
                return {
                    "success": True,
                    "library_type": library_type,
                    "library_id": library_id,
                    "json_path": json_path
                }

    # Method 3: Fallback to credentials from .env
    logger.warning("Could not extract library info from JSON items, checking .env credentials")

    zotero_user_id = os.getenv("ZOTERO_USER_ID")
    zotero_group_id = os.getenv("ZOTERO_GROUP_ID")

    if zotero_user_id:
        logger.info(f"Using ZOTERO_USER_ID from .env: {zotero_user_id}")
        return {
            "success": True,
            "library_type": "users",
            "library_id": zotero_user_id,
            "json_path": json_path
        }
    elif zotero_group_id:
        logger.info(f"Using ZOTERO_GROUP_ID from .env: {zotero_group_id}")
        return {
            "success": True,
            "library_type": "groups",
            "library_id": zotero_group_id,
            "json_path": json_path
        }

    # If all methods fail, return error
    return {
        "success": False,
        "error": "Could not extract library information. Please set ZOTERO_USER_ID or ZOTERO_GROUP_ID in Settings."
    }


def extract_item_keys_from_json(json_path: str) -> List[Dict]:
    """
    Extract all item keys and basic metadata from a Zotero JSON file.

    Supports both modern ("key") and legacy ("itemKey") field names.

    Args:
        json_path: Path to the Zotero JSON file

    Returns:
        List of dictionaries with item information:
        - itemKey: The item's key (extracted from "key" or "itemKey" field)
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

        # Extract itemKey (try multiple field names)
        item_key = item.get("itemKey") or item.get("key")  # Modern exports use "key", legacy use "itemKey"

        if not item_key:
            # Try to extract from URI as last resort
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
        else:
            logger.warning(f"Could not extract itemKey for item: {item.get('title', 'Untitled')}")

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
