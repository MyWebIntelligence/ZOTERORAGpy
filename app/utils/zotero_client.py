"""
Zotero API v3 client for RAGpy.

This module provides functions to interact with Zotero API v3 for:
- Verifying API keys
- Retrieving library versions
- Checking if notes exist
- Creating child notes with automatic retry and concurrency control

API Documentation: https://www.zotero.org/support/dev/web_api/v3/start
"""

import time
import uuid
import logging
from typing import Optional, Dict, List
import requests

logger = logging.getLogger(__name__)

# Constants
ZOTERO_API_BASE = "https://api.zotero.org"
ZOTERO_API_VERSION = "3"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


class ZoteroAPIError(Exception):
    """Custom exception for Zotero API errors."""

    def __init__(self, status_code: int, message: str, response: Optional[requests.Response] = None):
        self.status_code = status_code
        self.message = message
        self.response = response
        super().__init__(f"Zotero API Error {status_code}: {message}")


def _build_headers(api_key: str, additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Build standard headers for Zotero API requests.

    Args:
        api_key: Zotero API key
        additional_headers: Optional additional headers to merge

    Returns:
        Dictionary of headers
    """
    headers = {
        "Zotero-API-Key": api_key,
        "Zotero-API-Version": ZOTERO_API_VERSION,
        "Content-Type": "application/json",
    }

    if additional_headers:
        headers.update(additional_headers)

    return headers


def _build_library_prefix(library_type: str, library_id: str) -> str:
    """
    Build the library prefix for API URLs.

    Args:
        library_type: "users" or "groups"
        library_id: The library ID (user ID or group ID)

    Returns:
        Library prefix string (e.g., "users/12345" or "groups/67890")
    """
    if library_type not in ("users", "groups"):
        raise ValueError(f"Invalid library_type: {library_type}. Must be 'users' or 'groups'")

    return f"{library_type}/{library_id}"


def verify_api_key(api_key: str) -> Dict:
    """
    Verify that the API key is valid and return its permissions.

    Args:
        api_key: Zotero API key to verify

    Returns:
        Dictionary with key information (username, userID, access permissions)

    Raises:
        ZoteroAPIError: If the key is invalid or API request fails
    """
    url = f"{ZOTERO_API_BASE}/keys/current"
    headers = _build_headers(api_key)

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            logger.info("Zotero API key verified successfully")
            return response.json()
        elif response.status_code == 403:
            raise ZoteroAPIError(403, "Invalid API key", response)
        else:
            raise ZoteroAPIError(
                response.status_code,
                f"Failed to verify API key: {response.text}",
                response
            )
    except requests.RequestException as e:
        logger.error(f"Network error while verifying API key: {e}")
        raise ZoteroAPIError(0, f"Network error: {str(e)}")


def get_library_version(library_type: str, library_id: str, api_key: str) -> str:
    """
    Get the current version of the library.

    This is used for concurrency control with If-Unmodified-Since-Version header.

    Args:
        library_type: "users" or "groups"
        library_id: The library ID
        api_key: Zotero API key

    Returns:
        Version string (e.g., "12345")

    Raises:
        ZoteroAPIError: If the request fails
    """
    prefix = _build_library_prefix(library_type, library_id)
    url = f"{ZOTERO_API_BASE}/{prefix}/items/top"
    headers = _build_headers(api_key)

    try:
        response = requests.get(url, headers=headers, params={"limit": 1}, timeout=10)

        if response.status_code == 200:
            version = response.headers.get("Last-Modified-Version", "0")
            logger.debug(f"Retrieved library version: {version}")
            return version
        else:
            raise ZoteroAPIError(
                response.status_code,
                f"Failed to get library version: {response.text}",
                response
            )
    except requests.RequestException as e:
        logger.error(f"Network error while getting library version: {e}")
        raise ZoteroAPIError(0, f"Network error: {str(e)}")


def check_note_exists(
    library_type: str,
    library_id: str,
    item_key: str,
    sentinel: str,
    api_key: str
) -> bool:
    """
    Check if a note with the given sentinel already exists as a child of the item.

    Args:
        library_type: "users" or "groups"
        library_id: The library ID
        item_key: The parent item key
        sentinel: The unique sentinel to search for (e.g., "ragpy-note-id:uuid")
        api_key: Zotero API key

    Returns:
        True if a note with the sentinel exists, False otherwise

    Raises:
        ZoteroAPIError: If the request fails
    """
    prefix = _build_library_prefix(library_type, library_id)
    url = f"{ZOTERO_API_BASE}/{prefix}/items/{item_key}/children"
    headers = _build_headers(api_key)

    try:
        response = requests.get(
            url,
            headers=headers,
            params={"itemType": "note"},
            timeout=15
        )

        if response.status_code == 200:
            notes = response.json()
            for note in notes:
                note_content = note.get("data", {}).get("note", "")
                if sentinel in note_content:
                    logger.info(f"Found existing note with sentinel {sentinel}")
                    return True
            return False
        elif response.status_code == 404:
            logger.warning(f"Parent item {item_key} not found")
            raise ZoteroAPIError(404, f"Parent item {item_key} not found", response)
        else:
            raise ZoteroAPIError(
                response.status_code,
                f"Failed to check child notes: {response.text}",
                response
            )
    except requests.RequestException as e:
        logger.error(f"Network error while checking notes: {e}")
        raise ZoteroAPIError(0, f"Network error: {str(e)}")


def create_child_note(
    library_type: str,
    library_id: str,
    item_key: str,
    note_html: str,
    tags: Optional[List[str]] = None,
    api_key: str = "",
    library_version: Optional[str] = None
) -> Dict:
    """
    Create a child note for a Zotero item.

    This function implements:
    - Automatic retry on 412 (version conflict)
    - Backoff on 429 (rate limit)
    - Write token for idempotence

    Args:
        library_type: "users" or "groups"
        library_id: The library ID
        item_key: The parent item key
        note_html: HTML content of the note
        tags: Optional list of tags to add to the note
        api_key: Zotero API key
        library_version: Optional library version for concurrency control

    Returns:
        Dictionary with response data including:
        - success: bool
        - note_key: str (if successful)
        - message: str
        - new_version: str (if successful)

    Raises:
        ZoteroAPIError: If all retry attempts fail
    """
    prefix = _build_library_prefix(library_type, library_id)
    url = f"{ZOTERO_API_BASE}/{prefix}/items"

    # Build the note item payload
    note_item = {
        "itemType": "note",
        "note": note_html,
        "parentItem": item_key,
        "tags": [{"tag": tag} for tag in (tags or [])]
    }

    # Generate write token for idempotence
    write_token = uuid.uuid4().hex

    for attempt in range(MAX_RETRIES):
        try:
            # Build headers with version control
            additional_headers = {"Zotero-Write-Token": write_token}

            # Try to get current version if not provided or on retry
            if library_version is None or attempt > 0:
                try:
                    library_version = get_library_version(library_type, library_id, api_key)
                    additional_headers["If-Unmodified-Since-Version"] = library_version
                except ZoteroAPIError:
                    logger.warning("Could not get library version, proceeding without it")
            else:
                additional_headers["If-Unmodified-Since-Version"] = library_version

            headers = _build_headers(api_key, additional_headers)

            # Make the request
            response = requests.post(
                url,
                headers=headers,
                json=[note_item],  # API expects an array
                timeout=30
            )

            # Handle response
            if response.status_code in (200, 201):
                result = response.json()
                new_version = response.headers.get("Last-Modified-Version")

                # Extract the created note key
                # According to Zotero API docs, successful["0"] contains the itemKey directly
                # Response format: {"successful": {"0": "<itemKey>"}, "unchanged": {}, "failed": {}}
                if "successful" in result and "0" in result["successful"]:
                    note_key = result["successful"]["0"]
                    logger.info(f"Successfully created note {note_key} for item {item_key}")
                    return {
                        "success": True,
                        "note_key": note_key,
                        "message": "Note created successfully",
                        "new_version": new_version
                    }
                else:
                    logger.error(f"Unexpected response format: {result}")
                    return {
                        "success": False,
                        "message": "Note created but could not extract note key",
                        "raw_response": result
                    }

            elif response.status_code == 412:
                # Version conflict - retry with new version
                logger.warning(f"Version conflict (412), retrying (attempt {attempt + 1}/{MAX_RETRIES})")
                library_version = None  # Force refresh on next iteration
                time.sleep(RETRY_DELAY)
                continue

            elif response.status_code == 429:
                # Rate limit - respect Retry-After header
                retry_after = int(response.headers.get("Retry-After", RETRY_DELAY))
                logger.warning(f"Rate limit (429), waiting {retry_after}s")
                time.sleep(retry_after)
                continue

            elif response.status_code == 409:
                # Conflict - possibly locked library
                logger.warning(f"Conflict (409), retrying (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY * 2)
                continue

            elif response.status_code == 404:
                # Parent item not found
                raise ZoteroAPIError(404, f"Parent item {item_key} not found", response)

            elif response.status_code in (401, 403):
                # Authentication/permission error - no point in retrying
                raise ZoteroAPIError(
                    response.status_code,
                    "Invalid API key or insufficient permissions",
                    response
                )

            elif response.status_code == 400:
                # Bad request - probably malformed HTML or payload
                raise ZoteroAPIError(
                    400,
                    f"Bad request: {response.text}",
                    response
                )

            else:
                # Other error
                raise ZoteroAPIError(
                    response.status_code,
                    f"Failed to create note: {response.text}",
                    response
                )

        except requests.RequestException as e:
            logger.error(f"Network error on attempt {attempt + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            else:
                raise ZoteroAPIError(0, f"Network error after {MAX_RETRIES} attempts: {str(e)}")

    # Should not reach here, but just in case
    return {
        "success": False,
        "message": f"Failed to create note after {MAX_RETRIES} attempts"
    }


def update_item_abstract(
    library_type: str,
    library_id: str,
    item_key: str,
    new_abstract: str,
    api_key: str,
    separator: str = "\n\n---\n\n",
    mode: str = "append"
) -> Dict:
    """
    Update an item's abstractNote field.

    Args:
        library_type: "users" or "groups"
        library_id: The library ID
        item_key: The item key to update
        new_abstract: New abstract text
        api_key: Zotero API key
        separator: Separator between existing and new abstract (only used in append mode)
        mode: "append" (default) appends to existing, "replace" replaces entirely

    Returns:
        Dictionary with response data including:
        - success: bool
        - message: str
        - new_version: str (if successful)
        - previous_abstract: str (the abstract before update)

    Raises:
        ZoteroAPIError: If the request fails
    """
    prefix = _build_library_prefix(library_type, library_id)

    # First, get the current item to retrieve existing abstract and version
    try:
        item_data = get_item(library_type, library_id, item_key, api_key)
    except ZoteroAPIError as e:
        logger.error(f"Failed to retrieve item {item_key}: {e}")
        raise

    # Extract current abstract and version
    current_data = item_data.get("data", {})
    current_abstract = current_data.get("abstractNote", "")
    item_version = str(item_data.get("version", "0"))

    logger.info(f"Current abstract length: {len(current_abstract)} chars, version: {item_version}")

    # Build new abstract based on mode
    if mode == "append" and current_abstract and current_abstract.strip():
        updated_abstract = current_abstract + separator + new_abstract
        logger.info(f"Appending to existing abstract (mode={mode})")
    else:
        updated_abstract = new_abstract
        logger.info(f"Replacing abstract (mode={mode})")

    # Build the PATCH payload (only update abstractNote)
    patch_payload = {
        "abstractNote": updated_abstract
    }

    url = f"{ZOTERO_API_BASE}/{prefix}/items/{item_key}"

    for attempt in range(MAX_RETRIES):
        try:
            # Build headers with version control
            additional_headers = {
                "If-Unmodified-Since-Version": item_version
            }
            headers = _build_headers(api_key, additional_headers)

            # Make PATCH request
            response = requests.patch(
                url,
                headers=headers,
                json=patch_payload,
                timeout=30
            )

            # Handle response
            if response.status_code in (200, 204):
                new_version = response.headers.get("Last-Modified-Version", item_version)
                logger.info(f"Successfully updated abstract for item {item_key}")
                return {
                    "success": True,
                    "message": "Abstract updated successfully",
                    "new_version": new_version,
                    "previous_abstract": current_abstract,
                    "new_abstract_length": len(updated_abstract)
                }

            elif response.status_code == 412:
                # Version conflict - refresh item and retry
                logger.warning(f"Version conflict (412), refreshing item (attempt {attempt + 1}/{MAX_RETRIES})")
                try:
                    item_data = get_item(library_type, library_id, item_key, api_key)
                    current_data = item_data.get("data", {})
                    current_abstract = current_data.get("abstractNote", "")
                    item_version = str(item_data.get("version", "0"))

                    # Rebuild abstract with fresh data
                    if current_abstract and current_abstract.strip():
                        updated_abstract = current_abstract + separator + new_abstract
                    else:
                        updated_abstract = new_abstract
                    patch_payload["abstractNote"] = updated_abstract

                except ZoteroAPIError:
                    pass
                time.sleep(RETRY_DELAY)
                continue

            elif response.status_code == 429:
                # Rate limit
                retry_after = int(response.headers.get("Retry-After", RETRY_DELAY))
                logger.warning(f"Rate limit (429), waiting {retry_after}s")
                time.sleep(retry_after)
                continue

            elif response.status_code == 404:
                raise ZoteroAPIError(404, f"Item {item_key} not found", response)

            elif response.status_code in (401, 403):
                raise ZoteroAPIError(
                    response.status_code,
                    "Invalid API key or insufficient permissions",
                    response
                )

            else:
                raise ZoteroAPIError(
                    response.status_code,
                    f"Failed to update abstract: {response.text}",
                    response
                )

        except requests.RequestException as e:
            logger.error(f"Network error on attempt {attempt + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            else:
                raise ZoteroAPIError(0, f"Network error after {MAX_RETRIES} attempts: {str(e)}")

    return {
        "success": False,
        "message": f"Failed to update abstract after {MAX_RETRIES} attempts"
    }


def get_item(
    library_type: str,
    library_id: str,
    item_key: str,
    api_key: str
) -> Dict:
    """
    Get a single item by its key.

    Args:
        library_type: "users" or "groups"
        library_id: The library ID
        item_key: The item key
        api_key: Zotero API key

    Returns:
        Dictionary with item data

    Raises:
        ZoteroAPIError: If the request fails
    """
    prefix = _build_library_prefix(library_type, library_id)
    url = f"{ZOTERO_API_BASE}/{prefix}/items/{item_key}"
    headers = _build_headers(api_key)

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            raise ZoteroAPIError(404, f"Item {item_key} not found", response)
        else:
            raise ZoteroAPIError(
                response.status_code,
                f"Failed to get item: {response.text}",
                response
            )
    except requests.RequestException as e:
        logger.error(f"Network error while getting item: {e}")
        raise ZoteroAPIError(0, f"Network error: {str(e)}")
