"""
Zotero API Client
=================

This module implements a client for the Zotero Web API v3. It handles authentication,
requests, and error handling for interacting with Zotero libraries (users and groups).

Key Features:
- Authentication: Verifies API keys and permissions.
- Note Management: Checks for existing notes and creates new child notes.
- Concurrency Control: Handles Zotero's versioning system (If-Unmodified-Since-Version).
- Robustness: Implements retries with exponential backoff for rate limits and errors.
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
        A dictionary summarizing the outcome of the operation:
        - `success` (bool): `True` if the note was created successfully,
          otherwise `False`.
        - `note_key` (str): The unique key of the newly created note,
          present only on success.
        - `message` (str): A human-readable status message.
        - `new_version` (str): The new library version after the update,
          which can be used for subsequent requests.
        - `raw_response` (dict, optional): The raw response from the API
          in case of an unexpected success format.

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
        A dictionary summarizing the outcome of the operation:
        - `success` (bool): `True` if the abstract was updated successfully,
          otherwise `False`.
        - `message` (str): A human-readable status message.
        - `new_version` (str): The new library version after the update.
        - `previous_abstract` (str): The content of the abstract field
          before the update was applied.
        - `new_abstract_length` (int): The character length of the new
          abstract.

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


# ============================================================================
# Citation Import Extensions (Publish or Perish → Zotero)
# ============================================================================
# The following functions support the citation import feature:
# - Collection management (get_or_create_collection)
# - Item deduplication (search_item_by_doi/url/title)
# - Item creation/update with intelligent merging (create_or_update_item)
# - Field validation per item type (_clean_item_data_for_type)
# ============================================================================


# Mapping of "publication title" field per Zotero item type
# Different item types use different field names for the publication venue
PUBLICATION_TITLE_FIELD_MAP = {
    "journalArticle": "publicationTitle",
    "magazineArticle": "publicationTitle",
    "newspaperArticle": "publicationTitle",
    "conferencePaper": "proceedingsTitle",
    "bookSection": "bookTitle",
    "encyclopediaArticle": "encyclopediaTitle",
    "dictionaryEntry": "dictionaryTitle",
    "webpage": "websiteTitle",
    "blogPost": "blogTitle",
    "forumPost": "forumTitle",
    "audioRecording": "albumTitle",
    "videoRecording": "albumTitle",
    "tvBroadcast": "programTitle",
    "radioBroadcast": "programTitle",
    "podcast": "seriesTitle",
    "map": "seriesTitle",
}

# Item types that don't have any "publication title" type field
ITEM_TYPES_WITHOUT_PUBLICATION_TITLE = {
    "preprint",  # Has 'repository' instead
    "report",    # Has 'institution' instead
    "thesis",    # Has 'university' instead
    "patent",    # Has 'issuingAuthority' instead
    "statute",   # Has 'code' instead
    "case",      # Has 'reporter' instead
    "bill",      # No publication title
    "hearing",   # No publication title
    "letter",    # No publication title
    "interview", # No publication title
    "email",     # No publication title
    "instantMessage",  # No publication title
    "presentation",    # No publication title
    "manuscript",      # No publication title
    "document",        # No publication title
    "note",            # No publication title
    "attachment",      # No publication title
    "artwork",         # No publication title
    "film",            # No publication title
    "book",            # No publication title (it IS the publication)
}

# All possible "publication title" field names in Zotero
ALL_PUBLICATION_TITLE_FIELDS = {
    "publicationTitle",
    "proceedingsTitle",
    "bookTitle",
    "encyclopediaTitle",
    "dictionaryTitle",
    "websiteTitle",
    "blogTitle",
    "forumTitle",
    "albumTitle",
    "programTitle",
    "seriesTitle",
}


def _clean_item_data_for_type(item_data: Dict) -> Dict:
    """
    Clean and validate item data based on the Zotero item type.

    This function:
    1. Maps generic 'publicationTitle' to the correct field for the item type
    2. Removes invalid fields that would cause Zotero API errors
    3. Preserves all other valid fields

    Args:
        item_data: Raw item data dictionary (may contain invalid fields)

    Returns:
        Cleaned item data dictionary with only valid fields for the item type

    Examples:
        >>> data = {"itemType": "conferencePaper", "publicationTitle": "ACM Conference"}
        >>> cleaned = _clean_item_data_for_type(data)
        >>> # Result: {"itemType": "conferencePaper", "proceedingsTitle": "ACM Conference"}

        >>> data = {"itemType": "preprint", "publicationTitle": "arXiv"}
        >>> cleaned = _clean_item_data_for_type(data)
        >>> # Result: {"itemType": "preprint"} (publicationTitle removed, no equivalent)
    """
    cleaned = item_data.copy()
    item_type = cleaned.get("itemType", "")

    # Get the source publication title value (if any)
    source_pub_title = None
    for field in ALL_PUBLICATION_TITLE_FIELDS:
        if field in cleaned and cleaned[field]:
            source_pub_title = cleaned[field]
            break

    # Remove all publication title fields first
    for field in ALL_PUBLICATION_TITLE_FIELDS:
        cleaned.pop(field, None)

    # If we have a publication title value and this item type supports it, add it back
    if source_pub_title and item_type not in ITEM_TYPES_WITHOUT_PUBLICATION_TITLE:
        correct_field = PUBLICATION_TITLE_FIELD_MAP.get(item_type)
        if correct_field:
            cleaned[correct_field] = source_pub_title
            logger.debug(f"Mapped publicationTitle to {correct_field} for {item_type}")
        else:
            # Item type not in our map but also not in "without" list
            # Default to keeping publicationTitle (for any new/unknown types)
            cleaned["publicationTitle"] = source_pub_title
            logger.debug(f"Keeping publicationTitle for unknown type: {item_type}")
    elif source_pub_title:
        logger.debug(f"Removed publicationTitle for {item_type} (not supported)")

    return cleaned


def get_or_create_collection(
    library_type: str,
    library_id: str,
    collection_name: str,
    description: str = "",
    api_key: str = ""
) -> Dict:
    """
    Get or create a Zotero collection by name (case-insensitive search).

    This function searches for an existing collection with the given name.
    If found, returns its key. If not found, creates a new collection.

    Args:
        library_type: "users" or "groups"
        library_id: The library ID
        collection_name: Name of the collection to find or create
        description: Ignored (Zotero API doesn't support collection descriptions)
        api_key: Zotero API key

    Returns:
        Dictionary with:
        - key (str): Collection key
        - created (bool): True if collection was created, False if existing
        - name (str): Collection name
        - version (str): Library version after operation

    Raises:
        ZoteroAPIError: If API requests fail

    Examples:
        >>> result = get_or_create_collection(
        ...     library_type="users",
        ...     library_id="12345",
        ...     collection_name="My Papers",
        ...     api_key="..."
        ... )
        >>> print(f"Collection key: {result['key']}, Created: {result['created']}")
    """
    prefix = _build_library_prefix(library_type, library_id)
    url = f"{ZOTERO_API_BASE}/{prefix}/collections"
    headers = _build_headers(api_key)

    # Step 1: Search for existing collection (case-insensitive)
    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            collections = response.json()

            # Search for collection by name (case-insensitive)
            for coll in collections:
                if coll.get("data", {}).get("name", "").lower() == collection_name.lower():
                    logger.info(f"Found existing collection: {coll['key']} - {coll['data']['name']}")
                    return {
                        "key": coll["key"],
                        "created": False,
                        "name": coll["data"]["name"],
                        "version": response.headers.get("Last-Modified-Version", "")
                    }

            # Collection not found - proceed to create
            logger.info(f"Collection '{collection_name}' not found, creating new one")

        elif response.status_code != 200:
            raise ZoteroAPIError(
                response.status_code,
                f"Failed to list collections: {response.text}",
                response
            )

    except requests.RequestException as e:
        logger.error(f"Network error while searching collections: {e}")
        raise ZoteroAPIError(0, f"Network error: {str(e)}")

    # Step 2: Create new collection
    # Note: Zotero collections only support 'name' and 'parentCollection'
    # The 'description' parameter is kept for API compatibility but not sent to Zotero
    # (Zotero API does not support collection descriptions)
    collection_data = {
        "name": collection_name,
        "parentCollection": False
    }

    # Generate write token for idempotence
    write_token = uuid.uuid4().hex

    for attempt in range(MAX_RETRIES):
        try:
            # Get current library version
            library_version = get_library_version(library_type, library_id, api_key)

            additional_headers = {
                "Zotero-Write-Token": write_token,
                "If-Unmodified-Since-Version": library_version
            }
            headers = _build_headers(api_key, additional_headers)

            # Create collection
            response = requests.post(
                url,
                headers=headers,
                json=[collection_data],
                timeout=30
            )

            if response.status_code in (200, 201):
                result = response.json()
                new_version = response.headers.get("Last-Modified-Version")

                # Extract collection key from response
                if "successful" in result and "0" in result["successful"]:
                    collection_key = result["successful"]["0"]
                    logger.info(f"Created collection '{collection_name}' with key {collection_key}")
                    return {
                        "key": collection_key,
                        "created": True,
                        "name": collection_name,
                        "version": new_version
                    }
                else:
                    logger.error(f"Unexpected create response: {result}")
                    raise ZoteroAPIError(500, "Collection created but could not extract key")

            elif response.status_code == 412:
                logger.warning(f"Version conflict (412), retrying (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
                continue

            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", RETRY_DELAY))
                logger.warning(f"Rate limit (429), waiting {retry_after}s")
                time.sleep(retry_after)
                continue

            else:
                raise ZoteroAPIError(
                    response.status_code,
                    f"Failed to create collection: {response.text}",
                    response
                )

        except requests.RequestException as e:
            logger.error(f"Network error while creating collection: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            else:
                raise ZoteroAPIError(0, f"Network error: {str(e)}")

    # All retries exhausted
    raise ZoteroAPIError(500, f"Failed to create collection after {MAX_RETRIES} attempts")


def search_item_by_doi(
    library_type: str,
    library_id: str,
    doi: str,
    api_key: str
) -> Optional[str]:
    """
    Search for an existing Zotero item by DOI.

    Args:
        library_type: "users" or "groups"
        library_id: The library ID
        doi: DOI to search for
        api_key: Zotero API key

    Returns:
        Item key (str) if found, None if not found

    Raises:
        ZoteroAPIError: If API request fails (except 404)

    Examples:
        >>> item_key = search_item_by_doi("users", "12345", "10.1234/test", "...")
        >>> if item_key:
        ...     print(f"Found existing item: {item_key}")
    """
    if not doi:
        return None

    prefix = _build_library_prefix(library_type, library_id)
    # Search using itemType filter to exclude attachments
    url = f"{ZOTERO_API_BASE}/{prefix}/items?itemType=-attachment&q={doi}"
    headers = _build_headers(api_key)

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            items = response.json()

            # Check each result for matching DOI (case-insensitive)
            for item in items:
                item_doi = item.get("data", {}).get("DOI", "")
                if item_doi and item_doi.lower() == doi.lower():
                    logger.info(f"Found item with matching DOI: {item['key']}")
                    return item["key"]

            logger.debug(f"No item found with DOI: {doi}")
            return None

        else:
            logger.warning(f"Search by DOI failed with status {response.status_code}")
            return None

    except requests.RequestException as e:
        logger.error(f"Network error while searching by DOI: {e}")
        return None


def search_item_by_url(
    library_type: str,
    library_id: str,
    url: str,
    api_key: str
) -> Optional[str]:
    """
    Search for an existing Zotero item by URL.

    Args:
        library_type: "users" or "groups"
        library_id: The library ID
        url: URL to search for
        api_key: Zotero API key

    Returns:
        Item key (str) if found, None if not found

    Raises:
        ZoteroAPIError: If API request fails (except 404)

    Examples:
        >>> item_key = search_item_by_url("users", "12345", "https://example.com/article", "...")
        >>> if item_key:
        ...     print(f"Found existing item: {item_key}")
    """
    if not url:
        return None

    prefix = _build_library_prefix(library_type, library_id)
    # Extract domain for better search
    search_query = url.split("//")[-1].split("/")[0]  # Extract domain
    search_url = f"{ZOTERO_API_BASE}/{prefix}/items?itemType=-attachment&q={search_query}"
    headers = _build_headers(api_key)

    try:
        response = requests.get(search_url, headers=headers, timeout=10)

        if response.status_code == 200:
            items = response.json()

            # Check each result for matching URL (exact match)
            for item in items:
                item_url = item.get("data", {}).get("url", "")
                if item_url and item_url.lower() == url.lower():
                    logger.info(f"Found item with matching URL: {item['key']}")
                    return item["key"]

            logger.debug(f"No item found with URL: {url}")
            return None

        else:
            logger.warning(f"Search by URL failed with status {response.status_code}")
            return None

    except requests.RequestException as e:
        logger.error(f"Network error while searching by URL: {e}")
        return None


def _normalize_title_for_search(title: str) -> str:
    """
    Normalize title for fuzzy matching (lowercase, alphanum only).

    Args:
        title: Original title

    Returns:
        Normalized title (lowercase, alphanum + spaces)

    Examples:
        >>> _normalize_title_for_search("Machine Learning: A Survey (2024)")
        'machine learning a survey 2024'
    """
    import re
    # Convert to lowercase
    title = title.lower()
    # Keep only alphanumeric and spaces
    title = re.sub(r'[^a-z0-9\s]', '', title)
    # Collapse multiple spaces
    title = re.sub(r'\s+', ' ', title)
    return title.strip()


def search_item_by_title(
    library_type: str,
    library_id: str,
    title: str,
    api_key: str
) -> Optional[str]:
    """
    Search for an existing Zotero item by normalized title.

    Uses fuzzy matching (lowercase, alphanum only) for better recall.

    Args:
        library_type: "users" or "groups"
        library_id: The library ID
        title: Title to search for
        api_key: Zotero API key

    Returns:
        Item key (str) if found, None if not found

    Examples:
        >>> item_key = search_item_by_title("users", "12345", "Machine Learning Review", "...")
    """
    if not title or len(title) < 5:  # Skip very short titles
        return None

    normalized_search = _normalize_title_for_search(title)
    if len(normalized_search) < 5:
        return None

    prefix = _build_library_prefix(library_type, library_id)
    # Search with first significant words
    search_terms = " ".join(normalized_search.split()[:5])  # First 5 words
    url = f"{ZOTERO_API_BASE}/{prefix}/items?itemType=-attachment&q={search_terms}"
    headers = _build_headers(api_key)

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            items = response.json()

            # Check each result for normalized title match
            for item in items:
                item_title = item.get("data", {}).get("title", "")
                if item_title:
                    normalized_item_title = _normalize_title_for_search(item_title)
                    if normalized_item_title == normalized_search:
                        logger.info(f"Found item with matching title: {item['key']}")
                        return item["key"]

            logger.debug(f"No item found with title: {title}")
            return None

        else:
            logger.warning(f"Search by title failed with status {response.status_code}")
            return None

    except requests.RequestException as e:
        logger.error(f"Network error while searching by title: {e}")
        return None


def _merge_item_data(existing_data: Dict, new_data: Dict) -> Dict:
    """
    Intelligently merge new item data into existing item data.

    Strategy:
    - Keep existing data if more complete (non-empty)
    - Merge tags (union of both sets)
    - Never overwrite complete existing data with empty new data

    Args:
        existing_data: Current item data from Zotero
        new_data: New data to merge in

    Returns:
        Merged item data dictionary

    Examples:
        >>> existing = {"title": "Full Title", "abstractNote": "..."}
        >>> new = {"title": "Full Title", "DOI": "10.1234/test"}
        >>> merged = _merge_item_data(existing, new)
        >>> # Result has both abstractNote (kept) and DOI (added)
    """
    merged = existing_data.copy()

    # Fields to merge (prefer existing if both non-empty)
    # Include all publication title variants for different item types
    mergeable_fields = [
        "title", "abstractNote", "DOI", "url", "date",
        # Publication title variants (will be normalized by _clean_item_data_for_type)
        "publicationTitle", "proceedingsTitle", "bookTitle", "websiteTitle",
        "encyclopediaTitle", "dictionaryTitle", "blogTitle", "forumTitle",
        "albumTitle", "programTitle", "seriesTitle",
        # Other common fields
        "language", "pages", "volume", "issue", "ISSN", "ISBN",
        "repository", "institution", "university", "publisher", "place"
    ]

    for field in mergeable_fields:
        existing_value = existing_data.get(field, "")
        new_value = new_data.get(field, "")

        # Only update if new value is non-empty and (existing is empty OR new is longer)
        if new_value and (not existing_value or len(str(new_value)) > len(str(existing_value))):
            merged[field] = new_value

    # Merge tags (union)
    existing_tags = set(tag.get("tag", "") for tag in existing_data.get("tags", []))
    new_tags = set(tag.get("tag", "") for tag in new_data.get("tags", []))
    merged_tags = existing_tags | new_tags
    merged["tags"] = [{"tag": tag} for tag in sorted(merged_tags) if tag]

    # Merge creators if new list is longer
    existing_creators = existing_data.get("creators", [])
    new_creators = new_data.get("creators", [])
    if len(new_creators) > len(existing_creators):
        merged["creators"] = new_creators

    return merged


def create_or_update_item(
    library_type: str,
    library_id: str,
    item_data: Dict,
    api_key: str,
    library_version: Optional[str] = None
) -> Dict:
    """
    Create or update a Zotero item with triple deduplication strategy.

    Deduplication Strategy (priority order):
    1. Search by DOI (if present)
    2. Search by URL (if present)
    3. Search by normalized title

    If existing item found:
    - Update with intelligent merge (keep existing complete data)
    - Merge tags (union)
    - Return action="updated"

    If not found:
    - Create new item
    - Return action="created"

    Args:
        library_type: "users" or "groups"
        library_id: The library ID
        item_data: Complete item data dictionary (Zotero format)
        api_key: Zotero API key
        library_version: Optional library version for concurrency control

    Returns:
        Dictionary with:
        - success (bool): True if operation succeeded
        - item_key (str): Item key (created or updated)
        - action (str): "created" or "updated"
        - message (str): Status message
        - new_version (str): New library version

    Raises:
        ZoteroAPIError: If all retry attempts fail

    Examples:
        >>> result = create_or_update_item(
        ...     library_type="users",
        ...     library_id="12345",
        ...     item_data={
        ...         "itemType": "journalArticle",
        ...         "title": "Test Article",
        ...         "DOI": "10.1234/test",
        ...         ...
        ...     },
        ...     api_key="..."
        ... )
        >>> print(f"{result['action']}: {result['item_key']}")
    """
    prefix = _build_library_prefix(library_type, library_id)
    url = f"{ZOTERO_API_BASE}/{prefix}/items"

    # Step 0: Clean item data (validate fields for item type)
    item_data = _clean_item_data_for_type(item_data)
    logger.debug(f"Cleaned item data for type '{item_data.get('itemType')}': {list(item_data.keys())}")

    # Step 1: Triple deduplication search
    existing_key = None

    # Priority 1: DOI
    doi = item_data.get("DOI")
    if doi:
        existing_key = search_item_by_doi(library_type, library_id, doi, api_key)
        if existing_key:
            logger.info(f"Found existing item by DOI: {existing_key}")

    # Priority 2: URL (if DOI search failed)
    if not existing_key:
        item_url = item_data.get("url")
        if item_url:
            existing_key = search_item_by_url(library_type, library_id, item_url, api_key)
            if existing_key:
                logger.info(f"Found existing item by URL: {existing_key}")

    # Priority 3: Normalized title (if both DOI and URL failed)
    if not existing_key:
        title = item_data.get("title")
        if title:
            existing_key = search_item_by_title(library_type, library_id, title, api_key)
            if existing_key:
                logger.info(f"Found existing item by title: {existing_key}")

    # Step 2: Update existing item if found
    if existing_key:
        try:
            # Get current item data
            item_url = f"{ZOTERO_API_BASE}/{prefix}/items/{existing_key}"
            headers = _build_headers(api_key)
            response = requests.get(item_url, headers=headers, timeout=10)

            if response.status_code == 200:
                existing_item = response.json()
                existing_data = existing_item.get("data", {})
                existing_version = existing_item.get("version")

                # Merge data intelligently and clean for item type
                merged_data = _merge_item_data(existing_data, item_data)
                merged_data = _clean_item_data_for_type(merged_data)

                # Update item
                for attempt in range(MAX_RETRIES):
                    try:
                        update_headers = _build_headers(api_key, {
                            "If-Unmodified-Since-Version": str(existing_version)
                        })

                        update_response = requests.patch(
                            item_url,
                            headers=update_headers,
                            json=merged_data,
                            timeout=30
                        )

                        if update_response.status_code == 204:
                            new_version = update_response.headers.get("Last-Modified-Version")
                            logger.info(f"Updated existing item: {existing_key}")
                            return {
                                "success": True,
                                "item_key": existing_key,
                                "action": "updated",
                                "message": "Item updated successfully",
                                "new_version": new_version
                            }

                        elif update_response.status_code == 412:
                            logger.warning(f"Update version conflict, retrying (attempt {attempt + 1}/{MAX_RETRIES})")
                            # Refresh item data
                            response = requests.get(item_url, headers=headers, timeout=10)
                            if response.status_code == 200:
                                existing_item = response.json()
                                existing_version = existing_item.get("version")
                            time.sleep(RETRY_DELAY)
                            continue

                        elif update_response.status_code == 429:
                            retry_after = int(update_response.headers.get("Retry-After", RETRY_DELAY))
                            logger.warning(f"Rate limit during update, waiting {retry_after}s")
                            time.sleep(retry_after)
                            continue

                        else:
                            logger.warning(f"Update failed with status {update_response.status_code}, falling back to create")
                            break  # Fall through to create

                    except requests.RequestException as e:
                        logger.error(f"Network error during update: {e}")
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(RETRY_DELAY)
                            continue
                        else:
                            break  # Fall through to create

        except Exception as e:
            logger.warning(f"Failed to update existing item: {e}, falling back to create")

    # Step 3: Create new item (if not found or update failed)
    write_token = uuid.uuid4().hex

    for attempt in range(MAX_RETRIES):
        try:
            # Get current library version
            if library_version is None or attempt > 0:
                try:
                    library_version = get_library_version(library_type, library_id, api_key)
                except ZoteroAPIError:
                    logger.warning("Could not get library version, proceeding without it")
                    library_version = None

            additional_headers = {"Zotero-Write-Token": write_token}
            if library_version:
                additional_headers["If-Unmodified-Since-Version"] = library_version

            headers = _build_headers(api_key, additional_headers)

            # Create item
            response = requests.post(
                url,
                headers=headers,
                json=[item_data],  # API expects array
                timeout=30
            )

            if response.status_code in (200, 201):
                result = response.json()
                new_version = response.headers.get("Last-Modified-Version")

                # Extract created item key
                if "successful" in result and "0" in result["successful"]:
                    item_key = result["successful"]["0"]
                    logger.info(f"Created new item: {item_key}")
                    return {
                        "success": True,
                        "item_key": item_key,
                        "action": "created",
                        "message": "Item created successfully",
                        "new_version": new_version
                    }
                else:
                    logger.error(f"Unexpected create response: {result}")
                    return {
                        "success": False,
                        "message": "Item created but could not extract key",
                        "raw_response": result
                    }

            elif response.status_code == 412:
                logger.warning(f"Version conflict (412), retrying (attempt {attempt + 1}/{MAX_RETRIES})")
                library_version = None  # Force refresh
                time.sleep(RETRY_DELAY)
                continue

            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", RETRY_DELAY))
                logger.warning(f"Rate limit (429), waiting {retry_after}s")
                time.sleep(retry_after)
                continue

            elif response.status_code == 409:
                logger.warning(f"Conflict (409), retrying (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY * 2)
                continue

            else:
                raise ZoteroAPIError(
                    response.status_code,
                    f"Failed to create item: {response.text}",
                    response
                )

        except requests.RequestException as e:
            logger.error(f"Network error while creating item: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            else:
                raise ZoteroAPIError(0, f"Network error: {str(e)}")

    # All retries exhausted
    raise ZoteroAPIError(500, f"Failed to create item after {MAX_RETRIES} attempts")
