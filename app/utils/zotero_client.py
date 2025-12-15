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

import json
import time
import uuid
import logging
import random
from typing import Optional, Dict, List
import requests

logger = logging.getLogger(__name__)

# Constants
ZOTERO_API_BASE = "https://api.zotero.org"
ZOTERO_API_VERSION = "3"
MAX_RETRIES = 5  # Increased for library contention scenarios
RETRY_DELAY = 2  # seconds (base delay)


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
                # Version conflict - retry with new version and exponential backoff
                backoff_delay = RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Version conflict (412), retrying in {backoff_delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                library_version = None  # Force refresh on next iteration
                time.sleep(backoff_delay)
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

# Mapping for item types that use an alternative field instead of publicationTitle
# These types have a specific field where we should put the publication source
ALTERNATIVE_PUBLICATION_FIELD_MAP = {
    "preprint": "repository",      # e.g., "arXiv", "bioRxiv", "SSRN", "medRxiv"
    "report": "institution",       # e.g., "RAND Corporation", "World Bank"
    "thesis": "university",        # e.g., "MIT", "Stanford University"
    "patent": "issuingAuthority",  # e.g., "US Patent Office", "EPO"
    "statute": "code",             # e.g., "US Code", "CFR"
    "case": "reporter",            # e.g., "Supreme Court Reporter"
}

# Item types that truly don't have any "publication title" type field
ITEM_TYPES_WITHOUT_PUBLICATION_TITLE = {
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
    2. For special item types (preprint, thesis, report), maps to alternative fields
    3. Removes invalid fields that would cause Zotero API errors
    4. Preserves all other valid fields

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
        >>> # Result: {"itemType": "preprint", "repository": "arXiv"}

        >>> data = {"itemType": "thesis", "publicationTitle": "MIT"}
        >>> cleaned = _clean_item_data_for_type(data)
        >>> # Result: {"itemType": "thesis", "university": "MIT"}
    """
    cleaned = item_data.copy()
    item_type = cleaned.get("itemType", "")

    # Get the source publication title value (if any)
    source_pub_title = None
    for field in ALL_PUBLICATION_TITLE_FIELDS:
        if field in cleaned and cleaned[field]:
            source_pub_title = cleaned[field]
            break

    # Also check alternative fields that might have been set directly
    # (in case LLM or other source already set the correct field)
    alternative_fields_present = {}
    for alt_field in ALTERNATIVE_PUBLICATION_FIELD_MAP.values():
        if alt_field in cleaned and cleaned[alt_field]:
            alternative_fields_present[alt_field] = cleaned[alt_field]

    # Remove all publication title fields first
    for field in ALL_PUBLICATION_TITLE_FIELDS:
        cleaned.pop(field, None)

    # Handle item types with alternative fields (preprint, thesis, report, etc.)
    if item_type in ALTERNATIVE_PUBLICATION_FIELD_MAP:
        alt_field = ALTERNATIVE_PUBLICATION_FIELD_MAP[item_type]

        # If we have a publication title value, map it to the alternative field
        if source_pub_title:
            cleaned[alt_field] = source_pub_title
            logger.debug(f"Mapped publicationTitle to {alt_field} for {item_type}: '{source_pub_title}'")
        # If alternative field was already set, keep it
        elif alt_field in alternative_fields_present:
            cleaned[alt_field] = alternative_fields_present[alt_field]
            logger.debug(f"Kept existing {alt_field} for {item_type}")

    # Handle item types without any publication title support
    elif item_type in ITEM_TYPES_WITHOUT_PUBLICATION_TITLE:
        if source_pub_title:
            logger.debug(f"Removed publicationTitle for {item_type} (not supported)")

    # Handle standard item types with publication title variants
    elif source_pub_title:
        correct_field = PUBLICATION_TITLE_FIELD_MAP.get(item_type)
        if correct_field:
            cleaned[correct_field] = source_pub_title
            logger.debug(f"Mapped publicationTitle to {correct_field} for {item_type}")
        else:
            # Item type not in our map - default to keeping publicationTitle
            cleaned["publicationTitle"] = source_pub_title
            logger.debug(f"Keeping publicationTitle for type: {item_type}")

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


def fetch_collection_items(
    library_type: str,
    library_id: str,
    collection_key: str,
    api_key: str,
    limit: int = 100
) -> List[Dict]:
    """
    Fetch all items in a Zotero collection for deduplication.

    This function retrieves all items in a collection, handling pagination
    automatically. Useful for checking existing items before import.

    Args:
        library_type: "users" or "groups"
        library_id: The library ID
        collection_key: Key of the collection to fetch items from
        api_key: Zotero API key
        limit: Items per page (max 100)

    Returns:
        List of item dictionaries with keys: key, DOI, title, itemType

    Raises:
        ZoteroAPIError: If API request fails

    Examples:
        >>> items = fetch_collection_items("users", "12345", "ABC123", "...")
        >>> existing_dois = {item["DOI"] for item in items if item.get("DOI")}
        >>> print(f"Found {len(existing_dois)} items with DOIs")
    """
    prefix = _build_library_prefix(library_type, library_id)
    url = f"{ZOTERO_API_BASE}/{prefix}/collections/{collection_key}/items"
    headers = _build_headers(api_key)

    all_items = []
    start = 0

    while True:
        params = {
            "start": start,
            "limit": min(limit, 100),
            "itemType": "-attachment"  # Exclude attachments
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)

            if response.status_code == 200:
                items = response.json()

                if not items:
                    break

                # Extract relevant fields for deduplication
                for item in items:
                    data = item.get("data", {})
                    all_items.append({
                        "key": item.get("key"),
                        "DOI": data.get("DOI", ""),
                        "title": data.get("title", ""),
                        "itemType": data.get("itemType", "")
                    })

                # Check if there are more items
                total_results = int(response.headers.get("Total-Results", 0))
                start += len(items)

                if start >= total_results:
                    break

                logger.debug(f"Fetched {start}/{total_results} items from collection {collection_key}")

            elif response.status_code == 404:
                logger.warning(f"Collection {collection_key} not found")
                return []

            else:
                raise ZoteroAPIError(
                    response.status_code,
                    f"Failed to fetch collection items: {response.text}",
                    response
                )

        except requests.RequestException as e:
            logger.error(f"Network error fetching collection items: {e}")
            raise ZoteroAPIError(0, f"Network error: {str(e)}")

    logger.info(f"Fetched {len(all_items)} items from collection {collection_key}")
    return all_items


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
    api_key: str,
    target_creators: Optional[List[Dict]] = None,
    target_date: Optional[str] = None
) -> Optional[str]:
    """
    Search for an existing Zotero item by normalized title with optional creator/date validation.

    Uses fuzzy matching (lowercase, alphanum only) for better recall,
    then validates against authors and date to prevent false positives.

    Args:
        library_type: "users" or "groups"
        library_id: The library ID
        title: Title to search for
        api_key: Zotero API key
        target_creators: Optional list of creator dicts (to validate match)
        target_date: Optional date string (to validate match)

    Returns:
        Item key (str) if found and validated, None otherwise

    Examples:
        >>> item_key = search_item_by_title(
        ...     "users", "12345", "Machine Learning", "...",
        ...     target_creators=[{"lastName": "Smith", "creatorType": "author"}]
        ... )
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

            # Check each result for normalized title match AND validation
            for item in items:
                item_data = item.get("data", {})
                item_title = item_data.get("title", "")
                
                if not item_title:
                    continue

                normalized_item_title = _normalize_title_for_search(item_title)
                
                if normalized_item_title == normalized_search:
                    # Title matches! Now strictly validate other fields if provided
                    
                    # 1. Author Validation
                    if target_creators:
                        item_creators = item_data.get("creators", [])
                        
                        # Extract last names from target
                        target_last_names = {
                            c.get("lastName", "").lower() 
                            for c in target_creators 
                            if c.get("lastName")
                        }
                        
                        # Extract last names from candidate item
                        item_last_names = {
                            c.get("lastName", "").lower() 
                            for c in item_creators 
                            if c.get("lastName")
                        }
                        
                        # If both have authors, require overlap
                        if target_last_names and item_last_names:
                            if not target_last_names.intersection(item_last_names):
                                logger.info(
                                    f"Title match '{title}' REJECTED: Author mismatch "
                                    f"(Target: {target_last_names}, Found: {item_last_names})"
                                )
                                continue  # Distinct authors → likely different paper
                    
                    # 2. Date/Year Validation
                    if target_date:
                        item_date = item_data.get("date", "")
                        # Simple 4-digit year extraction
                        import re
                        target_year_match = re.search(r'\d{4}', str(target_date))
                        item_year_match = re.search(r'\d{4}', str(item_date))
                        
                        if target_year_match and item_year_match:
                            target_year = int(target_year_match.group(0))
                            item_year = int(item_year_match.group(0))
                            
                            # Allow 1 year variance (prepints vs publication dates)
                            if abs(target_year - item_year) > 1:
                                logger.info(
                                    f"Title match '{title}' REJECTED: Year mismatch "
                                    f"(Target: {target_year}, Found: {item_year})"
                                )
                                continue

                    logger.info(f"Found existing item by title (validated): {item['key']}")
                    return item["key"]

            logger.debug(f"No valid item found with title: {title}")
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
            # Pass creators and date for stronger validation
            existing_key = search_item_by_title(
                library_type, 
                library_id, 
                title, 
                api_key,
                target_creators=item_data.get("creators"),
                target_date=item_data.get("date")
            )
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

            # DEBUG: Log item data being sent
            logger.info(f"Creating item with itemType={item_data.get('itemType')}, title={item_data.get('title', '')[:50]}")
            logger.debug(f"Full item_data: {json.dumps(item_data, indent=2, default=str)}")

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
                # Log full error details before raising
                logger.error(f"Zotero API error {response.status_code}: {response.text}")
                logger.error(f"Item data that caused error: itemType={item_data.get('itemType')}, creators={item_data.get('creators')}")
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


# ============================================================================
# Clustering Tag Management
# ============================================================================
# The following function supports the document clustering feature (Step 4.b):
# - Adds cluster tags to multiple Zotero items
# - Handles version conflicts and rate limits
# - Returns detailed success/failure counts
# ============================================================================


def add_tags_to_items(
    library_type: str,
    library_id: str,
    tag_mapping: Dict[str, str],
    api_key: str,
    replace_all: bool = True
) -> Dict:
    """
    Add or replace cluster tags on multiple Zotero items.

    This function manages tags on Zotero items for organizing documents by cluster.
    It's designed to work with the clustering feature (Step 4.b) which generates
    tags like '_MaBiblio_01', '_MaBiblio_02', etc.

    Args:
        library_type: "users" or "groups" depending on library ownership.
        library_id: The library ID (user ID or group ID).
        tag_mapping: Dictionary mapping Zotero item keys to cluster tags.
                    Example: {"ABC123": "_MaBiblio_01", "DEF456": "_MaBiblio_02"}
        api_key: Zotero API key with write permissions.
        replace_all: If True (default), replaces ALL existing tags with only the
                    cluster tag. If False, adds the cluster tag to existing tags.

    Returns:
        Dictionary with:
            - success_count (int): Number of items successfully tagged
            - failed_count (int): Number of items that failed
            - skipped_count (int): Number of items where tag already existed (replace_all=False only)
            - errors (list): List of error details for failed items

    Example:
        >>> result = add_tags_to_items(
        ...     library_type="users",
        ...     library_id="12345",
        ...     tag_mapping={"ABC": "_MaBiblio_01", "DEF": "_MaBiblio_02"},
        ...     api_key="your_api_key",
        ...     replace_all=True  # Remove all existing tags
        ... )
        >>> print(f"Tagged {result['success_count']} items")

    Note:
        - When replace_all=True: ALL existing tags are removed and replaced by cluster tag.
        - When replace_all=False: Skips items where the tag already exists.
        - Uses If-Unmodified-Since-Version header for safe concurrent updates.
        - Implements retry logic for version conflicts (412) and rate limits (429).
    """
    prefix = _build_library_prefix(library_type, library_id)

    success_count = 0
    failed_count = 0
    skipped_count = 0
    errors = []

    total_items = len(tag_mapping)
    mode_str = "REPLACE ALL tags with" if replace_all else "ADD tags to"
    logger.info(f"Starting to {mode_str} cluster tags on {total_items} Zotero items")

    for idx, (item_key, tag) in enumerate(tag_mapping.items(), 1):
        try:
            # Get current item to retrieve existing tags and version
            item = get_item(library_type, library_id, item_key, api_key)
            item_data = item.get("data", {})
            item_version = str(item.get("version", "0"))

            # Get existing tags
            existing_tags = item_data.get("tags", [])
            existing_tag_values = {t.get("tag") for t in existing_tags if t.get("tag")}

            if replace_all:
                # REPLACE mode: Remove all tags, set only the cluster tag
                updated_tags = [{"tag": tag}]
                # Skip only if this exact tag is the ONLY existing tag
                if existing_tag_values == {tag}:
                    logger.debug(f"Item {item_key} already has only tag '{tag}', skipping")
                    skipped_count += 1
                    success_count += 1
                    continue
            else:
                # ADD mode: Skip if tag already exists
                if tag in existing_tag_values:
                    logger.debug(f"Tag '{tag}' already exists on item {item_key}")
                    skipped_count += 1
                    success_count += 1  # Count as success since goal is achieved
                    continue
                # Add new tag to existing tags
                updated_tags = existing_tags + [{"tag": tag}]

            # Build PATCH request
            url = f"{ZOTERO_API_BASE}/{prefix}/items/{item_key}"

            # Retry loop for version conflicts
            for attempt in range(MAX_RETRIES):
                headers = _build_headers(api_key, {
                    "If-Unmodified-Since-Version": item_version
                })

                response = requests.patch(
                    url,
                    headers=headers,
                    json={"tags": updated_tags},
                    timeout=30
                )

                if response.status_code == 204:
                    # Success - no content returned
                    action = "Replaced tags with" if replace_all else "Added tag"
                    logger.debug(f"[{idx}/{total_items}] {action} '{tag}' on item {item_key}")
                    success_count += 1
                    break

                elif response.status_code == 412:
                    # Version conflict - refresh version and retry
                    logger.warning(f"Version conflict for {item_key}, retrying (attempt {attempt + 1})")
                    item = get_item(library_type, library_id, item_key, api_key)
                    item_version = str(item.get("version", "0"))
                    # Update tags in case they changed
                    existing_tags = item.get("data", {}).get("tags", [])
                    existing_tag_values = {t.get("tag") for t in existing_tags}

                    if replace_all:
                        # REPLACE mode: Check if already replaced correctly
                        if existing_tag_values == {tag}:
                            skipped_count += 1
                            success_count += 1
                            break
                        updated_tags = [{"tag": tag}]
                    else:
                        # ADD mode: Check if tag was added by concurrent operation
                        if tag in existing_tag_values:
                            skipped_count += 1
                            success_count += 1
                            break
                        updated_tags = existing_tags + [{"tag": tag}]

                    time.sleep(RETRY_DELAY)
                    continue

                elif response.status_code == 429:
                    # Rate limit - wait and retry
                    retry_after = int(response.headers.get("Retry-After", RETRY_DELAY * 2))
                    logger.warning(f"Rate limit hit, waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue

                elif response.status_code == 404:
                    # Item not found
                    logger.warning(f"Item {item_key} not found in Zotero")
                    failed_count += 1
                    errors.append({
                        "item_key": item_key,
                        "tag": tag,
                        "error": "Item not found in Zotero"
                    })
                    break

                else:
                    # Other error
                    logger.error(f"Failed to tag item {item_key}: {response.status_code} - {response.text}")
                    failed_count += 1
                    errors.append({
                        "item_key": item_key,
                        "tag": tag,
                        "error": f"HTTP {response.status_code}: {response.text[:100]}"
                    })
                    break
            else:
                # All retries exhausted
                failed_count += 1
                errors.append({
                    "item_key": item_key,
                    "tag": tag,
                    "error": f"Failed after {MAX_RETRIES} attempts"
                })

        except ZoteroAPIError as e:
            failed_count += 1
            errors.append({
                "item_key": item_key,
                "tag": tag,
                "error": str(e)
            })
            logger.error(f"Zotero API error for item {item_key}: {e}")

        except Exception as e:
            failed_count += 1
            errors.append({
                "item_key": item_key,
                "tag": tag,
                "error": str(e)
            })
            logger.error(f"Unexpected error tagging item {item_key}: {e}")

    logger.info(
        f"Tagging complete: {success_count} succeeded, {failed_count} failed, "
        f"{skipped_count} already had tag"
    )

    return {
        "success_count": success_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "errors": errors
    }


# =============================================================================
# FILE ATTACHMENT UPLOAD API
# =============================================================================


def upload_file_attachment(
    library_type: str,
    library_id: str,
    parent_item_key: str,
    pdf_bytes: bytes,
    filename: str,
    md5_hash: str,
    mtime: int,
    api_key: str,
    original_url: str = "",
    content_type: str = "application/pdf"
) -> Dict:
    """
    Upload a file as an attachment to a Zotero item.

    Zotero file upload is a 4-step process:
    1. Create an attachment item (imported_url link mode)
    2. Request upload authorization from Zotero
    3. Upload file to Zotero's storage (S3)
    4. Register upload completion with Zotero

    Args:
        library_type: "users" or "groups"
        library_id: Library ID (user ID or group ID)
        parent_item_key: Key of the parent item to attach file to
        pdf_bytes: Raw file content as bytes
        filename: Name for the attachment file
        md5_hash: MD5 hash of pdf_bytes
        mtime: Modification time in milliseconds (Unix timestamp * 1000)
        api_key: Zotero API key
        original_url: Original URL the file was downloaded from (optional)
        content_type: MIME type of the file (default: application/pdf)

    Returns:
        Dictionary with:
        - success (bool): Whether upload was successful
        - attachment_key (str): Key of the created attachment item
        - message (str): Status/error message

    Raises:
        ZoteroAPIError: If any step of the upload process fails

    Example:
        >>> result = upload_file_attachment(
        ...     library_type="users",
        ...     library_id="12345",
        ...     parent_item_key="ABC123XY",
        ...     pdf_bytes=b'%PDF-...',
        ...     filename="article.pdf",
        ...     md5_hash="a1b2c3...",
        ...     mtime=1702300000000,
        ...     api_key="xxx"
        ... )
        >>> print(result)
        {'success': True, 'attachment_key': 'DEF456GH', 'message': 'File uploaded successfully'}
    """
    from urllib.parse import unquote

    # Decode URL-encoded filename and sanitize
    filename = unquote(filename)
    # Remove or replace characters that might cause issues
    filename = filename.replace('"', "'").replace('\n', ' ').replace('\r', '')
    # Limit filename length (Zotero has issues with very long filenames)
    if len(filename) > 200:
        base, ext = filename.rsplit('.', 1) if '.' in filename else (filename, 'pdf')
        filename = f"{base[:190]}.{ext}"

    logger.info(f"Starting file upload to Zotero: {filename} ({len(pdf_bytes)} bytes)")

    try:
        # Step 1: Create attachment item
        attachment_key, library_version = _create_attachment_item(
            library_type=library_type,
            library_id=library_id,
            parent_item_key=parent_item_key,
            filename=filename,
            content_type=content_type,
            original_url=original_url,
            api_key=api_key
        )
        logger.info(f"Created attachment item: {attachment_key}")

        # Step 2: Request upload authorization
        upload_auth = _request_upload_authorization(
            library_type=library_type,
            library_id=library_id,
            attachment_key=attachment_key,
            md5_hash=md5_hash,
            filename=filename,
            filesize=len(pdf_bytes),
            mtime=mtime,
            api_key=api_key
        )

        # Check if file already exists (Zotero returns 200 with "exists": 1)
        if upload_auth.get("exists"):
            logger.info(f"File already exists in Zotero: {attachment_key}")
            return {
                "success": True,
                "attachment_key": attachment_key,
                "message": "File already exists in Zotero (MD5 match)"
            }

        logger.info(f"Upload authorized, uploading to storage...")

        # Step 3: Upload to storage
        # Zotero provides pre-built multipart body components
        upload_url = upload_auth.get("url")
        upload_prefix = upload_auth.get("prefix", "")
        upload_suffix = upload_auth.get("suffix", "")
        upload_content_type = upload_auth.get("contentType", "")
        upload_key = upload_auth.get("uploadKey")

        if not upload_url or not upload_key or not upload_prefix:
            logger.error(f"Invalid upload authorization response: url={bool(upload_url)}, "
                        f"key={bool(upload_key)}, prefix={bool(upload_prefix)}")
            raise ZoteroAPIError(500, "Invalid upload authorization response")

        upload_success = _upload_to_storage(
            upload_url=upload_url,
            pdf_bytes=pdf_bytes,
            prefix=upload_prefix,
            suffix=upload_suffix,
            content_type_header=upload_content_type
        )

        if not upload_success:
            raise ZoteroAPIError(500, "File upload to storage failed")

        logger.info("File uploaded to storage, registering upload...")

        # Step 4: Register upload
        register_success = _register_upload(
            library_type=library_type,
            library_id=library_id,
            attachment_key=attachment_key,
            upload_key=upload_key,
            api_key=api_key
        )

        if not register_success:
            raise ZoteroAPIError(500, "Failed to register upload with Zotero")

        logger.info(f"File upload complete: {attachment_key}")

        return {
            "success": True,
            "attachment_key": attachment_key,
            "message": "File uploaded successfully"
        }

    except ZoteroAPIError as e:
        logger.error(f"Zotero API error during file upload: {e}")
        return {
            "success": False,
            "attachment_key": "",
            "message": str(e)
        }

    except Exception as e:
        logger.error(f"Unexpected error during file upload: {e}")
        return {
            "success": False,
            "attachment_key": "",
            "message": f"Unexpected error: {str(e)}"
        }


def _create_attachment_item(
    library_type: str,
    library_id: str,
    parent_item_key: str,
    filename: str,
    content_type: str,
    original_url: str,
    api_key: str
) -> tuple:
    """
    Step 1: Create an attachment item in Zotero.

    Creates an imported_url attachment item that will hold the uploaded file.

    Args:
        library_type: "users" or "groups"
        library_id: Library ID
        parent_item_key: Parent item to attach to
        filename: Attachment filename
        content_type: MIME type
        original_url: Source URL of the file
        api_key: Zotero API key

    Returns:
        Tuple of (attachment_key, library_version)

    Raises:
        ZoteroAPIError: If creation fails
    """
    prefix = _build_library_prefix(library_type, library_id)
    url = f"{ZOTERO_API_BASE}/{prefix}/items"

    # Sanitize filename for Zotero (remove characters that cause issues)
    safe_filename = filename.replace('"', "'").replace('\n', ' ').replace('\r', '')
    # Truncate if too long
    if len(safe_filename) > 200:
        base, ext = safe_filename.rsplit('.', 1) if '.' in safe_filename else (safe_filename, 'pdf')
        safe_filename = f"{base[:190]}.{ext}"

    logger.debug(f"Creating attachment item for parent {parent_item_key}, filename: {safe_filename}")

    # Attachment item data
    # For file uploads, use "imported_file" linkMode (not "imported_url" which is for links)
    attachment_data = [{
        "itemType": "attachment",
        "parentItem": parent_item_key,
        "linkMode": "imported_file",
        "title": safe_filename,
        "contentType": content_type
    }]

    # Generate write token for idempotency (Zotero requires 5-32 chars, .hex gives 32)
    write_token = uuid.uuid4().hex

    headers = _build_headers(api_key, {
        "Zotero-Write-Token": write_token
    })

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=attachment_data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                # Success response format: {"successful": {"0": {...}}, "unchanged": {}, "failed": {}}
                if "successful" in result and "0" in result["successful"]:
                    attachment_key = result["successful"]["0"]["key"]
                    library_version = response.headers.get("Last-Modified-Version", "0")
                    return (attachment_key, library_version)

                # Check if creation failed
                if "failed" in result and result["failed"]:
                    error_msg = str(result["failed"])
                    raise ZoteroAPIError(response.status_code, f"Failed to create attachment: {error_msg}")

            elif response.status_code == 412:
                # Version conflict - retry
                logger.warning(f"Version conflict creating attachment (attempt {attempt + 1})")
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue

            elif response.status_code == 429:
                # Rate limit
                retry_after = int(response.headers.get("Retry-After", RETRY_DELAY * 2))
                logger.warning(f"Rate limit hit, waiting {retry_after}s")
                time.sleep(retry_after)
                continue

            else:
                # Log detailed error info for debugging
                logger.error(f"Attachment creation failed: HTTP {response.status_code}")
                logger.error(f"Request payload: {json.dumps(attachment_data, indent=2)}")
                logger.error(f"Response: {response.text[:500]}")
                raise ZoteroAPIError(
                    response.status_code,
                    f"Failed to create attachment item: {response.text}",
                    response
                )

        except requests.RequestException as e:
            logger.warning(f"Network error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            raise ZoteroAPIError(0, f"Network error: {str(e)}")

    raise ZoteroAPIError(500, f"Failed to create attachment after {MAX_RETRIES} attempts")


def _request_upload_authorization(
    library_type: str,
    library_id: str,
    attachment_key: str,
    md5_hash: str,
    filename: str,
    filesize: int,
    mtime: int,
    api_key: str
) -> Dict:
    """
    Step 2: Request authorization to upload a file to Zotero storage.

    This step asks Zotero for permission to upload a file and returns
    the S3 upload URL and parameters.

    Args:
        library_type: "users" or "groups"
        library_id: Library ID
        attachment_key: Key of the attachment item created in step 1
        md5_hash: MD5 hash of the file content
        filename: Name of the file
        filesize: Size of the file in bytes
        mtime: Modification time in milliseconds
        api_key: Zotero API key

    Returns:
        Dictionary with upload authorization:
        - url: S3 upload URL
        - uploadKey: Key to use when registering upload
        - params: Parameters to include in S3 upload request
        - exists: 1 if file already exists (no upload needed)

    Raises:
        ZoteroAPIError: If authorization fails
    """
    prefix = _build_library_prefix(library_type, library_id)
    url = f"{ZOTERO_API_BASE}/{prefix}/items/{attachment_key}/file"

    # Request body (URL-encoded form data)
    data = {
        "md5": md5_hash,
        "filename": filename,
        "filesize": str(filesize),
        "mtime": str(mtime)
    }

    headers = {
        "Zotero-API-Key": api_key,
        "Zotero-API-Version": ZOTERO_API_VERSION,
        "Content-Type": "application/x-www-form-urlencoded",
        "If-None-Match": "*"  # Indicates new file upload
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                url,
                headers=headers,
                data=data,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()

            elif response.status_code == 412:
                # File already exists with this MD5
                logger.info("File with same MD5 already exists")
                return {"exists": 1}

            elif response.status_code == 413:
                raise ZoteroAPIError(413, "File too large for Zotero storage")

            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", RETRY_DELAY * 2))
                logger.warning(f"Rate limit hit, waiting {retry_after}s")
                time.sleep(retry_after)
                continue

            else:
                raise ZoteroAPIError(
                    response.status_code,
                    f"Upload authorization failed: {response.text}",
                    response
                )

        except requests.RequestException as e:
            logger.warning(f"Network error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            raise ZoteroAPIError(0, f"Network error: {str(e)}")

    raise ZoteroAPIError(500, f"Upload authorization failed after {MAX_RETRIES} attempts")


def _upload_to_storage(
    upload_url: str,
    pdf_bytes: bytes,
    prefix: str,
    suffix: str,
    content_type_header: str
) -> bool:
    """
    Step 3: Upload file to Zotero's S3 storage.

    Uses the pre-built multipart body from Zotero's authorization response
    to upload the file content to S3. Zotero provides the exact multipart
    format including boundary, form fields, and structure.

    Args:
        upload_url: S3 upload URL from authorization response
        pdf_bytes: File content to upload
        prefix: Pre-built multipart body prefix (includes all form fields)
        suffix: Pre-built multipart body suffix (closing boundary)
        content_type_header: Content-Type header with boundary from authorization

    Returns:
        True if upload successful, False otherwise

    Raises:
        ZoteroAPIError: If upload fails
    """
    # Zotero provides a pre-built multipart body:
    # - prefix: Contains all form fields (key, policy, signature, etc.)
    # - We insert the file bytes in between
    # - suffix: Contains the closing boundary
    #
    # The body is: prefix + file_bytes + suffix
    # Content-Type must match exactly what Zotero provided (with correct boundary)

    # Build the complete multipart body
    body = prefix.encode('utf-8') + pdf_bytes + suffix.encode('utf-8')

    headers = {
        "Content-Type": content_type_header,
        "Content-Length": str(len(body))
    }

    logger.debug(f"Uploading to S3: {len(body)} bytes, Content-Type: {content_type_header[:60]}...")

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                upload_url,
                headers=headers,
                data=body,
                timeout=120  # Longer timeout for file upload
            )

            # S3 returns 201 on success (as specified in success_action_status)
            if response.status_code in (200, 201, 204):
                logger.info("File uploaded to storage successfully")
                return True

            # S3 errors
            logger.warning(f"Storage upload failed: HTTP {response.status_code}")
            logger.debug(f"Response: {response.text[:500] if response.text else 'empty'}")

            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue

        except requests.RequestException as e:
            logger.warning(f"Network error during storage upload (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue

    return False


def _register_upload(
    library_type: str,
    library_id: str,
    attachment_key: str,
    upload_key: str,
    api_key: str
) -> bool:
    """
    Step 4: Register completed upload with Zotero.

    After uploading to S3, this step notifies Zotero that the upload
    is complete and associates the file with the attachment item.

    Args:
        library_type: "users" or "groups"
        library_id: Library ID
        attachment_key: Key of the attachment item
        upload_key: Upload key from authorization response
        api_key: Zotero API key

    Returns:
        True if registration successful, False otherwise

    Raises:
        ZoteroAPIError: If registration fails
    """
    prefix = _build_library_prefix(library_type, library_id)
    url = f"{ZOTERO_API_BASE}/{prefix}/items/{attachment_key}/file"

    data = {"upload": upload_key}

    headers = {
        "Zotero-API-Key": api_key,
        "Zotero-API-Version": ZOTERO_API_VERSION,
        "Content-Type": "application/x-www-form-urlencoded",
        "If-None-Match": "*"
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                url,
                headers=headers,
                data=data,
                timeout=30
            )

            if response.status_code == 204:
                logger.info("Upload registered successfully")
                return True

            elif response.status_code == 412:
                # Version conflict or file already registered
                logger.warning("Upload registration conflict (may already be registered)")
                return True  # Consider success if already registered

            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", RETRY_DELAY * 2))
                logger.warning(f"Rate limit hit, waiting {retry_after}s")
                time.sleep(retry_after)
                continue

            else:
                logger.error(f"Upload registration failed: {response.status_code} - {response.text}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue

        except requests.RequestException as e:
            logger.warning(f"Network error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue

    return False
