"""
Zotero Client Extensions for Citation Import
=============================================

Extensions to zotero_client.py for Publish or Perish citation import feature.

New functions:
- get_or_create_collection(): Manage Zotero collections
- search_item_by_doi/url(): Search for existing items (deduplication)
- create_or_update_item(): Create or update items with intelligent merging

Author: RAGpy Team
Date: 2025-12-05
"""

import time
import uuid
import logging
import re
from typing import Optional, Dict, List
import requests

# Import from main zotero_client
from app.utils.zotero_client import (
    ZOTERO_API_BASE,
    ZOTERO_API_VERSION,
    MAX_RETRIES,
    RETRY_DELAY,
    ZoteroAPIError,
    _build_headers,
    _build_library_prefix,
    get_library_version
)

logger = logging.getLogger(__name__)


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
        description: Optional description for new collections
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
    collection_data = {
        "name": collection_name,
        "parentCollection": False
    }

    # Add description if provided (via relations or extra field)
    if description:
        collection_data["description"] = description

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
    merged = existing.copy()

    # Fields to merge (prefer existing if both non-empty)
    mergeable_fields = [
        "title", "abstractNote", "DOI", "url", "date", "publicationTitle",
        "language", "pages", "volume", "issue", "ISSN"
    ]

    for field in mergeable_fields:
        existing_value = existing.get(field, "")
        new_value = new_data.get(field, "")

        # Only update if new value is non-empty and (existing is empty OR new is longer)
        if new_value and (not existing_value or len(str(new_value)) > len(str(existing_value))):
            merged[field] = new_value

    # Merge tags (union)
    existing_tags = set(tag.get("tag", "") for tag in existing.get("tags", []))
    new_tags = set(tag.get("tag", "") for tag in new_data.get("tags", []))
    merged_tags = existing_tags | new_tags
    merged["tags"] = [{"tag": tag} for tag in sorted(merged_tags) if tag]

    # Merge creators if new list is longer
    existing_creators = existing.get("creators", [])
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

                # Merge data intelligently
                merged_data = _merge_item_data(existing_data, item_data)

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
