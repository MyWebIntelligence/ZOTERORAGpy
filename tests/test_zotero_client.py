"""
Unit tests for Zotero API client.

These tests use mocking to avoid real API calls.
Run with: pytest tests/test_zotero_client.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from app.utils import zotero_client


class TestBuildHeaders:
    """Test header building function."""

    def test_basic_headers(self):
        """Test basic header construction."""
        headers = zotero_client._build_headers("test_key")

        assert headers["Zotero-API-Key"] == "test_key"
        assert headers["Zotero-API-Version"] == "3"
        assert headers["Content-Type"] == "application/json"

    def test_additional_headers(self):
        """Test merging additional headers."""
        headers = zotero_client._build_headers(
            "test_key",
            {"Custom-Header": "value"}
        )

        assert headers["Custom-Header"] == "value"
        assert headers["Zotero-API-Key"] == "test_key"


class TestBuildLibraryPrefix:
    """Test library prefix building."""

    def test_users_prefix(self):
        """Test users library prefix."""
        prefix = zotero_client._build_library_prefix("users", "12345")
        assert prefix == "users/12345"

    def test_groups_prefix(self):
        """Test groups library prefix."""
        prefix = zotero_client._build_library_prefix("groups", "67890")
        assert prefix == "groups/67890"

    def test_invalid_type(self):
        """Test invalid library type raises error."""
        with pytest.raises(ValueError):
            zotero_client._build_library_prefix("invalid", "12345")


class TestVerifyApiKey:
    """Test API key verification."""

    @patch('app.utils.zotero_client.requests.get')
    def test_valid_key(self, mock_get):
        """Test successful key verification."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "username": "testuser",
            "userID": "12345"
        }
        mock_get.return_value = mock_response

        result = zotero_client.verify_api_key("valid_key")

        assert result["username"] == "testuser"
        assert result["userID"] == "12345"
        mock_get.assert_called_once()

    @patch('app.utils.zotero_client.requests.get')
    def test_invalid_key(self, mock_get):
        """Test invalid key raises error."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"
        mock_get.return_value = mock_response

        with pytest.raises(zotero_client.ZoteroAPIError) as exc_info:
            zotero_client.verify_api_key("invalid_key")

        assert exc_info.value.status_code == 403


class TestGetLibraryVersion:
    """Test library version retrieval."""

    @patch('app.utils.zotero_client.requests.get')
    def test_get_version(self, mock_get):
        """Test retrieving library version."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Last-Modified-Version": "12345"}
        mock_get.return_value = mock_response

        version = zotero_client.get_library_version("users", "123", "test_key")

        assert version == "12345"

    @patch('app.utils.zotero_client.requests.get')
    def test_version_error(self, mock_get):
        """Test error when retrieving version."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_get.return_value = mock_response

        with pytest.raises(zotero_client.ZoteroAPIError):
            zotero_client.get_library_version("users", "123", "test_key")


class TestCheckNoteExists:
    """Test note existence checking."""

    @patch('app.utils.zotero_client.requests.get')
    def test_note_exists(self, mock_get):
        """Test finding existing note with sentinel."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "data": {
                    "note": "<!-- ragpy-note-id:test-uuid --><p>Note content</p>"
                }
            }
        ]
        mock_get.return_value = mock_response

        exists = zotero_client.check_note_exists(
            "users", "123", "ITEMKEY", "ragpy-note-id:test-uuid", "test_key"
        )

        assert exists is True

    @patch('app.utils.zotero_client.requests.get')
    def test_note_not_exists(self, mock_get):
        """Test when note does not exist."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "data": {
                    "note": "<p>Different note</p>"
                }
            }
        ]
        mock_get.return_value = mock_response

        exists = zotero_client.check_note_exists(
            "users", "123", "ITEMKEY", "ragpy-note-id:test-uuid", "test_key"
        )

        assert exists is False


class TestCreateChildNote:
    """Test child note creation."""

    @patch('app.utils.zotero_client.get_library_version')
    @patch('app.utils.zotero_client.requests.post')
    def test_successful_creation(self, mock_post, mock_get_version):
        """Test successful note creation."""
        mock_get_version.return_value = "100"

        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.headers = {"Last-Modified-Version": "101"}
        # According to Zotero API docs, successful["0"] is the itemKey directly
        mock_response.json.return_value = {
            "successful": {
                "0": "NOTEKEY123"
            },
            "unchanged": {},
            "failed": {}
        }
        mock_post.return_value = mock_response

        result = zotero_client.create_child_note(
            library_type="users",
            library_id="123",
            item_key="ITEMKEY",
            note_html="<p>Test note</p>",
            tags=["test"],
            api_key="test_key"
        )

        assert result["success"] is True
        assert result["note_key"] == "NOTEKEY123"
        assert result["new_version"] == "101"

    @patch('app.utils.zotero_client.get_library_version')
    @patch('app.utils.zotero_client.requests.post')
    def test_parent_not_found(self, mock_post, mock_get_version):
        """Test error when parent item not found."""
        mock_get_version.return_value = "100"

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Parent item not found"
        mock_post.return_value = mock_response

        with pytest.raises(zotero_client.ZoteroAPIError) as exc_info:
            zotero_client.create_child_note(
                library_type="users",
                library_id="123",
                item_key="INVALID",
                note_html="<p>Test</p>",
                api_key="test_key"
            )

        assert exc_info.value.status_code == 404

    @patch('app.utils.zotero_client.get_library_version')
    @patch('app.utils.zotero_client.requests.post')
    def test_version_conflict_retry(self, mock_post, mock_get_version):
        """Test retry on version conflict (412)."""
        # First call returns old version, second returns new
        mock_get_version.side_effect = ["100", "101"]

        # First POST returns 412, second succeeds
        mock_response_412 = Mock()
        mock_response_412.status_code = 412
        mock_response_412.text = "Version conflict"

        mock_response_success = Mock()
        mock_response_success.status_code = 201
        mock_response_success.headers = {"Last-Modified-Version": "102"}
        # Correct format according to Zotero API docs
        mock_response_success.json.return_value = {
            "successful": {"0": "NOTEKEY"},
            "unchanged": {},
            "failed": {}
        }

        mock_post.side_effect = [mock_response_412, mock_response_success]

        result = zotero_client.create_child_note(
            library_type="users",
            library_id="123",
            item_key="ITEM",
            note_html="<p>Test</p>",
            api_key="test_key"
        )

        assert result["success"] is True
        assert mock_post.call_count == 2


# ============================================================================
# Citation Import Extensions Tests
# ============================================================================


class TestGetOrCreateCollection:
    """Test collection management (get or create)."""

    @patch('app.utils.zotero_client.requests.get')
    def test_existing_collection_found(self, mock_get):
        """Test finding existing collection (case-insensitive)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Last-Modified-Version": "100"}
        mock_response.json.return_value = [
            {
                "key": "COLL123",
                "data": {"name": "My Collection"}
            }
        ]
        mock_get.return_value = mock_response

        result = zotero_client.get_or_create_collection(
            "users", "123", "my collection", api_key="test_key"
        )

        assert result["key"] == "COLL123"
        assert result["created"] is False
        assert result["name"] == "My Collection"

    @patch('app.utils.zotero_client.get_library_version')
    @patch('app.utils.zotero_client.requests.post')
    @patch('app.utils.zotero_client.requests.get')
    def test_create_new_collection(self, mock_get, mock_post, mock_get_version):
        """Test creating new collection when not found."""
        # GET returns empty list (no existing collection)
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = []
        mock_get.return_value = mock_get_response

        # POST creates collection
        mock_get_version.return_value = "100"
        mock_post_response = Mock()
        mock_post_response.status_code = 201
        mock_post_response.headers = {"Last-Modified-Version": "101"}
        mock_post_response.json.return_value = {
            "successful": {"0": "NEWCOLL456"}
        }
        mock_post.return_value = mock_post_response

        result = zotero_client.get_or_create_collection(
            "users", "123", "New Collection", description="Test", api_key="test_key"
        )

        assert result["key"] == "NEWCOLL456"
        assert result["created"] is True
        assert result["name"] == "New Collection"

    @patch('app.utils.zotero_client.get_library_version')
    @patch('app.utils.zotero_client.requests.post')
    @patch('app.utils.zotero_client.requests.get')
    def test_collection_creation_retry_on_412(self, mock_get, mock_post, mock_get_version):
        """Test retry logic on version conflict during creation."""
        # GET returns empty (no existing)
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = []
        mock_get.return_value = mock_get_response

        # Version increases on retry
        mock_get_version.side_effect = ["100", "101"]

        # First POST fails with 412, second succeeds
        mock_412 = Mock()
        mock_412.status_code = 412

        mock_success = Mock()
        mock_success.status_code = 201
        mock_success.headers = {"Last-Modified-Version": "102"}
        mock_success.json.return_value = {"successful": {"0": "COLL789"}}

        mock_post.side_effect = [mock_412, mock_success]

        result = zotero_client.get_or_create_collection(
            "users", "123", "Test", api_key="test_key"
        )

        assert result["created"] is True
        assert mock_post.call_count == 2


class TestSearchItemByDoi:
    """Test DOI-based item search."""

    @patch('app.utils.zotero_client.requests.get')
    def test_item_found_by_doi(self, mock_get):
        """Test finding item by DOI (case-insensitive)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "key": "ITEM123",
                "data": {"DOI": "10.1234/test"}
            }
        ]
        mock_get.return_value = mock_response

        item_key = zotero_client.search_item_by_doi(
            "users", "123", "10.1234/TEST", "test_key"
        )

        assert item_key == "ITEM123"

    @patch('app.utils.zotero_client.requests.get')
    def test_item_not_found_by_doi(self, mock_get):
        """Test when no item matches DOI."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        item_key = zotero_client.search_item_by_doi(
            "users", "123", "10.9999/notfound", "test_key"
        )

        assert item_key is None

    def test_empty_doi_returns_none(self):
        """Test that empty DOI returns None immediately."""
        assert zotero_client.search_item_by_doi("users", "123", "", "key") is None
        assert zotero_client.search_item_by_doi("users", "123", None, "key") is None


class TestSearchItemByUrl:
    """Test URL-based item search."""

    @patch('app.utils.zotero_client.requests.get')
    def test_item_found_by_url(self, mock_get):
        """Test finding item by exact URL match."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "key": "ITEM456",
                "data": {"url": "https://example.com/article"}
            }
        ]
        mock_get.return_value = mock_response

        item_key = zotero_client.search_item_by_url(
            "users", "123", "https://example.com/article", "test_key"
        )

        assert item_key == "ITEM456"

    @patch('app.utils.zotero_client.requests.get')
    def test_item_not_found_by_url(self, mock_get):
        """Test when no item matches URL."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        item_key = zotero_client.search_item_by_url(
            "users", "123", "https://notfound.com", "test_key"
        )

        assert item_key is None


class TestSearchItemByTitle:
    """Test title-based item search with normalization."""

    @patch('app.utils.zotero_client.requests.get')
    def test_item_found_by_normalized_title(self, mock_get):
        """Test finding item with normalized title matching."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "key": "ITEM789",
                "data": {"title": "Machine Learning: A Survey (2024)"}
            }
        ]
        mock_get.return_value = mock_response

        item_key = zotero_client.search_item_by_title(
            "users", "123", "Machine Learning  A Survey 2024", "test_key"
        )

        assert item_key == "ITEM789"

    @patch('app.utils.zotero_client.requests.get')
    def test_item_not_found_by_title(self, mock_get):
        """Test when no title matches."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        item_key = zotero_client.search_item_by_title(
            "users", "123", "Completely Different Title", "test_key"
        )

        assert item_key is None

    def test_short_title_returns_none(self):
        """Test that very short titles are skipped."""
        # Less than 5 chars
        assert zotero_client.search_item_by_title("users", "123", "AI", "key") is None
        assert zotero_client.search_item_by_title("users", "123", "Test", "key") is None


class TestNormalizeTitleForSearch:
    """Test title normalization helper."""

    def test_lowercase_conversion(self):
        """Test conversion to lowercase."""
        result = zotero_client._normalize_title_for_search("UPPERCASE TITLE")
        assert result == "uppercase title"

    def test_remove_punctuation(self):
        """Test removal of punctuation."""
        result = zotero_client._normalize_title_for_search("Title: With, Punctuation!")
        assert result == "title with punctuation"

    def test_collapse_spaces(self):
        """Test collapsing multiple spaces."""
        result = zotero_client._normalize_title_for_search("Too    many     spaces")
        assert result == "too many spaces"

    def test_keep_numbers(self):
        """Test that numbers are preserved."""
        result = zotero_client._normalize_title_for_search("Study 2024 Results")
        assert result == "study 2024 results"


class TestMergeItemData:
    """Test intelligent item data merging."""

    def test_merge_adds_missing_fields(self):
        """Test that missing fields from new data are added."""
        existing = {"title": "Test", "abstractNote": "Existing abstract"}
        new = {"title": "Test", "DOI": "10.1234/test"}

        merged = zotero_client._merge_item_data(existing, new)

        assert merged["abstractNote"] == "Existing abstract"
        assert merged["DOI"] == "10.1234/test"

    def test_merge_prefers_existing_complete_data(self):
        """Test that existing complete data is preserved."""
        existing = {"title": "Full Title", "abstractNote": "Long existing abstract"}
        new = {"title": "Short", "abstractNote": "Short"}

        merged = zotero_client._merge_item_data(existing, new)

        # Longer existing values are kept
        assert merged["title"] == "Full Title"
        assert merged["abstractNote"] == "Long existing abstract"

    def test_merge_updates_empty_fields(self):
        """Test that empty existing fields are updated."""
        existing = {"title": "Test", "DOI": ""}
        new = {"title": "Test", "DOI": "10.1234/new"}

        merged = zotero_client._merge_item_data(existing, new)

        assert merged["DOI"] == "10.1234/new"

    def test_merge_tags_union(self):
        """Test that tags are merged as union."""
        existing = {"tags": [{"tag": "existing"}, {"tag": "common"}]}
        new = {"tags": [{"tag": "new"}, {"tag": "common"}]}

        merged = zotero_client._merge_item_data(existing, new)

        tag_values = {tag["tag"] for tag in merged["tags"]}
        assert tag_values == {"common", "existing", "new"}

    def test_merge_creators_prefers_longer_list(self):
        """Test that longer creator list is used."""
        existing = {"creators": [{"firstName": "A", "lastName": "B"}]}
        new = {
            "creators": [
                {"firstName": "A", "lastName": "B"},
                {"firstName": "C", "lastName": "D"}
            ]
        }

        merged = zotero_client._merge_item_data(existing, new)

        assert len(merged["creators"]) == 2


class TestCreateOrUpdateItem:
    """Test create or update with triple deduplication."""

    @patch('app.utils.zotero_client.search_item_by_doi')
    @patch('app.utils.zotero_client.get_library_version')
    @patch('app.utils.zotero_client.requests.post')
    def test_create_new_item_no_duplicate(self, mock_post, mock_get_version, mock_search_doi):
        """Test creating new item when no duplicate found."""
        # No duplicates
        mock_search_doi.return_value = None

        mock_get_version.return_value = "100"

        # Create succeeds
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.headers = {"Last-Modified-Version": "101"}
        mock_response.json.return_value = {
            "successful": {"0": "NEWITEM123"}
        }
        mock_post.return_value = mock_response

        result = zotero_client.create_or_update_item(
            "users", "123",
            {"itemType": "journalArticle", "title": "New Article", "DOI": "10.1234/new"},
            "test_key"
        )

        assert result["success"] is True
        assert result["action"] == "created"
        assert result["item_key"] == "NEWITEM123"

    @patch('app.utils.zotero_client.search_item_by_doi')
    @patch('app.utils.zotero_client.requests.get')
    @patch('app.utils.zotero_client.requests.patch')
    def test_update_existing_item_found_by_doi(self, mock_patch, mock_get, mock_search_doi):
        """Test updating existing item when found by DOI."""
        # Duplicate found by DOI
        mock_search_doi.return_value = "EXISTING123"

        # GET existing item
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            "key": "EXISTING123",
            "version": 50,
            "data": {
                "title": "Original Title",
                "DOI": "10.1234/existing",
                "tags": [{"tag": "old"}]
            }
        }
        mock_get.return_value = mock_get_response

        # PATCH update succeeds
        mock_patch_response = Mock()
        mock_patch_response.status_code = 204
        mock_patch_response.headers = {"Last-Modified-Version": "51"}
        mock_patch.return_value = mock_patch_response

        result = zotero_client.create_or_update_item(
            "users", "123",
            {
                "itemType": "journalArticle",
                "title": "Updated Title",
                "DOI": "10.1234/existing",
                "tags": [{"tag": "new"}]
            },
            "test_key"
        )

        assert result["success"] is True
        assert result["action"] == "updated"
        assert result["item_key"] == "EXISTING123"
        mock_patch.assert_called_once()

    @patch('app.utils.zotero_client.search_item_by_doi')
    @patch('app.utils.zotero_client.search_item_by_url')
    @patch('app.utils.zotero_client.requests.get')
    @patch('app.utils.zotero_client.requests.patch')
    def test_fallback_to_url_search(self, mock_patch, mock_get, mock_search_url, mock_search_doi):
        """Test fallback to URL search when DOI fails."""
        # DOI search fails
        mock_search_doi.return_value = None
        # URL search finds item
        mock_search_url.return_value = "FOUNDBYURL"

        # GET existing
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            "version": 10,
            "data": {"title": "Existing", "url": "https://example.com/article"}
        }
        mock_get.return_value = mock_get_response

        # Update succeeds
        mock_patch_response = Mock()
        mock_patch_response.status_code = 204
        mock_patch_response.headers = {"Last-Modified-Version": "11"}
        mock_patch.return_value = mock_patch_response

        result = zotero_client.create_or_update_item(
            "users", "123",
            {"title": "Updated", "url": "https://example.com/article"},
            "test_key"
        )

        assert result["action"] == "updated"
        assert result["item_key"] == "FOUNDBYURL"

    @patch('app.utils.zotero_client.search_item_by_doi')
    @patch('app.utils.zotero_client.search_item_by_url')
    @patch('app.utils.zotero_client.search_item_by_title')
    @patch('app.utils.zotero_client.requests.get')
    @patch('app.utils.zotero_client.requests.patch')
    def test_fallback_to_title_search(
        self, mock_patch, mock_get, mock_search_title, mock_search_url, mock_search_doi
    ):
        """Test fallback to title search when DOI and URL fail."""
        # DOI and URL fail
        mock_search_doi.return_value = None
        mock_search_url.return_value = None
        # Title search finds item
        mock_search_title.return_value = "FOUNDBYTITLE"

        # GET existing
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            "version": 20,
            "data": {"title": "Matching Title"}
        }
        mock_get.return_value = mock_get_response

        # Update succeeds
        mock_patch_response = Mock()
        mock_patch_response.status_code = 204
        mock_patch_response.headers = {"Last-Modified-Version": "21"}
        mock_patch.return_value = mock_patch_response

        result = zotero_client.create_or_update_item(
            "users", "123",
            {"title": "matching title"},
            "test_key"
        )

        assert result["action"] == "updated"
        assert result["item_key"] == "FOUNDBYTITLE"

    @patch('app.utils.zotero_client.search_item_by_doi')
    @patch('app.utils.zotero_client.get_library_version')
    @patch('app.utils.zotero_client.requests.post')
    def test_create_retry_on_version_conflict(
        self, mock_post, mock_get_version, mock_search_doi
    ):
        """Test retry on 412 version conflict during creation."""
        mock_search_doi.return_value = None

        # Version updates on retry
        mock_get_version.side_effect = ["100", "101"]

        # First POST fails with 412, second succeeds
        mock_412 = Mock()
        mock_412.status_code = 412

        mock_success = Mock()
        mock_success.status_code = 201
        mock_success.headers = {"Last-Modified-Version": "102"}
        mock_success.json.return_value = {"successful": {"0": "CREATED"}}

        mock_post.side_effect = [mock_412, mock_success]

        result = zotero_client.create_or_update_item(
            "users", "123",
            {"title": "Test"},
            "test_key"
        )

        assert result["success"] is True
        assert mock_post.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
