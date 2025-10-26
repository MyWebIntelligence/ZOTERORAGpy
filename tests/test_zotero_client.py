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
        mock_response.json.return_value = {
            "successful": {
                "0": {"key": "NOTEKEY123"}
            }
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
        mock_response_success.json.return_value = {
            "successful": {"0": {"key": "NOTEKEY"}}
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
