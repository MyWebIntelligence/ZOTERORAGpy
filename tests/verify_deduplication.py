
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add app to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.utils.zotero_client import search_item_by_title

class TestZoteroDeduplication(unittest.TestCase):
    def setUp(self):
        self.api_key = "fake_key"
        self.library_id = "123"
        self.library_type = "users"

    @patch('app.utils.zotero_client.requests.get')
    def test_generic_title_no_authors_no_date_rejects(self, mock_get):
        """Test that a generic short title with no authors/date in target is REJECTED even if title matches."""
        # Setup mock to return an existing item with the same title
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{
            "key": "EXISTING_KEY",
            "data": {
                "title": "Introduction",
                "creators": [{"lastName": "Smith", "creatorType": "author"}],
                "date": "2020"
            }
        }]
        mock_get.return_value = mock_response

        # Target: "Introduction", no authors, no date
        result_key = search_item_by_title(
            self.library_type, self.library_id, 
            "Introduction", self.api_key, 
            target_creators=[], target_date=None
        )

        # Should be None (Rejected) because title is short (<40 chars) and no authors/date provided
        self.assertIsNone(result_key, "Should reject generic title match when authors/date missing")

    @patch('app.utils.zotero_client.requests.get')
    def test_long_title_no_authors_accepts(self, mock_get):
        """Test that a LONG unique title matches even without authors."""
        long_title = "A Very Specific and Long Title Analysis of Zotero Deduplication Logic in Complex Systems"
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{
            "key": "EXISTING_KEY",
            "data": {
                "title": long_title,
                "creators": [{"lastName": "Smith", "creatorType": "author"}] 
            }
        }]
        mock_get.return_value = mock_response

        # Target: Long title, no authors
        result_key = search_item_by_title(
            self.library_type, self.library_id, 
            long_title, self.api_key, 
            target_creators=[], target_date=None
        )

        self.assertEqual(result_key, "EXISTING_KEY", "Should match long unique title even without authors")

    @patch('app.utils.zotero_client.requests.get')
    def test_generic_title_with_matching_year_accepts(self, mock_get):
        """Test that a generic title matches IF the year matches."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{
            "key": "EXISTING_KEY",
            "data": {
                "title": "Introduction",
                "date": "2023-01-01"
            }
        }]
        mock_get.return_value = mock_response

        # Target: "Introduction", 2023
        result_key = search_item_by_title(
            self.library_type, self.library_id, 
            "Introduction", self.api_key, 
            target_creators=[], target_date="2023"
        )

        self.assertEqual(result_key, "EXISTING_KEY", "Should match generic title if year matches")

    @patch('app.utils.zotero_client.requests.get')
    def test_generic_title_with_mismatching_year_rejects(self, mock_get):
        """Test that a generic title is rejected if year mismatches."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{
            "key": "EXISTING_KEY",
            "data": {
                "title": "Introduction",
                "date": "2020"
            }
        }]
        mock_get.return_value = mock_response

        # Target: "Introduction", 2023 (differs > 1 year)
        result_key = search_item_by_title(
            self.library_type, self.library_id, 
            "Introduction", self.api_key, 
            target_creators=[], target_date="2023"
        )

        self.assertIsNone(result_key, "Should reject generic title if year mismatches")

if __name__ == '__main__':
    unittest.main()
