
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import requests

# Add app to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.utils.zotero_client import search_item_by_title, create_or_update_item

class TestZoteroMultiUpdate(unittest.TestCase):
    def setUp(self):
        self.api_key = "fake_key"
        self.library_id = "123"
        self.library_type = "users"

    @patch('app.utils.zotero_client.requests.get')
    def test_search_returns_multiple_matches(self, mock_get):
        """Test search_item_by_title returns ALL matches for 'Update All' strategy."""
        
        title = "Duplicate Paper Title"
        # Mock finding 2 items with same title and same 1st author
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "key": "KEY_1",
                "data": {"title": title, "creators": [{"lastName": "Smith"}]}
            },
             {
                "key": "KEY_2",
                "data": {"title": title, "creators": [{"lastName": "Smith"}]}
            }
        ]
        mock_get.return_value = mock_response

        # Target: Same title, matching author
        target_creators = [{"lastName": "Smith", "creatorType": "author"}]
        
        result_keys = search_item_by_title(
            self.library_type, self.library_id, 
            title, self.api_key, 
            target_creators=target_creators
        )

        self.assertEqual(len(result_keys), 2, "Should return both keys")
        self.assertIn("KEY_1", result_keys)
        self.assertIn("KEY_2", result_keys)

    @patch('app.utils.zotero_client.requests.patch')
    @patch('app.utils.zotero_client.requests.get')
    def test_create_or_update_syncs_all_items(self, mock_get, mock_patch):
        """Test create_or_update_item updates ALL found duplicates."""
        
        # 1. Setup Search Mock (Find 2 items)
        search_res = MagicMock()
        search_res.status_code = 200
        search_res.json.return_value = [
             {"key": "KEY_A", "data": {"title": "Title", "creators": [{"lastName": "Doe"}]}, "version": 1},
             {"key": "KEY_B", "data": {"title": "Title", "creators": [{"lastName": "Doe"}]}, "version": 5}
        ]
        
        # 2. Setup Get Item Mock (for the update loop fetching current version)
        item_res = MagicMock()
        item_res.status_code = 200
        # Simple generic item returned for detail fetch
        item_res.json.return_value = {"data": {}, "version": 10} 

        mock_get.side_effect = [search_res, item_res, item_res] # 1 search call, 2 fetch calls

        # 3. Setup Patch Mock (Update success)
        patch_res = MagicMock()
        patch_res.status_code = 204
        mock_patch.return_value = patch_res

        # Execute
        result = create_or_update_item(
            self.library_type, self.library_id,
            {"title": "Title", "creators": [{"lastName": "Doe", "creatorType": "author"}]},
            self.api_key
        )

        # Assertions
        self.assertTrue(result["success"])
        self.assertEqual(len(result["synced_keys"]), 2, "Should report 2 synced keys")
        self.assertIn("KEY_A", result["synced_keys"])
        self.assertIn("KEY_B", result["synced_keys"])
        
        # Verify PATCH was called twice
        self.assertEqual(mock_patch.call_count, 2, "Should send update PATCH for each item")

if __name__ == '__main__':
    unittest.main()
