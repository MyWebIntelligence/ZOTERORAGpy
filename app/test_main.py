"""
Integration Tests for Main Application
======================================

This module contains integration tests for the FastAPI application defined in `app.main`.
It uses `fastapi.testclient.TestClient` to simulate HTTP requests and verify
endpoint behavior, particularly for the vector database upload functionality.

Key Tests:
- Pinecone upload success and error handling.
- Weaviate upload success.
- Qdrant upload success.
- Input validation (missing files, missing parameters).
"""
import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import json # <--- Added import json
from fastapi.testclient import TestClient

# Add the 'ragpy' directory to sys.path to allow 'from app.main import app'
# and 'from scripts.rad_vectordb import ...'
# This assumes the test file is in ragpy/app/ and ragpy is the project root for scripts.
# Or, more robustly, go up two levels from this file's dir to reach __RAG, then down to ragpy.
current_file_dir = os.path.dirname(os.path.abspath(__file__))
ragpy_dir = os.path.dirname(current_file_dir) # This should be __RAG/ragpy
project_root_dir = os.path.dirname(ragpy_dir) # This should be __RAG

# Add __RAG/ragpy to sys.path for 'from app.main import app'
if ragpy_dir not in sys.path:
    sys.path.insert(0, ragpy_dir)

# Add __RAG (project root) to sys.path for 'from scripts.rad_vectordb import ...'
# This allows 'scripts.rad_vectordb' to be found if 'ragpy' is the top-level package for scripts.
# However, the imports in main.py are 'from scripts.rad_vectordb', implying 'ragpy' is already a package root.
# Let's ensure the directory containing 'scripts' is discoverable.
# If main.py is in app/, and it does 'from scripts.rad_vectordb', then 'ragpy' (parent of app and scripts)
# must be in PYTHONPATH or be the CWD.
# For testing, we can add 'ragpy' to the path.
if ragpy_dir not in sys.path: # If ragpy_dir is /path/to/__RAG/ragpy
    sys.path.insert(0, ragpy_dir)


try:
    from app.main import app
except ModuleNotFoundError as e:
    print(f"Failed to import app from app.main. Current sys.path: {sys.path}")
    print(f"Error: {e}")
    # If app is in the same directory as test_main.py, this should work.
    # If main.py is in ragpy/app/ and test_main.py is also in ragpy/app/
    # then 'from main import app' might be needed if 'app' is not a package.
    # Given the structure, 'from app.main import app' implies 'ragpy' is a root.
    raise

client = TestClient(app)

# Dummy path for uploads, ensure it exists for tests that need it
TEST_UPLOAD_PATH = os.path.join(project_root_dir, "ragpy", "test_uploads_temp")
TEST_CHUNKS_JSON_FILENAME = "output_chunks_with_embeddings_sparse.json"
TEST_CHUNKS_JSON_FULLPATH = os.path.join(TEST_UPLOAD_PATH, TEST_CHUNKS_JSON_FILENAME)


class TestMainApp(unittest.TestCase):
    """
    Test Suite for Main Application Endpoints.
    
    This class groups all integration tests for the main application.
    It sets up a temporary environment (dummy .env, chunks file) to isolate tests.
    """

    @classmethod
    def setUpClass(cls):
        # Create a dummy .env file for testing credential loading
        cls.test_env_path = os.path.join(ragpy_dir, ".env.test_main")
        with open(cls.test_env_path, "w") as f:
            f.write("PINECONE_API_KEY=test_pinecone_key\n")
            f.write("WEAVIATE_API_KEY=test_weaviate_key\n")
            f.write("WEAVIATE_URL=http://testweaviate.url\n")
            f.write("QDRANT_API_KEY=test_qdrant_key\n")
            f.write("QDRANT_URL=http://testqdrant.url\n")
        
        # Create a dummy chunks file
        os.makedirs(TEST_UPLOAD_PATH, exist_ok=True)
        with open(TEST_CHUNKS_JSON_FULLPATH, "w") as f:
            json.dump([{"id": "test_chunk", "embedding": [0.1, 0.2]}], f)


    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_env_path):
            os.remove(cls.test_env_path)
        if os.path.exists(TEST_CHUNKS_JSON_FULLPATH):
            os.remove(TEST_CHUNKS_JSON_FULLPATH)
        if os.path.exists(TEST_UPLOAD_PATH):
            os.rmdir(TEST_UPLOAD_PATH)


    def _get_mock_env_path(self):
        # This function will be used by the patch to return our test .env path
        return self.test_env_path

    @patch('scripts.rad_vectordb.insert_to_pinecone') # Patching at source
    @patch('os.path.abspath') # To control where .env is looked for
    def test_upload_db_pinecone_success(self, mock_abspath, mock_insert_pinecone):
        mock_abspath.side_effect = lambda path: self._get_mock_env_path() if ".env" in path else os.path.normpath(path)
        
        mock_insert_pinecone.return_value = {
            "status": "success",
            "message": "Pinecone upload successful.",
            "inserted_count": 10
        }

        response = client.post("/upload_db", data={
            "path": TEST_UPLOAD_PATH,
            "db_choice": "pinecone",
            "pinecone_index_name": "test_index"
        })
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertEqual(json_response["status"], "success")
        self.assertEqual(json_response["message"], "Pinecone upload successful.")
        self.assertEqual(json_response["inserted_count"], 10)
        self.assertEqual(json_response["db_choice"], "pinecone")
        mock_insert_pinecone.assert_called_once_with(
            embeddings_json_file=TEST_CHUNKS_JSON_FULLPATH,
            index_name="test_index",
            pinecone_api_key="test_pinecone_key"
        )

    @patch('scripts.rad_vectordb.insert_to_pinecone') # Patching at source
    @patch('os.path.abspath')
    def test_upload_db_pinecone_script_error(self, mock_abspath, mock_insert_pinecone):
        mock_abspath.side_effect = lambda path: self._get_mock_env_path() if ".env" in path else os.path.normpath(path)
        
        mock_insert_pinecone.return_value = {
            "status": "error",
            "message": "Pinecone script internal error.",
            "inserted_count": 0
        }
        response = client.post("/upload_db", data={
            "path": TEST_UPLOAD_PATH,
            "db_choice": "pinecone",
            "pinecone_index_name": "test_index"
        })
        self.assertEqual(response.status_code, 500) # As per main.py logic
        json_response = response.json()
        self.assertEqual(json_response["status"], "error") # This is the nested status from rad_vectordb
        self.assertEqual(json_response["error"], "Pinecone script internal error.")
        self.assertEqual(json_response["inserted_count"], 0)

    @patch('os.path.abspath')
    def test_upload_db_pinecone_missing_index_name(self, mock_abspath):
        mock_abspath.side_effect = lambda path: self._get_mock_env_path() if ".env" in path else os.path.normpath(path)
        response = client.post("/upload_db", data={
            "path": TEST_UPLOAD_PATH,
            "db_choice": "pinecone"
            # pinecone_index_name is missing
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("Pinecone Index Name is required", response.json()["error"])

    @patch('os.path.abspath')
    def test_upload_db_chunks_file_not_found(self, mock_abspath):
        mock_abspath.side_effect = lambda path: self._get_mock_env_path() if ".env" in path else os.path.normpath(path)
        response = client.post("/upload_db", data={
            "path": "/non/existent/path", # This path won't have the chunks file
            "db_choice": "pinecone",
            "pinecone_index_name": "test_index"
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("Required chunks file not found", response.json()["error"])

    @patch('os.path.abspath')
    def test_upload_db_pinecone_missing_api_key(self, mock_abspath):
        # Simulate .env file without PINECONE_API_KEY
        temp_env_content = "OTHER_KEY=some_value\n"
        with patch('builtins.open', unittest.mock.mock_open(read_data=temp_env_content)) as mock_open_env:
            # Ensure abspath still points to a "valid" .env for the open mock to be used
            mock_abspath.side_effect = lambda path: self._get_mock_env_path() if ".env" in path else os.path.normpath(path)
            
            response = client.post("/upload_db", data={
                "path": TEST_UPLOAD_PATH,
                "db_choice": "pinecone",
                "pinecone_index_name": "test_index"
            })
        self.assertEqual(response.status_code, 400)
        self.assertIn("Pinecone API Key not found in credentials", response.json()["error"])

    # Example for Weaviate - can be expanded
    @patch('scripts.rad_vectordb.insert_to_weaviate_hybrid') # Patching at source
    @patch('os.path.abspath')
    def test_upload_db_weaviate_success(self, mock_abspath, mock_insert_weaviate):
        mock_abspath.side_effect = lambda path: self._get_mock_env_path() if ".env" in path else os.path.normpath(path)
        mock_insert_weaviate.return_value = 5 # Returns count of inserted items

        response = client.post("/upload_db", data={
            "path": TEST_UPLOAD_PATH,
            "db_choice": "weaviate",
            "weaviate_class_name": "TestClass",
            "weaviate_tenant_name": "test_tenant"
        })
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertEqual(json_response["status"], "success")
        self.assertIn("Weaviate upload successful. 5 items inserted.", json_response["message"])
        self.assertEqual(json_response["inserted_count"], 5)
        mock_insert_weaviate.assert_called_once_with(
            embeddings_json_file=TEST_CHUNKS_JSON_FULLPATH,
            url="http://testweaviate.url",
            api_key="test_weaviate_key",
            class_name="TestClass",
            tenant_name="test_tenant"
        )

    # Example for Qdrant - can be expanded
    @patch('scripts.rad_vectordb.insert_to_qdrant') # Patching at source
    @patch('os.path.abspath')
    def test_upload_db_qdrant_success(self, mock_abspath, mock_insert_qdrant):
        mock_abspath.side_effect = lambda path: self._get_mock_env_path() if ".env" in path else os.path.normpath(path)
        mock_insert_qdrant.return_value = 3 # Returns count

        response = client.post("/upload_db", data={
            "path": TEST_UPLOAD_PATH,
            "db_choice": "qdrant",
            "qdrant_collection_name": "test_collection"
        })
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertEqual(json_response["status"], "success")
        self.assertIn("Qdrant upload successful. 3 items inserted.", json_response["message"])
        self.assertEqual(json_response["inserted_count"], 3)
        mock_insert_qdrant.assert_called_once_with(
            embeddings_json_file=TEST_CHUNKS_JSON_FULLPATH,
            collection_name="test_collection",
            qdrant_url="http://testqdrant.url",
            qdrant_api_key="test_qdrant_key"
        )

if __name__ == "__main__":
    # Adjust sys.path for standalone execution if necessary
    # This setup is primarily for running with `python -m unittest ragpy.app.test_main`
    # from the __RAG directory, or similar.
    unittest.main()
