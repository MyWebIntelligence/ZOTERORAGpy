import unittest
from fastapi.testclient import TestClient
import sys
import os
from pathlib import Path

# Add root to path
SCRIPT_DIR = Path(__file__).parent.absolute()
RAGPY_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(RAGPY_ROOT))

from app.main import app

class TestIntegrationAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_read_main(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        # It returns HTML, so we check content type
        self.assertIn("text/html", response.headers["content-type"])

    def test_get_credentials(self):
        response = self.client.get("/get_credentials")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("OPENAI_API_KEY", data)
        self.assertIn("PINECONE_API_KEY", data)

    def test_upload_csv_invalid_extension(self):
        # Test uploading a non-csv file
        files = {'file': ('test.txt', b'some content', 'text/plain')}
        response = self.client.post("/upload_csv", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": "Only .csv files are accepted."})

    def test_upload_csv_success(self):
        # Test uploading a valid csv file
        csv_content = b"header1,text\nval1,some text content"
        files = {'file': ('test.csv', csv_content, 'text/csv')}
        response = self.client.post("/upload_csv", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("path", data)
        self.assertIn("tree", data)
        self.assertIn("output.csv", data["tree"])
