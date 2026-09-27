"""
Integration tests for a few public API routes (home page, settings, CSV upload).

``/get_credentials`` is ADMIN only and ``/upload_csv`` requires an
authenticated user: the fake admin is injected with ``app.dependency_overrides``
and the settings ``.env`` is a temporary file holding fake values, so the host
``.env`` is never read. CSV uploads land in a temporary uploads directory.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

# Add root to path
SCRIPT_DIR = Path(__file__).parent.absolute()
RAGPY_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(RAGPY_ROOT))

from app.main import app
from app.middleware.auth import get_current_active_user, require_admin
from app.models.user import User
from app.routes import ingestion as ingestion_routes
from app.routes import settings as settings_routes

FAKE_DOTENV = (
    "OPENAI_API_KEY=fake-openai-key\n"
    "PINECONE_API_KEY=fake-pinecone-key\n"
    "# COMMENTED_KEY=ignored\n"
    "OPENROUTER_DEFAULT_MODEL=google/gemini-2.5-flash\n"
)


def _fake_admin():
    """Admin user (not persisted) without personal credentials."""
    return User(
        id=1,
        email="admin@example.com",
        hashed_password="x",
        roles=["ADMIN"],
        is_active=True,
        is_verified=True,
        api_credentials=None,
    )


class TestIntegrationAPI(unittest.TestCase):
    """Home page, admin credentials form and CSV upload routes."""

    def setUp(self):
        """Temporary settings home and uploads directory, fresh test client."""
        self._tmp = tempfile.TemporaryDirectory()
        self.settings_home = os.path.join(self._tmp.name, "ragpy_home")
        self.uploads = os.path.join(self._tmp.name, "uploads")
        os.makedirs(self.settings_home)
        os.makedirs(self.uploads)
        for patcher in (
            patch.object(settings_routes, "RAGPY_DIR", self.settings_home),
            patch.object(ingestion_routes, "UPLOAD_DIR", self.uploads),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(app.dependency_overrides.pop, require_admin, None)
        self.addCleanup(app.dependency_overrides.pop, get_current_active_user, None)
        self.client = TestClient(app)

    def _as_admin(self):
        """Authenticate every request as the fake admin."""
        app.dependency_overrides[require_admin] = _fake_admin

    def test_read_main(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        # It returns HTML, so we check content type
        self.assertIn("text/html", response.headers["content-type"])

    def test_get_credentials(self):
        with open(os.path.join(self.settings_home, ".env"), "w", encoding="utf-8") as f:
            f.write(FAKE_DOTENV)
        self._as_admin()

        response = self.client.get("/get_credentials")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("OPENAI_API_KEY", data)
        self.assertIn("PINECONE_API_KEY", data)
        # Values come from the settings .env (fake file), unknown keys are empty
        self.assertTrue(data["OPENAI_API_KEY"] == "fake-openai-key")
        self.assertTrue(data["PINECONE_API_KEY"] == "fake-pinecone-key")
        self.assertEqual(data["OPENROUTER_DEFAULT_MODEL"], "google/gemini-2.5-flash")
        self.assertEqual(data["ZOTERO_API_KEY"], "")
        self.assertNotIn("COMMENTED_KEY", data)

    def test_get_credentials_without_dotenv_returns_empty_form(self):
        self._as_admin()

        response = self.client.get("/get_credentials")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("OPENAI_API_KEY", data)
        self.assertTrue(all(value == "" for value in data.values()))

    def test_get_credentials_requires_authentication(self):
        response = self.client.get("/get_credentials")
        self.assertEqual(response.status_code, 401)

    def _as_user(self):
        """Authenticate upload requests as the fake admin (no project_id: no database access)."""
        app.dependency_overrides[get_current_active_user] = _fake_admin

    def test_upload_csv_requires_authentication(self):
        """Anonymous CSV upload: 401 and nothing written."""
        files = {'file': ('test.csv', b"header1,text\nval1,some text", 'text/csv')}
        response = self.client.post("/upload_csv", files=files)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(os.listdir(self.uploads), [])

    def test_upload_csv_invalid_extension(self):
        self._as_user()
        # Test uploading a non-csv file
        files = {'file': ('test.txt', b'some content', 'text/plain')}
        response = self.client.post("/upload_csv", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": "Only .csv files are accepted."})
        self.assertEqual(os.listdir(self.uploads), [])

    def test_upload_csv_success(self):
        self._as_user()
        # Test uploading a valid csv file
        csv_content = b"header1,text\nval1,some text content"
        files = {'file': ('test.csv', csv_content, 'text/csv')}
        response = self.client.post("/upload_csv", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("path", data)
        self.assertIn("tree", data)
        self.assertIn("output.csv", data["tree"])
        # The session folder is created under the (temporary) uploads directory
        self.assertTrue(os.path.isfile(os.path.join(self.uploads, data["path"], "output.csv")))


if __name__ == "__main__":
    unittest.main()
