"""
Integration Tests for Main Application
======================================

This module contains integration tests for the FastAPI application defined in `app.main`.
It uses `fastapi.testclient.TestClient` to simulate HTTP requests and verify
endpoint behavior, particularly for the vector database upload route (`POST /upload_db`).

The route is authenticated and launches `scripts/rad_vectordb.py` in a subprocess
through `run_tracked_subprocess`, with an environment built by
`build_subprocess_env` from the user's personal credentials. The tests:

- authenticate a non-admin user through `app.dependency_overrides` (its fake
  credentials are encrypted exactly as in the database, and no `.env` value can
  reach a non-admin subprocess);
- replace `run_tracked_subprocess` with an `AsyncMock` (no process is started)
  and check the command line, the environment and how the CLI output is parsed;
- work in a temporary uploads directory and a temporary SQLite database with
  the full schema (audit A13: no pre-existing ``data/ragpy.db`` is needed, and
  the developer's database is never touched); the session folder is recorded
  as an upload of the test user (``SessionOwner``, audit A02).

Key Tests:
- Pinecone upload success and error handling.
- Weaviate upload success.
- Qdrant upload success.
- Input validation (missing files, missing parameters, missing credentials, authentication).
"""
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add the 'ragpy' directory (parent of app/ and scripts/) to sys.path so that
# 'from app.main import app' works when this file is run on its own.
current_file_dir = os.path.dirname(os.path.abspath(__file__))
ragpy_dir = os.path.dirname(current_file_dir)
if ragpy_dir not in sys.path:
    sys.path.insert(0, ragpy_dir)

from app.main import app
from app.core.config import RAGPY_DIR
from app.core.credentials import encrypt_credentials
from app.database.base import Base
from app.database.session import get_db
from app.middleware.auth import get_current_active_user
from app.models import audit, background_task, pipeline_session, project  # noqa: F401  (tables)
from app.models.pipeline_session import SessionOwner
from app.models.user import User
from app.routes import processing as processing_routes

client = TestClient(app)

SESSION = "sess_vectordb"
TEST_CHUNKS_JSON_FILENAME = "output_chunks_with_embeddings_sparse.json"
VECTORDB_SCRIPT = os.path.join(RAGPY_DIR, "scripts", "rad_vectordb.py")

# Fake personal credentials of the test user (never real values).
FAKE_CREDENTIALS = {
    "pinecone_api_key": "fake-pinecone-key",
    "weaviate_url": "http://weaviate.invalid",
    "weaviate_api_key": "fake-weaviate-key",
    "qdrant_url": "http://qdrant.invalid",
    "qdrant_api_key": "fake-qdrant-key",
}


def _make_user(credentials):
    """Non-admin user (not persisted) whose credentials are stored encrypted."""
    return User(
        id=4242,
        email="vectordb-tests@example.com",
        hashed_password="x",
        roles=["USER"],
        is_active=True,
        is_verified=True,
        api_credentials=encrypt_credentials(credentials) if credentials else None,
    )


def _completed(returncode=0, stdout="", stderr=""):
    """Result of the (fake) rad_vectordb.py subprocess."""
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _result_block(status, inserted, skipped=0):
    """stdout of rad_vectordb.py ending with its canonical '=== Result ===' block."""
    return (
        "=== rad_vectordb.py ===\n"
        "Input file: ...\n"
        "Status line from a log dated 2026 must not be parsed\n"
        "\n=== Result ===\n"
        f"Status: {status}\n"
        "Message: done\n"
        f"Inserted: {inserted}\n"
        f"Skipped (dedup): {skipped}\n"
    )


class TestMainApp(unittest.TestCase):
    """
    Test Suite for Main Application Endpoints.

    This class groups all integration tests for the main application.
    Each test gets a temporary uploads directory holding one session folder
    with a chunks file, an authenticated non-admin user and a fake subprocess
    launcher.
    """

    def setUp(self):
        """Temporary uploads dir, authenticated user and fake subprocess launcher."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.uploads = self._tmp.name
        self.session_dir = os.path.join(self.uploads, SESSION)
        os.makedirs(self.session_dir)
        self.chunks_file = os.path.join(self.session_dir, TEST_CHUNKS_JSON_FILENAME)
        with open(self.chunks_file, "w", encoding="utf-8") as f:
            f.write('[{"id": "test_chunk", "embedding": [0.1, 0.2]}]')

        upload_patcher = patch.object(processing_routes, "UPLOAD_DIR", self.uploads)
        upload_patcher.start()
        self.addCleanup(upload_patcher.stop)

        self.run_subprocess = AsyncMock(return_value=_completed(0, _result_block("success", 0)))
        run_patcher = patch.object(processing_routes, "run_tracked_subprocess", self.run_subprocess)
        run_patcher.start()
        self.addCleanup(run_patcher.stop)

        # Temporary database with the full schema; the session folder is the
        # test user's own upload (SessionOwner), as /upload_zip records it.
        engine = create_engine(
            f"sqlite:///{os.path.join(self.uploads, 'test.db')}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(bind=engine)
        self.addCleanup(engine.dispose)
        self.db = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
        self.addCleanup(self.db.close)
        self.db.add(SessionOwner(session_folder=SESSION, user_id=4242, source_type="zip"))
        self.db.commit()
        app.dependency_overrides[get_db] = lambda: self.db
        self.addCleanup(app.dependency_overrides.pop, get_db, None)

        self.user = _make_user(FAKE_CREDENTIALS)
        app.dependency_overrides[get_current_active_user] = lambda: self.user
        self.addCleanup(app.dependency_overrides.pop, get_current_active_user, None)

    def _call(self):
        """Keyword arguments of the single run_tracked_subprocess call."""
        self.run_subprocess.assert_awaited_once()
        return self.run_subprocess.await_args.kwargs

    def _assert_command(self, db_choice, extra_args):
        """The CLI is rad_vectordb.py on the session chunks file, with ``extra_args``."""
        cmd = self._call()["cmd"]
        self.assertEqual(os.path.normpath(cmd[1]), os.path.normpath(VECTORDB_SCRIPT))
        self.assertEqual(cmd[2:], ["--input", self.chunks_file, "--db", db_choice] + extra_args)

    def test_upload_db_pinecone_success(self):
        self.run_subprocess.return_value = _completed(0, _result_block("success", 10))

        response = client.post("/upload_db", data={
            "path": SESSION,
            "db_choice": "pinecone",
            "pinecone_index_name": "test_index"
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "status": "success",
            "message": "Uploaded to pinecone",
            "inserted_count": 10,
        })
        self._assert_command("pinecone", ["--index", "test_index"])
        call = self._call()
        self.assertEqual(call["session_folder"], SESSION)
        self.assertEqual(call["timeout"], 3600)
        env = call["env"]
        self.assertTrue(env.get("PINECONE_API_KEY") == FAKE_CREDENTIALS["pinecone_api_key"])
        # A non-admin subprocess may not reload the stripped secrets from .env
        self.assertIn("OPENAI_API_KEY", env["RAGPY_DOTENV_DENY"].split(","))
        self.assertNotIn("OPENAI_API_KEY", env)

    def test_upload_db_pinecone_script_error(self):
        cli_output = (
            "=== rad_vectordb.py ===\n"
            "\n=== Result ===\n"
            "Status: error\n"
            "Message: Pinecone script internal error.\n"
            "Inserted: 0\n"
        )
        self.run_subprocess.return_value = _completed(1, cli_output, "tqdm progress bar")

        response = client.post("/upload_db", data={
            "path": SESSION,
            "db_choice": "pinecone",
            "pinecone_index_name": "test_index"
        })

        self.assertEqual(response.status_code, 500)
        json_response = response.json()
        self.assertEqual(json_response["error"], "Vector DB upload failed")
        # The diagnostic comes from stdout (stderr only holds the progress bar)
        self.assertIn("Pinecone script internal error.", json_response["details"])
        self.assertNotIn("tqdm", json_response["details"])
        self.assertNotIn("inserted_count", json_response)

    def test_upload_db_pinecone_missing_index_name(self):
        # The route does not check the index name itself: rad_vectordb.py
        # refuses to run without --index and its message is returned.
        self.run_subprocess.return_value = _completed(
            1, "=== rad_vectordb.py ===\nERROR: --index is required for Pinecone\n"
        )

        response = client.post("/upload_db", data={
            "path": SESSION,
            "db_choice": "pinecone"
            # pinecone_index_name is missing
        })

        cmd = self._call()["cmd"]
        self.assertNotIn("--index", cmd)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"], "Vector DB upload failed")
        self.assertIn("--index is required for Pinecone", response.json()["details"])

    def test_upload_db_chunks_file_not_found(self):
        # Unknown folder without any owner record: refused before any lookup (audit A02)
        response = client.post("/upload_db", data={
            "path": "non_existent_session",
            "db_choice": "pinecone",
            "pinecone_index_name": "test_index"
        })
        self.assertEqual(response.status_code, 403)

        # Recorded upload of the user whose folder no longer exists
        self.db.add(SessionOwner(session_folder="non_existent_session", user_id=4242, source_type="zip"))
        self.db.commit()
        response = client.post("/upload_db", data={
            "path": "non_existent_session",
            "db_choice": "pinecone",
            "pinecone_index_name": "test_index"
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("Directory not found", response.json()["error"])

        # Existing session folder without the sparse embeddings file
        os.remove(self.chunks_file)
        response = client.post("/upload_db", data={
            "path": SESSION,
            "db_choice": "pinecone",
            "pinecone_index_name": "test_index"
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("Embeddings file not found", response.json()["error"])
        self.run_subprocess.assert_not_awaited()

    def test_upload_db_pinecone_missing_api_key(self):
        # User without a personal Pinecone key: 403 before any subprocess
        credentials = {k: v for k, v in FAKE_CREDENTIALS.items() if k != "pinecone_api_key"}
        self.user = _make_user(credentials)

        with patch.dict(os.environ, {"PINECONE_API_KEY": "fake-server-pinecone-key"}):
            response = client.post("/upload_db", data={
                "path": SESSION,
                "db_choice": "pinecone",
                "pinecone_index_name": "test_index"
            })

        # A server-side key never stands in for a non-admin user's own key
        self.assertEqual(response.status_code, 403)
        json_response = response.json()
        self.assertEqual(json_response["credential_required"], "pinecone_api_key")
        self.assertEqual(json_response["configure_url"], "/settings/credentials")
        self.assertTrue(json_response["error"])
        self.run_subprocess.assert_not_awaited()

    def test_upload_db_requires_authentication(self):
        app.dependency_overrides.pop(get_current_active_user, None)

        response = client.post("/upload_db", data={
            "path": SESSION,
            "db_choice": "pinecone",
            "pinecone_index_name": "test_index"
        })

        self.assertEqual(response.status_code, 401)
        self.run_subprocess.assert_not_awaited()

    def test_upload_db_unknown_database(self):
        response = client.post("/upload_db", data={
            "path": SESSION,
            "db_choice": "milvus"
        })

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unknown database type", response.json()["error"])
        self.run_subprocess.assert_not_awaited()

    def test_upload_db_weaviate_success(self):
        self.run_subprocess.return_value = _completed(0, _result_block("success", 5, skipped=2))

        response = client.post("/upload_db", data={
            "path": SESSION,
            "db_choice": "weaviate",
            "weaviate_class_name": "TestClass",
            "weaviate_tenant_name": "test_tenant"
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "status": "success",
            "message": "Uploaded to weaviate",
            "inserted_count": 5,
            "skipped_count": 2,
        })
        self._assert_command("weaviate", ["--class", "TestClass", "--tenant", "test_tenant"])
        env = self._call()["env"]
        self.assertTrue(env.get("WEAVIATE_URL") == FAKE_CREDENTIALS["weaviate_url"])
        self.assertTrue(env.get("WEAVIATE_API_KEY") == FAKE_CREDENTIALS["weaviate_api_key"])

    def test_upload_db_qdrant_success(self):
        self.run_subprocess.return_value = _completed(0, _result_block("success", 3))

        response = client.post("/upload_db", data={
            "path": SESSION,
            "db_choice": "qdrant",
            "qdrant_collection_name": "test_collection"
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "status": "success",
            "message": "Uploaded to qdrant",
            "inserted_count": 3,
        })
        self._assert_command("qdrant", ["--collection", "test_collection"])
        env = self._call()["env"]
        self.assertTrue(env.get("QDRANT_URL") == FAKE_CREDENTIALS["qdrant_url"])
        self.assertTrue(env.get("QDRANT_API_KEY") == FAKE_CREDENTIALS["qdrant_api_key"])


if __name__ == "__main__":
    unittest.main()
