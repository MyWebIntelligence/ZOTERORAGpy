"""
Tests d'intégration pour les routes citations (Publish or Perish → Zotero).

Ce module teste les 4 endpoints API du workflow citation import :
1. POST /api/projects/{project_id}/upload_pop_json - Upload JSON PoP
2. POST /api/projects/{project_id}/filter_citations_sse - Filtrage LLM (SSE)
3. POST /api/projects/{project_id}/import_citations_sse - Import Zotero (SSE)
4. GET /api/projects/sessions/{session_id}/preview - Charger preview

Author: Claude Code
Date: 2025-12-05
"""

import json
import os
import tempfile
from io import BytesIO
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database.session import get_db
from app.database.base import Base
from app.models.user import User
from app.models.project import Project
from app.models.pipeline_session import PipelineSession, SessionStatus
from app.models.audit import AuditLog  # Import all models for metadata registration
from app.core.security import create_access_token


# ============================================================================
# Test Database Setup
# ============================================================================

@pytest.fixture(scope="function")
def test_db():
    """Create temporary file-based SQLite database for testing."""
    import tempfile
    import os

    # Create temporary database file
    db_fd, db_path = tempfile.mkstemp(suffix=".db")

    # Create test engine with file-based database
    test_engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=None  # Disable connection pooling for tests
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    # Import all models to register them with Base.metadata
    from app.models import user, project, audit, pipeline_session  # noqa: F401

    # Create tables
    Base.metadata.create_all(bind=test_engine)

    db = TestingSessionLocal()
    try:
        yield db, test_engine
    finally:
        db.close()
        Base.metadata.drop_all(bind=test_engine)
        test_engine.dispose()
        os.close(db_fd)
        os.unlink(db_path)


@pytest.fixture(scope="function")
def test_client(test_db):
    """Create FastAPI test client with overridden DB dependency and engine."""
    db_session, test_engine = test_db

    # Create scoped session factory for the test engine
    from sqlalchemy.orm import sessionmaker, scoped_session
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    TestSession = scoped_session(session_factory)

    # Override the database dependency to return scoped sessions
    def override_get_db():
        session = TestSession()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    # Patch the engine used by init_database
    with patch("app.database.init_db.engine", test_engine):
        with TestClient(app) as client:
            yield client

    # Clean up
    app.dependency_overrides.clear()
    TestSession.remove()


@pytest.fixture(scope="function")
def test_user(test_db):
    """Create test user."""
    db_session, _ = test_db  # Unpack tuple
    user = User(
        email="test@example.com",
        hashed_password="fakehash",
        is_active=True,
        is_verified=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture(scope="function")
def test_project(test_db, test_user):
    """Create test project."""
    db_session, _ = test_db  # Unpack tuple
    project = Project(
        name="Test Project",
        description="Test project for citation import",
        owner_id=test_user.id
    )
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)
    return project


@pytest.fixture(scope="function")
def auth_headers(test_user):
    """Create authentication headers with valid JWT token."""
    token = create_access_token(subject=str(test_user.id))
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="function")
def sample_pop_json():
    """Create sample Publish or Perish JSON data (array format)."""
    return [
        {
            "uid": "GS:1",
            "title": "Machine Learning for NLP: A Survey",
            "authors": ["Smith J", "Doe A"],
            "year": 2024,
            "source": "Journal of AI Research",
            "publisher": "IEEE",
            "article_url": "https://example.com/article1",
            "fulltext_url": None,
            "abstract": "This is a survey on ML for NLP...",
            "doi": "10.1234/ml-nlp-2024",
            "cites": 150,
            "type": "article"
        },
        {
            "uid": "GS:2",
            "title": "Deep Learning Advances",
            "authors": ["Brown C"],
            "year": 2023,
            "source": "ACM Computing Surveys",
            "article_url": None,
            "fulltext_url": "https://example.com/paper2.pdf",
            "abstract": "Recent advances in deep learning...",
            "doi": None,
            "cites": 89
        }
    ]


# ============================================================================
# Test: Upload PoP JSON
# ============================================================================

class TestUploadPopJson:
    """Test POST /api/projects/{project_id}/upload_pop_json endpoint."""

    def test_upload_valid_json(self, test_client, test_project, auth_headers, sample_pop_json):
        """Test uploading valid Publish or Perish JSON."""
        # Create JSON file
        json_data = json.dumps(sample_pop_json).encode("utf-8")
        files = {"json_file": ("test.json", BytesIO(json_data), "application/json")}

        form_data = {
            "collection_name": "ML Papers",
            "collection_description": "Machine learning research papers",
            "model": "gpt-4o-mini"
        }

        response = test_client.post(
            f"/api/projects/{test_project.id}/upload_pop_json",
            files=files,
            data=form_data,
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        assert "session_id" in data
        assert "session_folder" in data
        assert data["total_citations"] == 2
        assert data["requires_confirmation"] is False  # < 100

    def test_upload_large_batch_requires_confirmation(
        self, test_client, test_project, auth_headers
    ):
        """Test that > 100 citations triggers confirmation."""
        # Create 101 citations (direct array format)
        large_json = [
            {
                "uid": f"GS:{i}",
                "title": f"Paper {i}",
                "authors": ["Author X"],
                "year": 2024
            }
            for i in range(101)
        ]

        json_data = json.dumps(large_json).encode("utf-8")
        files = {"json_file": ("large.json", BytesIO(json_data), "application/json")}

        form_data = {
            "collection_name": "Large Collection",
            "collection_description": "",
            "model": "gpt-4o-mini"
        }

        response = test_client.post(
            f"/api/projects/{test_project.id}/upload_pop_json",
            files=files,
            data=form_data,
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        assert data["total_citations"] == 101
        assert data["requires_confirmation"] is True

    def test_upload_invalid_json(self, test_client, test_project, auth_headers):
        """Test uploading invalid JSON file."""
        invalid_data = b"Not valid JSON content"
        files = {"json_file": ("invalid.json", BytesIO(invalid_data), "application/json")}

        form_data = {
            "collection_name": "Test",
            "collection_description": "",
            "model": "gpt-4o-mini"
        }

        response = test_client.post(
            f"/api/projects/{test_project.id}/upload_pop_json",
            files=files,
            data=form_data,
            headers=auth_headers
        )

        assert response.status_code == 400
        assert "Invalid JSON" in response.json()["detail"]

    def test_upload_without_auth(self, test_client, test_project, sample_pop_json):
        """Test upload without authentication fails."""
        json_data = json.dumps(sample_pop_json).encode("utf-8")
        files = {"json_file": ("test.json", BytesIO(json_data), "application/json")}

        form_data = {
            "collection_name": "Test",
            "collection_description": "",
            "model": "gpt-4o-mini"
        }

        response = test_client.post(
            f"/api/projects/{test_project.id}/upload_pop_json",
            files=files,
            data=form_data
        )

        assert response.status_code == 401  # Unauthorized

    def test_upload_wrong_project_owner(
        self, test_client, test_db, test_project, auth_headers, sample_pop_json
    ):
        """Test upload to project owned by different user fails."""
        db_session, _ = test_db  # Unpack tuple
        # Create another user
        other_user = User(
            email="other@example.com",
            hashed_password="fakehash",
            is_active=True,
            is_verified=True  # Add is_verified
        )
        db_session.add(other_user)
        db_session.commit()
        db_session.refresh(other_user)

        # Create token for other user
        other_token = create_access_token(subject=str(other_user.id))
        other_headers = {"Authorization": f"Bearer {other_token}"}

        json_data = json.dumps(sample_pop_json).encode("utf-8")
        files = {"json_file": ("test.json", BytesIO(json_data), "application/json")}

        form_data = {
            "collection_name": "Test",
            "collection_description": "",
            "model": "gpt-4o-mini"
        }

        response = test_client.post(
            f"/api/projects/{test_project.id}/upload_pop_json",
            files=files,
            data=form_data,
            headers=other_headers
        )

        assert response.status_code == 403  # Forbidden


# ============================================================================
# Test: Filter Citations SSE
# ============================================================================

class TestFilterCitationsSSE:
    """Test POST /api/projects/{project_id}/filter_citations_sse endpoint."""

    @pytest.fixture
    def mock_session_with_files(self, test_db, test_project, sample_pop_json, tmp_path):
        """Create pipeline session with saved files."""
        db_session, _ = test_db  # Unpack tuple
        # Create session
        session = PipelineSession(
            project_id=test_project.id,
            session_folder=str(tmp_path.name),
            source_type="pop_json",
            status=SessionStatus.CREATED
        )
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)

        # Create files
        session_folder = tmp_path / session.session_folder
        session_folder.mkdir(exist_ok=True)

        # Save JSON
        json_path = session_folder / "publishorperish.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sample_pop_json, f)

        # Save config
        config_path = session_folder / "config.json"
        config = {
            "project_name": test_project.name,
            "project_description": test_project.description,
            "collection_name": "Test Collection",
            "collection_description": "Test",
            "model": "gpt-4o-mini"
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

        return session

    @patch("app.routes.citations.filter_citation_with_llm")
    @patch("app.routes.citations.fetch_citation_content")
    @patch("app.routes.citations.get_llm_semaphore")
    def test_filter_citations_success(
        self,
        mock_semaphore,
        mock_fetch,
        mock_filter,
        test_client,
        test_project,
        auth_headers,
        mock_session_with_files,
        tmp_path
    ):
        """Test successful citation filtering with SSE."""
        # Mock semaphore
        mock_sem = AsyncMock()
        mock_sem._value = 5
        mock_sem.__aenter__ = AsyncMock(return_value=None)
        mock_sem.__aexit__ = AsyncMock(return_value=None)
        mock_semaphore.return_value = mock_sem

        # Mock fetch (returns empty content)
        mock_fetch.return_value = ("", "none")

        # Mock LLM filter (first relevant, second skipped)
        mock_filter.side_effect = [
            {
                "itemType": "journalArticle",
                "title": "Machine Learning for NLP: A Survey",
                "creators": [{"firstName": "J", "lastName": "Smith", "creatorType": "author"}],
                "DOI": "10.1234/ml-nlp-2024"
            },
            "NA"  # Skipped
        ]

        # Patch UPLOAD_DIR to use tmp_path
        with patch("app.routes.citations.UPLOAD_DIR", str(tmp_path)):
            response = test_client.post(
                f"/api/projects/{test_project.id}/filter_citations_sse",
                data={"session_id": mock_session_with_files.id},
                headers=auth_headers,
                stream=True
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Parse SSE events
        events = []
        for line in response.iter_lines():
            line = line.decode("utf-8")
            if line.startswith("data:"):
                event_data = json.loads(line[5:].strip())
                events.append(event_data)

        # Check events sequence
        assert events[0]["type"] == "init"
        assert events[0]["total"] == 2

        assert events[1]["type"] == "progress"
        assert events[1]["status"] == "relevant"

        assert events[2]["type"] == "progress"
        assert events[2]["status"] == "skipped"

        assert events[3]["type"] == "complete"
        assert events[3]["relevant"] == 1
        assert events[3]["skipped"] == 1

        # Verify preview.json was created
        preview_path = tmp_path / mock_session_with_files.session_folder / "preview.json"
        assert preview_path.exists()

        with open(preview_path, "r", encoding="utf-8") as f:
            preview = json.load(f)

        assert len(preview["relevant"]) == 1
        assert len(preview["skipped"]) == 1

    def test_filter_without_upload(
        self, test_client, test_db, test_project, auth_headers
    ):
        """Test filtering without prior upload fails."""
        db_session, _ = test_db  # Unpack tuple
        # Create session without files
        session = PipelineSession(
            project_id=test_project.id,
            session_folder="nonexistent",
            source_type="pop_json",
            status=SessionStatus.CREATED
        )
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)

        response = test_client.post(
            f"/api/projects/{test_project.id}/filter_citations_sse",
            data={"session_id": session.id},
            headers=auth_headers,
            stream=True
        )

        # Should return error event
        events = []
        for line in response.iter_lines():
            line = line.decode("utf-8")
            if line.startswith("data:"):
                event_data = json.loads(line[5:].strip())
                events.append(event_data)

        assert any(e.get("type") == "error" for e in events)


# ============================================================================
# Test: Import Citations SSE
# ============================================================================

class TestImportCitationsSSE:
    """Test POST /api/projects/{project_id}/import_citations_sse endpoint."""

    @pytest.fixture
    def mock_session_with_preview(self, test_db, test_project, tmp_path):
        """Create session with preview.json."""
        db_session, _ = test_db  # Unpack tuple
        session = PipelineSession(
            project_id=test_project.id,
            session_folder=str(tmp_path.name),
            source_type="pop_json",
            status=SessionStatus.CREATED
        )
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)

        # Create session folder
        session_folder = tmp_path / session.session_folder
        session_folder.mkdir(exist_ok=True)

        # Save preview.json
        preview = {
            "relevant": [
                {
                    "index": 0,
                    "citation": {
                        "title": "Test Paper 1",
                        "authors": ["Smith J"]
                    },
                    "zotero_data": {
                        "itemType": "journalArticle",
                        "title": "Test Paper 1",
                        "DOI": "10.1234/test1"
                    },
                    "web_source": "none"
                },
                {
                    "index": 1,
                    "citation": {
                        "title": "Test Paper 2",
                        "authors": ["Doe A"]
                    },
                    "zotero_data": {
                        "itemType": "journalArticle",
                        "title": "Test Paper 2",
                        "url": "https://example.com/2"
                    },
                    "web_source": "html"
                }
            ],
            "skipped": []
        }

        preview_path = session_folder / "preview.json"
        with open(preview_path, "w", encoding="utf-8") as f:
            json.dump(preview, f)

        # Save config.json
        config = {
            "collection_name": "Test Collection",
            "collection_description": "Test"
        }
        config_path = session_folder / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

        return session

    @patch("app.routes.citations.create_or_update_item")
    @patch("app.routes.citations.get_or_create_collection")
    @patch.dict(os.environ, {
        "ZOTERO_API_KEY": "test_key",
        "ZOTERO_LIBRARY_TYPE": "user",
        "ZOTERO_USER_ID": "12345"
    })
    def test_import_citations_success(
        self,
        mock_get_collection,
        mock_create_item,
        test_client,
        test_project,
        auth_headers,
        mock_session_with_preview,
        tmp_path
    ):
        """Test successful citation import to Zotero."""
        # Mock collection creation
        mock_get_collection.return_value = {
            "key": "COLL123",
            "created": True,
            "name": "Test Collection"
        }

        # Mock item creation (first created, second updated)
        mock_create_item.side_effect = [
            {
                "success": True,
                "item_key": "ITEM1",
                "action": "created",
                "message": "Created"
            },
            {
                "success": True,
                "item_key": "ITEM2",
                "action": "updated",
                "message": "Updated"
            }
        ]

        # Import all citations (indices [0, 1])
        selected_indices = json.dumps([0, 1])

        with patch("app.routes.citations.UPLOAD_DIR", str(tmp_path)):
            response = test_client.post(
                f"/api/projects/{test_project.id}/import_citations_sse",
                data={
                    "session_id": mock_session_with_preview.id,
                    "selected_indices": selected_indices
                },
                headers=auth_headers,
                stream=True
            )

        assert response.status_code == 200

        # Parse SSE events
        events = []
        for line in response.iter_lines():
            line = line.decode("utf-8")
            if line.startswith("data:"):
                event_data = json.loads(line[5:].strip())
                events.append(event_data)

        # Check events
        assert events[0]["type"] == "init"
        assert events[0]["total"] == 2

        assert events[1]["type"] == "progress"
        assert events[1]["action"] == "created"

        assert events[2]["type"] == "progress"
        assert events[2]["action"] == "updated"

        assert events[3]["type"] == "complete"
        assert events[3]["created"] == 1
        assert events[3]["updated"] == 1
        assert events[3]["errors"] == 0

        # Verify session updated
        test_db = next(get_db())
        session = test_db.query(PipelineSession).get(mock_session_with_preview.id)
        assert session.status == SessionStatus.COMPLETED
        assert session.chunk_count == 2  # Created + Updated

    @patch.dict(os.environ, {}, clear=True)  # Clear env vars
    def test_import_without_zotero_credentials(
        self, test_client, test_project, auth_headers, mock_session_with_preview, tmp_path
    ):
        """Test import without Zotero credentials fails."""
        selected_indices = json.dumps([0])

        with patch("app.routes.citations.UPLOAD_DIR", str(tmp_path)):
            response = test_client.post(
                f"/api/projects/{test_project.id}/import_citations_sse",
                data={
                    "session_id": mock_session_with_preview.id,
                    "selected_indices": selected_indices
                },
                headers=auth_headers,
                stream=True
            )

        # Should return error event
        events = []
        for line in response.iter_lines():
            line = line.decode("utf-8")
            if line.startswith("data:"):
                event_data = json.loads(line[5:].strip())
                events.append(event_data)

        assert any("credentials not configured" in e.get("message", "").lower() for e in events)


# ============================================================================
# Test: Get Preview
# ============================================================================

class TestGetPreview:
    """Test GET /api/projects/sessions/{session_id}/preview endpoint."""

    @pytest.fixture
    def session_with_preview(self, test_db, test_project, tmp_path):
        """Create session with preview results."""
        db_session, _ = test_db  # Unpack tuple
        session = PipelineSession(
            project_id=test_project.id,
            session_folder=str(tmp_path.name),
            source_type="pop_json",
            status=SessionStatus.CREATED
        )
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)

        # Create preview file
        session_folder = tmp_path / session.session_folder
        session_folder.mkdir(exist_ok=True)

        preview = {
            "relevant": [
                {
                    "index": 0,
                    "citation": {"title": "Relevant Paper"},
                    "zotero_data": {"itemType": "journalArticle"}
                }
            ],
            "skipped": [
                {
                    "index": 1,
                    "citation": {"title": "Skipped Paper"},
                    "reason": "Not relevant"
                }
            ]
        }

        preview_path = session_folder / "preview.json"
        with open(preview_path, "w", encoding="utf-8") as f:
            json.dump(preview, f)

        return session

    def test_get_preview_success(
        self, test_client, auth_headers, session_with_preview, tmp_path
    ):
        """Test retrieving preview results."""
        with patch("app.routes.citations.UPLOAD_DIR", str(tmp_path)):
            response = test_client.get(
                f"/api/projects/sessions/{session_with_preview.id}/preview",
                headers=auth_headers
            )

        assert response.status_code == 200
        data = response.json()

        assert "relevant" in data
        assert "skipped" in data
        assert len(data["relevant"]) == 1
        assert len(data["skipped"]) == 1

    def test_get_preview_not_found(self, test_client, test_db, test_project, auth_headers):
        """Test getting preview when file doesn't exist."""
        db_session, _ = test_db  # Unpack tuple
        session = PipelineSession(
            project_id=test_project.id,
            session_folder="nonexistent",
            source_type="pop_json",
            status=SessionStatus.CREATED
        )
        db_session.add(session)
        db_session.commit()

        response = test_client.get(
            f"/api/projects/sessions/{session.id}/preview",
            headers=auth_headers
        )

        assert response.status_code == 404
        assert "Preview not available" in response.json()["detail"]

    def test_get_preview_wrong_user(
        self, test_client, test_db, session_with_preview
    ):
        """Test accessing preview from different user fails."""
        db_session, _ = test_db  # Unpack tuple
        # Create another user
        other_user = User(
            email="other@example.com",
            hashed_password="fakehash",
            is_active=True
        )
        db_session.add(other_user)
        db_session.commit()

        other_token = create_access_token(subject=str(other_user.id))
        other_headers = {"Authorization": f"Bearer {other_token}"}

        response = test_client.get(
            f"/api/projects/sessions/{session_with_preview.id}/preview",
            headers=other_headers
        )

        assert response.status_code == 403  # Forbidden


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
