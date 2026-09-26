"""
Tests for credential UI <-> pipeline consistency.

Regression coverage for the bug where an admin updated their API key in the
settings modal (which wrote only the ``.env`` file) but the pipeline kept using
a stale personal credential from the database, because
``build_subprocess_env`` / ``get_credential_or_env`` read the per-user database
store first and only fall back to ``.env`` for admins.

The fix makes ``POST /save_credentials`` mirror values into the admin's personal
database credentials and makes ``GET /get_credentials`` return the *effective*
value (database first, then ``.env``). These tests assert the resulting
invariant: **the value entered in the UI is the value the pipeline uses.**

With Albert enabled (``ALBERT_ENABLED=1``), ``ALBERT_API_KEY`` follows the same
round trip, except that the form only ever receives it masked (``••••`` plus
its last 4 characters) and posting that mask back never overwrites the key.

A form value carrying a control or line separator character (``\\r``, ``\\n``…)
is refused with a 400 before anything is written, so the form can never inject
extra ``.env`` lines.

Date: 2026-05-27
"""

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database.session import get_db
from app.database.base import Base
from app.models.user import User
from app.models.audit import AuditLog  # noqa: F401 (register metadata)
from app.middleware.auth import require_admin
from app.core.credentials import (
    build_subprocess_env,
    update_user_credentials,
    get_user_credentials,
    encrypt_credentials,
    CREDENTIAL_ENV_MAPPING,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def db_session():
    """File-based SQLite session (matches the repo's other route tests)."""
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=None,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        os.close(db_fd)
        os.unlink(db_path)


@pytest.fixture()
def admin_user(db_session):
    user = User(
        email="admin@example.com",
        hashed_password="x",
        roles=["USER", "ADMIN"],
        is_active=True,
        is_verified=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    assert user.is_admin is True
    return user


@pytest.fixture()
def client(db_session, admin_user):
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[require_admin] = lambda: admin_user
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(require_admin, None)


# ---------------------------------------------------------------------------
# Core invariant: the database store is what the pipeline uses
# ---------------------------------------------------------------------------
def test_db_credential_overrides_env_for_admin(admin_user, monkeypatch):
    """A personal (DB) Mistral key shadows .env for admins — the root cause."""
    monkeypatch.setenv("MISTRAL_API_KEY", "ENV_KEY")
    admin_user.api_credentials = encrypt_credentials({"mistral_api_key": "DB_OLD"})

    env = build_subprocess_env(admin_user)
    assert env["MISTRAL_API_KEY"] == "DB_OLD"


def test_updating_db_credential_changes_what_pipeline_uses(admin_user, db_session, monkeypatch):
    """After update_user_credentials, build_subprocess_env reflects the new key."""
    monkeypatch.setenv("MISTRAL_API_KEY", "ENV_KEY")
    admin_user.api_credentials = encrypt_credentials({"mistral_api_key": "DB_OLD"})

    update_user_credentials(admin_user, {"mistral_api_key": "NEW_KEY"}, db_session)

    env = build_subprocess_env(admin_user)
    assert env["MISTRAL_API_KEY"] == "NEW_KEY"


def test_clearing_db_credential_falls_back_to_env_for_admin(admin_user, db_session, monkeypatch):
    """Empty value deletes the personal credential, so .env fallback applies."""
    monkeypatch.setenv("MISTRAL_API_KEY", "ENV_KEY")
    admin_user.api_credentials = encrypt_credentials({"mistral_api_key": "DB_OLD"})

    update_user_credentials(admin_user, {"mistral_api_key": ""}, db_session)

    env = build_subprocess_env(admin_user)
    assert env["MISTRAL_API_KEY"] == "ENV_KEY"


# ---------------------------------------------------------------------------
# Route behaviour: /save_credentials mirrors to DB, /get_credentials shows it
# ---------------------------------------------------------------------------
def test_save_credentials_mirrors_to_personal_db_store(client, admin_user, db_session, tmp_path, monkeypatch):
    # Redirect the .env write to a temp dir so the real project .env is untouched.
    monkeypatch.setattr("app.routes.settings.RAGPY_DIR", str(tmp_path))

    resp = client.post("/save_credentials", json={"MISTRAL_API_KEY": "UI_KEY"})
    assert resp.status_code == 200
    assert resp.json().get("status") == "success"

    # .env written
    env_file = tmp_path / ".env"
    assert env_file.exists()
    assert "MISTRAL_API_KEY=UI_KEY" in env_file.read_text()

    # AND the personal DB credential now holds the same value (the store the
    # pipeline reads first).
    db_session.refresh(admin_user)
    assert get_user_credentials(admin_user).get("mistral_api_key") == "UI_KEY"

    # End-to-end: the pipeline subprocess env uses the UI value.
    env = build_subprocess_env(admin_user)
    assert env["MISTRAL_API_KEY"] == "UI_KEY"


def test_get_credentials_returns_effective_value_db_first(client, admin_user, db_session, tmp_path, monkeypatch):
    monkeypatch.setattr("app.routes.settings.RAGPY_DIR", str(tmp_path))
    # .env has one value, the personal DB store another (the effective one).
    (tmp_path / ".env").write_text("MISTRAL_API_KEY=ENV_VALUE\n")
    update_user_credentials(admin_user, {"mistral_api_key": "DB_VALUE"}, db_session)

    resp = client.get("/get_credentials")
    assert resp.status_code == 200
    assert resp.json()["MISTRAL_API_KEY"] == "DB_VALUE"


def test_save_credentials_refuses_line_injection(client, admin_user, db_session, tmp_path, monkeypatch):
    monkeypatch.setattr("app.routes.settings.RAGPY_DIR", str(tmp_path))
    env_file = tmp_path / ".env"
    env_file.write_text("MISTRAL_API_KEY=ENV_VALUE\n", encoding="utf-8")

    resp = client.post(
        "/save_credentials",
        json={"MISTRAL_API_KEY": "UI_KEY\r\nOPENAI_API_KEY=INJECTED", "PINECONE_ENV": "us-east-1"},
    )
    assert resp.status_code == 400
    assert resp.json()["invalid_keys"] == ["MISTRAL_API_KEY"]
    assert "INJECTED" not in resp.text

    # Nothing written: neither the .env file nor the personal DB store.
    assert env_file.read_text(encoding="utf-8") == "MISTRAL_API_KEY=ENV_VALUE\n"
    db_session.refresh(admin_user)
    assert get_user_credentials(admin_user) == {}


def test_env_to_cred_reverse_mapping_is_unambiguous():
    """The reverse mapping used by save_credentials must be bijective."""
    env_to_cred = {env: cred for cred, env in CREDENTIAL_ENV_MAPPING.items()}
    assert len(env_to_cred) == len(CREDENTIAL_ENV_MAPPING)
    assert env_to_cred["MISTRAL_API_KEY"] == "mistral_api_key"


# ---------------------------------------------------------------------------
# Albert (ALBERT_ENABLED=1): the masked key never overwrites the stored key
# ---------------------------------------------------------------------------
ALBERT_UI_KEY = "fake-albert-ui-key-4242"
ALBERT_MASK_PREFIX = "\u2022\u2022\u2022\u2022"


def _env_file_values(path):
    """Parse a ``.env`` file into a dict (``NAME=value`` lines)."""
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            name, value = line.split("=", 1)
            values[name.strip()] = value.strip()
    return values


def test_albert_key_roundtrip_when_enabled_mask_never_overwrites(client, admin_user, db_session, tmp_path, monkeypatch):
    monkeypatch.setattr("app.routes.settings.RAGPY_DIR", str(tmp_path))
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    monkeypatch.delenv("ALBERT_API_KEY", raising=False)

    # The UI saves a new key: .env and the personal DB store both get it.
    resp = client.post("/save_credentials", json={"ALBERT_API_KEY": ALBERT_UI_KEY})
    assert resp.status_code == 200 and resp.json().get("status") == "success"
    db_session.refresh(admin_user)
    assert (get_user_credentials(admin_user).get("albert_api_key") == ALBERT_UI_KEY) is True
    assert (_env_file_values(tmp_path / ".env").get("ALBERT_API_KEY") == ALBERT_UI_KEY) is True

    # The form only ever receives the mask, never the key.
    form = client.get("/get_credentials").json()
    assert form["ALBERT_API_KEY"] == ALBERT_MASK_PREFIX + ALBERT_UI_KEY[-4:]
    assert ALBERT_UI_KEY not in json.dumps(form)

    # Posting the whole form back (mask included) keeps the stored key.
    resp = client.post("/save_credentials", json=form)
    assert resp.status_code == 200
    db_session.refresh(admin_user)
    assert (get_user_credentials(admin_user).get("albert_api_key") == ALBERT_UI_KEY) is True
    assert (_env_file_values(tmp_path / ".env").get("ALBERT_API_KEY") == ALBERT_UI_KEY) is True

    # End-to-end: the pipeline subprocess env uses the UI value, not the mask.
    uses_ui_key = build_subprocess_env(admin_user).get("ALBERT_API_KEY") == ALBERT_UI_KEY
    assert uses_ui_key is True
