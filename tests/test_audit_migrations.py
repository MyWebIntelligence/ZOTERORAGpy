"""Audit A14 (2026-09-27) : migration d'une base existante vers le schéma corrigé.

Une base d'avant l'audit n'a ni la table ``session_owners`` (A02) ni les index
``ix_project_members_user_id`` / ``ix_project_members_project_id`` (A11).
``init_database()`` (``create_all`` puis ``run_migrations``) doit les ajouter
sans toucher aux données, et rester idempotent.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect, text

import app.database.init_db as init_db
from app.database.base import Base
from app.models import audit, background_task, pipeline_session, project, user  # noqa: F401  (tables)


@pytest.fixture
def old_database(tmp_path, monkeypatch):
    """Base SQLite au schéma d'avant l'audit, avec un utilisateur, un projet et un membre."""
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE session_owners"))
        conn.execute(text("DROP INDEX ix_project_members_user_id"))
        conn.execute(text("DROP INDEX ix_project_members_project_id"))
        conn.execute(text(
            "INSERT INTO users (id, email, hashed_password, roles, is_active, is_verified, "
            "failed_login_attempts, is_pending_approval, created_at, updated_at) "
            "VALUES (1, 'a@example.org', 'x', '[\"USER\"]', 1, 1, 0, 0, '2025-01-01', '2025-01-01')"
        ))
        conn.execute(text(
            "INSERT INTO projects (id, name, owner_id, is_archived, created_at, updated_at) "
            "VALUES (1, 'P', 1, 0, '2025-01-01', '2025-01-01')"
        ))
        conn.execute(text(
            "INSERT INTO project_members (id, project_id, user_id, role, created_at, updated_at) "
            "VALUES (1, 1, 1, 'viewer', '2025-01-01', '2025-01-01')"
        ))
    monkeypatch.setattr(init_db, "engine", engine)
    yield engine
    engine.dispose()


def _schema(engine):
    """Tables et index de ``project_members`` de la base."""
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    member_indexes = {idx["name"] for idx in inspector.get_indexes("project_members")}
    return tables, member_indexes


def test_old_database_is_migrated_without_data_loss(old_database):
    """``session_owners`` créée, index des adhésions ajoutés, données intactes ; deuxième passage sans effet."""
    tables, indexes = _schema(old_database)
    assert "session_owners" not in tables and "ix_project_members_user_id" not in indexes

    for _ in range(2):  # idempotent
        init_db.Base.metadata.create_all(bind=old_database)
        init_db.run_migrations()

    tables, indexes = _schema(old_database)
    assert "session_owners" in tables
    assert {"ix_project_members_user_id", "ix_project_members_project_id"} <= indexes
    with old_database.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM users")).scalar() == 1
        assert conn.execute(text("SELECT role FROM project_members")).scalar() == "viewer"
        conn.execute(text(
            "INSERT INTO session_owners (session_folder, user_id, source_type) VALUES ('abcd1234_x', 1, 'zip')"
        ))
