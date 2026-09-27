"""Suites de la relecture des corrections de l'audit du 2026-09-27.

* Suppression d'un utilisateur : ses imports hors projet (``session_owners``)
  partent avec lui ; un compte créé ensuite avec le même id SQLite n'en hérite pas.
* Nettoyage des dossiers orphelins (admin) : un import hors projet plus ancien
  que ``SESSION_TTL_HOURS`` et non verrouillé est retiré avec sa ligne ; un
  import récent, un import verrouillé par un traitement et une session de
  projet sont conservés.
* Pièces jointes du JSON Zotero (lecture, A01) : un chemin absolu ou ``..`` hors
  du dossier de la session n'est jamais lu ; la recherche approchée ne parcourt
  que ce dossier et retrouve les fichiers de ses sous-dossiers.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

_THIS = os.path.dirname(os.path.abspath(__file__))
RAGPY_ROOT = os.path.dirname(_THIS)
for _p in (RAGPY_ROOT, os.path.join(RAGPY_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.database.base import Base  # noqa: E402
from app.models import audit, background_task, pipeline_session, project  # noqa: E402,F401  (tables)
from app.models.pipeline_session import PipelineSession, SessionOwner  # noqa: E402
from app.models.project import Project  # noqa: E402
from app.models.user import User  # noqa: E402


@pytest.fixture
def db(tmp_path):
    """Base SQLite temporaire au schéma complet."""
    engine = create_engine(f"sqlite:///{tmp_path / 'followups.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = factory()
    session.factory = factory
    yield session
    session.close()
    engine.dispose()


def _user(db, email):
    """Utilisateur persistant."""
    user = User(email=email, hashed_password="x", roles=["USER"], is_active=True, is_verified=True)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# ======================================================================
# Suppression d'un utilisateur
# ======================================================================
def test_deleting_a_user_removes_his_upload_owner_rows(db):
    """``db.delete(user)`` (routes admin et « supprimer mon compte ») retire ses lignes ``session_owners``."""
    from app.core.session_access import SESSION_UNOWNED_MESSAGE, SessionAction, session_refusal

    eve = _user(db, "eve@example.org")
    db.add(SessionOwner(session_folder="abcd1234_eve", user_id=eve.id, source_type="zip"))
    db.commit()
    eve_id = eve.id
    db.delete(eve)
    db.commit()
    assert db.query(SessionOwner).count() == 0
    # Un compte qui réutiliserait l'id n'hérite de rien : dossier sans propriétaire.
    frank = User(id=eve_id, email="frank@example.org", hashed_password="x", roles=["USER"],
                 is_active=True, is_verified=True)
    db.add(frank)
    db.commit()
    uploads = os.path.dirname(db.bind.url.database)
    refusal = session_refusal(db, "abcd1234_eve", frank, SessionAction.WRITE, upload_dir=uploads)
    assert refusal == (403, SESSION_UNOWNED_MESSAGE)


# ======================================================================
# Nettoyage des dossiers orphelins
# ======================================================================
def test_orphan_cleanup_reclaims_stale_standalone_uploads(db, tmp_path, monkeypatch):
    """Import ancien et inactif non verrouillé : retiré ; récent, actif, verrouillé ou de projet : conservé."""
    import time as _time

    from app.services import job_control, session_cleanup

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    names = ("old_owned", "recent_owned", "old_busy", "old_but_active", "project_sess", "orphan")
    for name in names:
        (uploads / name).mkdir()
        (uploads / name / "output.csv").write_text("x", encoding="utf-8")
    two_days_ago = _time.time() - 48 * 3600
    for name in ("old_owned", "old_busy", "orphan"):
        for path in (uploads / name / "output.csv", uploads / name):
            os.utime(path, (two_days_ago, two_days_ago))
    monkeypatch.setenv("UPLOADS_DIR", str(uploads))
    monkeypatch.setenv("SESSION_TTL_HOURS", "24")
    monkeypatch.setattr(session_cleanup, "SessionLocal", db.factory)

    owner = _user(db, "owner@example.org")
    old = datetime.utcnow() - timedelta(hours=48)
    db.add_all([
        SessionOwner(session_folder="old_owned", user_id=owner.id, created_at=old),
        SessionOwner(session_folder="recent_owned", user_id=owner.id),
        SessionOwner(session_folder="old_busy", user_id=owner.id, created_at=old),
        SessionOwner(session_folder="old_but_active", user_id=owner.id, created_at=old),  # fichiers récents
    ])
    proj = Project(name="P", owner_id=owner.id)
    db.add(proj)
    db.commit()
    db.add(PipelineSession(project_id=proj.id, session_folder="project_sess"))
    db.commit()

    ticket, busy = job_control.acquire_job("old_busy", job_control.GROUP_PIPELINE, None, admission=False)
    assert busy is None
    try:
        result = session_cleanup.cleanup_orphaned_folders()
    finally:
        ticket.release()

    assert sorted(os.listdir(uploads)) == sorted(["old_but_active", "old_busy", "project_sess", "recent_owned"])
    assert result["deleted"] == 2 and result.get("skipped_busy") == 1
    db.expire_all()
    assert sorted(row.session_folder for row in db.query(SessionOwner).all()) == sorted([
        "old_busy", "old_but_active", "recent_owned"])
    # Aucun verrou laissé derrière : la session occupée redevient libre.
    assert not job_control.session_busy("old_busy", job_control.GROUP_PIPELINE)


# ======================================================================
# Pièces jointes du JSON Zotero confinées au dossier de la session
# ======================================================================
@pytest.fixture
def rad_dataframe(monkeypatch):
    """Module ``rad_dataframe`` importé comme par la CLI."""
    import rad_dataframe as module

    return module


def _item(path):
    """Élément Zotero minimal avec une pièce jointe texte."""
    return {"key": "ITEM1", "title": "Titre", "creators": [], "attachments": [{"path": path, "title": "Txt"}]}


@pytest.mark.parametrize("kind", ["absolute", "parent", "deep_parent"])
def test_attachment_outside_session_is_never_read(rad_dataframe, tmp_path, kind):
    """Chemin absolu ou ``..`` hors du dossier : fichier jamais lu, erreur ``PATH_OUTSIDE_DIR`` tracée."""
    session = tmp_path / "session"
    session.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("SECRET-DU-SERVEUR", encoding="utf-8")
    path = {"absolute": str(secret), "parent": "../secret.txt", "deep_parent": "files/../../secret.txt"}[kind]
    result = rad_dataframe._process_single_zotero_item(_item(path), str(session))
    assert all("SECRET-DU-SERVEUR" not in str(record.get("texteocr", "")) for record in result.records)
    assert any(error["error_type"] == "PATH_OUTSIDE_DIR" for error in result.errors)


def test_attachment_inside_session_is_read(rad_dataframe, tmp_path):
    """Chemin relatif ou absolu DANS le dossier de la session : lu normalement."""
    session = tmp_path / "session"
    (session / "files").mkdir(parents=True)
    note = session / "files" / "note.txt"
    note.write_text("Texte de la note autorisée.", encoding="utf-8")
    for path in ("files/note.txt", str(note)):
        result = rad_dataframe._process_single_zotero_item(_item(path), str(session))
        assert any("Texte de la note autorisée." in str(r.get("texteocr", "")) for r in result.records), path
        assert not any(error["error_type"] == "PATH_OUTSIDE_DIR" for error in result.errors)


def test_fuzzy_search_never_walks_outside_and_finds_nested_files(rad_dataframe, tmp_path):
    """Recherche approchée : jamais hors de la session ; dans la session, le vrai chemin d'un sous-dossier."""
    session = tmp_path / "session"
    nested = session / "files" / "123"
    nested.mkdir(parents=True)
    (nested / "Étude.pdf").write_bytes(b"%PDF-1.4")
    outside = tmp_path / "ailleurs"
    outside.mkdir()
    (outside / "Etude.pdf").write_bytes(b"%PDF-1.4")
    assert rad_dataframe._find_pdf_fuzzy(str(outside / "Etude.pdf"), str(outside / "Etude.pdf"), str(session)) is None
    found = rad_dataframe._find_pdf_fuzzy(str(session / "files" / "Etude.pdf"), "files/Etude.pdf", str(session))
    assert found == str(nested / "Étude.pdf")


def test_refused_absolute_paths_match_their_own_key_folder(rad_dataframe, tmp_path):
    """Export Better BibTeX (chemins absolus d'une autre machine) : chaque élément retrouve SON fichier.

    Même nom de fichier dans chaque ``storage/<CLÉ>`` : la correspondance se fait sur
    ``CLÉ/fichier`` (jamais le nom seul, jamais une recherche approchée) ; une clé
    absente de l'archive donne une erreur, pas le texte d'un autre document.
    """
    session = tmp_path / "session"
    for key, text in (("KEY1", "Texte du document un."), ("KEY2", "Texte du document deux.")):
        folder = session / "storage" / key
        folder.mkdir(parents=True)
        (folder / "Full Text.txt").write_text(text, encoding="utf-8")
    expected = {"KEY1": "Texte du document un.", "KEY2": "Texte du document deux."}
    for key, text in expected.items():
        item = _item(f"/Users/autre/Zotero/storage/{key}/Full Text.txt")
        result = rad_dataframe._process_single_zotero_item(item, str(session))
        texts = [str(r.get("texteocr", "")) for r in result.records]
        assert any(text in t for t in texts), (key, texts)
        assert not any(e["error_type"] == "PATH_OUTSIDE_DIR" for e in result.errors)
    missing = rad_dataframe._process_single_zotero_item(_item("/Users/autre/Zotero/storage/KEY9/Full Text.txt"),
                                                       str(session))
    assert missing.records == [] or all("Texte du document" not in str(r.get("texteocr", "")) for r in missing.records)
    assert any(e["error_type"] == "PATH_OUTSIDE_DIR" for e in missing.errors)


def test_cli_opt_in_follows_outside_paths(rad_dataframe, tmp_path, monkeypatch):
    """``--allow-outside-dir`` (CLI seulement) : un chemin absolu hors de ``--dir`` est suivi."""
    session = tmp_path / "session"
    session.mkdir()
    local = tmp_path / "local.txt"
    local.write_text("Fichier local de l'utilisateur de la CLI.", encoding="utf-8")
    monkeypatch.setattr(rad_dataframe, "ALLOW_ATTACHMENTS_OUTSIDE_DIR", True)
    result = rad_dataframe._process_single_zotero_item(_item(str(local)), str(session))
    assert any("Fichier local" in str(r.get("texteocr", "")) for r in result.records)
