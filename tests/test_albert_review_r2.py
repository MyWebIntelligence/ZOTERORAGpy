"""Tests de non-régression des constats de la revue indépendante du sprint Albert R2.

(Constats 1 à 3, sur les réponses sourcées, retirés avec la fonction le 2026-10-03.)

4. enregistrement des corpus par la tâche Celery ;
5. seules les lignes du manifeste écrites par l'envoi en cours sont enregistrées ;
6. corpus orphelins (compte supprimé) et partage retiré (projet supprimé) ;
7. collection supprimée → corpus marqué ``deleted`` ;
9. politique inconnue = ``albert_only`` ; images seulement avec une clé Albert ; plafond total audio.
"""

from __future__ import annotations

import json

import pytest

import scripts.rad_dataframe as rad
from app.core import albert_policy as web_policy
from app.models.albert_corpus import AlbertCorpus
from app.models.project import Project
from app.models.user import User
from app.services import albert_access
from scripts.rad_albert.config import DEFAULT_BASE_URL, normalise_data_policy
from tests.albert_fakes import FAKE_ALBERT_KEY
from tests.test_albert_corpora_routes import web  # noqa: F401


# ---------------------------------------------------------------------------
# 4 et 5. Enregistrement : lignes de l'envoi seulement, voie Celery
# ---------------------------------------------------------------------------
def _manifest_line(**row):
    """Ligne JSON de manifeste."""
    return json.dumps(row) + "\n"


def test_register_manifest_run_ignores_earlier_runs(web, tmp_path):
    manifest = tmp_path / "albert_manifest.jsonl"
    manifest.write_text(_manifest_line(collection_id=111, document_id=1, document_name="a", count=3), encoding="utf-8")
    offset = albert_access.manifest_offset(str(manifest))
    with open(manifest, "a", encoding="utf-8") as handle:
        handle.write(_manifest_line(collection_id=222, document_id=2, document_name="b", count=4))
    corpora = albert_access.register_manifest_run(
        web.db, user=web.users["collab"], manifest_path=str(manifest), offset=offset,
        base_url=DEFAULT_BASE_URL, key_fingerprint="f", session_folder="s",
    )
    assert [c.collection_id for c in corpora] == [222]
    assert web.db.query(AlbertCorpus).filter_by(collection_id=111).count() == 0


def test_celery_upload_registers_the_corpus(web, tmp_path, monkeypatch):
    from app.tasks import runner, vectordb

    manifest = tmp_path / "albert_manifest.jsonl"
    manifest.write_text(_manifest_line(collection_id=333, document_id=3, document_name="c", count=2), encoding="utf-8")
    monkeypatch.setattr(runner, "SessionLocal", lambda: web.db.__class__(bind=web.db.get_bind()))
    vectordb._register_albert_upload(web.users["owner"].id, str(manifest), 0, str(tmp_path / "output_chunks.json"),
                                     {"ALBERT_API_KEY": FAKE_ALBERT_KEY})
    web.db.expire_all()
    corpus = web.db.query(AlbertCorpus).filter_by(collection_id=333).one()
    assert corpus.owner_user_id == web.users["owner"].id and corpus.key_fingerprint


# ---------------------------------------------------------------------------
# 6 et 7. Orphelins, partage retiré, collection supprimée
# ---------------------------------------------------------------------------
def test_deleted_owner_corpora_become_orphaned_and_invisible(web):
    owner_id = web.users["owner"].id
    count = albert_access.orphan_user_corpora(web.db, owner_id)
    web.db.commit()
    assert count == 1
    assert albert_access.list_corpora(web.db, web.users["owner"]) == []
    _corpus, refusal = albert_access.get_corpus(web.db, web.users["owner"], web.corpus.id, albert_access.CorpusAction.READ)
    assert refusal[0] == 404


def test_deleted_project_unshares_its_corpora(web):
    project = Project(name="Autre", owner_id=web.users["other"].id)
    web.db.add(project)
    web.db.commit()
    assert albert_access.detach_project_corpora(web.db, web.project.id) == 1
    web.db.commit()
    web.db.expire_all()
    assert web.db.query(AlbertCorpus).get(web.corpus.id).project_id is None
    assert albert_access.list_corpora(web.db, web.users["viewer"]) == []


def test_project_deletion_route_unshares_corpora(web):
    response = web.client.delete(f"/projects/{web.project.id}", headers=web.headers["owner"])
    assert response.status_code == 200, response.text
    web.db.expire_all()
    assert web.db.query(AlbertCorpus).get(web.corpus.id).project_id is None


def test_collection_deletion_marks_corpus_deleted(web):
    response = web.client.delete(f"/api/albert/collections/{web.cid}?confirm=true", headers=web.headers["owner"])
    assert response.status_code == 200, response.text
    web.db.expire_all()
    assert web.db.query(AlbertCorpus).get(web.corpus.id).status == "deleted"
    listing = web.client.get("/api/albert/corpora", headers=web.headers["owner"]).json()["corpora"]
    assert web.corpus.id not in [c["id"] for c in listing]


# ---------------------------------------------------------------------------
# 9. Durcissements
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value,expected", [
    (None, "compatible"), ("", "compatible"), ("Compatible", "compatible"), ("albert_only", "albert_only"),
    ("albert-only", "albert_only"), ("Albert Only", "albert_only"), ("strict", "albert_only"),
    ("tout-ouvert", "albert_only"),
])
def test_unknown_policy_value_fails_closed(value, expected, monkeypatch):
    assert normalise_data_policy(value) == expected
    if value is None:
        monkeypatch.delenv("ALBERT_DATA_POLICY", raising=False)
    else:
        monkeypatch.setenv("ALBERT_DATA_POLICY", value)
    assert web_policy.strict_policy() is (expected == "albert_only")


def test_images_need_an_albert_key(monkeypatch):
    monkeypatch.setattr(rad, "OCR_ENABLE_ALBERT", True)
    monkeypatch.setattr(rad, "ALBERT_API_KEY", None)
    assert ".png" not in rad.supported_attachment_extensions()
    monkeypatch.setattr(rad, "ALBERT_API_KEY", FAKE_ALBERT_KEY)
    assert ".png" in rad.supported_attachment_extensions()


def test_user_model_still_importable():
    assert User.__tablename__ == "users"
