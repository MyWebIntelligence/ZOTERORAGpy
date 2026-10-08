"""Tests des routes de gestion des corpus Albert (``app/routes/albert_corpora.py``) et du
service d'accès (``app/services/albert_access.py``). La recherche et les réponses sourcées
ont été retirées le 2026-10-03 : leurs routes répondent 404.

Base SQLite temporaire, faux serveur Albert (``FakeAlbert`` via une sous-classe d'``AlbertClient``),
clés factices seulement. Personas : propriétaire, autre compte (même clé Albert : aucun accès croisé
implicite), lecteur et collaborateur d'un projet, utilisateur sans clé.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.routes.albert_corpora as corpora_routes
import scripts.rad_albert.client as albert_client_module
from app.core.credentials import encrypt_credentials
from app.core.security import create_access_token
from app.database.base import Base
from app.database.session import get_db
from app.main import app
from app.models.albert_corpus import AlbertCorpus, AlbertCorpusSource
from app.models.audit import AuditLog
from app.models.pipeline_session import PipelineSession
from app.models.project import Project, ProjectMember
from app.models.user import User
from app.services import albert_access
from scripts.rad_albert.config import DEFAULT_BASE_URL
from tests.albert_fakes import FAKE_ALBERT_KEY, FakeAlbert

PERSONAS = {
    "owner": (["USER"], {"albert_api_key": FAKE_ALBERT_KEY}),
    "other": (["USER"], {"albert_api_key": FAKE_ALBERT_KEY}),
    "viewer": (["USER"], {"albert_api_key": FAKE_ALBERT_KEY}),
    "collab": (["USER"], {"albert_api_key": FAKE_ALBERT_KEY}),
    "nokey": (["USER"], {}),
}
CHUNKS = [
    ("<!-- Page 12 -->\nBourdieu définit le champ scientifique comme un espace de concurrence.",
     {"title": "Science de la science", "authors": "Bourdieu", "year": 2001, "content_id": "a_0"}),
    ("Latour décrit la fabrication des faits scientifiques au laboratoire.",
     {"title": "La vie de laboratoire", "authors": "Latour", "year": 1979, "content_id": "b_0"}),
]


def _recording_client_class(fake):
    """``AlbertClient`` réel branché sur le transport du faux, sans attente réelle."""
    base = albert_client_module.AlbertClient

    class _FakeTransportClient(base):
        """Client sur ``FakeAlbert``."""

        def __init__(self, cfg, api_key, **kwargs):
            """Branche le transport factice."""
            kwargs.setdefault("transport", fake.transport)
            kwargs.setdefault("sleep", lambda seconds: None)
            super().__init__(cfg, api_key, **kwargs)

    return _FakeTransportClient


@pytest.fixture
def web(tmp_path, monkeypatch):
    """Application sous test : Albert activé, base temporaire, faux Albert."""
    for name in ("ALBERT_ENABLED", "ALBERT_BASE_URL", "ALBERT_API_KEY", "ALBERT_DATA_POLICY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ALBERT_ENABLED", "1")

    engine = create_engine(f"sqlite:///{tmp_path / 'rag.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    users = {}
    for persona, (roles, creds) in PERSONAS.items():
        user = User(email=f"{persona}@rag.test", hashed_password="x", roles=roles, is_active=True,
                    is_verified=True, api_credentials=encrypt_credentials(creds) if creds else None)
        db.add(user)
        db.commit()
        db.refresh(user)
        users[persona] = user
    project = Project(name="Projet", owner_id=users["owner"].id)
    db.add(project)
    db.commit()
    db.add(ProjectMember(project_id=project.id, user_id=users["viewer"].id, role="viewer"))
    db.add(ProjectMember(project_id=project.id, user_id=users["collab"].id, role="collaborator"))
    db.commit()

    fake = FakeAlbert()
    monkeypatch.setattr(albert_client_module, "AlbertClient", _recording_client_class(fake))
    with fake.client() as http:
        cid = http.post("/collections", json={"name": "corpus-test", "visibility": "private"}).json()["id"]
        did = http.post("/documents", files={"name": (None, "ragpy:K1:a.pdf"),
                                             "collection_id": (None, str(cid))}).json()["id"]
        http.post(f"/documents/{did}/chunks", json={"chunks": [{"content": c, "metadata": m} for c, m in CHUNKS]})
        other_cid = http.post("/collections", json={"name": "autre", "visibility": "private"}).json()["id"]
        other_did = http.post("/documents", files={"name": (None, "ragpy:K9:b.pdf"),
                                                   "collection_id": (None, str(other_cid))}).json()["id"]
        http.post(f"/documents/{other_did}/chunks", json={"chunks": [{"content": "Texte secret d'un autre corpus."}]})
        public_cid = http.post("/collections", json={"name": "mediatech-test", "visibility": "public"}).json()["id"]
    fake.reset_calls()
    corpus = albert_access.register_corpus(db, owner_user_id=users["owner"].id, collection_id=cid,
                                           base_url=DEFAULT_BASE_URL, collection_name="corpus-test",
                                           project_id=project.id, gdpr_ack=True)
    albert_access.upsert_sources(db, corpus, [{"document_id": did, "document_name": "ragpy:K1:a.pdf",
                                               "title": "Science de la science", "chunk_count": 2}])
    other = albert_access.register_corpus(db, owner_user_id=users["other"].id, collection_id=other_cid,
                                          base_url=DEFAULT_BASE_URL, collection_name="autre")
    db.commit()

    app.dependency_overrides[get_db] = lambda: db
    try:
        yield SimpleNamespace(
            client=TestClient(app), db=db, users=users, fake=fake, corpus=corpus, other=other,
            cid=cid, did=did, other_did=other_did, public_cid=public_cid, project=project, tmp=tmp_path,
            headers={p: {"Authorization": "Bearer " + create_access_token(subject=str(u.id))} for p, u in users.items()},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)
        db.close()
        engine.dispose()


# ---------------------------------------------------------------------------
# Portes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method,path", [
    ("GET", "/api/albert/corpora"), ("GET", "/api/albert/usage"), ("GET", "/api/albert/capabilities"),
    ("DELETE", "/api/albert/corpora/1"),
])
def test_routes_are_unknown_while_albert_is_off(web, monkeypatch, method, path):
    monkeypatch.setenv("ALBERT_ENABLED", "0")
    response = web.client.request(method, path, headers=web.headers["owner"])
    assert response.status_code == 404 and response.json() == {"detail": "Not Found"}


@pytest.mark.parametrize("method,path", [
    ("POST", "/api/albert/search"), ("POST", "/api/albert/rag"), ("POST", "/api/albert/rag/stream"),
    ("POST", "/api/albert/rag/abc/cancel"), ("GET", "/api/albert/public_collections"), ("GET", "/albert/rag"),
])
def test_removed_search_and_answer_routes_are_gone(web, method, path):
    """Recherche et réponses sourcées retirées (2026-10-03) : 404 même avec Albert activé, aucun appel."""
    response = web.client.request(method, path, json={}, headers=web.headers["owner"])
    assert response.status_code in (404, 405)
    assert web.fake.calls == []


def test_routes_hidden_from_openapi(web):
    paths = web.client.get("/openapi.json").json()["paths"]
    assert not any(p.startswith("/api/albert/") or p == "/albert/rag" for p in paths)


def test_capabilities_reports_policy_without_search(web):
    body = web.client.get("/api/albert/capabilities", headers=web.headers["owner"]).json()
    assert body["policy"]["policy"] == "compatible" and body["audio"] is False
    assert "rag" not in body and "rerank" not in body and "defaults" not in body


# ---------------------------------------------------------------------------
# Corpus et droits
# ---------------------------------------------------------------------------
def test_corpora_listing_respects_owner_and_project(web):
    owner = web.client.get("/api/albert/corpora", headers=web.headers["owner"]).json()["corpora"]
    assert [c["id"] for c in owner] == [web.corpus.id] and owner[0]["can_write"] is True
    assert owner[0]["documents"] == 1 and owner[0]["chunks"] == 2
    viewer = web.client.get("/api/albert/corpora", headers=web.headers["viewer"]).json()["corpora"]
    assert [c["id"] for c in viewer] == [web.corpus.id] and viewer[0]["can_write"] is False
    other = web.client.get("/api/albert/corpora", headers=web.headers["other"]).json()["corpora"]
    assert [c["id"] for c in other] == [web.other.id]


def test_same_remote_key_never_opens_another_users_corpus(web):
    response = web.client.get(f"/api/albert/corpora/{web.corpus.id}/documents", headers=web.headers["other"])
    assert response.status_code == 404
    assert web.fake.calls == []


def test_viewer_can_read_documents_but_not_delete(web):
    listing = web.client.get(f"/api/albert/corpora/{web.corpus.id}/documents", headers=web.headers["viewer"])
    assert listing.status_code == 200
    delete = web.client.delete(f"/api/albert/corpora/{web.corpus.id}/documents/{web.did}?confirm=true",
                               headers=web.headers["viewer"])
    assert delete.status_code == 403
    assert web.fake.calls_to("DELETE", "/v1/documents/*") == []


def test_missing_key_gives_403_before_any_call(web):
    response = web.client.get("/api/albert/usage?days=7", headers=web.headers["nokey"])
    assert response.status_code == 403
    assert response.json()["credential_required"] == "albert_api_key"
    assert web.fake.calls == []


# ---------------------------------------------------------------------------
# Documents, rattachement, usage
# ---------------------------------------------------------------------------
def test_documents_listing_joins_catalogue(web):
    body = web.client.get(f"/api/albert/corpora/{web.corpus.id}/documents", headers=web.headers["viewer"]).json()
    assert body["total"] == 1 and body["documents"][0]["in_catalogue"] is True
    assert body["documents"][0]["title"] == "Science de la science"


def test_document_deletion_requires_confirmation_membership_and_audits(web):
    no_confirm = web.client.delete(f"/api/albert/corpora/{web.corpus.id}/documents/{web.did}",
                                   headers=web.headers["owner"])
    assert no_confirm.status_code == 400 and no_confirm.json()["confirm_required"] is True
    foreign = web.client.delete(f"/api/albert/corpora/{web.corpus.id}/documents/{web.other_did}?confirm=true",
                                headers=web.headers["owner"])
    assert foreign.status_code == 404
    assert web.fake.calls_to("DELETE", "/v1/documents/*") == []
    ok = web.client.delete(f"/api/albert/corpora/{web.corpus.id}/documents/{web.did}?confirm=true",
                           headers=web.headers["collab"])
    assert ok.status_code == 200, ok.json()
    web.db.expire_all()
    rows = [r for r in web.db.query(AuditLog).all() if r.action == "ALBERT_DOCUMENT_DELETE"]
    assert len(rows) == 1 and rows[0].success == 1 and rows[0].details["outcome"] == "deleted"
    assert web.db.query(AlbertCorpusSource).filter_by(document_id=web.did).count() == 0


def test_bind_private_collection_and_refuse_public(web):
    public = web.client.post("/api/albert/corpora/bind", json={"collection_id": web.public_cid},
                             headers=web.headers["owner"])
    assert public.status_code == 400
    bound = web.client.post("/api/albert/corpora/bind", json={"collection_id": web.other.collection_id},
                            headers=web.headers["owner"])
    assert bound.status_code == 201, bound.json()
    again = web.client.post("/api/albert/corpora/bind", json={"collection_id": web.other.collection_id},
                            headers=web.headers["owner"])
    assert again.status_code == 200 and again.json()["existing"] is True
    viewer_bind = web.client.post("/api/albert/corpora/bind",
                                  json={"collection_id": web.cid, "project_id": web.project.id},
                                  headers=web.headers["viewer"])
    assert viewer_bind.status_code == 403


def test_detach_is_owner_only_and_keeps_remote_collection(web):
    refused = web.client.delete(f"/api/albert/corpora/{web.corpus.id}", headers=web.headers["collab"])
    assert refused.status_code == 403
    ok = web.client.delete(f"/api/albert/corpora/{web.corpus.id}", headers=web.headers["owner"])
    assert ok.status_code == 200
    assert web.fake.calls_to("DELETE", "/v1/collections/*") == []
    assert web.cid in web.fake.collections


def test_usage_route_summarises_buckets(web):
    web.fake.usage_buckets = [{"object": "usage.bucket", "start_time": 1790899200, "end_time": 1790985600,
                               "requests": 3, "prompt_tokens": 145, "completion_tokens": 50, "total_tokens": 195,
                               "cost": 0.0, "impacts": {"kWh": 1e-6, "kgCO2eq": 2e-7}}]
    body = web.client.get("/api/albert/usage?days=7", headers=web.headers["owner"]).json()
    assert body["success"] is True
    assert body["totals"]["requests"] == 3 and body["days"][0]["date_utc"] == "2026-10-02"
    call = web.fake.calls_to("GET", "/v1/usage")[0]
    assert int(call.params["end_time"]) - int(call.params["start_time"]) == 7 * 86400
    bad = web.client.get("/api/albert/usage?days=400", headers=web.headers["owner"])
    assert bad.status_code == 400


def test_summarise_usage_unknown_fields_stay_none():
    summary = corpora_routes.summarise_usage([{"start_time": 86400, "requests": 2}])
    assert summary["days"][0]["date_utc"] == "1970-01-02"
    assert summary["totals"]["requests"] == 2 and summary["totals"]["cost"] is None


# ---------------------------------------------------------------------------
# Service : enregistrement après envoi
# ---------------------------------------------------------------------------
def test_live_manifest_slices_drop_rolled_back_documents():
    rows = [
        {"collection_id": 1, "document_id": 10, "count": 64},
        {"collection_id": 1, "document_id": 11, "count": 5},
        {"event": "rollback", "collection_id": 1, "document_id": 11},
        {"collection_id": 1, "document_id": 10, "count": 3},
    ]
    live = albert_access.live_manifest_slices(rows)
    assert [(r["document_id"], r["count"]) for r in live] == [(10, 64), (10, 3)]


def test_register_upload_creates_corpus_sources_and_shares_project(web, tmp_path):
    session = "abcd1234_corpus"
    web.db.add(PipelineSession(project_id=web.project.id, session_folder=session))
    web.db.commit()
    chunks = tmp_path / "output_chunks.json"
    chunks.write_text(json.dumps([
        {"itemKey": "K7", "filename": "/home/x/secret/c.pdf", "title": "Titre C", "authors": "Auteur",
         "date": "2019-05-01", "text": "texte"}]), encoding="utf-8")
    rows = [{"collection_id": 4242, "collection_name": "nouvelle", "document_id": 77,
             "document_name": "ragpy:K7:c.pdf", "count": 12, "slice": 1}]
    corpora = albert_access.register_upload(web.db, user=web.users["owner"], manifest_rows=rows,
                                            base_url=DEFAULT_BASE_URL, key_fingerprint="abc",
                                            session_folder=session, chunks_path=str(chunks))
    assert len(corpora) == 1
    corpus = corpora[0]
    assert corpus.project_id == web.project.id and corpus.collection_name == "nouvelle"
    source = corpus.sources[0]
    assert (source.title, source.year, source.filename, source.chunk_count) == ("Titre C", "2019", "c.pdf", 12)
    assert "/home" not in json.dumps(source.to_dict())
    viewer = albert_access.list_corpora(web.db, web.users["viewer"])
    assert corpus.id in [c.id for c in viewer]


def test_upload_db_hook_registers_corpus_from_manifest(web, tmp_path):
    import asyncio

    from app.routes import processing

    manifest = tmp_path / "albert_manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in [
        {"collection_id": 5151, "collection_name": "hook", "document_id": 1, "document_name": "ragpy::x", "count": 3},
    ]) + "\n", encoding="utf-8")
    asyncio.run(processing._register_albert_corpus(web.db, web.users["owner"], str(manifest), "sess_hook", None))
    corpus = web.db.query(AlbertCorpus).filter_by(collection_id=5151).one()
    assert corpus.owner_user_id == web.users["owner"].id and corpus.key_fingerprint
    assert len(corpus.sources) == 1


def test_user_deletion_requires_confirmation_when_albert_corpora_exist(web):
    admin = User(email="admin@rag.test", hashed_password="x", roles=["USER", "ADMIN"], is_active=True,
                 is_verified=True)
    web.db.add(admin)
    web.db.commit()
    headers = {"Authorization": "Bearer " + create_access_token(subject=str(admin.id))}
    owner_id = web.users["owner"].id
    refused = web.client.delete(f"/api/admin/users/{owner_id}", headers=headers)
    assert refused.status_code == 409
    inventory = refused.json()["detail"]["albert_collections"]
    assert [item["collection_id"] for item in inventory] == [web.cid]
    assert web.db.query(User).filter_by(id=owner_id).count() == 1
    # Sans corpus Albert : suppression inchangée.
    plain = web.client.delete(f"/api/admin/users/{web.users['nokey'].id}", headers=headers)
    assert plain.status_code == 200
    kept = web.client.delete(f"/api/admin/users/{owner_id}?albert_data=keep", headers=headers)
    assert kept.status_code == 200
    web.db.expire_all()
    audit = [r for r in web.db.query(AuditLog).all() if r.resource_id == owner_id and r.resource_type == "user"]
    assert audit and audit[-1].details["albert_collections_kept"][0]["collection_id"] == web.cid
