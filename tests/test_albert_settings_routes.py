"""Tests des routes de collections Albert de ``app/routes/settings.py`` (lot 7, tâche 8).

Couverture (``.claude/tasks/SPRINT_albert.md``, lot 7) :

* ``GET /api/albert/collections`` et ``DELETE /api/albert/collections/{id}``
  répondent 404 ``{"detail": "Not Found"}`` quand Albert est OFF, pour toute
  méthode, exactement comme un chemin inconnu, sans aucun appel amont ;
* 403 ``credential_required=albert_api_key`` sans clé personnelle, même quand
  le serveur a une clé (non-admin) ; repli ``.env`` pour un admin ;
* le listing ne renvoie que des collections dont la visibilité est confirmée
  ``private`` (jamais le propriétaire), parcourt toutes les pages amont
  (``limit`` <= 100) et se pagine côté RAGpy ;
* la suppression exige ``?confirm=true`` et est journalisée par
  ``create_audit_log`` ;
* la clé n'apparaît jamais dans une URL, un paramètre, une réponse ni un log.

Aucun appel réseau : ``scripts.rad_albert.client.AlbertClient`` est remplacé
par une sous-classe branchée sur le ``httpx.MockTransport`` de
``tests/albert_fakes.FakeAlbert`` (même motif que
``tests/test_albert_credentials.py``). Seules des clés factices sont
utilisées ; les assertions portent sur des noms, des booléens et des
compteurs.

Run: pytest tests/test_albert_settings_routes.py -q
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database.base import Base
from app.database.session import get_db
from app.models import user as _user_models, project as _project_models  # noqa: F401
from app.models import audit as _audit_models, pipeline_session as _ps_models  # noqa: F401
from app.models import background_task as _bg_models  # noqa: F401
from app.models.audit import AuditLog
from app.models.user import User
from app.core.security import create_access_token
from app.core.credentials import CREDENTIAL_ENV_MAPPING, CREDENTIAL_ERROR_MESSAGES, encrypt_credentials
import app.routes.settings as settings_routes
import scripts.rad_albert.client as albert_client_module
from tests.albert_fakes import FAKE_ALBERT_KEY, FakeAlbert

# ---------------------------------------------------------------------------
# Constantes littérales
# ---------------------------------------------------------------------------
LIST_PATH = "/api/albert/collections"
ITEM_PATH = "/api/albert/collections/{cid}"
CONFIGURE_URL = "/settings/credentials"
ALBERT_CREDENTIAL = "albert_api_key"
DELETE_AUDIT_PREFIX = "ALBERT_COLLECTION"
PROBED_METHODS = ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS")
MAX_UPSTREAM_LIMIT = 100

ADMIN_DB_ALBERT_KEY = "fake-albert-key-admin-db-0002"
SERVER_ENV_ALBERT_KEY = "fake-albert-server-env-0002"
QUERY_ALBERT_KEY = "fake-albert-query-key-0002"
ALL_FAKE_ALBERT_KEYS = (FAKE_ALBERT_KEY, ADMIN_DB_ALBERT_KEY, SERVER_ENV_ALBERT_KEY, QUERY_ALBERT_KEY)

PERSONA_ROLES = {
    "admin": ["USER", "ADMIN"],
    "admin_nokeys": ["USER", "ADMIN"],
    "member_nokeys": ["USER"],
    "member_albert": ["USER"],
}
# member_albert porte la clé attendue par FakeAlbert ; celle de l'admin est refusée (401 amont).
PERSONA_CREDENTIALS = {
    "admin": {"albert_api_key": ADMIN_DB_ALBERT_KEY},
    "admin_nokeys": {},
    "member_nokeys": {},
    "member_albert": {"albert_api_key": FAKE_ALBERT_KEY},
}

N_PRIVATE = 105
N_PUBLIC = 3
FORBIDDEN_SUMMARY_FIELDS = ("owner",)


# ---------------------------------------------------------------------------
# Aides
# ---------------------------------------------------------------------------
def _contains_fake_key(text):
    """Vrai si une des clés Albert factices apparaît dans ``text``."""
    return any(key in text for key in ALL_FAKE_ALBERT_KEYS)


def _recording_client_class(fake, builds):
    """Sous-classe d'``AlbertClient`` branchée sur ``fake`` ; ``builds`` note si la clé est celle du faux."""
    base = albert_client_module.AlbertClient

    class _FakeTransportClient(base):
        """``AlbertClient`` réel sur le transport de ``FakeAlbert``, sans attente réelle."""

        def __init__(self, cfg, api_key, **kwargs):
            """Note la construction puis branche le transport factice."""
            builds.append(api_key == FAKE_ALBERT_KEY)
            kwargs.setdefault("transport", fake.transport)
            kwargs.setdefault("sleep", lambda seconds: None)
            super().__init__(cfg, api_key, **kwargs)

    return _FakeTransportClient


def _populate(fake, n_private, n_public):
    """Crée des collections dans ``fake`` (via son client) ; renvoie ``(ids privés, ids publics)``."""
    private, public = [], []
    with fake.client() as http:
        for idx in range(n_private):
            resp = http.post("/collections", json={"name": f"ragpy-private-{idx:03d}", "visibility": "private"})
            private.append(resp.json()["id"])
        for idx in range(n_public):
            resp = http.post("/collections", json={"name": f"public-{idx}", "visibility": "public"})
            public.append(resp.json()["id"])
    fake.reset_calls()
    return private, public


def _listed(body):
    """Collections d'une réponse de listing."""
    items = body.get("collections") if isinstance(body, dict) else None
    assert isinstance(items, list), body
    return items


def _audit_rows(web, collection_id=None):
    """Entrées d'audit de suppression Albert (optionnellement pour une collection)."""
    web.db.expire_all()
    rows = [
        r for r in web.db.query(AuditLog).all()
        if (r.action or "").upper().startswith(DELETE_AUDIT_PREFIX) and "DELETE" in (r.action or "").upper()
    ]
    if collection_id is None:
        return rows
    return [r for r in rows if str((r.details or {}).get("collection_id")) == str(collection_id)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _credential_env_hygiene(monkeypatch):
    """Retire les identifiants réels de l'env (seules des valeurs factices restent)."""
    for name in CREDENTIAL_ENV_MAPPING.values():
        monkeypatch.delenv(name, raising=False)
    for name in ("ALBERT_ENABLED", "ALBERT_BASE_URL", "RAGPY_DOTENV_DENY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def web(tmp_path, monkeypatch):
    """Application sous test : base SQLite temporaire, 4 personas, faux Albert."""
    home = tmp_path / "ragpy_home"
    home.mkdir()
    monkeypatch.setattr(settings_routes, "RAGPY_DIR", str(home))

    engine = create_engine(f"sqlite:///{tmp_path / 'albert_settings.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    users = {}
    for persona, roles in PERSONA_ROLES.items():
        creds = PERSONA_CREDENTIALS[persona]
        user = User(
            email=f"{persona.replace('_', '.')}@albert-settings.test",
            hashed_password="x",
            roles=list(roles),
            is_active=True,
            is_verified=True,
            api_credentials=encrypt_credentials(creds) if creds else None,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        users[persona] = user

    fake = FakeAlbert()
    builds = []
    monkeypatch.setattr(albert_client_module, "AlbertClient", _recording_client_class(fake, builds))

    def _override_get_db():
        """Renvoie la session de test (surcharge de dépendance)."""
        return db

    app.dependency_overrides[get_db] = _override_get_db
    try:
        yield SimpleNamespace(
            client=TestClient(app),
            db=db,
            users=users,
            headers={
                persona: {"Authorization": "Bearer " + create_access_token(subject=str(user.id))}
                for persona, user in users.items()
            },
            fake=fake,
            builds=builds,
        )
    finally:
        app.dependency_overrides.pop(get_db, None)
        db.close()
        engine.dispose()


# ---------------------------------------------------------------------------
# Albert OFF : routes invisibles
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("switch", [None, "0", "off"])
def test_collections_404_when_off(web, monkeypatch, switch):
    if switch is not None:
        monkeypatch.setenv("ALBERT_ENABLED", switch)
    monkeypatch.setenv("ALBERT_API_KEY", SERVER_ENV_ALBERT_KEY)
    paths = (LIST_PATH, ITEM_PATH.format(cid=302360), ITEM_PATH.format(cid=302360) + "?confirm=true")
    for method in PROBED_METHODS:
        reference = web.client.request(method, "/api/albert/does-not-exist", follow_redirects=False)
        assert reference.status_code == 404
        for path in paths:
            for headers in ({}, web.headers["member_albert"], web.headers["admin"]):
                response = web.client.request(method, path, headers=headers, follow_redirects=False)
                assert (response.status_code, response.content) == (reference.status_code, reference.content)
                assert "allow" not in response.headers and "location" not in response.headers
                if method == "GET":
                    assert response.json() == {"detail": "Not Found"}
    assert web.builds == []
    assert web.fake.calls == []
    assert _audit_rows(web) == []


# ---------------------------------------------------------------------------
# Albert ON : clé exigée
# ---------------------------------------------------------------------------
def test_collections_403_without_key(web, monkeypatch):
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    monkeypatch.setenv("ALBERT_API_KEY", SERVER_ENV_ALBERT_KEY)
    expected = {
        "error": CREDENTIAL_ERROR_MESSAGES[ALBERT_CREDENTIAL],
        "credential_required": ALBERT_CREDENTIAL,
        "configure_url": CONFIGURE_URL,
    }
    target = ITEM_PATH.format(cid=302360) + "?confirm=true"
    for method, path in (("GET", LIST_PATH), ("DELETE", target)):
        response = web.client.request(method, path, headers=web.headers["member_nokeys"])
        assert response.status_code == 403, (method, response.text)
        assert response.json() == expected
        assert web.client.request(method, path).status_code == 401
    assert web.builds == []
    assert web.fake.calls == []
    assert _audit_rows(web) == []

    # Admin sans clé personnelle : la clé du serveur (.env) est utilisée ; sans aucune clé : 403.
    monkeypatch.setenv("ALBERT_API_KEY", FAKE_ALBERT_KEY)
    response = web.client.get(LIST_PATH, headers=web.headers["admin_nokeys"])
    assert response.status_code == 200
    assert web.builds == [True]
    monkeypatch.delenv("ALBERT_API_KEY")
    response = web.client.get(LIST_PATH, headers=web.headers["admin_nokeys"])
    assert response.status_code == 403 and response.json() == expected


# ---------------------------------------------------------------------------
# Albert ON : listing privé et paginé
# ---------------------------------------------------------------------------
def test_collections_private_paginated(web, monkeypatch):
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    private, public = _populate(web.fake, N_PRIVATE, N_PUBLIC)

    seen = []
    pages = []
    page = 1
    while True:
        response = web.client.get(f"{LIST_PATH}?page={page}&per_page={MAX_UPSTREAM_LIMIT}",
                                  headers=web.headers["member_albert"])
        assert response.status_code == 200, response.text[:300]
        body = response.json()
        items = _listed(body)
        pages.append(len(items))
        seen.extend(items)
        if len(items) < MAX_UPSTREAM_LIMIT:
            break
        page += 1
        assert page <= 5, "pagination sans fin"
    assert pages == [MAX_UPSTREAM_LIMIT, N_PRIVATE - MAX_UPSTREAM_LIMIT]
    ids = [item["id"] for item in seen]
    assert sorted(ids) == sorted(private)
    assert not set(ids) & set(public)
    assert all(item.get("visibility") in (None, "private") for item in seen)
    for item in seen:
        assert not set(FORBIDDEN_SUMMARY_FIELDS) & set(item)

    # Toutes les pages amont ont été parcourues, jamais au-delà de 100 par page.
    upstream = web.fake.calls_to("GET", "/v1/collections")
    assert upstream, "aucun appel amont"
    assert all(int(call.params.get("limit", 0)) <= MAX_UPSTREAM_LIMIT for call in upstream)
    offsets = sorted({int(call.params.get("offset", 0)) for call in upstream})
    assert offsets[0] == 0 and len(offsets) >= 2

    # Réponse par défaut : une page d'au plus 100 collections, toutes privées.
    response = web.client.get(LIST_PATH, headers=web.headers["member_albert"])
    assert response.status_code == 200
    default_items = _listed(response.json())
    assert 0 < len(default_items) <= MAX_UPSTREAM_LIMIT
    assert set(i["id"] for i in default_items) <= set(private)

    # Pagination invalide : 400 sans appel amont.
    web.fake.reset_calls()
    for query in ("page=0", "per_page=0", f"per_page={MAX_UPSTREAM_LIMIT + 1}", "page=abc"):
        response = web.client.get(f"{LIST_PATH}?{query}", headers=web.headers["member_albert"])
        assert response.status_code == 400, query
    assert web.fake.calls == []


def test_collections_listing_fails_closed_on_visibility(web, monkeypatch):
    """Une collection de visibilité absente ou non privée n'est jamais listée, même renvoyée par l'amont."""
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    page = {"object": "list", "data": [
        {"object": "collection", "id": 11, "name": "a", "visibility": "private", "owner": "<masqué>"},
        {"object": "collection", "id": 12, "name": "b", "visibility": "public", "owner": "<masqué>"},
        {"object": "collection", "id": 13, "name": "c", "owner": "<masqué>"},
        {"object": "collection", "id": 14, "name": "d", "visibility": "PRIVATE-ish", "owner": "<masqué>"},
    ]}
    web.fake.inject("GET", "/v1/collections", (200, page))
    response = web.client.get(LIST_PATH, headers=web.headers["member_albert"])
    assert response.status_code == 200
    items = _listed(response.json())
    assert [item["id"] for item in items] == [11]
    assert "owner" not in items[0]


# ---------------------------------------------------------------------------
# Albert ON : suppression confirmée et journalisée
# ---------------------------------------------------------------------------
def test_delete_collection(web, monkeypatch):
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    private, _public = _populate(web.fake, 2, 0)
    cid = private[0]
    user = web.users["member_albert"]

    # Sans confirmation (absente, fausse ou vide) : 400, rien n'est supprimé ni envoyé.
    for query in ("", "?confirm=false", "?confirm=", "?confirm=0"):
        response = web.client.delete(ITEM_PATH.format(cid=cid) + query, headers=web.headers["member_albert"])
        assert response.status_code == 400, query
    assert cid in web.fake.collections
    assert web.fake.calls_to("DELETE") == []
    assert _audit_rows(web) == []

    # Identifiant invalide : 400 sans appel amont.
    for bad in ("abc", "0", "-3"):
        response = web.client.delete(ITEM_PATH.format(cid=bad) + "?confirm=true", headers=web.headers["member_albert"])
        assert response.status_code in (400, 404, 422), bad
    assert web.fake.calls_to("DELETE") == []

    # Confirmée : supprimée chez Albert et journalisée.
    response = web.client.delete(ITEM_PATH.format(cid=cid) + "?confirm=true", headers=web.headers["member_albert"])
    assert response.status_code in (200, 204), response.text[:300]
    assert cid not in web.fake.collections
    assert private[1] in web.fake.collections
    deletes = web.fake.calls_to("DELETE", f"/v1/collections/{cid}")
    assert len(deletes) == 1
    rows = _audit_rows(web, cid)
    assert len(rows) == 1
    row = rows[0]
    assert row.user_id == user.id
    assert row.resource_type == "albert_collection"
    assert bool(row.success) is True
    assert not _contains_fake_key(json.dumps(row.details or {}))

    # Collection inconnue chez Albert : erreur, jamais journalisée comme un succès.
    response = web.client.delete(ITEM_PATH.format(cid=999999) + "?confirm=true", headers=web.headers["member_albert"])
    assert response.status_code >= 400
    assert [r for r in _audit_rows(web, 999999) if bool(r.success)] == []


# ---------------------------------------------------------------------------
# Clé jamais dans une URL, un paramètre, une réponse ou un log
# ---------------------------------------------------------------------------
def test_key_not_in_urls_or_logs(web, monkeypatch, caplog):
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    caplog.set_level(logging.DEBUG)
    private, _ = _populate(web.fake, 3, 0)

    # Aucun paramètre de requête ne porte une clé.
    for route in app.routes:
        if getattr(route, "path", "").startswith(LIST_PATH):
            names = [param.name for param in route.dependant.query_params]
            assert not [n for n in names if "key" in n.lower() or "token" in n.lower()], (route.path, names)

    responses = [
        web.client.get(f"{LIST_PATH}?api_key={QUERY_ALBERT_KEY}", headers=web.headers["member_albert"]),
        web.client.delete(ITEM_PATH.format(cid=private[0]) + f"?confirm=true&api_key={QUERY_ALBERT_KEY}",
                          headers=web.headers["member_albert"]),
        # Clé refusée en amont (401) : la réponse d'erreur ne la recopie pas non plus.
        web.client.get(LIST_PATH, headers=web.headers["admin"]),
        web.client.delete(ITEM_PATH.format(cid=private[1]) + "?confirm=true", headers=web.headers["admin"]),
        # Une clé passée en paramètre ne remplace jamais l'identifiant manquant.
        web.client.get(f"{LIST_PATH}?api_key={QUERY_ALBERT_KEY}", headers=web.headers["member_nokeys"]),
    ]
    assert [r.status_code for r in responses[:2]] == [200, 200]
    assert responses[2].status_code >= 400 and responses[3].status_code >= 400
    assert responses[4].status_code == 403
    for response in responses:
        assert not _contains_fake_key(response.text)
        assert not _contains_fake_key(json.dumps(dict(response.headers)))
    assert web.fake.calls, "aucun appel amont"
    for call in web.fake.calls:
        assert not _contains_fake_key(call.url)
        assert not _contains_fake_key(json.dumps(call.params))
        assert "api_key" not in call.params
        assert not _contains_fake_key(call.content.decode("utf-8", "replace"))
        assert (call.headers.get("authorization", "").startswith("Bearer ")) is True
    assert web.builds.count(True) == 2 and web.builds.count(False) == 2
    # Journaux de l'application (la ligne du client de test, qui décrit la requête
    # du test elle-même avec son paramètre, est exclue).
    app_logs = [r.getMessage() for r in caplog.records if "http://testserver" not in r.getMessage()]
    assert app_logs, "aucun journal capturé"
    assert not [True for message in app_logs if _contains_fake_key(message)]
    for row in web.db.query(AuditLog).all():
        assert not _contains_fake_key(json.dumps(row.details or {}))
        assert not _contains_fake_key(row.error_message or "")
