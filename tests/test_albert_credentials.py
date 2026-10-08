"""Tests de l'identifiant Albert (``albert_api_key``) et des routes ``/api/albert/*``.

Couverture (Lot 1, tâches 11 à 14) :

* registres de ``app/core/credentials.py`` égaux et bijectifs, ``ALBERT_BASE_URL``
  jamais identifiant utilisateur ;
* environnement des sous-processus : clé Albert retirée pour les non-admins,
  ``RAGPY_DOTENV_DENY`` trié et séparé par des virgules, jamais posé pour un admin ;
* JSON de ``GET /users/me/credentials`` identique au golden G8 quand Albert est
  OFF, clé masquée quand il est ON ;
* ``GET /get_credentials`` : superposition base filtrée par le formulaire, clé
  Albert masquée côté serveur ; ``POST /save_credentials`` ignore le masque ;
* ``GET /api/albert/status`` et ``/api/albert/models`` : 404 quand OFF, 403 sans
  clé (même avec une clé serveur pour un non-admin), succès sur ``FakeAlbert``,
  401 amont → 400, 429 / 5xx → 502, aucune clé dans l'URL ni dans la réponse.

Durcissements (revue W1) :

* un seul analyseur de l'interrupteur ``ALBERT_ENABLED`` pour le web,
  ``AlbertConfig`` et ``EmbeddingConfig`` ;
* secrets serveur (``SERVER_SECRET_ENV_VARS``) retirés de l'env des
  sous-processus non-admins ;
* ``POST /save_credentials`` refuse les caractères de contrôle (400) ;
* routes Albert invisibles quand OFF pour toute méthode et avec une barre
  finale, absentes du schéma OpenAPI ; un seul essai amont, sans attente ;
* routes de listing Pinecone / Weaviate / Qdrant : ni clé ni URL lues dans la
  requête (fermeture SSRF).

Aucun appel réseau : la vraie fabrique ``app.routes.settings._albert_client_factory``
construit une sous-classe d'``AlbertClient`` branchée sur le ``httpx.MockTransport``
de ``tests/albert_fakes.FakeAlbert`` ; les bibliothèques de bases vectorielles
sont remplacées par des modules factices. Seules des clés factices sont
utilisées ; les assertions portent sur des noms, des booléens et des
empreintes, jamais sur l'affichage d'une clé.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
import types
from pathlib import Path
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
from app.models.user import User
from app.core.security import create_access_token
from app.core.credentials import (
    CREDENTIAL_ENV_MAPPING,
    CREDENTIAL_ERROR_MESSAGES,
    CREDENTIAL_KEYS,
    DOTENV_DENY_ENV_VAR,
    SERVER_CREDENTIAL_BASE_URLS,
    SERVER_SECRET_ENV_VARS,
    CredentialMissingError,
    albert_enabled,
    build_subprocess_env,
    encrypt_credentials,
    get_user_credentials,
    mask_credential,
    visible_credential_keys,
)
import app.routes.settings as settings_routes
import scripts.rad_albert.client as albert_client_module
from scripts.rad_albert.config import AlbertConfig
from scripts.rad_albert.errors import AlbertDisabledError
from scripts.rad_providers import EmbeddingConfig
from tests.albert_fakes import FAKE_ALBERT_KEY, FakeAlbert

# ---------------------------------------------------------------------------
# Constantes (littérales, jamais dérivées du code testé)
# ---------------------------------------------------------------------------
GOLDEN_ME_CREDENTIALS = (
    Path(__file__).parent / "fixtures" / "albert" / "golden_off" / "routes" / "g8_users_me_credentials.json"
)

# Les 15 variables historiques du formulaire admin, dans l'ordre historique.
HISTORICAL_FORM_ENV_KEYS = [
    "OPENAI_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_DEFAULT_MODEL",
    "MISTRAL_API_KEY", "MISTRAL_OCR_MODEL", "MISTRAL_API_BASE_URL",
    "PINECONE_API_KEY", "PINECONE_ENV",
    "WEAVIATE_API_KEY", "WEAVIATE_URL",
    "QDRANT_API_KEY", "QDRANT_URL",
    "ZOTERO_API_KEY", "ZOTERO_USER_ID", "ZOTERO_GROUP_ID",
]
MASK_PREFIX = "••••"
CONFIGURE_URL = "/settings/credentials"
ALBERT_ROUTES = ("/api/albert/status", "/api/albert/models")

# Clés factices (suffixes distincts pour savoir quelle valeur est masquée).
ADMIN_DB_ALBERT_KEY = "fake-albert-key-admin-db-0001"
MEMBER_ALBERT_KEY = "fake-albert-key-member-0001"
SERVER_ENV_ALBERT_KEY = "fake-albert-server-env-5555"
ENV_FILE_ALBERT_KEY = "fake-albert-env-file-7777"
NEW_ALBERT_KEY = "fake-albert-key-new-8888"
QUERY_ALBERT_KEY = "fake-albert-query-key-9999"
ALL_FAKE_ALBERT_KEYS = (
    FAKE_ALBERT_KEY, ADMIN_DB_ALBERT_KEY, MEMBER_ALBERT_KEY, SERVER_ENV_ALBERT_KEY,
    ENV_FILE_ALBERT_KEY, NEW_ALBERT_KEY, QUERY_ALBERT_KEY,
)

# Personas. admin / member_keys / member_nokeys reprennent à l'identique le
# harnais des goldens G8 (même base, mêmes empreintes de masques).
PERSONA_ROLES = {
    "admin": ["USER", "ADMIN"],
    "admin_nokeys": ["USER", "ADMIN"],
    "member_keys": ["USER"],
    "member_nokeys": ["USER"],
    "member_albert": ["USER"],
}
PERSONA_CREDENTIALS = {
    "admin": {"openai_api_key": "fake-openai-admin-db-0001", "albert_api_key": ADMIN_DB_ALBERT_KEY},
    "admin_nokeys": {},
    "member_keys": {
        "openai_api_key": "fake-openai-member-0001",
        "mistral_api_key": "fake-mistral-member-0001",
        "pinecone_api_key": "fake-pinecone-member-0001",
        "zotero_api_key": "fake-zotero-member-0001",
        "zotero_user_id": "fake-zotero-user-member-0001",
        "albert_api_key": MEMBER_ALBERT_KEY,
    },
    "member_nokeys": {},
    "member_albert": {"albert_api_key": FAKE_ALBERT_KEY},
}
GOLDEN_PERSONAS = ("admin", "member_keys", "member_nokeys")

# Champs identifiants de /v1/me qui ne doivent jamais être renvoyés.
IDENTIFYING_ME_FIELDS = ("email", "name", "id", "organization_id")

# Valeurs de l'interrupteur et lecture attendue (analyseur d'AlbertConfig).
SWITCH_VALUES_ON = ("1", " 1 ", "true", "TRUE", "yes", "on")
SWITCH_VALUES_OFF = ("", " ", "0", "false", "no", "off", "2")

# Secrets serveur factices (jamais utiles aux scripts du pipeline).
FAKE_SERVER_SECRETS = {
    "LOCAL_API_KEY": "fake-local-key-0001",
    "FLOWER_PASSWORD": "fake-flower-password-0001",
    "JWT_SECRET_KEY": "fake-jwt-secret-0001",
    "JWT_SECRET_KEY_PREVIOUS": "fake-jwt-previous-secret-0001",
    "RESEND_API_KEY": "fake-resend-key-0001",
}

# Méthodes HTTP sondées sur les routes Albert désactivées.
PROBED_METHODS = ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS")

# Routes de listing des bases vectorielles et identifiant exigé sans clé.
VECTOR_DB_LISTING_ROUTES = {
    "/api/pinecone/indexes": "pinecone_api_key",
    "/api/weaviate/collections": "weaviate_url",
    "/api/qdrant/collections": "qdrant_url",
}
SERVER_VECTOR_DB_ENV = {
    "PINECONE_API_KEY": "fake-pinecone-server-0001",
    "WEAVIATE_URL": "https://weaviate.fake.example.test",
    "WEAVIATE_API_KEY": "fake-weaviate-server-0001",
    "QDRANT_URL": "https://qdrant.fake.example.test",
    "QDRANT_API_KEY": "fake-qdrant-server-0001",
}
EVIL_HOST = "evil.example.test"
QUERY_VECTOR_DB_KEY = "fake-query-vector-db-key-0002"


# ---------------------------------------------------------------------------
# Aides
# ---------------------------------------------------------------------------
def _fp(value):
    """Empreinte ``fp:`` + ``sha256(value)[:12]`` (format des goldens G8)."""
    return "fp:" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def _fingerprint_masks(body):
    """Remplace chaque valeur ``masked`` non vide par son empreinte (comme G8)."""
    out = {}
    for key, value in body.items():
        value = dict(value)
        if value.get("masked"):
            value["masked"] = _fp(value["masked"])
        out[key] = value
    return out


def _user(roles, creds):
    """``User`` hors base, avec des identifiants chiffrés (pour ``build_subprocess_env``)."""
    return User(
        email="env.persona@example.test",
        hashed_password="x",
        roles=list(roles),
        is_active=True,
        is_verified=True,
        api_credentials=encrypt_credentials(creds) if creds else None,
    )


def _env_facts(env):
    """Réduit un env de sous-processus à des noms et des booléens.

    Les assertions ne portent que sur ce résumé : en cas d'échec, pytest
    n'affiche jamais l'env complet (qui recopie tout ``os.environ``).
    """
    deny = env.get(DOTENV_DENY_ENV_VAR)
    return SimpleNamespace(
        has_albert="ALBERT_API_KEY" in env,
        albert_is_fake_key=env.get("ALBERT_API_KEY") == FAKE_ALBERT_KEY,
        albert_is_server_key=env.get("ALBERT_API_KEY") == SERVER_ENV_ALBERT_KEY,
        has_deny=deny is not None,
        deny=deny,
        deny_names=deny.split(",") if deny else [],
    )


def _read_env_file(path):
    """Lit un fichier ``.env`` en dictionnaire (lignes ``NOM=valeur``)."""
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            name, value = line.split("=", 1)
            values[name.strip()] = value.strip()
    return values


def _json_keys(value):
    """Ensemble de toutes les clés de dictionnaire d'une structure JSON."""
    keys = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(key)
            keys |= _json_keys(item)
    elif isinstance(value, list):
        for item in value:
            keys |= _json_keys(item)
    return keys


def _contains_fake_key(text):
    """Vrai si une des clés Albert factices apparaît dans ``text``."""
    return any(key in text for key in ALL_FAKE_ALBERT_KEYS)


def _recording_client_class(fake, record, builds, sleeps):
    """Sous-classe d'``AlbertClient`` branchée sur ``fake``, construite par la vraie fabrique des routes.

    ``record`` reçoit un booléen par construction (la clé transmise est-elle
    ``FAKE_ALBERT_KEY`` ?) ; ``builds`` reçoit la politique, le limiteur et le
    délai demandés par la fabrique ; ``sleeps`` reçoit chaque attente demandée
    par le client (le sommeil injecté rend son temps virtuel).
    """
    base = albert_client_module.AlbertClient

    class _FakeTransportClient(base):
        """``AlbertClient`` réel sur le transport de ``FakeAlbert``, sommeil enregistré."""

        def __init__(self, cfg, api_key, **kwargs):
            """Note la construction, puis branche le transport factice et le sommeil enregistré."""
            record.append(api_key == FAKE_ALBERT_KEY)
            policy = kwargs.get("policy")
            builds.append(SimpleNamespace(
                single_attempt=bool(policy is not None and policy.single_attempt),
                use_limiter=kwargs.get("use_limiter", True),
                timeout=cfg.timeout_collections,
            ))
            kwargs.setdefault("transport", fake.transport)
            kwargs.setdefault("sleep", sleeps.append)
            super().__init__(cfg, api_key, **kwargs)

    return _FakeTransportClient


def _fake_vector_db_modules(captured):
    """Modules factices ``pinecone`` / ``weaviate`` / ``qdrant_client`` des routes de listing.

    Chaque client note dans ``captured`` l'hôte et la clé reçus, sans réseau,
    et renvoie une liste vide.
    """

    class Pinecone:
        """Client Pinecone factice."""

        def __init__(self, api_key=None, **_kwargs):
            """Note la clé reçue."""
            captured["pinecone"] = {"api_key": api_key}

        def list_indexes(self):
            """Aucun index."""
            return []

    class _Collections:
        """Collections Weaviate factices."""

        def list_all(self):
            """Aucune collection."""
            return []

    class _WeaviateClient:
        """Client Weaviate factice."""

        collections = _Collections()

        def close(self):
            """Rien à fermer."""

    def connect_to_weaviate_cloud(cluster_url=None, auth_credentials=None, **_kwargs):
        """Note l'hôte et l'authentification reçus."""
        captured["weaviate"] = {"url": cluster_url, "auth": auth_credentials}
        return _WeaviateClient()

    class Auth:
        """Fabrique d'authentification Weaviate factice."""

        @staticmethod
        def api_key(value):
            """Authentification par clé : ``("api_key", clé)``."""
            return ("api_key", value)

    class QdrantClient:
        """Client Qdrant factice."""

        def __init__(self, url=None, api_key=None, **_kwargs):
            """Note l'hôte et la clé reçus."""
            captured["qdrant"] = {"url": url, "api_key": api_key}

        def get_collections(self):
            """Aucune collection."""
            return SimpleNamespace(collections=[])

    modules = {name: types.ModuleType(name) for name in (
        "pinecone", "weaviate", "weaviate.classes", "weaviate.classes.init", "qdrant_client",
    )}
    modules["pinecone"].Pinecone = Pinecone
    modules["weaviate"].connect_to_weaviate_cloud = connect_to_weaviate_cloud
    modules["weaviate"].classes = modules["weaviate.classes"]
    modules["weaviate.classes"].init = modules["weaviate.classes.init"]
    modules["weaviate.classes.init"].Auth = Auth
    modules["qdrant_client"].QdrantClient = QdrantClient
    return modules


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _credential_env_hygiene(monkeypatch):
    """Retire les identifiants réels de l'env (seules des valeurs factices restent)."""
    for name in CREDENTIAL_ENV_MAPPING.values():
        monkeypatch.delenv(name, raising=False)
    for name in ("ALBERT_ENABLED", "ALBERT_BASE_URL", DOTENV_DENY_ENV_VAR):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def web(tmp_path, monkeypatch):
    """Application sous test : base SQLite temporaire, 5 personas, faux Albert, ``.env`` isolé."""
    home = tmp_path / "ragpy_home"
    home.mkdir()
    monkeypatch.setattr(settings_routes, "RAGPY_DIR", str(home))

    engine = create_engine(f"sqlite:///{tmp_path / 'albert_credentials.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    users = {}
    for persona, roles in PERSONA_ROLES.items():
        creds = PERSONA_CREDENTIALS[persona]
        user = User(
            email=f"{persona.replace('_', '.')}@example.test",
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
    factory_calls = []
    client_builds = []
    sleeps = []
    monkeypatch.setattr(
        albert_client_module,
        "AlbertClient",
        _recording_client_class(fake, factory_calls, client_builds, sleeps),
    )

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
            factory_calls=factory_calls,
            client_builds=client_builds,
            sleeps=sleeps,
            home=home,
        )
    finally:
        app.dependency_overrides.pop(get_db, None)
        db.close()
        engine.dispose()


def _write_settings_env(web, albert_value=ENV_FILE_ALBERT_KEY):
    """Écrit le ``.env`` du formulaire admin (valeurs factices, clé Albert incluse)."""
    lines = [
        "# settings file of the Albert credential tests",
        "OPENAI_API_KEY=fake-openai-env-file-0001",
        "MISTRAL_API_KEY=fake-mistral-env-file-0001",
        f"ALBERT_API_KEY={albert_value}",
        "OTHER_SETTING=fake-other-0001",
        "",
    ]
    (web.home / ".env").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Registres et environnement des sous-processus
# ---------------------------------------------------------------------------
def test_registries_equal_and_bijective():
    keys = list(CREDENTIAL_KEYS)
    assert len(keys) == len(set(keys))
    assert set(keys) == set(CREDENTIAL_ENV_MAPPING) == set(CREDENTIAL_ERROR_MESSAGES)
    env_names = list(CREDENTIAL_ENV_MAPPING.values())
    assert len(env_names) == len(set(env_names))
    assert CREDENTIAL_ENV_MAPPING["albert_api_key"] == "ALBERT_API_KEY"
    assert keys.index("albert_api_key") == keys.index("mistral_url") + 1
    assert CREDENTIAL_ERROR_MESSAGES["albert_api_key"] == (
        "Clé API Albert (DINUM) requise. Configurez-la dans Paramètres > Mes Identifiants."
    )


@pytest.mark.parametrize("value, expected", [(None, False), ("0", False), ("", False), (" 1 ", True), ("1", True)])
def test_visible_credential_keys_follow_albert_switch(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv("ALBERT_ENABLED", value)
    assert albert_enabled() is expected
    visible = visible_credential_keys()
    assert ("albert_api_key" in visible) is expected
    # Clés des serveurs du lot L9 : listées seulement si leur adresse est déclarée (aucune ici).
    assert visible == [k for k in CREDENTIAL_KEYS
                       if (expected or k != "albert_api_key") and k not in SERVER_CREDENTIAL_BASE_URLS]
    assert settings_routes._admin_form_env_keys() == (
        HISTORICAL_FORM_ENV_KEYS[:6] + ["ALBERT_API_KEY"] + HISTORICAL_FORM_ENV_KEYS[6:]
        if expected else HISTORICAL_FORM_ENV_KEYS
    )


@pytest.mark.parametrize("value", SWITCH_VALUES_ON + SWITCH_VALUES_OFF)
def test_albert_switch_single_parser(monkeypatch, value):
    monkeypatch.setenv("ALBERT_ENABLED", value)
    expected = value in SWITCH_VALUES_ON
    library = AlbertConfig.from_env({"ALBERT_ENABLED": value}).enabled
    process = AlbertConfig.from_env().enabled
    try:
        EmbeddingConfig.from_env({"ALBERT_ENABLED": value, "EMBEDDING_PROVIDER": "albert"})
        embedding = True
    except AlbertDisabledError:
        embedding = False
    assert [albert_enabled(), library, process, embedding] == [expected] * 4
    assert ("albert_api_key" in visible_credential_keys()) is expected
    assert ("ALBERT_API_KEY" in settings_routes._admin_form_env_keys()) is expected


def test_albert_switch_never_raises_on_invalid_base_url(monkeypatch):
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    monkeypatch.setenv("ALBERT_BASE_URL", f"https://{EVIL_HOST}/v1")
    with pytest.raises(ValueError):
        AlbertConfig.from_env()
    assert albert_enabled() is True
    monkeypatch.setenv("ALBERT_ENABLED", "0")
    assert albert_enabled() is False


def test_deny_format_sorted_comma():
    member = _user(["USER"], {"openai_api_key": "fake-openai-member-0001", "zotero_api_key": "fake-zotero-0001"})
    facts = _env_facts(build_subprocess_env(member))
    deny = facts.deny
    names = facts.deny_names
    assert facts.has_deny is True
    assert " " not in deny
    assert names == sorted(names)
    assert len(names) == len(set(names))
    assert all(name.endswith("_API_KEY") for name in names)
    expected = sorted(
        name for name in CREDENTIAL_ENV_MAPPING.values()
        if name.endswith("_API_KEY") and name not in ("OPENAI_API_KEY", "ZOTERO_API_KEY")
    )
    assert names == expected
    assert "ALBERT_API_KEY" in names

    every_key = {cred: "fake-" + cred for cred, env_name in CREDENTIAL_ENV_MAPPING.items() if env_name.endswith("_API_KEY")}
    facts = _env_facts(build_subprocess_env(_user(["USER"], every_key)))
    assert facts.deny == ""


@pytest.mark.parametrize("switch", ["0", "1"])
def test_nonadmin_env_strips_albert_key_and_sets_deny(monkeypatch, switch):
    monkeypatch.setenv("ALBERT_ENABLED", switch)
    monkeypatch.setenv("ALBERT_API_KEY", SERVER_ENV_ALBERT_KEY)

    facts = _env_facts(build_subprocess_env(_user(["USER"], {})))
    assert facts.has_albert is False
    assert "ALBERT_API_KEY" in facts.deny_names

    facts = _env_facts(build_subprocess_env(_user(["USER"], {"albert_api_key": FAKE_ALBERT_KEY})))
    assert facts.albert_is_fake_key is True
    assert "ALBERT_API_KEY" not in facts.deny_names

    with pytest.raises(CredentialMissingError) as excinfo:
        build_subprocess_env(_user(["USER"], {}), required_keys=["albert_api_key"])
    assert excinfo.value.credential_key == "albert_api_key"
    assert str(excinfo.value) == CREDENTIAL_ERROR_MESSAGES["albert_api_key"]


def test_admin_env_has_no_deny(monkeypatch):
    monkeypatch.setenv("ALBERT_API_KEY", SERVER_ENV_ALBERT_KEY)
    monkeypatch.setenv(DOTENV_DENY_ENV_VAR, "OPENAI_API_KEY")  # valeur périmée héritée du parent
    facts = _env_facts(build_subprocess_env(_user(["USER", "ADMIN"], {})))
    assert facts.has_deny is False
    assert facts.albert_is_server_key is True

    facts = _env_facts(build_subprocess_env(_user(["USER", "ADMIN"], {"albert_api_key": FAKE_ALBERT_KEY})))
    assert facts.has_deny is False
    assert facts.albert_is_fake_key is True


def test_nonadmin_env_strips_server_secrets(monkeypatch):
    assert tuple(SERVER_SECRET_ENV_VARS) == tuple(sorted(FAKE_SERVER_SECRETS))
    for name, value in FAKE_SERVER_SECRETS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("CELERY_BROKER_URL", "redis://broker.fake.example.test:6379/0")

    env = build_subprocess_env(_user(["USER"], {"openai_api_key": "fake-openai-member-0001"}))
    assert sorted(name for name in FAKE_SERVER_SECRETS if name in env) == []
    assert ("CELERY_BROKER_URL" in env) is True  # configuration non secrète conservée
    # Contrat de RAGPY_DOTENV_DENY inchangé : noms *_API_KEY du registre seulement.
    deny = env[DOTENV_DENY_ENV_VAR].split(",")
    assert sorted(set(deny) - set(CREDENTIAL_ENV_MAPPING.values())) == []

    admin_env = build_subprocess_env(_user(["USER", "ADMIN"], {}))
    kept = sorted(name for name, value in FAKE_SERVER_SECRETS.items() if admin_env.get(name) == value)
    assert kept == sorted(FAKE_SERVER_SECRETS)


# ---------------------------------------------------------------------------
# GET / PUT /users/me/credentials
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("switch", [None, "0"])
def test_me_credentials_off_equals_golden(web, monkeypatch, switch):
    if switch is not None:
        monkeypatch.setenv("ALBERT_ENABLED", switch)
    monkeypatch.setenv("ALBERT_API_KEY", FAKE_ALBERT_KEY)
    golden = {case["case"]: case for case in json.loads(GOLDEN_ME_CREDENTIALS.read_text(encoding="utf-8"))["cases"]}
    for persona in GOLDEN_PERSONAS:
        response = web.client.get("/users/me/credentials", headers=web.headers[persona])
        assert response.status_code == golden[persona]["status"] == 200
        body = response.json()
        assert list(body) == golden[persona]["key_order"]
        assert _fingerprint_masks(body) == golden[persona]["body"]
        assert "albert" not in response.text
    assert web.factory_calls == []


def test_me_credentials_on_masked(web, monkeypatch):
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    response = web.client.get("/users/me/credentials", headers=web.headers["member_keys"])
    assert response.status_code == 200
    body = response.json()
    keys = list(body)
    assert keys == [k for k in CREDENTIAL_KEYS if k not in SERVER_CREDENTIAL_BASE_URLS]
    assert keys.index("albert_api_key") == keys.index("mistral_url") + 1
    entry = body["albert_api_key"]
    assert entry["has_value"] is True
    assert entry["masked"] == mask_credential(MEMBER_ALBERT_KEY)
    assert entry["masked"].endswith(MEMBER_ALBERT_KEY[-4:])
    assert not _contains_fake_key(response.text)

    response = web.client.get("/users/me/credentials", headers=web.headers["member_nokeys"])
    assert response.json()["albert_api_key"] == {"has_value": False, "masked": ""}


def test_put_me_credentials_albert_ignored_when_off(web, monkeypatch):
    user = web.users["member_nokeys"]
    headers = web.headers["member_nokeys"]

    response = web.client.put("/users/me/credentials", json={"albert_api_key": NEW_ALBERT_KEY}, headers=headers)
    assert response.status_code == 200
    assert response.json() == {"message": "Aucune modification"}
    web.db.refresh(user)
    assert ("albert_api_key" in get_user_credentials(user)) is False

    monkeypatch.setenv("ALBERT_ENABLED", "1")
    response = web.client.put("/users/me/credentials", json={"albert_api_key": NEW_ALBERT_KEY}, headers=headers)
    assert response.status_code == 200
    assert response.json()["updated"] == ["albert_api_key"]
    web.db.refresh(user)
    assert (get_user_credentials(user).get("albert_api_key") == NEW_ALBERT_KEY) is True

    monkeypatch.setenv("ALBERT_ENABLED", "0")
    response = web.client.put(
        "/users/me/credentials", json={"albert_api_key": "", "openai_api_key": "fake-openai-put-0001"}, headers=headers
    )
    assert response.json()["updated"] == ["openai_api_key"]
    web.db.refresh(user)
    assert (get_user_credentials(user).get("albert_api_key") == NEW_ALBERT_KEY) is True


# ---------------------------------------------------------------------------
# GET /get_credentials et POST /save_credentials (formulaire admin)
# ---------------------------------------------------------------------------
def test_get_credentials_masks_albert_key(web, monkeypatch):
    monkeypatch.setenv("ALBERT_ENABLED", "1")

    response = web.client.get("/get_credentials", headers=web.headers["admin"])
    assert response.status_code == 200
    assert list(response.json()) == HISTORICAL_FORM_ENV_KEYS[:6] + ["ALBERT_API_KEY"] + HISTORICAL_FORM_ENV_KEYS[6:]
    assert response.json()["ALBERT_API_KEY"] == ""  # pas de .env : formulaire vide

    _write_settings_env(web)
    response = web.client.get("/get_credentials", headers=web.headers["admin"])
    body = response.json()
    assert list(body) == HISTORICAL_FORM_ENV_KEYS[:6] + ["ALBERT_API_KEY"] + HISTORICAL_FORM_ENV_KEYS[6:]
    assert body["ALBERT_API_KEY"] == MASK_PREFIX + ADMIN_DB_ALBERT_KEY[-4:]  # base prioritaire sur le .env
    assert not _contains_fake_key(response.text)

    response = web.client.get("/get_credentials", headers=web.headers["admin_nokeys"])
    assert response.json()["ALBERT_API_KEY"] == MASK_PREFIX + ENV_FILE_ALBERT_KEY[-4:]
    assert not _contains_fake_key(response.text)

    _write_settings_env(web, albert_value="")
    response = web.client.get("/get_credentials", headers=web.headers["admin_nokeys"])
    assert response.json()["ALBERT_API_KEY"] == ""

    assert settings_routes._mask_albert_key("abc") == MASK_PREFIX
    assert settings_routes._mask_albert_key(None) == ""


def test_save_credentials_ignores_mask_roundtrip(web, monkeypatch):
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    admin = web.users["admin"]
    headers = web.headers["admin"]
    env_file = web.home / ".env"
    _write_settings_env(web)

    form = web.client.get("/get_credentials", headers=headers).json()
    assert form["ALBERT_API_KEY"].startswith(MASK_PREFIX)
    response = web.client.post("/save_credentials", json=form, headers=headers)
    assert response.status_code == 200 and response.json()["status"] == "success"
    web.db.refresh(admin)
    assert (get_user_credentials(admin).get("albert_api_key") == ADMIN_DB_ALBERT_KEY) is True
    assert (_read_env_file(env_file).get("ALBERT_API_KEY") == ENV_FILE_ALBERT_KEY) is True
    assert (_read_env_file(env_file).get("OTHER_SETTING") == "fake-other-0001") is True

    web.client.post("/save_credentials", json={"ALBERT_API_KEY": NEW_ALBERT_KEY}, headers=headers)
    web.db.refresh(admin)
    assert (get_user_credentials(admin).get("albert_api_key") == NEW_ALBERT_KEY) is True
    assert (_read_env_file(env_file).get("ALBERT_API_KEY") == NEW_ALBERT_KEY) is True
    assert web.client.get("/get_credentials", headers=headers).json()["ALBERT_API_KEY"] == (
        MASK_PREFIX + NEW_ALBERT_KEY[-4:]
    )

    web.client.post("/save_credentials", json={"ALBERT_API_KEY": ""}, headers=headers)
    web.db.refresh(admin)
    assert ("albert_api_key" in get_user_credentials(admin)) is False
    assert _read_env_file(env_file).get("ALBERT_API_KEY") == ""
    assert web.client.get("/get_credentials", headers=headers).json()["ALBERT_API_KEY"] == ""


def test_save_credentials_rejects_control_characters(web, monkeypatch):
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    admin = web.users["admin"]
    headers = web.headers["admin"]
    env_file = web.home / ".env"
    _write_settings_env(web)
    env_before = env_file.read_bytes()
    creds_before = get_user_credentials(admin)

    cases = [
        ({"ALBERT_API_KEY": NEW_ALBERT_KEY + "\nALBERT_ALLOW_CUSTOM_HOST=1"}, ["ALBERT_API_KEY"]),
        ({"ALBERT_API_KEY": NEW_ALBERT_KEY + "\rALBERT_BASE_URL=https://" + EVIL_HOST}, ["ALBERT_API_KEY"]),
        ({"OPENAI_API_KEY": "fake-openai-0002 X=1", "ALBERT_API_KEY": NEW_ALBERT_KEY}, ["OPENAI_API_KEY"]),
        ({"MISTRAL_API_KEY": "fake-mistral\x00-0002", "ZOTERO_USER_ID": "12\x8534"}, ["MISTRAL_API_KEY", "ZOTERO_USER_ID"]),
        ({"QDRANT_URL": "https://qdrant.fake.example.test\x0bX"}, ["QDRANT_URL"]),
    ]
    for payload, invalid in cases:
        response = web.client.post("/save_credentials", json=payload, headers=headers)
        assert response.status_code == 400
        body = response.json()
        assert body["invalid_keys"] == invalid
        assert body["error"]
        assert not _contains_fake_key(response.text)
        assert (env_file.read_bytes() == env_before) is True
    web.db.refresh(admin)
    assert (get_user_credentials(admin) == creds_before) is True

    # Tabulation interne : acceptée comme avant Albert (elle ne coupe pas la ligne
    # NOM=valeur) ; la valeur est écrite telle quelle dans le .env et la base.
    tabbed = "https://qdrant.fake.example.test\tX"
    assert settings_routes._has_forbidden_chars(tabbed) is False
    response = web.client.post("/save_credentials", json={"QDRANT_URL": tabbed}, headers=headers)
    assert response.status_code == 200
    names = _read_env_file(env_file)
    assert names.get("QDRANT_URL") == tabbed
    assert "X" not in names
    web.db.refresh(admin)
    assert get_user_credentials(admin).get("qdrant_url") == tabbed

    # Blancs périphériques (retour à la ligne final compris) : retirés, comme avant.
    response = web.client.post("/save_credentials", json={"ALBERT_API_KEY": NEW_ALBERT_KEY + "\r\n"}, headers=headers)
    assert response.status_code == 200
    names = _read_env_file(env_file)
    assert (names.get("ALBERT_API_KEY") == NEW_ALBERT_KEY) is True
    assert "ALBERT_ALLOW_CUSTOM_HOST" not in names and "ALBERT_BASE_URL" not in names
    assert settings_routes._has_forbidden_chars("fake-plain-value-0001 with spaces") is False


@pytest.mark.parametrize("switch", [None, "0"])
def test_get_credentials_off_ignores_db_albert_key(web, monkeypatch, switch):
    if switch is not None:
        monkeypatch.setenv("ALBERT_ENABLED", switch)
    admin = web.users["admin"]
    headers = web.headers["admin"]
    env_file = web.home / ".env"

    response = web.client.get("/get_credentials", headers=headers)
    assert list(response.json()) == HISTORICAL_FORM_ENV_KEYS

    _write_settings_env(web)
    response = web.client.get("/get_credentials", headers=headers)
    assert response.status_code == 200
    assert list(response.json()) == HISTORICAL_FORM_ENV_KEYS
    assert "ALBERT" not in response.text
    assert not _contains_fake_key(response.text)

    # Albert OFF : ALBERT_API_KEY n'est pas un champ du formulaire ; ni le .env
    # ni la base ne sont modifiés.
    web.client.post("/save_credentials", json={"ALBERT_API_KEY": NEW_ALBERT_KEY}, headers=headers)
    web.db.refresh(admin)
    assert (get_user_credentials(admin).get("albert_api_key") == ADMIN_DB_ALBERT_KEY) is True
    assert (_read_env_file(env_file).get("ALBERT_API_KEY") == ENV_FILE_ALBERT_KEY) is True


def test_base_url_not_a_user_credential(web, monkeypatch):
    assert "ALBERT_BASE_URL" not in CREDENTIAL_ENV_MAPPING.values()
    assert not [key for key in CREDENTIAL_KEYS if "albert" in key and key != "albert_api_key"]
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    assert "ALBERT_BASE_URL" not in settings_routes._admin_form_env_keys()

    evil = "https://evil.example.com/v1"
    _write_settings_env(web)
    web.client.post("/save_credentials", json={"ALBERT_BASE_URL": evil}, headers=web.headers["admin"])
    assert ("ALBERT_BASE_URL" in _read_env_file(web.home / ".env")) is False
    assert "ALBERT_BASE_URL" not in web.client.get("/get_credentials", headers=web.headers["admin"]).json()

    user = web.users["member_nokeys"]
    response = web.client.put("/users/me/credentials", json={"albert_base_url": evil}, headers=web.headers["member_nokeys"])
    assert response.json() == {"message": "Aucune modification"}
    web.db.refresh(user)
    assert get_user_credentials(user) == {}
    assert "albert_base_url" not in web.client.get("/users/me/credentials", headers=web.headers["member_nokeys"]).json()


# ---------------------------------------------------------------------------
# GET /api/albert/status et /api/albert/models
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("switch", [None, "0"])
def test_status_404_when_off(web, monkeypatch, switch):
    if switch is not None:
        monkeypatch.setenv("ALBERT_ENABLED", switch)
    monkeypatch.setenv("ALBERT_API_KEY", FAKE_ALBERT_KEY)
    reference = web.client.get("/api/albert/does-not-exist")
    assert reference.status_code == 404
    for path in ALBERT_ROUTES:
        for headers in ({}, web.headers["admin"], web.headers["member_albert"]):
            response = web.client.get(path, headers=headers)
            assert response.status_code == 404
            assert response.json() == {"detail": "Not Found"}
            assert response.content == reference.content
            posted = web.client.post(path, headers=headers)
            assert posted.status_code == 404
            assert posted.content == reference.content
    assert web.factory_calls == []
    assert web.fake.calls == []


@pytest.mark.parametrize("switch", [None, "0", "off"])
def test_albert_routes_invisible_when_off_every_method(web, monkeypatch, switch):
    if switch is not None:
        monkeypatch.setenv("ALBERT_ENABLED", switch)
    for method in PROBED_METHODS:
        for suffix in ("", "/"):
            reference = web.client.request(method, "/api/albert/does-not-exist" + suffix, follow_redirects=False)
            assert reference.status_code == 404
            for path in ALBERT_ROUTES:
                for headers in ({}, web.headers["member_albert"]):
                    response = web.client.request(method, path + suffix, headers=headers, follow_redirects=False)
                    assert (response.status_code, response.content) == (reference.status_code, reference.content)
                    assert "allow" not in response.headers
                    assert "location" not in response.headers
    assert web.factory_calls == []
    assert web.fake.calls == []


def test_albert_routes_standard_when_on(web, monkeypatch):
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    for path in ALBERT_ROUTES:
        response = web.client.post(path, headers=web.headers["member_albert"])
        assert response.status_code == 405
        assert response.json() == {"detail": "Method Not Allowed"}
        assert response.headers["allow"] == "GET"
    assert web.factory_calls == []

    # Même interrupteur que les bibliothèques : « true » active aussi les routes.
    monkeypatch.setenv("ALBERT_ENABLED", "true")
    assert web.client.get("/api/albert/status", headers=web.headers["member_albert"]).status_code == 200
    assert web.factory_calls == [True]


def test_status_403_nonadmin_without_key_even_if_env_key(web, monkeypatch):
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    monkeypatch.setenv("ALBERT_API_KEY", SERVER_ENV_ALBERT_KEY)
    expected = {
        "error": CREDENTIAL_ERROR_MESSAGES["albert_api_key"],
        "credential_required": "albert_api_key",
        "configure_url": CONFIGURE_URL,
    }
    for path in ALBERT_ROUTES:
        response = web.client.get(path, headers=web.headers["member_nokeys"])
        assert response.status_code == 403
        assert response.json() == expected
    assert web.factory_calls == []
    assert web.fake.calls == []

    # Un admin sans clé personnelle retombe sur la clé serveur (.env).
    monkeypatch.setenv("ALBERT_API_KEY", FAKE_ALBERT_KEY)
    response = web.client.get("/api/albert/status", headers=web.headers["admin_nokeys"])
    assert response.status_code == 200
    assert web.factory_calls == [True]

    # Sans aucune clé, l'admin reçoit aussi le 403.
    monkeypatch.delenv("ALBERT_API_KEY")
    response = web.client.get("/api/albert/status", headers=web.headers["admin_nokeys"])
    assert response.status_code == 403
    assert response.json() == expected


def test_status_unauthenticated_401_when_on(web, monkeypatch):
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    for path in ALBERT_ROUTES:
        assert web.client.get(path).status_code == 401
    assert web.factory_calls == []


def test_status_success_fake(web, monkeypatch):
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    response = web.client.get("/api/albert/status", headers=web.headers["member_albert"])
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    account = body["account"]
    assert set(account) == {
        "expires", "expires_in_days", "expired", "expires_soon", "budget", "permissions", "limits",
    }
    assert account["expires"] is None and account["expired"] is False and account["expires_soon"] is False
    assert account["permissions"] == []
    assert len(account["limits"]) == len(web.fake.me["limits"])
    assert "warning" not in body
    assert not set(IDENTIFYING_ME_FIELDS) & _json_keys(body)
    assert web.factory_calls == [True]
    me_calls = web.fake.calls_to("GET", "/v1/me")
    assert len(me_calls) == 1
    assert me_calls[0].params == {}
    assert (me_calls[0].headers.get("authorization") == f"Bearer {FAKE_ALBERT_KEY}") is True

    response = web.client.get("/api/albert/models", headers=web.headers["member_albert"])
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["count"] == len(body["models"]) == len(web.fake.models)
    assert {m["id"] for m in body["models"]} == {m["id"] for m in web.fake.models}
    assert all(set(m) == {"id", "type", "aliases", "max_context_length"} for m in body["models"])
    by_id = {m["id"]: m for m in body["models"]}
    assert by_id["bge-m3"]["type"] == "text-embeddings-inference"
    assert web.factory_calls == [True, True]


def test_status_warns_before_expiry(web, monkeypatch):
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    web.fake.me["expires"] = int(time.time()) + 10 * 86400 + 3600
    body = web.client.get("/api/albert/status", headers=web.headers["member_albert"]).json()
    assert body["account"]["expires_soon"] is True
    assert body["account"]["expires_in_days"] == 10
    assert "albert.api@numerique.gouv.fr" in body["warning"]

    web.fake.me["expires"] = int(time.time()) + 200 * 86400
    body = web.client.get("/api/albert/status", headers=web.headers["member_albert"]).json()
    assert body["account"]["expires_soon"] is False
    assert "warning" not in body

    summary = settings_routes._albert_account_summary({"expires": 1000, "email": "x@example.test"}, now=2000)
    assert summary["expired"] is True and summary["expires_soon"] is False
    assert "email" not in summary


def test_status_upstream_401_to_400(web, monkeypatch, caplog):
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    caplog.set_level(logging.DEBUG)
    # La clé de l'admin (base) n'est pas celle qu'attend le faux : 401 amont.
    for path in ALBERT_ROUTES:
        response = web.client.get(path, headers=web.headers["admin"])
        assert response.status_code == 400
        body = response.json()
        assert body["credential_invalid"] == "albert_api_key"
        assert body["reason"] == "invalid_key"
        assert body["configure_url"] == CONFIGURE_URL
        assert body["error"]
        assert not _contains_fake_key(response.text)
    assert web.factory_calls == [False, False]
    assert [c.status for c in web.fake.calls] == [401, 401]

    web.fake.inject("GET", "/v1/me", "account_expired")
    response = web.client.get("/api/albert/status", headers=web.headers["member_albert"])
    assert response.status_code == 400
    assert response.json()["reason"] == "account_expired"
    assert response.json()["credential_invalid"] == "albert_api_key"
    assert not _contains_fake_key(caplog.text)


@pytest.mark.parametrize("entry, retry_after", [(500, None), (502, None), (429, 3600)])
def test_status_upstream_429_and_5xx_to_502(web, monkeypatch, entry, retry_after):
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    web.fake.inject("GET", "/v1/me", *([entry] * 10), retry_after=retry_after)
    response = web.client.get("/api/albert/status", headers=web.headers["member_albert"])
    assert response.status_code == 502
    body = response.json()
    assert body["error"]
    assert body["upstream_status"] == entry
    assert not _contains_fake_key(response.text)


def test_status_no_query_api_key_and_no_key_in_response(web, monkeypatch):
    for route in app.routes:
        if getattr(route, "path", None) in ALBERT_ROUTES:
            assert [param.name for param in route.dependant.query_params] == []
            assert [param.name for param in route.dependant.path_params] == []

    monkeypatch.setenv("ALBERT_ENABLED", "1")
    for path in ALBERT_ROUTES:
        response = web.client.get(f"{path}?api_key={QUERY_ALBERT_KEY}", headers=web.headers["member_nokeys"])
        assert response.status_code == 403
        assert response.json()["credential_required"] == "albert_api_key"
    assert web.factory_calls == []

    for path in ALBERT_ROUTES:
        response = web.client.get(f"{path}?api_key={QUERY_ALBERT_KEY}", headers=web.headers["member_albert"])
        assert response.status_code == 200
        assert not _contains_fake_key(response.text)
    assert web.factory_calls == [True, True]
    for call in web.fake.calls:
        assert not _contains_fake_key(call.url)
        assert call.params == {}


@pytest.mark.parametrize("entry, retry_after", [(503, None), (500, None), (429, 30), ("gateway_timeout", None)])
def test_albert_routes_single_attempt_no_sleep(web, monkeypatch, entry, retry_after):
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    monkeypatch.setenv("ALBERT_TIMEOUT_COLLECTIONS", "300")
    for path, upstream in (("/api/albert/status", "/v1/me"), ("/api/albert/models", "/v1/models")):
        web.fake.inject("GET", upstream, *([entry] * 5), retry_after=retry_after)
        response = web.client.get(path, headers=web.headers["member_albert"])
        assert response.status_code == 502
        assert not _contains_fake_key(response.text)
        assert len(web.fake.calls_to("GET", upstream)) == 1
    assert web.fake.pending_injections() == 8
    assert web.sleeps == []
    assert [build.single_attempt for build in web.client_builds] == [True, True]
    assert [build.use_limiter for build in web.client_builds] == [False, False]
    assert [build.timeout for build in web.client_builds] == [settings_routes._ALBERT_WEB_TIMEOUT] * 2
    assert settings_routes._ALBERT_WEB_TIMEOUT <= 15.0


@pytest.mark.parametrize("switch", [None, "0", "1"])
def test_openapi_never_mentions_albert(monkeypatch, switch):
    if switch is not None:
        monkeypatch.setenv("ALBERT_ENABLED", switch)
    monkeypatch.setattr(app, "openapi_schema", None)
    schema = app.openapi()
    assert "albert" not in json.dumps(schema).lower()
    assert not [path for path in schema["paths"] if path.startswith("/api/albert")]
    for name in ("UserCredentialsResponse", "UserCredentialsUpdate"):
        properties = schema["components"]["schemas"][name]["properties"]
        assert "albert_api_key" not in properties
        assert "mistral_url" in properties


# ---------------------------------------------------------------------------
# Routes de listing des bases vectorielles : ni clé ni URL lues dans la requête
# ---------------------------------------------------------------------------
def test_vector_db_listing_ignores_query_url_and_key(web, monkeypatch):
    for route in app.routes:
        if getattr(route, "path", None) in VECTOR_DB_LISTING_ROUTES:
            assert [param.name for param in route.dependant.query_params] == []

    captured = {}
    for name, module in _fake_vector_db_modules(captured).items():
        monkeypatch.setitem(sys.modules, name, module)
    for name, value in SERVER_VECTOR_DB_ENV.items():
        monkeypatch.setenv(name, value)
    query = f"?url=https://{EVIL_HOST}&api_key={QUERY_VECTOR_DB_KEY}"

    # Admin sans clé personnelle : identifiants serveur (.env) seulement, jamais l'URL
    # ni la clé de la requête.
    for path in VECTOR_DB_LISTING_ROUTES:
        response = web.client.get(path + query, headers=web.headers["admin_nokeys"])
        assert response.status_code == 200
        assert response.json()["success"] is True
    assert sorted(captured) == ["pinecone", "qdrant", "weaviate"]
    assert (captured["pinecone"]["api_key"] == SERVER_VECTOR_DB_ENV["PINECONE_API_KEY"]) is True
    assert (captured["weaviate"]["url"] == SERVER_VECTOR_DB_ENV["WEAVIATE_URL"]) is True
    assert (captured["weaviate"]["auth"] == ("api_key", SERVER_VECTOR_DB_ENV["WEAVIATE_API_KEY"])) is True
    assert (captured["qdrant"]["url"] == SERVER_VECTOR_DB_ENV["QDRANT_URL"]) is True
    assert (captured["qdrant"]["api_key"] == SERVER_VECTOR_DB_ENV["QDRANT_API_KEY"]) is True
    received = json.dumps(captured)
    assert (EVIL_HOST in received) is False
    assert (QUERY_VECTOR_DB_KEY in received) is False

    # Non-admin sans identifiants : 403, même avec une URL et une clé dans la requête.
    captured.clear()
    for path, credential in VECTOR_DB_LISTING_ROUTES.items():
        response = web.client.get(path + query, headers=web.headers["member_nokeys"])
        assert response.status_code == 403
        assert response.json()["credential_required"] == credential
    assert captured == {}


# ---------------------------------------------------------------------------
# Gabarits : blocs Albert en ligne (OFF identique à G9, ON présent, clé jamais rendue)
# ---------------------------------------------------------------------------
def test_pages_render_albert_blocks_only_when_on(web, monkeypatch):
    markers = {
        # Clé Albert et test de connexion : page Paramètres (lot L8 du sprint « configuration
        # unifiée ») et profil ; la page du pipeline garde ses blocs Albert (collections).
        "/pipeline": ('id="albertParams"', "/api/albert/collections"),
        "/profile": ('id="cred_albert_api_key"', "'albert_api_key',"),
    }
    for switch, expected in ((None, False), ("0", False), ("1", True)):
        if switch is None:
            monkeypatch.delenv("ALBERT_ENABLED", raising=False)
        else:
            monkeypatch.setenv("ALBERT_ENABLED", switch)
        for persona in ("admin", "member_keys"):
            for path, needles in markers.items():
                response = web.client.get(path, headers=web.headers[persona])
                assert response.status_code == 200
                for needle in needles:
                    assert (needle in response.text) is expected, (path, needle, switch)
                assert not _contains_fake_key(response.text)
