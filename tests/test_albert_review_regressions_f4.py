"""Régressions de la revue du lot 9, correcteur F4 : routes d'ingestion (``app/routes/ingestion.py``).

Deux constats confirmés, antérieurs à la branche (le fichier était identique
à ``main``), qui contournaient la décision du 2026-09-27 sur l'appartenance
des sessions et le confinement sous ``uploads/`` :

* ``/upload_stage_file/{stage}`` (bouton « Upload existing » de la page du
  pipeline) n'exigeait ni authentification ni appartenance : un client
  anonyme remplaçait un fichier d'étape d'une session enregistrée d'autrui,
  ou écrivait hors de ``uploads/`` (``path=../x``). La route exige désormais
  un utilisateur authentifié (en-tête ``Authorization`` ou cookie
  ``access_token``, celui qu'envoie la page) et applique, avant toute
  écriture, le même contrôle que les routes de ``processing.py`` : 400 hors
  de ``uploads/``, 403 pour une session enregistrée d'un projet inaccessible,
  comportement historique pour un dossier sans ``PipelineSession``.
* Zip slip de ``/upload_zip`` : ``extract_zip_with_encoding_fix`` joignait
  les noms bruts des membres au dossier de destination, sans vérifier qu'ils
  y restaient (``../../x``, chemin absolu). Une archive dont un membre sort
  du dossier de destination est désormais refusée en entier, avant toute
  écriture (400 par la branche « archive invalide » existante de la route).

Décision du 2026-09-27 étendue à toutes les routes qui touchent aux sessions
et aux projets (correcteur G3) : ``/upload_zip`` et ``/upload_csv``
acceptaient un client anonyme et rattachaient la nouvelle session au
``project_id`` fourni par le client, sans contrôle : n'importe qui créait une
``PipelineSession`` dans le projet d'autrui et remplaçait son
``project.session_folder`` (dossier actif du projet). Les deux routes
exigent désormais un utilisateur authentifié (en-tête ``Authorization`` ou
cookie ``access_token``, celui qu'envoie la page) : 401 sinon. Quand un
``project_id`` est fourni, le contrôle des routes ``/api/pipeline/projects/
{id}/upload_*`` (``verify_project_access`` puis droit d'édition) s'applique
avant toute écriture : 404 projet inconnu, 403 projet inaccessible ou
lecture seule (``viewer``), sans fichier écrit, sans ligne créée et sans
``session_folder`` remplacé ; propriétaire, collaborateur et
administrateur passent. Sans ``project_id`` : comportement historique pour
l'utilisateur authentifié.

Harnais : ``golden_app`` de ``tests/test_albert_off_golden_routes.py``
(personas ``admin``, ``member_keys``, ``member_nokeys`` ; ``gsess-full`` est
la session enregistrée du projet de ``member_nokeys``), avec
``UPLOAD_DIR`` du module d'ingestion redirigé vers le dossier temporaire
(la base de test est servie par ``dependency_overrides[get_db]``).
Identifiants factices seulement ; toutes les écritures restent sous
``tmp_path``.

Run: pytest tests/test_albert_review_regressions_f4.py -q
"""

from __future__ import annotations

import hashlib
import io
import os
import zipfile

import pytest

import app.routes.ingestion as ingestion_routes
from app.core.security import create_access_token
from app.models.pipeline_session import PipelineSession
from app.models.project import Project, ProjectMember, ProjectRole
# Fixtures réutilisées (l'hygiène d'env des goldens est autouse dans ce module).
from tests.test_albert_off_golden_routes import _golden_env_hygiene, golden_app  # noqa: F401

# ======================================================================
# Constantes littérales
# ======================================================================
DENIED_MESSAGE = "Accès non autorisé à cette session : elle appartient à un projet auquel vous n'avez pas accès."
INVALID_PATH_MESSAGE = "Chemin de session invalide : le dossier doit se trouver sous uploads/."
REGISTERED = "gsess-full"          # session enregistrée du projet de member_nokeys
FOREIGN_PERSONA = "member_keys"    # ni propriétaire ni membre de ce projet
OWNER_PERSONA = "member_nokeys"
UNREGISTERED_OWN = "gsess-fresh-member-keys"      # dossier sans PipelineSession
UNREGISTERED_OTHER = "gsess-fresh-member-nokeys"  # idem, créé pour une autre persona
OUTSIDE_DIR = "outside_dir"        # frère de uploads/ sous tmp_path

# stage -> (nom du fichier écrit, nom du fichier envoyé, contenu envoyé)
STAGES = {
    "initial": ("output.csv", "mine.csv", b"title,texteocr\nAttacker,injected text\n"),
    "dense": ("output_chunks.json", "mine.json", b'[{"id": "x", "text": "injected text"}]'),
    "sparse": ("output_chunks_with_embeddings.json", "mine.json", b'[{"id": "x", "text": "injected text"}]'),
}
STAGE_IDS = list(STAGES)


# ======================================================================
# Aides
# ======================================================================
@pytest.fixture
def env(golden_app, monkeypatch):
    """``golden_app`` avec le module d'ingestion redirigé vers le dossier et la base temporaires.

    ``UPLOAD_DIR`` du module d'ingestion n'est pas redirigé par le harnais des
    goldens : il pointe ici vers ``tmp_path``. ``get_db`` du module est aussi
    redirigé (garde pour un appel direct ``next(get_db())``, hors
    ``dependency_overrides``).
    """
    def _yield_db():
        """Rend la session de test partagée (remplace ``get_db()``)."""
        yield golden_app.db

    monkeypatch.setattr(ingestion_routes, "UPLOAD_DIR", str(golden_app.uploads))
    monkeypatch.setattr(ingestion_routes, "get_db", _yield_db)
    # Identifiants lus une fois : un ancien ``db.close()`` de route détacherait les objets.
    golden_app.project_id = golden_app.project.id
    golden_app.user_ids = {persona: user.id for persona, user in golden_app.users.items()}
    golden_app.client.cookies.clear()
    (golden_app.uploads.parent / OUTSIDE_DIR).mkdir()
    return golden_app


def _snapshot(folder):
    """Noms, tailles et empreintes sha256 des fichiers d'un dossier (récursif)."""
    out = {}
    for root, _dirs, files in os.walk(folder):
        for name in files:
            full = os.path.join(root, name)
            with open(full, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
            out[os.path.relpath(full, folder)] = (os.path.getsize(full), digest)
    return out


def _upload_stage(env, stage, path, headers=None):
    """POST ``/upload_stage_file/{stage}`` du fichier d'étape de ``STAGES``."""
    _target, sent_name, content = STAGES[stage]
    return env.client.post(
        f"/upload_stage_file/{stage}",
        data={"path": path},
        files={"file": (sent_name, content)},
        headers=headers or {},
    )


def _zip_bytes(members):
    """Archive ZIP en mémoire : ``members`` est une liste ``(nom, contenu)`` (noms bruts, non assainis)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members:
            archive.writestr(name, content)
    return buffer.getvalue()


def _session_rows(env):
    """Dossiers de toutes les lignes ``PipelineSession`` (tri stable)."""
    env.db.expire_all()
    return sorted((row.project_id, row.session_folder) for row in env.db.query(PipelineSession).all())


# ======================================================================
# /upload_stage_file/{stage} : authentification et appartenance
# ======================================================================
@pytest.mark.parametrize("stage", STAGE_IDS)
def test_upload_stage_file_requires_authentication(env, stage):
    """Sans en-tête ni cookie : 401, et la session enregistrée n'est pas modifiée."""
    before = _snapshot(env.uploads / REGISTERED)
    resp = _upload_stage(env, stage, REGISTERED)
    assert resp.status_code == 401
    assert _snapshot(env.uploads / REGISTERED) == before


@pytest.mark.parametrize("folder", [REGISTERED, REGISTERED + "/", "./" + REGISTERED])
@pytest.mark.parametrize("stage", STAGE_IDS)
def test_upload_stage_file_refuses_foreign_registered_session(env, stage, folder):
    """Non-membre sur la session enregistrée d'un autre projet (toute graphie) : 403, rien n'est écrit."""
    before = _snapshot(env.uploads)
    resp = _upload_stage(env, stage, folder, env.headers[FOREIGN_PERSONA])
    assert resp.status_code == 403
    assert resp.json() == {"error": DENIED_MESSAGE}
    assert _snapshot(env.uploads) == before


@pytest.mark.parametrize("persona", [OWNER_PERSONA, "admin"])
@pytest.mark.parametrize("stage", STAGE_IDS)
def test_upload_stage_file_owner_and_admin_allowed(env, stage, persona):
    """Propriétaire du projet et administrateur : 200 et le fichier d'étape est remplacé."""
    target, _sent, content = STAGES[stage]
    resp = _upload_stage(env, stage, REGISTERED, env.headers[persona])
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    assert resp.json()["filename"] == target
    assert (env.uploads / REGISTERED / target).read_bytes() == content


@pytest.mark.parametrize("folder", [UNREGISTERED_OWN, UNREGISTERED_OTHER])
def test_upload_stage_file_unregistered_folder_keeps_behaviour(env, folder):
    """Dossier sans ``PipelineSession`` (ancien flux) : comportement historique pour un utilisateur authentifié."""
    target, _sent, content = STAGES["initial"]
    resp = _upload_stage(env, "initial", folder, env.headers[FOREIGN_PERSONA])
    assert resp.status_code == 200
    assert (env.uploads / folder / target).read_bytes() == content


@pytest.mark.parametrize("persona", ["admin", FOREIGN_PERSONA])
@pytest.mark.parametrize("kind", ["parent", "absolute", "upload_root", "nul_byte"])
def test_upload_stage_file_refuses_paths_outside_uploads(env, kind, persona):
    """Dossier qui ne se résout pas strictement sous ``uploads/`` : 400 pour tous, aucun fichier créé."""
    outside = env.uploads.parent / OUTSIDE_DIR
    path = {
        "parent": "../" + OUTSIDE_DIR,
        "absolute": str(outside),
        "upload_root": ".",
        "nul_byte": REGISTERED + "\x00",
    }[kind]
    before_uploads = _snapshot(env.uploads)
    resp = _upload_stage(env, "initial", path, env.headers[persona])
    assert resp.status_code == 400
    assert resp.json() == {"error": INVALID_PATH_MESSAGE}
    assert os.listdir(outside) == []
    assert _snapshot(env.uploads) == before_uploads


def test_upload_stage_file_accepts_the_access_token_cookie(env):
    """La page appelle la route sans en-tête ``Authorization`` : le cookie ``access_token`` suffit.

    Garde du repli par cookie (le rendu de ``index.html`` reste gelé par G9) :
    le propriétaire passe, le non-membre reçoit le même 403 qu'avec l'en-tête.
    """
    owner = env.users[OWNER_PERSONA]
    env.client.cookies.set("access_token", create_access_token(subject=str(owner.id)))
    try:
        resp = _upload_stage(env, "sparse", REGISTERED)
        assert resp.status_code == 200
        foreign = env.users[FOREIGN_PERSONA]
        env.client.cookies.set("access_token", create_access_token(subject=str(foreign.id)))
        before = _snapshot(env.uploads / REGISTERED)
        denied = _upload_stage(env, "sparse", REGISTERED)
        assert denied.status_code == 403
        assert denied.json() == {"error": DENIED_MESSAGE}
        assert _snapshot(env.uploads / REGISTERED) == before
    finally:
        env.client.cookies.clear()


def test_index_page_stage_upload_relies_on_the_cookie():
    """Le gabarit de la page appelle ``/upload_stage_file`` en POST same-origin (cookie envoyé par défaut)."""
    template = os.path.join(os.path.dirname(ingestion_routes.__file__), "..", "templates", "index.html")
    with open(template, encoding="utf-8") as handle:
        source = handle.read()
    assert "fetch(`/upload_stage_file/${stage}`, { method: 'POST', body: formData });" in source
    assert "credentials: 'omit'" not in source


# ======================================================================
# /upload_zip : zip slip (membres hors du dossier de destination)
# ======================================================================
@pytest.mark.parametrize("bad_member", ["../escaped.txt", "../../escaped.txt", "sub/../../escaped.txt", "ABSOLUTE"])
def test_extract_zip_rejects_members_outside_destination(tmp_path, bad_member):
    """Un membre qui sortirait du dossier de destination fait refuser toute l'archive, avant toute écriture."""
    dst = tmp_path / "uploads" / "dst"
    dst.mkdir(parents=True)
    if bad_member == "ABSOLUTE":
        bad_member = str(tmp_path / "abs_escaped.txt")
    archive = tmp_path / "archive.zip"
    archive.write_bytes(_zip_bytes([("biblio/biblio.json", b"[]"), (bad_member, b"escaped")]))
    with pytest.raises(zipfile.BadZipFile):
        ingestion_routes.extract_zip_with_encoding_fix(str(archive), str(dst))
    assert list(dst.iterdir()) == []
    assert not (tmp_path / "uploads" / "escaped.txt").exists()
    assert not (tmp_path / "escaped.txt").exists()
    assert not (tmp_path / "abs_escaped.txt").exists()


def test_extract_zip_keeps_encoding_fix_and_directories(tmp_path):
    """Archive saine : répertoires, sous-dossiers et correction NFD -> NFC inchangés."""
    dst = tmp_path / "dst"
    dst.mkdir()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(_zip_bytes([
        ("biblio/", b""),
        ("biblio/biblio.json", b"[]"),
        ("biblio/files/1/étude.pdf", b"%PDF-1.4"),
        ("./biblio/notes.txt", b"n"),
    ]))
    corrected = ingestion_routes.extract_zip_with_encoding_fix(str(archive), str(dst))
    assert corrected == 1
    assert (dst / "biblio" / "biblio.json").read_bytes() == b"[]"
    assert (dst / "biblio" / "files" / "1" / "étude.pdf").read_bytes() == b"%PDF-1.4"
    assert (dst / "biblio" / "notes.txt").read_bytes() == b"n"


@pytest.mark.parametrize("persona", [OWNER_PERSONA, "admin"])
def test_upload_zip_rejects_traversal_members(env, persona):
    """Zip slip par la route : 400, aucun fichier hors du dossier d'extraction, aucune session créée.

    Membres visés : un marqueur à côté de ``uploads/`` et le fichier dense de
    la session enregistrée du projet ; ``project_id`` est celui de ce projet
    et l'appelant y a accès (propriétaire, administrateur), donc la requête
    atteint l'extraction.
    """
    marker = env.uploads.parent / "escaped_marker.txt"
    before_registered = _snapshot(env.uploads / REGISTERED)
    before_rows = _session_rows(env)
    before_entries = set(os.listdir(env.uploads))
    payload = _zip_bytes([
        ("biblio/biblio.json", b"[]"),
        ("../../escaped_marker.txt", b"escaped"),
        ("../" + REGISTERED + "/output_chunks_with_embeddings.json", b'[{"id": "x", "text": "injected"}]'),
    ])
    resp = env.client.post(
        "/upload_zip",
        data={"project_id": str(env.project_id)},
        files={"file": ("biblio.zip", payload, "application/zip")},
        headers=env.headers[persona],
    )
    assert resp.status_code == 400
    assert resp.json() == {"error": "Uploaded file is not a valid ZIP archive."}
    assert not marker.exists()
    assert _snapshot(env.uploads / REGISTERED) == before_registered
    assert _session_rows(env) == before_rows
    env.db.expire_all()
    assert env.db.query(Project).filter(Project.id == env.project_id).one().session_folder == REGISTERED
    new_dirs = [name for name in set(os.listdir(env.uploads)) - before_entries if (env.uploads / name).is_dir()]
    assert new_dirs == []


def test_upload_zip_regular_archive_unchanged(env):
    """Archive saine par la route : extraction et arborescence historiques (dossier racine unique)."""
    payload = _zip_bytes([("biblio/biblio.json", b"[]"), ("biblio/files/1/étude.pdf", b"%PDF-1.4")])
    resp = env.client.post(
        "/upload_zip", files={"file": ("biblio.zip", payload, "application/zip")}, headers=env.headers[FOREIGN_PERSONA]
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["path"].endswith("_biblio" + os.sep + "biblio")
    assert sorted(body["tree"]) == ["biblio.json", "files/", "files/1/", "files/1/étude.pdf"]
    assert body["session_id"] is None


# ======================================================================
# /upload_zip et /upload_csv : authentification et accès au projet (G3)
# ======================================================================
PROJECT_DENIED_MESSAGE = "Accès non autorisé à ce projet"
PROJECT_READ_ONLY_MESSAGE = "Vous n'avez pas les droits pour ajouter des fichiers à ce projet"
PROJECT_NOT_FOUND_MESSAGE = "Projet non trouvé"
UNKNOWN_PROJECT_ID = 987654
UPLOAD_ROUTES = ["/upload_zip", "/upload_csv"]
# route -> (nom du fichier envoyé, contenu, type MIME, source_type attendu)
UPLOAD_PAYLOADS = {
    "/upload_zip": ("biblio.zip", _zip_bytes([("biblio/biblio.json", b"[]")]), "application/zip", "zip"),
    "/upload_csv": ("corpus.csv", b"title,text\nDoc A,some text content\n", "text/csv", "csv"),
}


def _upload(env, route, project_id=None, headers=None):
    """POST ``route`` (``/upload_zip`` ou ``/upload_csv``) du fichier de ``UPLOAD_PAYLOADS``, avec ``project_id`` éventuel."""
    name, content, mime, _source = UPLOAD_PAYLOADS[route]
    data = {} if project_id is None else {"project_id": str(project_id)}
    return env.client.post(route, data=data, files={"file": (name, content, mime)}, headers=headers or {})


def _uploads_state(env):
    """État complet de ``uploads/`` et de la base : fichiers (empreintes), dossiers, lignes de session, dossier actif du projet."""
    dirs = sorted(
        os.path.relpath(os.path.join(root, name), env.uploads)
        for root, names, _files in os.walk(env.uploads) for name in names
    )
    env.db.expire_all()
    folder = env.db.query(Project).filter(Project.id == env.project_id).one().session_folder
    return _snapshot(env.uploads), dirs, _session_rows(env), folder


def _add_project_member(env, persona, role):
    """Ajoute ``persona`` au projet de ``gsess-full`` avec le rôle ``role`` (valeur de ``ProjectRole``)."""
    env.db.add(ProjectMember(project_id=env.project_id, user_id=env.user_ids[persona], role=role))
    env.db.commit()


def _assert_session_created(env, route, body):
    """La réponse 200 a créé la ``PipelineSession`` du projet et fait de son dossier le dossier actif du projet."""
    _name, _content, _mime, source = UPLOAD_PAYLOADS[route]
    assert body["project_id"] == env.project_id
    assert isinstance(body["session_id"], int)
    env.db.expire_all()
    row = env.db.query(PipelineSession).filter(PipelineSession.id == body["session_id"]).one()
    assert (row.project_id, row.session_folder, row.source_type) == (env.project_id, body["path"], source)
    assert env.db.query(Project).filter(Project.id == env.project_id).one().session_folder == body["path"]
    assert (env.uploads / body["path"]).is_dir()


@pytest.mark.parametrize("with_project", [False, True])
@pytest.mark.parametrize("route", UPLOAD_ROUTES)
def test_upload_routes_require_authentication(env, route, with_project):
    """Sans en-tête ni cookie : 401, rien n'est écrit, aucune session créée, dossier actif du projet inchangé."""
    before = _uploads_state(env)
    resp = _upload(env, route, env.project_id if with_project else None)
    assert resp.status_code == 401
    assert _uploads_state(env) == before
    assert before[3] == REGISTERED


@pytest.mark.parametrize("route", UPLOAD_ROUTES)
def test_upload_routes_refuse_foreign_project(env, route):
    """Non-membre avec le ``project_id`` d'un autre projet : 403, rien n'est écrit ni remplacé."""
    before = _uploads_state(env)
    resp = _upload(env, route, env.project_id, env.headers[FOREIGN_PERSONA])
    assert resp.status_code == 403
    assert resp.json() == {"error": PROJECT_DENIED_MESSAGE}
    assert _uploads_state(env) == before


@pytest.mark.parametrize("route", UPLOAD_ROUTES)
def test_upload_routes_refuse_read_only_member(env, route):
    """Membre en lecture seule (``viewer``) : 403 comme les routes ``/api/pipeline``, rien n'est écrit ni remplacé."""
    _add_project_member(env, FOREIGN_PERSONA, ProjectRole.VIEWER.value)
    before = _uploads_state(env)
    resp = _upload(env, route, env.project_id, env.headers[FOREIGN_PERSONA])
    assert resp.status_code == 403
    assert resp.json() == {"error": PROJECT_READ_ONLY_MESSAGE}
    assert _uploads_state(env) == before


@pytest.mark.parametrize("persona", ["admin", FOREIGN_PERSONA])
@pytest.mark.parametrize("route", UPLOAD_ROUTES)
def test_upload_routes_refuse_unknown_project(env, route, persona):
    """``project_id`` sans projet : 404, rien n'est écrit (avant : fichiers écrits, rattachement ignoré en silence)."""
    before = _uploads_state(env)
    resp = _upload(env, route, UNKNOWN_PROJECT_ID, env.headers[persona])
    assert resp.status_code == 404
    assert resp.json() == {"error": PROJECT_NOT_FOUND_MESSAGE}
    assert _uploads_state(env) == before


def test_upload_zip_checks_project_before_reading_the_archive(env):
    """Le contrôle du projet précède tout travail : archive invalide d'un non-membre -> 403 (pas le 400 de l'extraction)."""
    before = _uploads_state(env)
    resp = env.client.post(
        "/upload_zip",
        data={"project_id": str(env.project_id)},
        files={"file": ("broken.zip", b"not a zip", "application/zip")},
        headers=env.headers[FOREIGN_PERSONA],
    )
    assert resp.status_code == 403
    assert _uploads_state(env) == before


@pytest.mark.parametrize("persona", [OWNER_PERSONA, "admin", "collaborator"])
@pytest.mark.parametrize("route", UPLOAD_ROUTES)
def test_upload_routes_owner_collaborator_admin_allowed(env, route, persona):
    """Propriétaire, collaborateur et administrateur : 200, session créée et devenue le dossier actif du projet."""
    if persona == "collaborator":
        _add_project_member(env, FOREIGN_PERSONA, ProjectRole.COLLABORATOR.value)
        persona = FOREIGN_PERSONA
    resp = _upload(env, route, env.project_id, env.headers[persona])
    assert resp.status_code == 200
    _assert_session_created(env, route, resp.json())


@pytest.mark.parametrize("route", UPLOAD_ROUTES)
def test_upload_routes_without_project_keep_behaviour(env, route):
    """Utilisateur authentifié sans ``project_id`` : 200, dossier créé, aucune session, aucun projet touché."""
    _files, _dirs, rows_before, folder_before = _uploads_state(env)
    resp = _upload(env, route, None, env.headers[FOREIGN_PERSONA])
    assert resp.status_code == 200
    body = resp.json()
    assert body["project_id"] is None and body["session_id"] is None
    assert (env.uploads / body["path"]).is_dir()
    _files, _dirs, rows_after, folder_after = _uploads_state(env)
    assert (rows_after, folder_after) == (rows_before, folder_before)


@pytest.mark.parametrize("route", UPLOAD_ROUTES)
def test_upload_routes_accept_the_access_token_cookie(env, route):
    """La page appelle les routes sans en-tête ``Authorization`` : le cookie ``access_token`` suffit.

    Le propriétaire passe (session créée) ; le non-membre reçoit le même 403
    qu'avec l'en-tête, sans rien écrire.
    """
    env.client.cookies.set("access_token", create_access_token(subject=str(env.user_ids[OWNER_PERSONA])))
    try:
        resp = _upload(env, route, env.project_id)
        assert resp.status_code == 200
        _assert_session_created(env, route, resp.json())
        env.client.cookies.set("access_token", create_access_token(subject=str(env.user_ids[FOREIGN_PERSONA])))
        before = _uploads_state(env)
        denied = _upload(env, route, env.project_id)
        assert denied.status_code == 403
        assert denied.json() == {"error": PROJECT_DENIED_MESSAGE}
        assert _uploads_state(env) == before
    finally:
        env.client.cookies.clear()


def test_index_page_uploads_rely_on_the_cookie():
    """Le gabarit de la page appelle ``/upload_zip`` et ``/upload_csv`` en POST same-origin (cookie envoyé par défaut)."""
    template = os.path.join(os.path.dirname(ingestion_routes.__file__), "..", "templates", "index.html")
    with open(template, encoding="utf-8") as handle:
        source = handle.read()
    assert "fetch('/upload_zip', { method: 'POST', body: formData });" in source
    assert "fetch('/upload_csv', { method: 'POST', body: formData });" in source
    assert "(json.error || res.statusText)" in source
    assert "credentials: 'omit'" not in source
