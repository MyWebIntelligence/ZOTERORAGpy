"""Appartenance des sessions des routes pipeline et journal d'usage Albert web (lot 9, tâche 0).

Décision utilisateur du 2026-09-27 : les routes de ``app/routes/processing.py``
qui travaillent sur un dossier de session le contrôlent avant tout travail
(``_pipeline_session_refusal``) :

* session enregistrée (ligne ``PipelineSession``) d'un projet auquel un
  non-admin n'a pas accès : 403 ``{"error": …}`` (routes JSON, et
  ``/cluster_documents_sse`` dont la page lit du JSON sur un statut de refus)
  ou un unique événement SSE ``{"type": "error", "message": …}`` avec le
  statut 403, sans lancement de sous-processus, sans appel d'aide et sans
  lecture ni écriture dans le dossier ; autre graphie du même dossier
  (``sess/``, ``./sess``, ``autre/../sess``, autre casse ou forme Unicode
  sur un système de fichiers insensible), sous-dossier d'une session
  enregistrée et dossier qui en contient une : refusés de même ;
* propriétaire, collaborateur du projet et administrateur : accès inchangé,
  avec les clés du collaborateur lui-même (argv, env, délai, arguments des
  aides) ; les cas ``member_keys`` des goldens G7 d'avant l'amendement
  (``3790a64``) sont rejoués à l'octet quand il est collaborateur ;
* import hors projet (``SessionOwner``, audit A02) : réservé à son auteur ;
  dossier sans aucune ligne (import antérieur au contrôle, ou projet
  supprimé) : réservé aux administrateurs ; lecteur (``viewer``) d'un
  projet : lecture seule (403 sur toute route qui écrit ou arrête) ;
* ``/stop_all_scripts`` : même contrôle (droit d'arrêt) et identifiant
  canonique du dossier pour le registre des processus ;
* dossier qui ne se résout pas strictement sous ``uploads/`` (``..``, chemin
  absolu, ``.``, lien symbolique sortant, octet NUL) : 400, pour tous.

Journal d'usage côté routes (contrat figé du lot 9) : un ``UsageLedger`` par
requête, créé seulement quand le modèle se résout vers Albert, transmis aux
aides (``albert_usage_ledger``) et ajouté en fin de traitement à
``<session>/albert_usage.jsonl`` avec une ligne de synthèse dans les logs,
seulement si Albert a été appelé (``ALBERT_USAGE_LOG=0`` : synthèse seule).
Autre fournisseur ou Albert OFF : aucun argument, aucun fichier.

Harnais : ``golden_app`` de ``tests/test_albert_off_golden_routes.py``
(personas ``admin``, ``member_keys``, ``member_nokeys`` ; ``gsess-full`` est
la session enregistrée du projet de ``member_nokeys`` ; lanceurs et aides
remplacés par des enregistreurs). Identifiants factices seulement ; les
assertions portent sur des noms, des statuts et des booléens.

Run: pytest tests/test_session_ownership.py -q
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import unicodedata

import pytest
from sqlalchemy.orm import sessionmaker

import app.database.session as db_session_module
import app.routes.citations as citation_routes
import app.routes.processing as processing_routes
import app.services.background_task_manager as background_task_module
import app.utils.book_note_generator as book_note_generator
import app.utils.llm_note_generator as llm_note_generator
import app.utils.zotero_client as zotero_client
from app.models.pipeline_session import PipelineSession, SessionOwner
from app.models.project import ProjectMember, ProjectRole
from scripts.rad_albert.errors import AlbertAuthError
from tests import test_albert_off_golden_routes as golden
# Fixtures réutilisées (l'hygiène d'env des goldens est autouse dans ce module).
from tests.test_albert_off_golden_routes import _golden_env_hygiene, golden_app  # noqa: F401

# ======================================================================
# Constantes littérales
# ======================================================================
DENIED_MESSAGE = "Accès non autorisé à cette session : elle appartient à un projet auquel vous n'avez pas accès."
INVALID_PATH_MESSAGE = "Chemin de session invalide : le dossier doit se trouver sous uploads/."
READ_ONLY_MESSAGE = (
    "Accès en lecture seule à cette session : seuls le propriétaire et les collaborateurs "
    "du projet peuvent la modifier ou arrêter ses traitements."
)
OWNER_DENIED_MESSAGE = "Accès non autorisé à cette session : elle appartient à un autre utilisateur."
UNOWNED_MESSAGE = (
    "Session sans propriétaire enregistré (import antérieur au contrôle d'accès, ou projet supprimé) : "
    "réservée aux administrateurs. Importez de nouveau le fichier pour continuer."
)
REGISTERED = "gsess-full"          # session enregistrée du projet de member_nokeys
FOREIGN_PERSONA = "member_keys"    # ni propriétaire ni membre de ce projet
OWNER_PERSONA = "member_nokeys"
UNREGISTERED = "gsess-fresh-member-nokeys"  # import hors projet de member_nokeys (SessionOwner)
USAGE_FILE = "albert_usage.jsonl"
ALBERT_MODEL = "albert/gpt-oss-120b"
SERVED_MODEL = "openai/gpt-oss-120b"
CALL_COST = 0.0021

# (route, champ du dossier, champs supplémentaires, forme du refus : "json" = corps
# ``{"error": …}``, "sse" = un unique événement SSE). ``/cluster_documents_sse`` refuse
# en JSON : le gestionnaire de la page (index.html, gelé par G9) lit ``response.json()``
# sur un statut non OK et afficherait sinon le seul libellé du statut.
ROUTES = (
    ("/process_dataframe", "path", {}, "json"),
    ("/process_dataframe_sse", "path", {}, "sse"),
    ("/initial_text_chunking", "path", {"model": "gpt-4o-mini"}, "json"),
    ("/initial_text_chunking_sse", "path", {"model": "gpt-4o-mini"}, "sse"),
    ("/dense_embedding_generation", "path", {}, "json"),
    ("/dense_embedding_generation_sse", "path", {}, "sse"),
    ("/sparse_embedding_generation", "path", {}, "json"),
    ("/sparse_embedding_generation_sse", "path", {}, "sse"),
    ("/upload_db", "path", {"db_choice": "pinecone", "pinecone_index_name": "idx-golden"}, "json"),
    ("/generate_zotero_notes_sse", "session", {"note_mode": "extended", "model": "gpt-4o-mini"}, "sse"),
    ("/cluster_documents", "session_folder", {"session_name": "Golden"}, "json"),
    ("/cluster_documents_sse", "session_folder", {"session_name": "Golden"}, "json"),
    ("/apply_cluster_tags_zotero", "session_folder", {}, "json"),
)
ROUTE_IDS = [route for route, _, _, _ in ROUTES]


# ======================================================================
# Aides
# ======================================================================
def _form(field, folder, extra):
    """Formulaire d'une route pour le dossier ``folder``."""
    return dict({field: folder}, **extra)


def _post(env, persona, url, form):
    """POST de formulaire en tant que ``persona`` : (réponse, exceptions serveur, appels enregistrés)."""
    resp = env.client.post(url, data=form, headers=env.headers[persona])
    return resp, golden._take_server_errors(env), env.recorder.take()


def _events(resp):
    """Événements SSE décodés d'une réponse."""
    return [json.loads(raw) for raw in golden._sse_events(resp.text)]


def _is_refusal(resp, style, status, message):
    """Vrai si ``resp`` est le refus de session attendu (statut, type de contenu, corps ou unique événement)."""
    if resp.status_code != status:
        return False
    content_type = resp.headers.get("content-type", "")
    if style == "sse":
        return content_type.startswith("text/event-stream") and _events(resp) == [{"type": "error", "message": message}]
    return content_type.startswith("application/json") and resp.json() == {"error": message}


def _snapshot(folder):
    """Noms, tailles et dates de modification des fichiers d'un dossier."""
    out = {}
    for name in sorted(os.listdir(folder)):
        stat = os.stat(os.path.join(folder, name))
        out[name] = (stat.st_size, stat.st_mtime_ns)
    return out


def _add_member(env, persona):
    """Ajoute ``persona`` comme collaborateur du projet de ``gsess-full``."""
    env.db.add(ProjectMember(
        project_id=env.project.id, user_id=env.users[persona].id, role=ProjectRole.COLLABORATOR.value,
        created_at=golden.FIXED_CREATED_AT, updated_at=golden.FIXED_CREATED_AT,
    ))
    env.db.commit()
    env.db.refresh(env.project)


@pytest.fixture
def no_zotero_tags(monkeypatch):
    """Remplace ``add_tags_to_items`` (jamais d'appel Zotero réel) ; renvoie la liste des appels."""
    calls = []

    def _fake_add_tags(**kwargs):
        """Enregistre l'appel et renvoie un succès vide."""
        calls.append(sorted(kwargs))
        return {"success_count": 0, "failed_count": 0, "errors": []}

    monkeypatch.setattr(zotero_client, "add_tags_to_items", _fake_add_tags)
    return calls


# ======================================================================
# Session enregistrée d'un autre projet : 403 sans aucun travail
# ======================================================================
@pytest.mark.parametrize("switch", ["0", "1"])
@pytest.mark.parametrize("url, field, extra, style", ROUTES, ids=ROUTE_IDS)
def test_foreign_registered_session_refused(golden_app, monkeypatch, no_zotero_tags, url, field, extra, style, switch):
    """Non-admin hors projet : 403 (JSON ou unique événement SSE), aucun lancement ni appel, dossier intact."""
    monkeypatch.setenv("ALBERT_ENABLED", switch)
    env = golden_app
    folder = env.uploads / REGISTERED
    before = _snapshot(folder)
    form = _form(field, REGISTERED, extra)
    if url == "/process_dataframe":
        form["force_fresh"] = "true"  # sans le contrôle, output.csv serait supprimé
    resp, errors, calls = _post(env, FOREIGN_PERSONA, url, form)
    assert _is_refusal(resp, style, 403, DENIED_MESSAGE), (url, resp.status_code, resp.text[:300])
    assert errors == [] and calls == [] and no_zotero_tags == [], url
    assert "credential_required" not in resp.text
    assert _snapshot(folder) == before, url


@pytest.mark.parametrize("spelling", ["gsess-full/", "./gsess-full", "gsess-empty/../gsess-full", "gsess-full/."])
@pytest.mark.parametrize("url, field, extra, style", ROUTES, ids=ROUTE_IDS)
def test_foreign_registered_session_other_spellings_refused(golden_app, no_zotero_tags, url, field, extra, style,
                                                             spelling):
    """Autre graphie du dossier enregistré : même refus 403, aucun lancement."""
    env = golden_app
    resp, errors, calls = _post(env, FOREIGN_PERSONA, url, _form(field, spelling, extra))
    assert _is_refusal(resp, style, 403, DENIED_MESSAGE), (url, spelling, resp.status_code, resp.text[:300])
    assert errors == [] and calls == [] and no_zotero_tags == [], (url, spelling)


@pytest.mark.parametrize("url, field, extra, style", ROUTES, ids=ROUTE_IDS)
def test_foreign_session_refused_through_symlink(golden_app, no_zotero_tags, url, field, extra, style):
    """Lien symbolique sous uploads/ vers une session enregistrée d'autrui : refusé (403)."""
    env = golden_app
    os.symlink(env.uploads / REGISTERED, env.uploads / "gsess-alias")
    resp, errors, calls = _post(env, FOREIGN_PERSONA, url, _form(field, "gsess-alias", extra))
    assert _is_refusal(resp, style, 403, DENIED_MESSAGE), (url, resp.status_code, resp.text[:300])
    assert errors == [] and calls == [], url


def _register(env, folder):
    """Crée ``folder`` sous uploads/ (session complète + export Zotero) et l'enregistre dans le projet de gsess-full."""
    path = env.uploads / folder
    path.mkdir(parents=True)
    golden._write_full_session(str(path))
    golden._write_zotero_json(str(path))
    env.db.add(PipelineSession(
        project_id=env.project.id, session_folder=folder, source_type="zip",
        created_at=golden.FIXED_CREATED_AT, updated_at=golden.FIXED_CREATED_AT,
    ))
    env.db.commit()
    return path


def _garnish(path):
    """Crée ``path`` garni comme une session complète (tout lancement y serait possible)."""
    path.mkdir(parents=True, exist_ok=True)
    golden._write_full_session(str(path))
    golden._write_zotero_json(str(path))
    return path


def _opens_same_folder(env, spelling, folder):
    """Vrai si le système de fichiers ouvre ``folder`` sous la graphie ``spelling`` (casse ou forme Unicode ignorées)."""
    candidate = env.uploads / spelling
    return candidate.is_dir() and os.path.samefile(candidate, env.uploads / folder)


def _assert_variant_never_works(env, no_zotero_tags, url, field, extra, style, spelling, folder):
    """Graphie ``spelling`` d'un dossier enregistré d'autrui : refus 403 s'il l'ouvre, et jamais aucun travail."""
    before = _snapshot(env.uploads / folder)
    resp, errors, calls = _post(env, FOREIGN_PERSONA, url, _form(field, spelling, extra))
    assert errors == [] and calls == [] and no_zotero_tags == [], (url, spelling, resp.status_code, calls)
    if _opens_same_folder(env, spelling, folder):
        assert _is_refusal(resp, style, 403, DENIED_MESSAGE), (url, spelling, resp.status_code, resp.text[:300])
    assert _snapshot(env.uploads / folder) == before


@pytest.mark.parametrize("spelling", ["GSESS-FULL", "Gsess-Full/", "./gsess-FULL"])
@pytest.mark.parametrize("url, field, extra, style", ROUTES, ids=ROUTE_IDS)
def test_foreign_session_case_variant_refused(golden_app, no_zotero_tags, url, field, extra, style, spelling):
    """Autre casse du dossier enregistré (APFS, montages Docker Desktop) : 403 sans travail ; ailleurs, aucun travail."""
    _assert_variant_never_works(golden_app, no_zotero_tags, url, field, extra, style, spelling, REGISTERED)


ACCENTED = unicodedata.normalize("NFC", "gsess-bibliothèque")


@pytest.mark.parametrize("spelling", [unicodedata.normalize("NFD", ACCENTED),
                                      unicodedata.normalize("NFD", ACCENTED).upper()], ids=["nfd", "nfd_upper"])
@pytest.mark.parametrize("url, field, extra, style", ROUTES, ids=ROUTE_IDS)
def test_foreign_session_unicode_variant_refused(golden_app, no_zotero_tags, url, field, extra, style, spelling):
    """Forme NFD d'un dossier enregistré en NFC (système insensible à la normalisation) : 403 sans travail."""
    env = golden_app
    _register(env, ACCENTED)
    _assert_variant_never_works(env, no_zotero_tags, url, field, extra, style, spelling, ACCENTED)


def test_variant_spelling_resolves_to_registered_folder(golden_app):
    """La graphie listée sur disque est celle de la base ; système exact : graphie inchangée."""
    env = golden_app
    _register(env, ACCENTED)
    nfd = unicodedata.normalize("NFD", ACCENTED)
    for spelling, folder in (("GSESS-FULL", REGISTERED), (nfd, ACCENTED), (nfd.upper(), ACCENTED)):
        spellings = processing_routes._session_folder_spellings(spelling)
        assert spellings[0] == spelling
        assert (folder in spellings) == _opens_same_folder(env, spelling, folder), (spelling, spellings)
    assert processing_routes._session_folder_spellings(REGISTERED) == [REGISTERED]
    assert processing_routes._session_folder_spellings("gsess-missing") == ["gsess-missing"]


# Dossiers liés à une session enregistrée d'autrui : gsess-full/sub (sous-dossier) ;
# gsess-nest/root enregistré (ZIP à dossier racine unique) et ses parent et enfant.
RELATED_FOLDERS = {
    "under": "gsess-full/sub",
    "nested_equal": "gsess-nest/root",
    "nested_parent": "gsess-nest",
    "nested_under": "gsess-nest/root/deeper",
}


def _related_layout(env):
    """Crée gsess-full/sub, gsess-nest/root (enregistré dans le projet de gsess-full) et gsess-nest/root/deeper."""
    _garnish(env.uploads / "gsess-full" / "sub")
    _register(env, "gsess-nest/root")
    _garnish(env.uploads / "gsess-nest" / "root" / "deeper")
    golden._write_full_session(str(env.uploads / "gsess-nest"))
    golden._write_zotero_json(str(env.uploads / "gsess-nest"))


@pytest.mark.parametrize("kind", sorted(RELATED_FOLDERS))
@pytest.mark.parametrize("url, field, extra, style", ROUTES, ids=ROUTE_IDS)
def test_foreign_session_related_folders_refused(golden_app, no_zotero_tags, url, field, extra, style, kind):
    """Sous-dossier d'une session enregistrée d'autrui, ou dossier qui en contient une : 403, aucun travail."""
    env = golden_app
    _related_layout(env)
    folder = RELATED_FOLDERS[kind]
    before = _snapshot(env.uploads / folder)
    resp, errors, calls = _post(env, FOREIGN_PERSONA, url, _form(field, folder, extra))
    assert _is_refusal(resp, style, 403, DENIED_MESSAGE), (url, kind, resp.status_code, resp.text[:300])
    assert errors == [] and calls == [] and no_zotero_tags == [], (url, kind)
    assert _snapshot(env.uploads / folder) == before


@pytest.mark.parametrize("persona", ["admin", OWNER_PERSONA, "collaborator"])
@pytest.mark.parametrize("kind", sorted(RELATED_FOLDERS))
def test_related_folders_keep_access_for_members(golden_app, persona, kind):
    """Propriétaire, collaborateur et administrateur : dossiers liés accessibles (un lancement)."""
    env = golden_app
    _related_layout(env)
    if persona == "collaborator":
        _add_member(env, FOREIGN_PERSONA)
        persona = FOREIGN_PERSONA
    folder = RELATED_FOLDERS[kind]
    resp, errors, calls = _post(env, persona, "/sparse_embedding_generation", {"path": folder})
    assert resp.status_code == 200 and errors == [], (persona, kind, resp.text[:300])
    assert [c["session_folder"] for c in calls] == [folder], (persona, kind)


# ======================================================================
# Propriétaire, collaborateur, administrateur : accès inchangé
# ======================================================================
@pytest.mark.parametrize("persona", ["admin", OWNER_PERSONA, "collaborator"])
@pytest.mark.parametrize("url, field, extra, style", ROUTES, ids=ROUTE_IDS)
def test_project_members_and_admin_keep_access(golden_app, no_zotero_tags, url, field, extra, style, persona):
    """Propriétaire, collaborateur du projet et administrateur ne reçoivent jamais le refus de session."""
    env = golden_app
    if persona == "collaborator":
        _add_member(env, FOREIGN_PERSONA)
        persona = FOREIGN_PERSONA
    resp, errors, _calls = _post(env, persona, url, _form(field, REGISTERED, extra))
    assert not _is_refusal(resp, style, 403, DENIED_MESSAGE), (url, persona, resp.text[:300])
    assert DENIED_MESSAGE not in resp.text and INVALID_PATH_MESSAGE not in resp.text
    assert errors == [], (url, persona, errors)


# Lancements d'un collaborateur sur la session enregistrée : les valeurs figées par les
# goldens G7 d'avant l'amendement (cas member_keys de 3790a64), chemins normalisés.
SESSION_TMP = "<TMP>/uploads/" + REGISTERED
RAD_CHUNK = "<RAGPY>/scripts/rad_chunk.py"
RAD_VECTORDB = "<RAGPY>/scripts/rad_vectordb.py"
CHUNK_INITIAL_ARGS = ["--input", f"{SESSION_TMP}/output.csv", "--output", SESSION_TMP, "--phase", "initial",
                      "--model", "gpt-4o-mini"]
CHUNK_DENSE_ARGS = ["--input", f"{SESSION_TMP}/output_chunks.json", "--output", SESSION_TMP, "--phase", "dense"]
CHUNK_SPARSE_ARGS = ["--input", f"{SESSION_TMP}/output_chunks_with_embeddings.json", "--output", SESSION_TMP,
                     "--phase", "sparse"]
PINECONE_FORM = {"db_choice": "pinecone", "pinecone_index_name": "idx-golden", "pinecone_namespace": "ns-golden"}
SSE_COMPLETE = {"type": "complete", "message": "Process completed successfully", "count": 3}
# (cas, route, champs supplémentaires, lanceur, argv, délai, corps JSON attendu ; None pour une route SSE)
COLLABORATOR_LAUNCHES = (
    ("chunking_default", "/initial_text_chunking", {}, "run_tracked_subprocess",
     ["python3", RAD_CHUNK] + CHUNK_INITIAL_ARGS, 1800,
     {"status": "success", "file": f"{SESSION_TMP}/output_chunks.json", "count": 3,
      "message": "Generated 3 chunks using model gpt-4o-mini"}),
    ("chunking_gpt", "/initial_text_chunking", {"model": "gpt-4o-mini"}, "run_tracked_subprocess",
     ["python3", RAD_CHUNK] + CHUNK_INITIAL_ARGS, 1800,
     {"status": "success", "file": f"{SESSION_TMP}/output_chunks.json", "count": 3,
      "message": "Generated 3 chunks using model gpt-4o-mini"}),
    ("chunking_sse_default", "/initial_text_chunking_sse", {}, "run_subprocess_with_sse",
     ["python3", "-u", RAD_CHUNK] + CHUNK_INITIAL_ARGS, 1800, None),
    ("chunking_sse_gpt", "/initial_text_chunking_sse", {"model": "gpt-4o-mini"}, "run_subprocess_with_sse",
     ["python3", "-u", RAD_CHUNK] + CHUNK_INITIAL_ARGS, 1800, None),
    ("dense", "/dense_embedding_generation", {}, "run_tracked_subprocess",
     ["python3", RAD_CHUNK] + CHUNK_DENSE_ARGS, 1800,
     {"status": "success", "file": f"{SESSION_TMP}/output_chunks_with_embeddings.json", "count": 3}),
    ("dense_sse", "/dense_embedding_generation_sse", {}, "run_subprocess_with_sse",
     ["python3", "-u", RAD_CHUNK] + CHUNK_DENSE_ARGS, 1800, None),
    ("sparse", "/sparse_embedding_generation", {}, "run_tracked_subprocess",
     ["python3", RAD_CHUNK] + CHUNK_SPARSE_ARGS, 1800,
     {"status": "success", "file": f"{SESSION_TMP}/output_chunks_with_embeddings_sparse.json", "count": 3}),
    ("sparse_sse", "/sparse_embedding_generation_sse", {}, "run_subprocess_with_sse",
     ["python3", "-u", RAD_CHUNK] + CHUNK_SPARSE_ARGS, 1800, None),
    ("upload_db_pinecone", "/upload_db", PINECONE_FORM, "run_tracked_subprocess",
     ["python3", RAD_VECTORDB, "--input", f"{SESSION_TMP}/output_chunks_with_embeddings_sparse.json",
      "--db", "pinecone", "--index", "idx-golden", "--namespace", "ns-golden"], 3600,
     {"status": "success", "message": "Uploaded to pinecone", "inserted_count": 3, "skipped_count": 1,
      "journal_path": f"{SESSION_TMP}/dedup_journal.jsonl"}),
)
# Variables d'env des sous-processus alimentées par les clés personnelles du collaborateur.
MEMBER_ENV_CREDENTIALS = (
    ("OPENAI_API_KEY", "openai_api_key"),
    ("MISTRAL_API_KEY", "mistral_api_key"),
    ("PINECONE_API_KEY", "pinecone_api_key"),
    ("ZOTERO_API_KEY", "zotero_api_key"),
    ("ZOTERO_USER_ID", "zotero_user_id"),
)


def _member_fp(key):
    """Empreinte de la clé personnelle ``key`` du collaborateur (identifiant factice)."""
    return golden._fp(golden.FAKE_DB_CREDENTIALS[FOREIGN_PERSONA][key])


def _member_env_fp():
    """Env attendu d'un lancement du collaborateur : ses clés en base, rien du ``.env`` serveur."""
    creds = golden.FAKE_DB_CREDENTIALS[FOREIGN_PERSONA]
    return golden._env_fp({name: creds[key] for name, key in MEMBER_ENV_CREDENTIALS})


def _server_env_fps():
    """Empreintes des valeurs de l'env serveur factice."""
    return {golden._fp(value) for value in golden.FAKE_SERVER_ENV.values()}


@pytest.mark.parametrize("case, url, extra, launcher, argv, timeout, body", COLLABORATOR_LAUNCHES,
                         ids=[c[0] for c in COLLABORATOR_LAUNCHES])
def test_collaborator_launch_uses_own_keys(golden_app, case, url, extra, launcher, argv, timeout, body):
    """Collaborateur du projet : réponse, argv, délai, dossier et env (ses propres clés) inchangés."""
    env = golden_app
    _add_member(env, FOREIGN_PERSONA)
    sse = body is None
    out = env.norm(golden._post_case(env, case, FOREIGN_PERSONA, url, dict({"path": REGISTERED}, **extra), sse=sse))
    assert out["status"] == 200 and out["server_exceptions"] == [], out
    if sse:
        events = [json.loads(e) for e in out["events"]]
        assert out["content_type"].startswith("text/event-stream")
        assert events[-1] == SSE_COMPLETE and all(e["type"] != "error" for e in events), events
    else:
        assert out["body"] == body
    assert [c["fn"] for c in out["calls"]] == [launcher], out["calls"]
    call = out["calls"][0]
    assert call["argv"] == argv
    assert call["session_folder"] == REGISTERED and call["timeout"] == timeout
    if sse:
        assert call["error_keywords"] == "<default>"
    assert call["env"] == _member_env_fp()
    assert not set(call["env"].values()) & _server_env_fps()


def test_collaborator_notes_use_own_keys(golden_app):
    """Notes d'un collaborateur (étendues, gpt-4o-mini) : aides et appels Zotero avec ses propres clés."""
    env = golden_app
    _add_member(env, FOREIGN_PERSONA)
    form = {"session": REGISTERED, "note_mode": "extended", "model": "gpt-4o-mini"}
    out = env.norm(golden._post_case(env, "collaborator", FOREIGN_PERSONA, "/generate_zotero_notes_sse", form,
                                     sse=True))
    assert out["status"] == 200 and out["server_exceptions"] == [], out
    last = json.loads(out["events"][-1])
    assert last["type"] == "complete" and last["summary"] == {
        "created": 2, "exists": 0, "skipped": 0, "errors": 0, "mode": "api"}
    note_calls = ["build_note_html_async", "check_note_exists", "create_child_note"]
    assert [c["fn"] for c in out["calls"]] == ["verify_api_key"] + note_calls * 2
    zotero_key, zotero_user = _member_fp("zotero_api_key"), _member_fp("zotero_user_id")
    for call in out["calls"]:
        kwargs = call["kwargs"]
        if call["fn"] == "verify_api_key":
            assert kwargs == {"api_key": zotero_key}
        elif call["fn"] == "build_note_html_async":
            assert (kwargs["model"], kwargs["mode"], kwargs["use_llm"]) == ("gpt-4o-mini", "extended", True)
            assert (kwargs["openai_api_key"], kwargs["openrouter_api_key"]) == (_member_fp("openai_api_key"), None)
        else:
            assert (kwargs["library_type"], kwargs["library_id"], kwargs["api_key"]) == ("users", zotero_user,
                                                                                          zotero_key)
    fingerprints = set(re.findall(r"fp:[0-9a-f]{12}", json.dumps(out["calls"])))
    assert fingerprints and not fingerprints & _server_env_fps()


# Cas member_keys des goldens de routes G7 de 3790a64 (avant l'amendement du lot 9) :
# 24 premiers caractères hexadécimaux du sha256 de ``json.dumps(cas, indent=2,
# ensure_ascii=False)``, relevés une fois sur ``git show 3790a64:tests/fixtures/albert/
# golden_off/routes/<fichier>``. Ils figent à l'octet la réponse, l'argv, le délai, l'env
# (empreintes des clés en base de member_keys) et les arguments des aides ; l'amendement
# les a remplacés par le refus de session, ce test les rejoue avec member_keys collaborateur.
HEAD_MEMBER_CASES = {
    "g7_initial_text_chunking.json": {
        "member_keys:default": "0cf90ed26e62051030ac1e03",
        "member_keys:gpt-4o-mini": "7d01ca32409ee0a1b99a20c6",
        "member_keys:gemini_slug": "633b769ad86a7ae543f8cb1c",
    },
    "g7_initial_text_chunking_sse.json": {
        "member_keys:default": "3f79d79550fe19e84c8471f8",
        "member_keys:gpt-4o-mini": "3f43e5b1e09fb920e71e8fd5",
    },
    "g7_dense_embedding_generation.json": {"member_keys": "7676067b3ed398b8f59e69e0"},
    "g7_dense_embedding_generation_sse.json": {"member_keys": "baa0310ec5ec8540427c09aa"},
    "g7_sparse_embedding_generation.json": {"member_keys": "1e8f01ca36911b29305d2d92"},
    "g7_sparse_embedding_generation_sse.json": {"member_keys": "689b2cf9b6dfa18a2defcf2a"},
    "g7_upload_db.json": {
        "member_keys:pinecone": "6a4aaa7010a7a7bf5dd08077",
        "member_keys:weaviate": "db942996a4530cb2fc358a58",
        "member_keys:qdrant": "5c2f887a5b59d774fe0041ec",
        "member_keys:albert": "c865911ad6071a0d89c2f879",
    },
    "g7_generate_zotero_notes_sse.json": {
        "member_keys:extended:gpt-4o-mini": "14b98f45655099b03f173453",
        "member_keys:extended:gemini_slug": "61623dec47c59590ae39f5ea",
    },
}
# Constructeur G7 (tests/test_albert_off_golden_routes.py) de chaque golden.
HEAD_MEMBER_BUILDERS = {
    "g7_initial_text_chunking.json": "test_g7_initial_text_chunking",
    "g7_initial_text_chunking_sse.json": "test_g7_initial_text_chunking_sse",
    "g7_dense_embedding_generation.json": "test_g7_dense_embedding_generation",
    "g7_dense_embedding_generation_sse.json": "test_g7_dense_embedding_generation_sse",
    "g7_sparse_embedding_generation.json": "test_g7_sparse_embedding_generation",
    "g7_sparse_embedding_generation_sse.json": "test_g7_sparse_embedding_generation_sse",
    "g7_upload_db.json": "test_g7_upload_db",
    "g7_generate_zotero_notes_sse.json": "test_g7_generate_zotero_notes_sse",
}
# Issue ``fixed_event`` du cas member_keys du golden « défaut connu » de 3790a64.
HEAD_KNOWN_DEFECT_FIXED_EVENT = {
    "status": 200,
    "content_type": "text/event-stream; charset=utf-8",
    "events": [{
        "type": "error",
        "message": "Clé API OpenRouter requise pour ce modèle. Configurez-la dans Paramètres > Mes Identifiants.",
        "credential_required": "openrouter_api_key",
    }],
    "server_exceptions": [],
    "calls": [],
}


def _case_digest(case):
    """Empreinte d'un cas de golden, sérialisé comme dans les fichiers de goldens."""
    text = json.dumps(case, indent=2, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _labelled(case):
    """Cas sérialisé, empreintes remplacées par leur libellé (diagnostic d'un écart)."""
    text = json.dumps(case, indent=2, ensure_ascii=False)
    for fp, label in golden._fingerprint_legend().items():
        text = text.replace(fp, label)
    return text


def test_collaborator_replays_pre_amendment_member_cases(golden_app, monkeypatch):
    """member_keys collaborateur : les 8 constructeurs G7 rejouent à l'octet ses cas d'avant l'amendement.

    Les charges des goldens sont capturées (``_check_golden`` remplacé) au
    lieu d'être écrites ou comparées ; seuls les cas member_keys sont
    confrontés aux empreintes de ``HEAD_MEMBER_CASES``. En cas d'écart, le
    cas observé est affiché avec ses empreintes libellées, à comparer à
    ``git show 3790a64:tests/fixtures/albert/golden_off/routes/<fichier>``.
    """
    env = golden_app
    _add_member(env, FOREIGN_PERSONA)
    captured = {}

    def _capture(name, payload, hint=""):
        """Garde la charge d'un golden sans l'écrire ni la comparer."""
        captured[name] = payload

    monkeypatch.delenv("RAGPY_UPDATE_GOLDEN", raising=False)
    monkeypatch.setattr(golden, "_check_golden", _capture)
    for name, builder in HEAD_MEMBER_BUILDERS.items():
        getattr(golden, builder)(env)
    assert sorted(captured) == sorted(HEAD_MEMBER_CASES)

    mismatches = []
    member_labels = {f"db.{FOREIGN_PERSONA}.{key}" for key in golden.FAKE_DB_CREDENTIALS[FOREIGN_PERSONA]}
    legend = golden._fingerprint_legend()
    for name, expected in HEAD_MEMBER_CASES.items():
        cases = {c["case"]: c for c in captured[name]["cases"] if c["persona"] == FOREIGN_PERSONA}
        assert sorted(cases) == sorted(expected), name
        for case, digest in expected.items():
            if _case_digest(cases[case]) != digest:
                mismatches.append(f"{name} / {case}:\n{_labelled(cases[case])}")
            labels = {legend.get(fp, fp) for fp in re.findall(r"fp:[0-9a-f]{12}", json.dumps(cases[case]))}
            assert labels <= member_labels, (name, case, labels - member_labels)
    assert not mismatches, "\n\n".join(mismatches)

    # Golden « défaut connu » : l'événement corrigé figé dans 3790a64 pour member_keys.
    form = {"path": REGISTERED, "model": "google/gemini-2.5-flash"}
    out = env.norm(golden._post_case(env, "initial_text_chunking_sse:member_keys:gemini_slug", FOREIGN_PERSONA,
                                     "/initial_text_chunking_sse", form, sse=True))
    assert golden._parsed_events(golden._sse_outcome(out)) == HEAD_KNOWN_DEFECT_FIXED_EVENT


def test_owner_and_admin_launch_on_registered_session(golden_app):
    """Propriétaire et administrateur : lancement sparse sur la session enregistrée."""
    env = golden_app
    for persona in ("admin", OWNER_PERSONA):
        resp, errors, calls = _post(env, persona, "/sparse_embedding_generation", {"path": REGISTERED})
        assert resp.status_code == 200 and errors == [], persona
        assert [c["session_folder"] for c in calls] == [REGISTERED], persona


# ======================================================================
# Dossiers sans projet : propriétaire enregistré, sinon administrateurs seuls (audit A02)
# ======================================================================
def _add_owner(env, folder, persona):
    """Enregistre ``persona`` comme propriétaire du dossier ``folder`` (import hors projet)."""
    env.db.add(SessionOwner(session_folder=folder, user_id=env.users[persona].id, source_type="zip"))
    env.db.commit()


def test_owned_upload_reserved_to_its_uploader(golden_app):
    """Import hors projet (``SessionOwner``) : son auteur et l'administrateur passent, un tiers reçoit 403."""
    env = golden_app
    own = "gsess-fresh-member-keys"  # import de member_keys (qui a ses clés)
    for url in ("/process_dataframe", "/process_dataframe_sse"):
        resp, errors, calls = _post(env, "member_keys", url, {"path": own})
        assert resp.status_code == 200 and errors == [], (url, resp.text[:300])
        assert [c["session_folder"] for c in calls] == [own], url
        style = "sse" if url.endswith("_sse") else "json"
        resp, errors, calls = _post(env, OWNER_PERSONA, url, {"path": own})
        assert _is_refusal(resp, style, 403, OWNER_DENIED_MESSAGE), (url, resp.status_code, resp.text[:300])
        assert errors == [] and calls == [], url
    resp, errors, _calls = _post(env, "admin", "/process_dataframe", {"path": own})
    assert resp.status_code == 200 and errors == []


@pytest.mark.parametrize("url, field, extra, style", ROUTES, ids=ROUTE_IDS)
def test_unowned_legacy_folder_reserved_to_admins(golden_app, no_zotero_tags, url, field, extra, style):
    """Dossier sans aucune ligne (import antérieur au contrôle) : 403 pour un non-admin, sans aucun travail."""
    env = golden_app
    folder = env.uploads / "gsess-empty"
    before = _snapshot(folder)
    for persona in (FOREIGN_PERSONA, OWNER_PERSONA):
        resp, errors, calls = _post(env, persona, url, _form(field, "gsess-empty", extra))
        assert _is_refusal(resp, style, 403, UNOWNED_MESSAGE), (url, persona, resp.status_code, resp.text[:300])
        assert errors == [] and calls == [] and no_zotero_tags == [], (url, persona)
    assert _snapshot(folder) == before


def test_unowned_legacy_folder_admin_keeps_access(golden_app):
    """Administrateur sur un dossier sans ligne : réponses historiques (ici l'absence d'output.csv)."""
    env = golden_app
    resp, errors, calls = _post(env, "admin", "/initial_text_chunking", {"path": "gsess-empty"})
    assert resp.status_code == 400 and calls == [] and errors == []
    assert resp.json() == {"error": "output.csv not found. Please complete the extraction step first."}
    resp, errors, calls = _post(env, "admin", "/sparse_embedding_generation", {"path": "gsess-missing"})
    assert resp.status_code == 400 and calls == [] and resp.json() == {"error": "Directory not found: gsess-missing"}


def test_missing_folder_is_refused_before_existence_for_non_admins(golden_app):
    """Dossier absent pour un non-admin : 403 (pas d'owner), sans révéler s'il existe."""
    env = golden_app
    resp, errors, calls = _post(env, FOREIGN_PERSONA, "/sparse_embedding_generation", {"path": "gsess-missing"})
    assert resp.status_code == 403 and calls == [] and resp.json() == {"error": UNOWNED_MESSAGE}


def test_orphan_session_row_grants_nothing(golden_app):
    """Ligne PipelineSession dont le projet n'existe plus : n'accorde rien, dossier réservé aux administrateurs."""
    env = golden_app
    env.db.add(PipelineSession(
        project_id=987654, session_folder="gsess-empty", source_type="zip",
        created_at=golden.FIXED_CREATED_AT, updated_at=golden.FIXED_CREATED_AT,
    ))
    env.db.commit()
    resp, errors, calls = _post(env, FOREIGN_PERSONA, "/initial_text_chunking", {"path": "gsess-empty"})
    assert resp.status_code == 403 and calls == [] and resp.json() == {"error": UNOWNED_MESSAGE}
    resp, errors, calls = _post(env, "admin", "/initial_text_chunking", {"path": "gsess-empty"})
    assert resp.status_code == 400 and calls == []


def test_orphan_row_with_owner_keeps_owner_access(golden_app):
    """Ligne orpheline + propriétaire enregistré : le propriétaire garde l'accès, un tiers est refusé."""
    env = golden_app
    own = "gsess-fresh-member-keys"
    env.db.add(PipelineSession(
        project_id=987654, session_folder=own, source_type="zip",
        created_at=golden.FIXED_CREATED_AT, updated_at=golden.FIXED_CREATED_AT,
    ))
    env.db.commit()
    resp, errors, calls = _post(env, "member_keys", "/process_dataframe", {"path": own})
    assert resp.status_code == 200 and errors == [] and len(calls) == 1
    resp, errors, calls = _post(env, OWNER_PERSONA, "/process_dataframe", {"path": own})
    assert resp.status_code == 403 and calls == []


@pytest.mark.parametrize("folder", ["gsess-full-copy", "gsess-nest-other", "gsess_nest", "gsess-nes"])
def test_unrelated_neighbour_folders_follow_their_own_owner(golden_app, folder):
    """Voisins par préfixe de caractères (pas de composant) ou jokers LIKE (``_``) : seul leur propre owner compte."""
    env = golden_app
    _related_layout(env)
    _garnish(env.uploads / folder)
    _add_owner(env, folder, FOREIGN_PERSONA)
    resp, errors, calls = _post(env, FOREIGN_PERSONA, "/sparse_embedding_generation", {"path": folder})
    assert resp.status_code == 200 and errors == [], (folder, resp.text[:300])
    assert [c["session_folder"] for c in calls] == [folder]


def test_nested_folder_under_uploads_is_accepted(golden_app):
    """Un sous-dossier reste strictement sous uploads/ : accepté pour son propriétaire et l'administrateur."""
    env = golden_app
    nested = env.uploads / "nested" / "sub"
    nested.mkdir(parents=True)
    golden._write_output_csv(str(nested / "output.csv"))
    _add_owner(env, "nested/sub", FOREIGN_PERSONA)
    for persona in ("admin", FOREIGN_PERSONA):
        resp, errors, calls = _post(env, persona, "/initial_text_chunking", {"path": "nested/sub", "model": "gpt-4o-mini"})
        assert resp.status_code == 200 and errors == [] and [c["session_folder"] for c in calls] == ["nested/sub"]
    resp, errors, calls = _post(env, OWNER_PERSONA, "/initial_text_chunking", {"path": "nested/sub", "model": "gpt-4o-mini"})
    assert resp.status_code == 403 and calls == []
    # Le dossier parent contient la session enregistrée : même propriétaire exigé.
    resp, errors, calls = _post(env, OWNER_PERSONA, "/initial_text_chunking", {"path": "nested", "model": "gpt-4o-mini"})
    assert resp.status_code == 403 and calls == []


# ======================================================================
# Lecteur d'un projet : lecture seule (audit A02)
# ======================================================================
def _add_viewer(env, persona):
    """Ajoute ``persona`` comme lecteur (``viewer``) du projet de ``gsess-full``."""
    env.db.add(ProjectMember(
        project_id=env.project.id, user_id=env.users[persona].id, role=ProjectRole.VIEWER.value,
        created_at=golden.FIXED_CREATED_AT, updated_at=golden.FIXED_CREATED_AT,
    ))
    env.db.commit()
    env.db.refresh(env.project)


@pytest.mark.parametrize("url, field, extra, style", ROUTES, ids=ROUTE_IDS)
def test_viewer_cannot_modify_registered_session(golden_app, no_zotero_tags, url, field, extra, style):
    """Lecteur du projet : 403 lecture seule sur toute route qui écrit, aucun lancement, dossier intact."""
    env = golden_app
    _add_viewer(env, FOREIGN_PERSONA)
    folder = env.uploads / REGISTERED
    before = _snapshot(folder)
    resp, errors, calls = _post(env, FOREIGN_PERSONA, url, _form(field, REGISTERED, extra))
    assert _is_refusal(resp, style, 403, READ_ONLY_MESSAGE), (url, resp.status_code, resp.text[:300])
    assert errors == [] and calls == [] and no_zotero_tags == [], url
    assert _snapshot(folder) == before


def test_viewer_can_read_session_files(golden_app):
    """Lecteur du projet : la lecture des fichiers de session reste permise."""
    env = golden_app
    _add_viewer(env, FOREIGN_PERSONA)
    resp = env.client.get(f"/api/pipeline/sessions/{REGISTERED}/files", headers=env.headers[FOREIGN_PERSONA])
    assert resp.status_code == 200 and resp.json()["session_folder"] == REGISTERED
    resp = env.client.get(f"/api/pipeline/sessions/{REGISTERED}/verify", headers=env.headers[FOREIGN_PERSONA])
    assert resp.status_code == 200 and resp.json()["can_edit"] is False


def test_viewer_cannot_stop_or_change_status(golden_app, monkeypatch):
    """Lecteur du projet : ni arrêt des traitements ni changement de statut."""
    env = golden_app
    _add_viewer(env, FOREIGN_PERSONA)
    stopped = []
    monkeypatch.setattr(processing_routes.process_manager, "stop_session", lambda key: stopped.append(key) or {})
    resp, errors, _calls = _post(env, FOREIGN_PERSONA, "/stop_all_scripts", {"session": REGISTERED})
    assert resp.status_code == 403 and resp.json() == {"error": READ_ONLY_MESSAGE} and stopped == []
    session_id = env.db.query(PipelineSession).filter(PipelineSession.session_folder == REGISTERED).one().id
    resp = env.client.patch(f"/api/pipeline/sessions/{session_id}/status", data={"status": "error"},
                            headers=env.headers[FOREIGN_PERSONA])
    assert resp.status_code == 403


# ======================================================================
# Arrêt : identifiant canonique et contrôle identique aux routes (audit A02)
# ======================================================================
@pytest.mark.parametrize("spelling", ["./gsess-full", "gsess-full/", "gsess-empty/../gsess-full"])
def test_stop_other_spelling_refused_for_foreign_user(golden_app, monkeypatch, spelling):
    """Tiers : une autre graphie du dossier enregistré reçoit le même 403, aucun arrêt."""
    env = golden_app
    stopped = []
    monkeypatch.setattr(processing_routes.process_manager, "stop_session", lambda key: stopped.append(key) or {})
    resp, errors, _calls = _post(env, FOREIGN_PERSONA, "/stop_all_scripts", {"session": spelling})
    assert resp.status_code == 403 and resp.json() == {"error": DENIED_MESSAGE}
    assert stopped == [] and errors == []


@pytest.mark.parametrize("spelling", ["./gsess-full", "gsess-full/", "gsess-full"])
def test_stop_uses_canonical_key(golden_app, monkeypatch, spelling):
    """Propriétaire : toute graphie arrête le registre canonique ``gsess-full``."""
    env = golden_app
    stopped = []
    monkeypatch.setattr(processing_routes.process_manager, "stop_session",
                        lambda key: stopped.append(key) or {"status": "ok"})
    resp, errors, _calls = _post(env, OWNER_PERSONA, "/stop_all_scripts", {"session": spelling})
    assert resp.status_code == 200 and errors == []
    assert stopped == [REGISTERED]


def test_launch_registers_canonical_key(golden_app):
    """Lancement avec ``./gsess-full`` : le processus est enregistré sous ``gsess-full``."""
    env = golden_app
    resp, errors, calls = _post(env, OWNER_PERSONA, "/sparse_embedding_generation", {"path": "./gsess-full"})
    assert resp.status_code == 200 and errors == []
    assert [c["session_folder"] for c in calls] == [REGISTERED]


# ======================================================================
# Chemins hors de uploads/ : 400 pour tous
# ======================================================================
def _outside_folder(env):
    """Dossier voisin de uploads/, garni comme une session (tout lancement y serait possible)."""
    outside = env.uploads.parent / "outside"
    if not outside.exists():
        outside.mkdir()
        golden._write_full_session(str(outside))
        golden._write_zotero_json(str(outside))
    return outside


@pytest.mark.parametrize("kind", ["dotdot", "absolute", "upload_root", "dotdot_via_session", "symlink_out", "nul"])
@pytest.mark.parametrize("url, field, extra, style", ROUTES, ids=ROUTE_IDS)
def test_path_outside_uploads_rejected(golden_app, no_zotero_tags, url, field, extra, style, kind):
    """``..``, chemin absolu, ``.``, lien sortant, octet NUL : 400 (JSON ou événement), aucun lancement."""
    env = golden_app
    outside = _outside_folder(env)
    if kind == "symlink_out":
        os.symlink(outside, env.uploads / "gsess-escape")
    folder = {
        "dotdot": "../outside",
        "absolute": str(outside),
        "upload_root": ".",
        "dotdot_via_session": "gsess-full/../../outside",
        "symlink_out": "gsess-escape",
        "nul": "gsess-full\x00x",
    }[kind]
    before = _snapshot(outside)
    for persona in ("admin", FOREIGN_PERSONA):
        resp, errors, calls = _post(env, persona, url, _form(field, folder, extra))
        assert _is_refusal(resp, style, 400, INVALID_PATH_MESSAGE), (url, kind, persona, resp.status_code,
                                                                     resp.text[:300])
        assert errors == [] and calls == [] and no_zotero_tags == [], (url, kind, persona)
    assert _snapshot(outside) == before


# ======================================================================
# Journal d'usage Albert des routes web (contrat figé du lot 9)
# ======================================================================
class _LedgerFakes:
    """Aides de notes et de citations qui inscrivent un appel dans le ledger reçu.

    ``seen`` garde, par appel, le nom de l'aide et la présence de l'argument
    ``albert_usage_ledger`` ; ``record_calls`` règle l'inscription et
    ``error`` l'exception levée après l'inscription.
    """

    def __init__(self):
        """Aucun appel vu ; chaque appel est inscrit ; aucune erreur."""
        self.seen = []
        self.record_calls = True
        self.error = None

    def _use(self, fn, kwargs, role):
        """Note l'appel et, si un ledger est transmis, y inscrit un appel Albert."""
        ledger = kwargs.get("albert_usage_ledger")
        self.seen.append((fn, "albert_usage_ledger" in kwargs))
        if ledger is not None and self.record_calls:
            ledger.record(
                endpoint="/v1/chat/completions", model=SERVED_MODEL, response_model="gpt-oss-120b",
                usage={"usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15,
                                 "cost": CALL_COST}},
                role=role, status=200,
            )
        if self.error is not None:
            raise self.error

    async def note_html(self, *args, **kwargs):
        """Remplace ``build_note_html_async``."""
        self._use("build_note_html_async", kwargs, "notes")
        return ("<!-- ragpy-note:ledger -->", "<p>Note</p>")

    async def abstract(self, *args, **kwargs):
        """Remplace ``build_abstract_text_async``."""
        self._use("build_abstract_text_async", kwargs, "notes")
        return "Résumé."

    async def book_note(self, *args, **kwargs):
        """Remplace ``build_book_note_async``."""
        self._use("build_book_note_async", kwargs, "long_context")
        return ("<!-- ragpy-book:ledger -->", "<p>Fiche</p>")

    async def process_citations(self, *args, **kwargs):
        """Remplace ``process_citations_parallel`` : un appel inscrit par citation."""
        citations = list(kwargs.get("citations") or [])
        yield ("init", {"total": len(citations), "batch_size": kwargs.get("batch_size")})
        for idx, citation in enumerate(citations, start=1):
            self._use("process_citations_parallel", kwargs, "citations")
            citation_dict = citation.model_dump(mode="json")
            yield ("progress", {
                "current": idx, "total": len(citations), "status": "skipped",
                "title": citation_dict.get("title", "")[:50], "filter_result": None,
                "citation": citation_dict, "web_source": "article_url", "error_message": None,
            })
        yield ("complete", {"relevant": 0, "skipped": len(citations), "errors": 0})

    async def filter_citation(self, *args, **kwargs):
        """Remplace ``filter_citation_with_llm`` (citation jugée non pertinente)."""
        self._use("filter_citation_with_llm", kwargs, "citations")
        return "NA"


@pytest.fixture
def ledger_env(golden_app, monkeypatch):
    """Harnais des goldens avec des aides qui alimentent le ledger reçu, tâches de fond exécutées."""
    env = golden_app
    fakes = _LedgerFakes()
    monkeypatch.setattr(llm_note_generator, "build_note_html_async", fakes.note_html)
    monkeypatch.setattr(llm_note_generator, "build_abstract_text_async", fakes.abstract)
    monkeypatch.setattr(book_note_generator, "build_book_note_async", fakes.book_note)
    monkeypatch.setattr(citation_routes, "process_citations_parallel", fakes.process_citations)
    monkeypatch.setattr(citation_routes, "filter_citation_with_llm", fakes.filter_citation)

    async def _run_now(task_id, coroutine, db):
        """Exécute la tâche de fond tout de suite (sa fin écrit le journal)."""
        await coroutine

    async def _no_progress(*args, **kwargs):
        """Ignore les mises à jour de progression."""
        return None

    monkeypatch.setattr(citation_routes.background_task_manager, "start_task", _run_now)
    monkeypatch.setattr(citation_routes.background_task_manager, "update_progress", _no_progress)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=env.db.get_bind())
    monkeypatch.setattr(db_session_module, "SessionLocal", factory)
    monkeypatch.setattr(background_task_module, "SessionLocal", factory)
    env.fakes = fakes
    return env


def _usage_records(folder):
    """Enregistrements de ``albert_usage.jsonl`` d'un dossier ([] si absent)."""
    path = os.path.join(folder, USAGE_FILE)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _usage_log_lines(caplog, context):
    """Lignes de synthèse d'usage Albert journalisées pour ``context``."""
    marker = f"Albert usage ({context}):"
    return [r.getMessage() for r in caplog.records if marker in r.getMessage()]


@pytest.mark.parametrize("mode, helper, role", [
    ("extended", "build_note_html_async", "notes"),
    ("short", "build_abstract_text_async", "notes"),
    ("book", "build_book_note_async", "long_context"),
])
def test_notes_albert_writes_usage_journal(ledger_env, monkeypatch, caplog, mode, helper, role):
    """Notes ``albert/…`` : ledger transmis à chaque aide, journal ajouté dans la session, synthèse loguée."""
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    env = ledger_env
    caplog.set_level(logging.INFO, logger=processing_routes.logger.name)
    form = {"session": REGISTERED, "note_mode": mode, "model": ALBERT_MODEL}
    for run in (1, 2):
        resp, errors, _ = _post(env, "admin", "/generate_zotero_notes_sse", form)
        assert resp.status_code == 200 and errors == [] and _events(resp)[-1]["type"] == "complete"
        assert env.fakes.seen == [(helper, True)] * 2
        env.fakes.seen.clear()
        records = _usage_records(env.uploads / REGISTERED)
        assert len(records) == 2 * run  # ajout, jamais d'écrasement
        assert all(r["model"] == SERVED_MODEL and r["role"] == role for r in records)
        assert all(isinstance(r["cost"], float) and r["cost"] == CALL_COST for r in records)
        assert len(_usage_log_lines(caplog, "notes")) == run


def test_notes_albert_usage_log_off_keeps_summary_only(ledger_env, monkeypatch, caplog):
    """``ALBERT_USAGE_LOG=0`` : aucun fichier, la ligne de synthèse reste."""
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    monkeypatch.setenv("ALBERT_USAGE_LOG", "0")
    env = ledger_env
    caplog.set_level(logging.INFO, logger=processing_routes.logger.name)
    resp, errors, _ = _post(env, "admin", "/generate_zotero_notes_sse",
                            {"session": REGISTERED, "note_mode": "extended", "model": ALBERT_MODEL})
    assert resp.status_code == 200 and errors == []
    assert _usage_records(env.uploads / REGISTERED) == []
    assert not os.path.exists(env.uploads / REGISTERED / USAGE_FILE)
    assert len(_usage_log_lines(caplog, "notes")) == 1


def test_notes_albert_without_recorded_call_writes_nothing(ledger_env, monkeypatch, caplog):
    """Albert sélectionné mais aucun appel inscrit : ni fichier ni ligne de synthèse."""
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    env = ledger_env
    env.fakes.record_calls = False
    caplog.set_level(logging.INFO, logger=processing_routes.logger.name)
    resp, errors, _ = _post(env, "admin", "/generate_zotero_notes_sse",
                            {"session": REGISTERED, "note_mode": "extended", "model": ALBERT_MODEL})
    assert resp.status_code == 200 and errors == []
    assert env.fakes.seen == [("build_note_html_async", True)] * 2
    assert not os.path.exists(env.uploads / REGISTERED / USAGE_FILE)
    assert _usage_log_lines(caplog, "notes") == []


def test_notes_albert_account_error_still_writes_journal(ledger_env, monkeypatch):
    """Erreur de compte après un appel inscrit : événement d'erreur, journal écrit quand même (fin du job)."""
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    env = ledger_env
    env.fakes.error = AlbertAuthError(reason="invalid_key", status=401)
    resp, errors, _ = _post(env, "admin", "/generate_zotero_notes_sse",
                            {"session": REGISTERED, "note_mode": "extended", "model": ALBERT_MODEL})
    events = _events(resp)
    assert errors == [] and events[-1]["type"] == "error" and events[-1].get("credential_required")
    assert len(_usage_records(env.uploads / REGISTERED)) == 1


@pytest.mark.parametrize("switch, model", [("0", "gpt-4o-mini"), ("0", None), ("1", "gpt-4o-mini"),
                                           ("1", "google/gemini-2.5-flash")])
def test_notes_other_providers_get_no_ledger(ledger_env, monkeypatch, caplog, switch, model):
    """Albert OFF ou autre fournisseur : aucun argument ``albert_usage_ledger``, aucun fichier, aucun log."""
    monkeypatch.setenv("ALBERT_ENABLED", switch)
    env = ledger_env
    caplog.set_level(logging.INFO, logger=processing_routes.logger.name)
    form = {"session": REGISTERED, "note_mode": "extended"}
    if model:
        form["model"] = model
    resp, errors, _ = _post(env, "admin", "/generate_zotero_notes_sse", form)
    assert resp.status_code == 200 and errors == []
    assert env.fakes.seen == [("build_note_html_async", False)] * 2
    assert not os.path.exists(env.uploads / REGISTERED / USAGE_FILE)
    assert _usage_log_lines(caplog, "notes") == []


def _citation_folder(env):
    """Dossier de la dernière session de citations créée par le harnais."""
    return env.uploads / f"gsess-cit-{env.citation_count}"


@pytest.mark.parametrize("route, form_extra, n_citations, helper", [
    ("filter_citations_sse", {"batch_size": 5}, 2, "process_citations_parallel"),
    ("batch_import_citations_sse", {"batch_size": 10}, 1, "filter_citation_with_llm"),
    ("filter_citations_bg", {"batch_size": 5}, 2, "process_citations_parallel"),
])
def test_citations_albert_writes_usage_journal(ledger_env, monkeypatch, caplog, route, form_extra, n_citations,
                                               helper):
    """Citations ``albert/…`` (3 sites) : ledger transmis, journal dans la session, synthèse loguée."""
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    env = ledger_env
    caplog.set_level(logging.INFO, logger=processing_routes.logger.name)
    pid, sid = golden._make_citation_session(env, "admin", ALBERT_MODEL, n_citations=n_citations)
    resp, errors, _ = _post(env, "admin", f"/api/projects/{pid}/{route}", dict({"session_id": sid}, **form_extra))
    assert resp.status_code == 200 and errors == [], resp.text[:300]
    assert env.fakes.seen == [(helper, True)] * n_citations
    records = _usage_records(_citation_folder(env))
    assert len(records) == n_citations
    assert all(r["model"] == SERVED_MODEL and r["cost"] == CALL_COST for r in records)
    assert len(_usage_log_lines(caplog, "citations")) == 1


@pytest.mark.parametrize("route, form_extra, helper", [
    ("filter_citations_sse", {"batch_size": 5}, "process_citations_parallel"),
    ("batch_import_citations_sse", {"batch_size": 10}, "filter_citation_with_llm"),
    ("filter_citations_bg", {"batch_size": 5}, "process_citations_parallel"),
])
@pytest.mark.parametrize("switch", ["0", "1"])
def test_citations_other_providers_get_no_ledger(ledger_env, monkeypatch, caplog, route, form_extra, helper,
                                                 switch):
    """Citations hors Albert : aucun argument ``albert_usage_ledger``, aucun fichier, aucun log."""
    monkeypatch.setenv("ALBERT_ENABLED", switch)
    env = ledger_env
    caplog.set_level(logging.INFO, logger=processing_routes.logger.name)
    pid, sid = golden._make_citation_session(env, "admin", "gpt-4o-mini", n_citations=1)
    resp, errors, _ = _post(env, "admin", f"/api/projects/{pid}/{route}", dict({"session_id": sid}, **form_extra))
    assert resp.status_code == 200 and errors == [], resp.text[:300]
    assert env.fakes.seen == [(helper, False)]
    assert not os.path.exists(_citation_folder(env) / USAGE_FILE)
    assert _usage_log_lines(caplog, "citations") == []


def test_flush_helper_contract(tmp_path, monkeypatch, caplog):
    """``flush_albert_usage_ledger`` : None ou ledger vide → rien ; appels → ajout ; erreur d'écriture avalée."""
    caplog.set_level(logging.INFO, logger=processing_routes.logger.name)
    assert processing_routes.flush_albert_usage_ledger(None, str(tmp_path), "notes") is None
    ledger = processing_routes.new_albert_usage_ledger()
    assert processing_routes.flush_albert_usage_ledger(ledger, str(tmp_path), "notes") is None
    assert os.listdir(tmp_path) == [] and _usage_log_lines(caplog, "notes") == []

    ledger.record(endpoint="/v1/chat/completions", model=SERVED_MODEL, usage={"cost": CALL_COST}, role="notes")
    written = processing_routes.flush_albert_usage_ledger(ledger, str(tmp_path), "notes")
    assert written == os.path.join(str(tmp_path), USAGE_FILE) and len(_usage_records(tmp_path)) == 1
    # Déjà écrit : un second passage n'ajoute rien, mais journalise la synthèse.
    assert processing_routes.flush_albert_usage_ledger(ledger, str(tmp_path), "notes") is None
    assert len(_usage_records(tmp_path)) == 1

    # Erreur d'écriture : avalée (aucune exception), avertissement puis synthèse journalisés.
    other = processing_routes.new_albert_usage_ledger()
    other.record(endpoint="/v1/chat/completions", model=SERVED_MODEL, role="notes")
    monkeypatch.setattr(other, "write_jsonl", _raise_os_error)
    caplog.clear()
    assert processing_routes.flush_albert_usage_ledger(other, str(tmp_path / "err"), "citations") is None
    assert not (tmp_path / "err").exists()
    assert any("not written" in r.getMessage() for r in caplog.records)
    assert len(_usage_log_lines(caplog, "citations")) == 1


def _raise_os_error(*args, **kwargs):
    """Simule une erreur d'écriture du journal."""
    raise OSError("disque plein (simulé)")
