"""Tests du câblage Albert des routes (lot 7) : pipeline, notes, citations, Celery.

Couverture (``.claude/tasks/SPRINT_albert.md``, lot 7, tâches 0 à 9) :

* Albert OFF : les requêtes des goldens G7, complétées des champs de
  formulaire propres à Albert, donnent exactement les réponses, événements
  SSE, argv, empreintes d'env et délais figés (G7 étendu) ; ``albert/…`` donne
  un 400 (ou un unique événement SSE d'erreur) sans sous-processus ni appel
  d'aide, ``upload_pop_json`` compris ; notes avec un modèle vide et un
  défaut OpenRouter (slug) : contrôle historique de la clé OpenAI, sans
  lecture du ``.env`` (aucun golden ne couvre ce cas) ;
* défaut préexistant corrigé : les routes SSE de chunking et d'embeddings
  denses émettent, pour une clé manquante, l'unique événement accepté par le
  golden ``g7_sse_credential_error_known_defect.json`` ;
* Albert ON : délai ``ALBERT_SUBPROCESS_TIMEOUT`` seulement quand la requête
  sélectionne Albert ; clé Albert exigée selon la capacité (chunking, OCR,
  dense, notes, citations, ``upload_db``), jamais la clé serveur pour un
  non-admin ; ``EMBEDDING_PROVIDER`` posé explicitement ; erreur de compte
  Albert convertie en événement ``credential_required`` puis arrêt ;
* ``/upload_db`` albert : entrée ``resolve_albert_input`` avant le contrôle du
  fichier sparse, acquittement RGPD exigé, drapeaux CLI, lignes
  ``Skipped (existing)``, ``Albert manifest`` et ``Dedup journal`` lues en
  entier, audit et copie du manifeste hors ``uploads/`` ; session enregistrée
  d'un projet inaccessible refusée (403) sans lancement, audit ni manifeste ;
  les 3 autres bases inchangées ;
* citations : clé transmise aux 3 sites, aucun sémaphore externe sur le
  chemin Albert, erreur de compte en événement SSE ;
* Celery : branche albert gardée (OFF, acquittement, clé, accès à la
  session), jamais réessayée automatiquement, ``output_chunks.json`` accepté ;
  limites de temps Celery (``soft_time_limit``/``time_limit``) couvrant
  ``ALBERT_SUBPROCESS_TIMEOUT`` sur les seules soumissions qui sélectionnent
  Albert.

Harnais : ``golden_app`` (``tests/test_albert_off_golden_routes.py``) et
``world`` (``tests/test_celery_tasks.py``) sont réutilisés tels quels. Les
requêtes ON visent les sessions d'un « projet Albert » (propriétaire
``member_albert``, les autres personas non-admin collaborateurs) : elles ne
dépendent donc jamais d'un accès à une session étrangère ; ``gsess-full``
(projet de ``member_nokeys``) sert de session étrangère. Les
lanceurs ``app.routes.processing.run_tracked_subprocess`` et
``app.utils.sse_helpers.run_subprocess_with_sse`` (plus l'alias éventuel du
module de traitement) sont remplacés par des enregistreurs ; aucun processus
ni appel réseau. Seules des clés factices sont utilisées ; les assertions ne
portent que sur des noms, des étiquettes et des booléens, jamais sur une
valeur de clé.

Run: pytest tests/test_albert_routes.py -q
"""

from __future__ import annotations

import glob
import importlib
import json
import os
import subprocess
from types import SimpleNamespace

import pytest
from celery.app.task import Task as CeleryTask
from sqlalchemy.orm import sessionmaker

import app.core.config as core_config
import app.database.session as db_session_module
import app.routes.citations as citation_routes
import app.routes.processing as processing_routes
import app.services.background_task_manager as background_task_module
import app.utils.book_note_generator as book_note_generator
import app.utils.llm_note_generator as llm_note_generator
import app.utils.sse_helpers as sse_helpers
from app.core.credentials import CREDENTIAL_ERROR_MESSAGES, encrypt_credentials
from app.core.security import create_access_token
from app.models.audit import AuditLog
from app.models.pipeline_session import PipelineSession
from app.models.project import Project, ProjectMember, ProjectRole
from app.models.user import User
from app.tasks import runner
from scripts.rad_albert.errors import AlbertAuthError, AlbertQuotaExhausted
from tests import test_albert_off_golden_routes as golden
from tests import test_celery_tasks as celery_harness
# Fixtures réutilisées (l'hygiène d'env des goldens est autouse dans ce module).
from tests.test_albert_off_golden_routes import _golden_env_hygiene, golden_app  # noqa: F401
from tests.test_celery_tasks import world  # noqa: F401

# ======================================================================
# Constantes littérales
# ======================================================================
ALBERT_CHAT_MODEL = "albert/ministral-3-8b-instruct-2512"
ALBERT_NOTES_MODEL = "albert/gpt-oss-120b"
ALBERT_TIMEOUT_DEFAULT = 21600
ALBERT_TIMEOUT_CUSTOM = 43210
PIPELINE_TIMEOUT = 1800
UPLOAD_TIMEOUT = 3600
CONFIGURE_URL = "/settings/credentials"
ALBERT_CREDENTIAL = "albert_api_key"
# Défaut web de type slug OpenRouter (contient « / ») : cas qu'aucun golden ne couvre.
OPENROUTER_SLUG_DEFAULT = "google/gemini-2.5-flash"
# Options de soumission Celery qui bornent la durée d'une tâche.
CELERY_TIME_LIMIT_OPTIONS = ("soft_time_limit", "time_limit")

# Clé Albert du serveur (repli admin) : celle posée par l'hygiène des goldens.
SERVER_ALBERT_KEY = golden.FAKE_ALBERT_KEY
MEMBER_ALBERT_ONLY_KEY = "fake-albert-key-member-only-0001"

# Personas ajoutées au harnais des goldens (les 3 personas G7 restent intactes).
EXTRA_PERSONAS = {
    "member_albert": {
        "albert_api_key": MEMBER_ALBERT_ONLY_KEY,
        "zotero_api_key": "fake-zotero-albert-0001",
        "zotero_user_id": "fake-zotero-user-albert-0001",
    },
    "member_mistral": {"mistral_api_key": "fake-mistral-only-0001"},
    "member_openrouter": {"openrouter_api_key": "fake-openrouter-member-only-0001"},
}

# Projet Albert : propriétaire member_albert, les autres personas non-admin
# collaborateurs (admin : accès de droit). Ses sessions servent aux requêtes ON.
ALBERT_PROJECT_OWNER = "member_albert"
ALBERT_PROJECT_MEMBERS = ("member_keys", "member_nokeys", "member_mistral", "member_openrouter")
# Session complète du projet Albert (mêmes artefacts que gsess-full).
SESSION = "gsess-alb-full"
# Session enregistrée du projet de member_nokeys, sans aucun membre : étrangère
# pour member_albert et member_keys.
FOREIGN_SESSION = "gsess-full"

# Sessions du projet Albert : nom -> artefacts présents (output.csv toujours écrit).
UPLOAD_SESSIONS = {
    SESSION: ("plain", "dense", "sparse"),
    "gsess-alb-chunks": ("plain",),
    "gsess-alb-dense": ("plain", "dense"),
    "gsess albert space": ("plain", "dense", "sparse"),
    "gsess-alb-empty": (),
}
ARTEFACT_FILES = {
    "plain": "output_chunks.json",
    "dense": "output_chunks_with_embeddings.json",
    "sparse": "output_chunks_with_embeddings_sparse.json",
}
STUB_SCRIPTS = ("rad_dataframe.py", "rad_chunk.py", "rad_vectordb.py", "rad_clustering.py")
MANIFEST_NAME = "albert_manifest.jsonl"
MANIFEST_ROWS = (
    {"document": "ragpy:ITEMA001:a.pdf", "slice": 0, "count": 2},
    {"document": "ragpy:ITEMB002:b.pdf", "slice": 0, "count": 1},
)

# Noms d'env dont on garde une étiquette (jamais la valeur).
ENV_FACT_NAMES = (
    "OPENAI_API_KEY", "OPENROUTER_API_KEY", "MISTRAL_API_KEY", "ALBERT_API_KEY",
    "PINECONE_API_KEY", "WEAVIATE_URL", "QDRANT_URL",
)
ABSENT = "<absent>"

# Champs de formulaire propres à Albert, ajoutés aux requêtes G7 rejouées.
ALBERT_UPLOAD_FIELDS = {
    "albert_collection_id": "4242",
    "albert_collection_name": "Corpus RAGpy",
    "albert_create_collection": "true",
    "albert_gdpr_ack": "true",
}
REPLAYED_GOLDENS = (
    ("g7_process_dataframe.json", {}),
    ("g7_process_dataframe_sse.json", {}),
    ("g7_initial_text_chunking.json", {}),
    ("g7_initial_text_chunking_sse.json", {}),
    ("g7_dense_embedding_generation.json", {"embedding_provider": "openai"}),
    ("g7_dense_embedding_generation_sse.json", {"embedding_provider": "openai"}),
    ("g7_sparse_embedding_generation.json", {}),
    ("g7_sparse_embedding_generation_sse.json", {}),
    ("g7_upload_db.json", ALBERT_UPLOAD_FIELDS),
    ("g7_generate_zotero_notes_sse.json", {}),
)
# Cas du golden upload_db qui visent les 3 bases historiques.
HISTORICAL_DB_CHOICES = ("pinecone", "weaviate", "qdrant")

REAL_MANIFEST_BASE = os.path.join(golden._RAGPY_ROOT, "data", "albert_manifests")


# ======================================================================
# Étiquettes des clés factices (jamais de valeur dans une assertion)
# ======================================================================
def _key_labels():
    """Associe chaque valeur factice connue à une étiquette lisible."""
    labels = {}
    for name, value in golden.FAKE_SERVER_ENV.items():
        labels.setdefault(value, "server." + name)
    labels.setdefault(SERVER_ALBERT_KEY, "server.ALBERT_API_KEY")
    for name, value in celery_harness.FAKE_WORKER_ENV.items():
        labels.setdefault(value, "worker." + name)
    for persona, creds in golden.FAKE_DB_CREDENTIALS.items():
        for key, value in creds.items():
            labels.setdefault(value, f"{persona}.{key}")
    for persona, creds in EXTRA_PERSONAS.items():
        for key, value in creds.items():
            labels.setdefault(value, f"{persona}.{key}")
    return labels


KEY_LABELS = _key_labels()
ALL_FAKE_SECRETS = tuple(sorted(KEY_LABELS))


def _label(value):
    """Étiquette d'une valeur d'identifiant (``None`` absente, ``<autre>`` inconnue)."""
    if value is None:
        return None
    if value == "":
        return ""
    return KEY_LABELS.get(value, "<autre>")


def _env_facts(env):
    """Réduit un env de sous-processus à des étiquettes, des noms et des booléens."""
    if env is None:
        return None
    facts = {name: _label(env.get(name)) for name in ENV_FACT_NAMES}
    facts["EMBEDDING_PROVIDER"] = env.get("EMBEDDING_PROVIDER", ABSENT)
    deny = env.get("RAGPY_DOTENV_DENY")
    facts["deny"] = None if deny is None else [n for n in deny.split(",") if n]
    return facts


def _contains_secret(text):
    """Vrai si une valeur factice d'identifiant figure dans ``text``."""
    return any(value in text for value in ALL_FAKE_SECRETS if len(value) >= 12)


# ======================================================================
# Enregistreurs
# ======================================================================
def _argv_value(cmd, flag):
    """Valeur de ``flag`` dans ``cmd`` (``flag valeur`` ou ``flag=valeur`` ; None si absent)."""
    cmd = [str(c) for c in cmd]
    value = golden._argv_value(cmd, flag)
    if value is not None:
        return value
    prefix = flag + "="
    for item in cmd:
        if item.startswith(prefix):
            return item[len(prefix):]
    return None


def _has_flag(cmd, flag):
    """Vrai si ``flag`` figure dans ``cmd`` (seul ou sous la forme ``flag=valeur``)."""
    return any(str(c) == flag or str(c).startswith(flag + "=") for c in cmd)


def _is_albert_upload(cmd):
    """Vrai pour un argv ``rad_vectordb.py --db albert``."""
    return golden._script_name(cmd) == "rad_vectordb.py" and _argv_value(cmd, "--db") == "albert"


def _write_manifest(path):
    """Écrit un manifeste d'envoi Albert factice (une ligne JSON par tranche)."""
    with open(path, "w", encoding="utf-8") as fh:
        for row in MANIFEST_ROWS:
            fh.write(json.dumps(row) + "\n")


def _albert_stdout(cmd, *, existing=2, with_journal=True):
    """Sortie de ``rad_vectordb.py --db albert`` : bloc Result et lignes propres à Albert."""
    input_file = _argv_value(cmd, "--input") or ""
    directory = os.path.dirname(input_file)
    manifest = os.path.join(directory, MANIFEST_NAME)
    lines = [
        "=== rad_vectordb.py ===",
        f"Input file: {input_file}",
        "Target DB: albert",
        "",
        "=== Result ===",
        "Status: success",
        "Message: 3 chunks envoyés",
        "Inserted: 3",
        "Skipped (dedup): 1",
    ]
    if with_journal:
        lines.append(f"Dedup journal: {os.path.join(directory, 'dedup_journal.jsonl')}")
    lines.append(f"Skipped (existing): {existing}")
    lines.append(f"Albert manifest: {manifest}")
    return "\n".join(lines) + "\n"


class _RecordingSemaphore:
    """Sémaphore asynchrone factice : compte les acquisitions et sait s'il est tenu."""

    def __init__(self):
        """Aucune acquisition au départ."""
        self.acquired = 0
        self.held = 0

    async def __aenter__(self):
        """Note l'acquisition."""
        self.acquired += 1
        self.held += 1
        return self

    async def __aexit__(self, *exc):
        """Note la libération."""
        self.held -= 1
        return False

    async def acquire(self):
        """Acquisition explicite (même comptage que ``async with``)."""
        self.acquired += 1
        self.held += 1
        return True

    def release(self):
        """Libération explicite."""
        self.held -= 1

    def locked(self):
        """Jamais saturé."""
        return False


class _Capture:
    """Lanceurs et aides enregistreurs : argv, délai, env réduit à des étiquettes.

    ``launches`` reçoit un dict par lancement (``tracked`` ou ``sse``) ;
    ``helpers`` un dict par appel d'aide en processus (notes, citations).
    ``returncode``, ``stdout_for``, ``note_error``, ``citation_error`` et
    ``filter_error`` règlent la réponse des faux.
    """

    def __init__(self):
        """Journaux vides, lancements réussis."""
        self.launches = []
        self.helpers = []
        self.returncode = 0
        self.stdout_for = None
        self.note_error = None
        self.citation_error = None
        self.filter_error = None
        self.semaphore = _RecordingSemaphore()
        self.bg_errors = []
        self.bg_messages = []

    def take(self):
        """Renvoie ``(launches, helpers)`` puis vide les journaux."""
        launches, helpers = self.launches, self.helpers
        self.launches, self.helpers = [], []
        return launches, helpers

    # --- lanceurs ---------------------------------------------------------
    def _launch(self, kind, cmd, session_folder, timeout, env):
        """Enregistre un lancement puis simule les fichiers produits."""
        cmd = [str(c) for c in cmd]
        self.launches.append({
            "kind": kind,
            "script": golden._script_name(cmd),
            "argv": cmd,
            "session_folder": session_folder,
            "timeout": timeout,
            "env": _env_facts(env),
        })
        golden._simulate_script_outputs(cmd)
        if _is_albert_upload(cmd):
            directory = os.path.dirname(_argv_value(cmd, "--input") or "")
            if directory and os.path.isdir(directory):
                _write_manifest(os.path.join(directory, MANIFEST_NAME))

    def _stdout(self, cmd):
        """Sortie simulée du script lancé."""
        if self.stdout_for is not None:
            custom = self.stdout_for(cmd)
            if custom is not None:
                return custom
        if _is_albert_upload(cmd):
            return _albert_stdout(cmd)
        return golden._fake_stdout(cmd)

    async def tracked(self, *args, **kwargs):
        """Remplace ``processing.run_tracked_subprocess``."""
        bound = golden._bind(("cmd", "session_folder", "timeout", "env"), args, kwargs)
        cmd = list(bound.get("cmd") or [])
        self._launch("tracked", cmd, bound.get("session_folder"), bound.get("timeout", "<default>"), bound.get("env"))
        return subprocess.CompletedProcess(args=cmd, returncode=self.returncode, stdout=self._stdout(cmd), stderr="")

    async def sse(self, cmd, progress_parser, **kwargs):
        """Remplace ``sse_helpers.run_subprocess_with_sse``."""
        cmd = list(cmd)
        self._launch("sse", cmd, kwargs.get("session_folder"), kwargs.get("timeout", "<default>"), kwargs.get("env"))
        for line in golden.SAMPLE_SUBPROCESS_LINES:
            event = progress_parser(line)
            if event:
                yield f"data: {json.dumps(event)}\n\n"
        yield golden.SSE_COMPLETE_EVENT

    # --- aides de notes -----------------------------------------------------
    def _helper(self, fn, kwargs, **extra):
        """Enregistre un appel d'aide (modèle, étiquettes des clés reçues)."""
        entry = {
            "fn": fn,
            "model": kwargs.get("model", ABSENT),
            "albert": _label(kwargs.get("albert_api_key")),
            "has_albert_kwarg": "albert_api_key" in kwargs,
            "openai": _label(kwargs.get("openai_api_key")),
            "openrouter": _label(kwargs.get("openrouter_api_key")),
        }
        entry.update(extra)
        self.helpers.append(entry)

    async def note_html(self, *args, **kwargs):
        """Remplace ``build_note_html_async``."""
        self._helper("build_note_html_async", golden._bind(("metadata", "text_content"), args, kwargs))
        if self.note_error is not None:
            raise self.note_error
        return ("<!-- ragpy-note:albert-test -->", "<p>Note</p>")

    async def abstract(self, *args, **kwargs):
        """Remplace ``build_abstract_text_async``."""
        self._helper("build_abstract_text_async", golden._bind(("metadata", "text_content"), args, kwargs))
        if self.note_error is not None:
            raise self.note_error
        return "Résumé."

    async def book_note(self, *args, **kwargs):
        """Remplace ``build_book_note_async``."""
        self._helper("build_book_note_async", golden._bind(("metadata", "text_content"), args, kwargs))
        if self.note_error is not None:
            raise self.note_error
        return ("<!-- ragpy-book:albert-test -->", "<p>Fiche</p>")

    # --- aides de citations -------------------------------------------------
    async def process_citations(self, *args, **kwargs):
        """Remplace ``process_citations_parallel`` (générateur asynchrone)."""
        bound = golden._bind(golden.HELPER_KWARG_ALLOWLIST["process_citations_parallel"], args, kwargs)
        config = bound.get("config") or {}
        self._helper("process_citations_parallel", dict(bound, model=config.get("model", ABSENT)))
        citations = list(bound.get("citations") or [])
        yield ("init", {"total": len(citations), "batch_size": bound.get("batch_size")})
        if self.citation_error is not None:
            raise self.citation_error
        for idx, citation in enumerate(citations, start=1):
            citation_dict = citation.model_dump(mode="json")
            yield ("progress", {
                "current": idx,
                "total": len(citations),
                "status": "skipped",
                "title": citation_dict.get("title", "")[:50],
                "filter_result": None,
                "citation": citation_dict,
                "web_source": "article_url",
                "error_message": None,
            })
        yield ("complete", {"relevant": 0, "skipped": len(citations), "errors": 0})

    async def filter_citation(self, *args, **kwargs):
        """Remplace ``filter_citation_with_llm`` : note si un sémaphore externe est tenu."""
        global_sem = getattr(llm_note_generator, "_llm_semaphore", None)
        real_held = bool(global_sem is not None and getattr(global_sem, "_value", 1) < int(
            os.getenv("MAX_CONCURRENT_LLM_CALLS", "5")))
        self._helper(
            "filter_citation_with_llm", kwargs,
            outer_semaphore_held=self.semaphore.held > 0 or real_held,
        )
        if self.filter_error is not None:
            raise self.filter_error
        return "NA"

    # --- tâches de fond -----------------------------------------------------
    async def run_background_now(self, task_id, coroutine, db):
        """Remplace ``background_task_manager.start_task`` : exécute la coroutine tout de suite."""
        try:
            await coroutine
        except Exception as exc:  # la tâche relance après avoir noté l'erreur
            self.bg_errors.append(type(exc).__name__)

    async def update_progress(self, *args, **kwargs):
        """Remplace ``background_task_manager.update_progress`` (message seulement)."""
        self.bg_messages.append(kwargs.get("message"))


# ======================================================================
# Fixtures
# ======================================================================
def _write_session(folder, kinds):
    """Crée ``folder`` avec les artefacts ``kinds`` (plain, dense, sparse) et un ``output.csv``."""
    os.makedirs(folder, exist_ok=True)
    golden._write_output_csv(os.path.join(folder, "output.csv"))
    for kind in kinds:
        golden._write_json(os.path.join(folder, ARTEFACT_FILES[kind]), golden._chunks(kind))


def _register_session(env, name):
    """Enregistre le dossier ``name`` comme session du projet Albert (accès contrôlé par projet)."""
    row = PipelineSession(
        project_id=env.albert_project.id, session_folder=name, source_type="zip",
        created_at=golden.FIXED_CREATED_AT, updated_at=golden.FIXED_CREATED_AT,
    )
    env.db.add(row)
    env.db.commit()


def _create_albert_project(env):
    """Projet Albert : propriétaire ``member_albert``, collaborateurs ``ALBERT_PROJECT_MEMBERS``."""
    project = Project(
        name="Albert Project",
        description="Albert description",
        owner_id=env.users[ALBERT_PROJECT_OWNER].id,
        session_folder=SESSION,
        created_at=golden.FIXED_CREATED_AT,
        updated_at=golden.FIXED_CREATED_AT,
    )
    env.db.add(project)
    env.db.commit()
    env.db.refresh(project)
    for persona in ALBERT_PROJECT_MEMBERS:
        env.db.add(ProjectMember(
            project_id=project.id, user_id=env.users[persona].id, role=ProjectRole.COLLABORATOR.value,
            created_at=golden.FIXED_CREATED_AT, updated_at=golden.FIXED_CREATED_AT,
        ))
    env.db.commit()
    env.db.refresh(project)
    return project


def _redirect_manifest_constants(monkeypatch, root):
    """Redirige vers ``root`` toute constante de chemin de manifeste du module de traitement."""
    for name in dir(processing_routes):
        if "MANIFEST" not in name.upper():
            continue
        value = getattr(processing_routes, name)
        if isinstance(value, str) and value.startswith(golden._RAGPY_ROOT):
            monkeypatch.setattr(processing_routes, name, str(root) + value[len(golden._RAGPY_ROOT):])


@pytest.fixture
def albert_env(golden_app, monkeypatch, tmp_path):
    """Harnais des goldens complété : 3 personas, projet Albert et ses sessions, enregistreurs complets."""
    env = golden_app
    for persona, creds in EXTRA_PERSONAS.items():
        user = User(
            email=f"{persona.replace('_', '.')}@albert-routes.test",
            hashed_password="x",
            roles=["USER"],
            is_active=True,
            is_verified=True,
            api_credentials=encrypt_credentials(creds),
            created_at=golden.FIXED_CREATED_AT,
            updated_at=golden.FIXED_CREATED_AT,
        )
        env.db.add(user)
        env.db.commit()
        env.db.refresh(user)
        env.users[persona] = user
        env.headers[persona] = {"Authorization": "Bearer " + create_access_token(subject=str(user.id))}
        fresh = env.uploads / golden._fresh(persona)
        fresh.mkdir()
        golden._write_zotero_json(str(fresh))
    env.albert_project = _create_albert_project(env)
    for name, kinds in UPLOAD_SESSIONS.items():
        _write_session(str(env.uploads / name), kinds)
        _register_session(env, name)

    # Racine RAGpy de test : scripts factices (contrôles d'existence) et data/.
    root = tmp_path / "ragpy_root"
    (root / "scripts").mkdir(parents=True)
    for script in STUB_SCRIPTS:
        (root / "scripts" / script).write_text("# stub de test\n", encoding="utf-8")
    monkeypatch.setattr(processing_routes, "RAGPY_DIR", str(root))
    monkeypatch.setattr(core_config, "RAGPY_DIR", str(root))
    _redirect_manifest_constants(monkeypatch, root)

    capture = _Capture()
    monkeypatch.setattr(processing_routes, "run_tracked_subprocess", capture.tracked)
    monkeypatch.setattr(sse_helpers, "run_subprocess_with_sse", capture.sse)
    monkeypatch.setattr(processing_routes, "run_subprocess_with_sse", capture.sse, raising=False)
    monkeypatch.setattr(llm_note_generator, "build_note_html_async", capture.note_html)
    monkeypatch.setattr(llm_note_generator, "build_abstract_text_async", capture.abstract)
    monkeypatch.setattr(book_note_generator, "build_book_note_async", capture.book_note)
    monkeypatch.setattr(citation_routes, "process_citations_parallel", capture.process_citations)
    monkeypatch.setattr(citation_routes, "filter_citation_with_llm", capture.filter_citation)
    monkeypatch.setattr(citation_routes, "get_llm_semaphore", lambda: capture.semaphore)
    monkeypatch.setattr(llm_note_generator, "get_llm_semaphore", lambda: capture.semaphore)
    monkeypatch.setattr(citation_routes.background_task_manager, "start_task", capture.run_background_now)
    monkeypatch.setattr(citation_routes.background_task_manager, "update_progress", capture.update_progress)

    # Toute session ouverte hors dépendance (SessionLocal) vise la base de test.
    factory = sessionmaker(autocommit=False, autoflush=False, bind=env.db.get_bind())
    monkeypatch.setattr(db_session_module, "SessionLocal", factory)
    monkeypatch.setattr(background_task_module, "SessionLocal", factory)
    for module in (processing_routes, citation_routes):
        monkeypatch.setattr(module, "SessionLocal", factory, raising=False)

    env.capture = capture
    env.ragpy_root = root
    env.session_factory = factory
    yield env


@pytest.fixture
def albert_on(monkeypatch):
    """Albert activé (interrupteur maître), délai de sous-processus par défaut."""
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    monkeypatch.delenv("ALBERT_SUBPROCESS_TIMEOUT", raising=False)


# ======================================================================
# Aides de requête
# ======================================================================
def _parse_events(text):
    """Événements SSE décodés (``<non-json>`` pour un événement illisible)."""
    out = []
    for raw in golden._sse_events(text):
        try:
            out.append(json.loads(raw))
        except ValueError:
            out.append({"type": "<non-json>", "raw": raw[:200]})
    return out


def _post(env, persona, url, form, *, files=None):
    """POST de formulaire en tant que ``persona`` : (réponse, exceptions serveur)."""
    data = {k: v for k, v in form.items() if v is not None}
    resp = env.client.post(url, data=data, files=files, headers=env.headers[persona])
    return resp, golden._take_server_errors(env)


def _post_sse(env, persona, url, form):
    """POST d'une route SSE : (statut, événements décodés, exceptions serveur)."""
    resp, errors = _post(env, persona, url, form)
    return resp.status_code, _parse_events(resp.text), errors


def _json(resp):
    """Corps JSON d'une réponse (``{"text": …}`` sinon)."""
    return golden._body(resp)


def _single_error_event(events):
    """Vrai si ``events`` est un unique événement ``{"type": "error"}``."""
    return len(events) == 1 and events[0].get("type") == "error"


def _credential_error_event(event, credential):
    """Vrai si ``event`` est un événement d'erreur qui désigne ``credential``."""
    return (
        event.get("type") == "error"
        and event.get("credential_required") == credential
        and isinstance(event.get("message"), str)
        and bool(event.get("message"))
    )


def _upload_form(path, **extra):
    """Formulaire ``/upload_db`` albert (acquittement RGPD compris sauf surcharge)."""
    form = {"path": path, "db_choice": "albert", "albert_collection_name": "Corpus RAGpy", "albert_gdpr_ack": "true"}
    form.update(extra)
    return form


def _replay_golden(env, name, extra):
    """Rejoue les cas du golden ``name`` avec les champs ``extra`` ; renvoie les écarts.

    Chaque cas est rejoué dans l'ordre du golden (même évolution des dossiers
    de session) ; seul le formulaire diffère. Le résultat normalisé (statut,
    corps ou événements, exceptions, appels) doit être identique.
    """
    doc = golden._read_golden(name)
    mismatches = []
    for case in doc["cases"]:
        request = case["request"]
        form = dict(request["form"])
        form.update(extra)
        sse = "events" in case
        out = env.norm(golden._post_case(env, case["case"], case["persona"], request["path"], form, sse=sse))
        expected = {k: v for k, v in case.items() if k != "request"}
        observed = {k: v for k, v in out.items() if k != "request"}
        if observed != expected:
            mismatches.append(
                f"{name}:{case['case']}\n  expected: {json.dumps(expected, ensure_ascii=False)[:3000]}"
                f"\n  observed: {json.dumps(observed, ensure_ascii=False)[:3000]}"
            )
    return mismatches


def _env_names_recorder(env, monkeypatch):
    """Enveloppe les faux lanceurs des goldens pour noter les noms d'env transmis."""
    names = []
    tracked = env.recorder.fake_tracked
    sse = env.recorder.fake_sse

    async def _tracked(*args, **kwargs):
        """Note les noms d'env puis délègue au faux des goldens."""
        bound = golden._bind(("cmd", "session_folder", "timeout", "env"), args, kwargs)
        names.append(sorted(bound.get("env") or {}))
        return await tracked(*args, **kwargs)

    async def _sse(cmd, progress_parser, **kwargs):
        """Note les noms d'env puis délègue au faux des goldens."""
        names.append(sorted(kwargs.get("env") or {}))
        async for event in sse(cmd, progress_parser, **kwargs):
            yield event

    monkeypatch.setattr(processing_routes, "run_tracked_subprocess", _tracked)
    monkeypatch.setattr(sse_helpers, "run_subprocess_with_sse", _sse)
    monkeypatch.setattr(processing_routes, "run_subprocess_with_sse", _sse, raising=False)
    return names


# ======================================================================
# Albert OFF : identité avec G7, préfixe albert/ refusé
# ======================================================================
@pytest.mark.parametrize("switch", ["0", None, "off"])
@pytest.mark.parametrize("name, extra", REPLAYED_GOLDENS, ids=[n for n, _ in REPLAYED_GOLDENS])
def test_off_payloads_equal_golden(golden_app, monkeypatch, name, extra, switch):
    """G7 étendu : champs Albert présents, clé Albert en base et dans l'env, Albert OFF."""
    if switch is None:
        monkeypatch.delenv("ALBERT_ENABLED", raising=False)
    else:
        monkeypatch.setenv("ALBERT_ENABLED", switch)
    env_names = _env_names_recorder(golden_app, monkeypatch)
    mismatches = _replay_golden(golden_app, name, extra)
    assert not mismatches, "\n\n".join(mismatches)
    # L'env OFF ne gagne ni EMBEDDING_PROVIDER ni aucune variable Albert nouvelle.
    for names in env_names:
        assert "EMBEDDING_PROVIDER" not in names
        assert [n for n in names if n.startswith("ALBERT_") and n not in ("ALBERT_API_KEY", "ALBERT_ENABLED")] == []


@pytest.mark.parametrize("switch", ["0", "1"])
def test_sse_missing_key_emits_json_twin_error_event(golden_app, monkeypatch, switch):
    """Tâche 0 : plus de NameError ; unique événement identique au 403 de la route JSON jumelle."""
    monkeypatch.setenv("ALBERT_ENABLED", switch)
    env = golden_app
    doc = golden._read_golden(golden.KNOWN_DEFECT_GOLDEN)
    failures = []
    for gcase in doc["cases"]:
        request = gcase["request"]
        out = env.norm(golden._post_case(env, gcase["case"], gcase["persona"], request["path"], request["form"], sse=True))
        outcome = golden._parsed_events(golden._sse_outcome(out))
        if outcome != gcase["accepted_outcomes"]["fixed_event"]:
            failures.append(f"{gcase['case']}: {json.dumps(golden._sse_outcome(out), ensure_ascii=False)}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("switch", ["0", None])
def test_albert_prefix_disabled_400_no_subprocess(albert_env, monkeypatch, switch):
    """Albert OFF : ``albert/…`` → 400 JSON ou unique événement d'erreur, sans lancement ni aide."""
    if switch is None:
        monkeypatch.delenv("ALBERT_ENABLED", raising=False)
    else:
        monkeypatch.setenv("ALBERT_ENABLED", switch)
    env = albert_env
    for model in (ALBERT_CHAT_MODEL, "ALBERT/Ministral-3-8b-instruct-2512", "Albert/gpt-oss-120b"):
        for persona in ("admin", "member_albert"):
            resp, errors = _post(env, persona, "/initial_text_chunking", {"path": SESSION, "model": model})
            assert resp.status_code == 400, (model, persona, resp.status_code)
            assert "albert" in json.dumps(_json(resp)).lower()
            assert errors == []
            status, events, errors = _post_sse(env, persona, "/initial_text_chunking_sse",
                                               {"path": SESSION, "model": model})
            assert status == 200 and errors == []
            assert _single_error_event(events), events
            launches, helpers = env.capture.take()
            assert launches == [] and helpers == [], model

    # Notes : événement d'erreur, aucune aide appelée.
    status, events, errors = _post_sse(env, "admin", "/generate_zotero_notes_sse",
                                       {"session": SESSION, "note_mode": "extended", "model": ALBERT_NOTES_MODEL})
    assert status == 200 and errors == []
    assert [e.get("type") for e in events] == ["error"], events
    assert env.capture.take() == ([], [])

    # Citations (admin : toutes les clés serveur, dont OpenRouter et Zotero) : SSE → événement
    # d'erreur Albert ; tâche de fond → 400 ; aucune aide appelée, aucune tâche lancée.
    for route in ("filter_citations_sse", "batch_import_citations_sse"):
        pid, sid = golden._make_citation_session(env, "admin", ALBERT_CHAT_MODEL, n_citations=1)
        status, events, errors = _post_sse(env, "admin", f"/api/projects/{pid}/{route}",
                                           {"session_id": sid, "batch_size": 5})
        assert status == 200 and errors == []
        assert events and events[-1].get("type") == "error", (route, events)
        assert "albert" in str(events[-1].get("message", "")).lower(), (route, events)
        assert not [e for e in events if e.get("type") in ("complete", "progress", "filter_progress")], route
        assert env.capture.take() == ([], []), route
    pid, sid = golden._make_citation_session(env, "admin", ALBERT_CHAT_MODEL, n_citations=1)
    resp, errors = _post(env, "admin", f"/api/projects/{pid}/filter_citations_bg", {"session_id": sid, "batch_size": 5})
    assert resp.status_code == 400 and errors == []
    assert env.capture.take() == ([], [])
    assert env.capture.bg_messages == []


def _pop_json_bytes():
    """Export Publish or Perish minimal (2 citations)."""
    citations = [
        {"uid": f"POP:{i:04d}", "title": f"Citation {i}", "authors": ["A. Author"], "year": 2024,
         "article_url": f"https://example.org/pop-{i}", "doi": f"10.0000/pop.{i}"}
        for i in (1, 2)
    ]
    return json.dumps(citations).encode("utf-8")


def _config_files(uploads):
    """Chemins des ``config.json`` des sessions de citations (``pop_*``)."""
    return sorted(glob.glob(os.path.join(str(uploads), "pop_*", "config.json")))


def _pop_sessions(env):
    """Nombre de sessions ``pop_json`` en base."""
    env.db.expire_all()
    return env.db.query(PipelineSession).filter(PipelineSession.source_type == "pop_json").count()


def test_upload_pop_json_albert_off_400(albert_env, monkeypatch):
    """``upload_pop_json`` valide le modèle par le résolveur avant d'écrire ``config.json``."""
    env = albert_env
    url = f"/api/projects/{env.project.id}/upload_pop_json"

    def _upload(model):
        """Téléverse l'export avec ``model`` en tant que propriétaire du projet."""
        return _post(env, "member_nokeys", url, {"collection_name": "Coll", "model": model},
                     files={"json_file": ("biblio.json", _pop_json_bytes(), "application/json")})

    for model in (ALBERT_CHAT_MODEL, "ALBERT/gpt-oss-120b"):
        resp, errors = _upload(model)
        assert resp.status_code == 400, (model, resp.status_code)
        assert errors == []
        assert _config_files(env.uploads) == []
        assert _pop_sessions(env) == 0

    # Contrôle OFF : un modèle historique est accepté comme aujourd'hui.
    resp, errors = _upload("gpt-4o-mini")
    assert resp.status_code == 200 and errors == []
    assert len(_config_files(env.uploads)) == 1

    # ON : albert/<id> accepté et conservé tel quel ; « albert/ » vide refusé.
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    resp, errors = _upload(ALBERT_CHAT_MODEL)
    assert resp.status_code == 200 and errors == []
    folder = os.path.join(str(env.uploads), resp.json()["session_folder"])
    with open(os.path.join(folder, "config.json"), encoding="utf-8") as fh:
        assert json.load(fh)["model"] == ALBERT_CHAT_MODEL
    before = _config_files(env.uploads)
    resp, errors = _upload("albert/")
    assert resp.status_code == 400 and errors == []
    assert _config_files(env.uploads) == before


# ======================================================================
# Arrêt des scripts : authentification et propriété de la session (report W1)
# ======================================================================
def test_stop_all_scripts_requires_auth_and_session_access(albert_env, monkeypatch):
    """``/stop_all_scripts`` : 401 sans jeton ; session enregistrée d'un projet étranger → 403."""
    env = albert_env
    stopped = []

    def _fake_stop_session(session):
        """Note l'arrêt demandé (aucun signal envoyé)."""
        stopped.append(session)
        return {"success": True, "stopped": 0}

    monkeypatch.setattr(processing_routes.process_manager, "stop_session", _fake_stop_session)

    resp = env.client.post("/stop_all_scripts", data={"session": "gsess-full"})
    assert resp.status_code == 401
    assert stopped == []

    # gsess-full appartient au projet de member_nokeys.
    resp, errors = _post(env, "member_keys", "/stop_all_scripts", {"session": "gsess-full"})
    assert resp.status_code == 403 and errors == []
    assert stopped == []
    for persona in ("member_nokeys", "admin"):
        resp, errors = _post(env, persona, "/stop_all_scripts", {"session": "gsess-full"})
        assert resp.status_code == 200 and errors == [], persona
    assert stopped == ["gsess-full", "gsess-full"]

    # Dossier non enregistré : comportement historique ; dossier absent : 404.
    resp, errors = _post(env, "member_keys", "/stop_all_scripts", {"session": "gsess-empty"})
    assert resp.status_code == 200 and stopped[-1] == "gsess-empty"
    resp, errors = _post(env, "member_keys", "/stop_all_scripts", {"session": "gsess-missing"})
    assert resp.status_code == 404
    assert env.capture.take() == ([], [])


# ======================================================================
# Albert ON : délais
# ======================================================================
def _fresh_ocr_session(env, name):
    """Nouvelle session du projet Albert ne contenant qu'un export Zotero."""
    folder = env.uploads / name
    folder.mkdir()
    golden._write_zotero_json(str(folder))
    _register_session(env, name)
    return name


@pytest.mark.parametrize("custom", [None, str(ALBERT_TIMEOUT_CUSTOM)])
def test_albert_timeout_only_when_selected(albert_env, albert_on, monkeypatch, custom):
    """Délai Albert seulement pour ``albert/…``, ``embedding_provider=albert``, ``db_choice=albert``, OCR Albert actif."""
    env = albert_env
    expected_albert = ALBERT_TIMEOUT_DEFAULT
    if custom is not None:
        monkeypatch.setenv("ALBERT_SUBPROCESS_TIMEOUT", custom)
        expected_albert = int(custom)

    cases = [
        ("chunk albert", "member_albert", "/initial_text_chunking", {"path": SESSION, "model": ALBERT_CHAT_MODEL},
         expected_albert),
        ("chunk albert sse", "member_albert", "/initial_text_chunking_sse",
         {"path": SESSION, "model": ALBERT_CHAT_MODEL}, expected_albert),
        ("chunk openai", "admin", "/initial_text_chunking", {"path": SESSION, "model": "gpt-4o-mini"},
         PIPELINE_TIMEOUT),
        ("chunk openai sse", "admin", "/initial_text_chunking_sse", {"path": SESSION}, PIPELINE_TIMEOUT),
        ("dense albert", "member_albert", "/dense_embedding_generation",
         {"path": SESSION, "embedding_provider": "albert"}, expected_albert),
        ("dense albert sse", "member_albert", "/dense_embedding_generation_sse",
         {"path": SESSION, "embedding_provider": "albert"}, expected_albert),
        ("dense openai", "member_keys", "/dense_embedding_generation",
         {"path": SESSION, "embedding_provider": "openai"}, PIPELINE_TIMEOUT),
        ("dense default", "admin", "/dense_embedding_generation_sse", {"path": SESSION}, PIPELINE_TIMEOUT),
        ("sparse", "member_albert", "/sparse_embedding_generation", {"path": SESSION}, PIPELINE_TIMEOUT),
        ("upload albert", "member_albert", "/upload_db", _upload_form(SESSION), expected_albert),
        ("upload pinecone", "admin", "/upload_db",
         {"path": SESSION, "db_choice": "pinecone", "pinecone_index_name": "idx"}, UPLOAD_TIMEOUT),
    ]
    for label, persona, url, form, expected in cases:
        resp, errors = _post(env, persona, url, form)
        assert resp.status_code == 200 and errors == [], (label, resp.status_code, resp.text[:300])
        launches, _ = env.capture.take()
        assert [launch["timeout"] for launch in launches] == [expected], label

    # OCR : Albert sélectionné seulement quand le maillon est actif et qu'une clé est disponible.
    ocr_cases = [
        ("ocr inactive", "0", "member_keys", "/process_dataframe", PIPELINE_TIMEOUT),
        ("ocr active albert key", "1", "member_albert", "/process_dataframe", expected_albert),
        ("ocr active albert key sse", "1", "member_albert", "/process_dataframe_sse", expected_albert),
        ("ocr active db albert key", "1", "member_keys", "/process_dataframe", expected_albert),
        ("ocr active no albert key", "1", "member_mistral", "/process_dataframe", PIPELINE_TIMEOUT),
    ]
    for idx, (label, ocr, persona, url, expected) in enumerate(ocr_cases):
        monkeypatch.setenv("OCR_ENABLE_ALBERT", ocr)
        path = _fresh_ocr_session(env, f"gsess-ocr-timeout-{idx}")
        resp, errors = _post(env, persona, url, {"path": path})
        assert resp.status_code == 200 and errors == [], (label, resp.status_code, resp.text[:300])
        launches, _ = env.capture.take()
        assert [launch["timeout"] for launch in launches] == [expected], label


# ======================================================================
# Albert ON : chunking
# ======================================================================
def test_chunking_albert_nonadmin_403_even_with_env_key(albert_env, albert_on, monkeypatch):
    """Non-admin sans clé Albert personnelle : 403 ``albert_api_key`` malgré la clé du serveur."""
    env = albert_env
    monkeypatch.setenv("ALBERT_API_KEY", SERVER_ALBERT_KEY)
    for persona in ("member_nokeys", "member_mistral"):
        form = {"path": SESSION, "model": ALBERT_CHAT_MODEL}
        resp, errors = _post(env, persona, "/initial_text_chunking", form)
        assert resp.status_code == 403 and errors == [], persona
        body = resp.json()
        assert set(body) == {"error", "credential_required", "configure_url"}
        assert body["credential_required"] == ALBERT_CREDENTIAL
        assert body["configure_url"] == CONFIGURE_URL
        assert body["error"] == CREDENTIAL_ERROR_MESSAGES[ALBERT_CREDENTIAL]
        status, events, errors = _post_sse(env, persona, "/initial_text_chunking_sse", form)
        assert status == 200 and errors == []
        assert events == [{"type": "error", "message": body["error"], "credential_required": ALBERT_CREDENTIAL}]
        assert env.capture.take() == ([], []), persona


def test_chunking_albert_env_has_user_key(albert_env, albert_on):
    """La clé Albert personnelle est injectée ; aucune clé OpenAI n'est exigée pour ``albert/…``."""
    env = albert_env
    form = {"path": SESSION, "model": ALBERT_CHAT_MODEL}
    for url in ("/initial_text_chunking", "/initial_text_chunking_sse"):
        resp, errors = _post(env, "member_albert", url, form)
        assert resp.status_code == 200 and errors == [], (url, resp.text[:300])
        launches, _ = env.capture.take()
        assert len(launches) == 1, url
        launch = launches[0]
        assert launch["script"] == "rad_chunk.py"
        assert _argv_value(launch["argv"], "--phase") == "initial"
        assert _argv_value(launch["argv"], "--model") == ALBERT_CHAT_MODEL
        facts = launch["env"]
        assert facts["ALBERT_API_KEY"] == "member_albert.albert_api_key", url
        assert facts["OPENAI_API_KEY"] is None and facts["OPENROUTER_API_KEY"] is None
        assert "ALBERT_API_KEY" not in facts["deny"] and "OPENAI_API_KEY" in facts["deny"]

    # Admin : sa clé Albert personnelle (base) prime sur celle du serveur.
    resp, errors = _post(env, "admin", "/initial_text_chunking", form)
    assert resp.status_code == 200 and errors == []
    launches, _ = env.capture.take()
    assert launches[0]["env"]["ALBERT_API_KEY"] == "admin.albert_api_key"
    assert launches[0]["env"]["deny"] is None


# ======================================================================
# Albert ON : OCR
# ======================================================================
def test_ocr_albert_only_user_allowed_when_active(albert_env, albert_on, monkeypatch):
    """Maillon OCR Albert actif : un utilisateur qui n'a qu'une clé Albert peut lancer l'extraction."""
    env = albert_env
    monkeypatch.setenv("OCR_ENABLE_ALBERT", "1")
    for idx, url in enumerate(("/process_dataframe", "/process_dataframe_sse")):
        path = _fresh_ocr_session(env, f"gsess-ocr-albert-{idx}")
        resp, errors = _post(env, "member_albert", url, {"path": path})
        assert resp.status_code == 200 and errors == [], (url, resp.text[:300])
        if url.endswith("_sse"):
            assert not [e for e in _parse_events(resp.text) if e.get("type") == "error"]
        launches, _ = env.capture.take()
        assert len(launches) == 1, url
        facts = launches[0]["env"]
        assert launches[0]["script"] == "rad_dataframe.py"
        assert facts["ALBERT_API_KEY"] == "member_albert.albert_api_key"
        assert facts["MISTRAL_API_KEY"] is None and facts["OPENAI_API_KEY"] is None

    # Maillon inactif (OCR_ENABLE_ALBERT=0, ou Albert OFF) : 403 mistral_api_key inchangé.
    for switch, ocr in (("1", "0"), ("0", "1")):
        monkeypatch.setenv("ALBERT_ENABLED", switch)
        monkeypatch.setenv("OCR_ENABLE_ALBERT", ocr)
        path = _fresh_ocr_session(env, f"gsess-ocr-off-{switch}-{ocr}")
        resp, errors = _post(env, "member_albert", "/process_dataframe", {"path": path})
        assert resp.status_code == 403 and errors == [], (switch, ocr)
        assert resp.json() == {
            "error": CREDENTIAL_ERROR_MESSAGES["mistral_api_key"],
            "credential_required": "mistral_api_key",
            "configure_url": CONFIGURE_URL,
        }
        status, events, errors = _post_sse(env, "member_albert", "/process_dataframe_sse", {"path": path})
        assert status == 200 and errors == []
        assert events == [{
            "type": "error",
            "message": CREDENTIAL_ERROR_MESSAGES["mistral_api_key"],
            "credential_required": "mistral_api_key",
        }]
        assert env.capture.take() == ([], [])

    # Maillon actif mais aucune clé du tout : même 403 qu'aujourd'hui.
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    monkeypatch.setenv("OCR_ENABLE_ALBERT", "1")
    path = _fresh_ocr_session(env, "gsess-ocr-nokeys")
    resp, errors = _post(env, "member_nokeys", "/process_dataframe", {"path": path})
    assert resp.status_code == 403 and resp.json()["credential_required"] == "mistral_api_key"
    assert env.capture.take() == ([], [])


# ======================================================================
# Albert ON : embeddings denses
# ======================================================================
DENSE_URLS = ("/dense_embedding_generation", "/dense_embedding_generation_sse")


def _dense(env, persona, url, **form):
    """Lance la phase dense sur ``gsess-full`` : (statut, événements ou corps, lancements)."""
    resp, errors = _post(env, persona, url, dict({"path": SESSION}, **form))
    assert errors == [], url
    launches, _ = env.capture.take()
    payload = _parse_events(resp.text) if url.endswith("_sse") else _json(resp)
    return resp.status_code, payload, launches


def test_dense_albert_env_and_keys(albert_env, albert_on):
    """``embedding_provider`` du formulaire : clé exigée selon le fournisseur, env posé explicitement."""
    env = albert_env
    for url in DENSE_URLS:
        status, _, launches = _dense(env, "member_albert", url, embedding_provider="albert")
        assert status == 200 and len(launches) == 1, url
        facts = launches[0]["env"]
        assert facts["EMBEDDING_PROVIDER"] == "albert"
        assert facts["ALBERT_API_KEY"] == "member_albert.albert_api_key"
        assert facts["OPENAI_API_KEY"] is None
        assert _argv_value(launches[0]["argv"], "--phase") == "dense"

        status, _, launches = _dense(env, "member_keys", url, embedding_provider="openai")
        assert status == 200 and len(launches) == 1, url
        assert launches[0]["env"]["EMBEDDING_PROVIDER"] == "openai"
        assert launches[0]["env"]["OPENAI_API_KEY"] == "member_keys.openai_api_key"

    # Fournisseur choisi sans la clé correspondante : 403 sur la bonne clé.
    for persona, provider, credential in (("member_albert", "openai", "openai_api_key"),
                                          ("member_nokeys", "albert", ALBERT_CREDENTIAL),
                                          ("member_mistral", "albert", ALBERT_CREDENTIAL)):
        status, body, launches = _dense(env, persona, DENSE_URLS[0], embedding_provider=provider)
        assert status == 403 and body["credential_required"] == credential, (persona, provider)
        status, events, launches_sse = _dense(env, persona, DENSE_URLS[1], embedding_provider=provider)
        assert status == 200 and len(events) == 1 and _credential_error_event(events[0], credential)
        assert events[0]["message"] == body["error"]
        assert launches == [] and launches_sse == []


def test_dense_server_default_albert_requires_albert_key(albert_env, albert_on, monkeypatch):
    """``EMBEDDING_PROVIDER=albert`` côté serveur et champ absent : la clé Albert est exigée."""
    env = albert_env
    monkeypatch.setenv("EMBEDDING_PROVIDER", "albert")
    for url in DENSE_URLS:
        status, _, launches = _dense(env, "member_keys", url)
        assert status == 200 and len(launches) == 1, url
        assert launches[0]["env"]["EMBEDDING_PROVIDER"] == "albert"
        assert launches[0]["env"]["ALBERT_API_KEY"] == "member_keys.albert_api_key"

        status, _, launches = _dense(env, "member_albert", url)
        assert status == 200 and len(launches) == 1, url

    status, body, launches = _dense(env, "member_nokeys", DENSE_URLS[0])
    assert status == 403 and body["credential_required"] == ALBERT_CREDENTIAL and launches == []
    status, events, launches = _dense(env, "member_nokeys", DENSE_URLS[1])
    assert len(events) == 1 and _credential_error_event(events[0], ALBERT_CREDENTIAL) and launches == []

    # Le champ du formulaire prime sur le défaut du serveur.
    status, body, launches = _dense(env, "member_albert", DENSE_URLS[0], embedding_provider="openai")
    assert status == 403 and body["credential_required"] == "openai_api_key" and launches == []
    status, _, launches = _dense(env, "member_keys", DENSE_URLS[0], embedding_provider="openai")
    assert status == 200 and launches[0]["env"]["EMBEDDING_PROVIDER"] == "openai"


def test_dense_invalid_provider_400(albert_env, monkeypatch):
    """ON : fournisseur inconnu → 400 ; OFF : champ ignoré sauf ``albert`` (400), env inchangé."""
    env = albert_env
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    for value in ("cohere", "ALBERT-bge", "openai3072"):
        status, body, launches = _dense(env, "member_keys", DENSE_URLS[0], embedding_provider=value)
        assert status == 400 and launches == [], value
        assert isinstance(body, dict) and (body.get("error") or body.get("detail")), value
        status, events, launches = _dense(env, "member_keys", DENSE_URLS[1], embedding_provider=value)
        assert status == 200 and _single_error_event(events) and launches == [], value

    monkeypatch.setenv("ALBERT_ENABLED", "0")
    for url in DENSE_URLS:
        status, payload, launches = _dense(env, "member_keys", url, embedding_provider="albert")
        if url.endswith("_sse"):
            assert status == 200 and _single_error_event(payload), payload
        else:
            assert status == 400, payload
        assert launches == [], url
        # Toute autre valeur est ignorée : même lancement que sans le champ, env inchangé.
        status, _, launches = _dense(env, "member_keys", url, embedding_provider="cohere")
        assert status == 200 and len(launches) == 1, url
        assert launches[0]["env"]["EMBEDDING_PROVIDER"] == ABSENT
        assert launches[0]["timeout"] == PIPELINE_TIMEOUT


# ======================================================================
# Albert ON : notes Zotero
# ======================================================================
NOTE_HELPERS = {
    "extended": "build_note_html_async",
    "book": "build_book_note_async",
    "short": "build_abstract_text_async",
}


def _notes(env, persona, **form):
    """Lance ``/generate_zotero_notes_sse`` sur la session du projet Albert : (événements, appels d'aide)."""
    status, events, errors = _post_sse(env, persona, "/generate_zotero_notes_sse", dict({"session": SESSION}, **form))
    assert status == 200 and errors == []
    launches, helpers = env.capture.take()
    assert launches == []
    return events, helpers


@pytest.mark.parametrize("mode", sorted(NOTE_HELPERS))
def test_notes_sse_albert_key_forwarded_and_account_error_event(albert_env, albert_on, mode):
    """Notes ``albert/…`` : clé Albert transmise ; erreur de compte → événement d'erreur puis arrêt."""
    env = albert_env
    events, helpers = _notes(env, "member_albert", note_mode=mode, model=ALBERT_NOTES_MODEL)
    assert events[-1].get("type") == "complete", events
    assert [h["fn"] for h in helpers] == [NOTE_HELPERS[mode]] * 2
    for helper in helpers:
        assert helper["model"] == ALBERT_NOTES_MODEL
        assert helper["albert"] == "member_albert.albert_api_key"

    # Sans clé Albert personnelle (clé serveur présente) : événement credential_required, aucune génération.
    for persona in ("member_nokeys", "member_mistral"):
        events, helpers = _notes(env, persona, note_mode=mode, model=ALBERT_NOTES_MODEL)
        assert len(events) == 1 and _credential_error_event(events[0], ALBERT_CREDENTIAL), (persona, events)
        assert helpers == [], persona

    # Erreur de compte au 1er document : un seul appel, événement d'erreur, pas de « complete ».
    for error in (AlbertAuthError(reason="invalid_key", status=401), AlbertQuotaExhausted(status=429)):
        env.capture.note_error = error
        events, helpers = _notes(env, "member_albert", note_mode=mode, model=ALBERT_NOTES_MODEL)
        assert len(helpers) == 1, (type(error).__name__, helpers)
        assert _credential_error_event(events[-1], ALBERT_CREDENTIAL), events
        assert not [e for e in events if e.get("type") == "complete"]
        assert not [e for e in events if e.get("type") == "progress"], events
        assert not _contains_secret(json.dumps(events))
    env.capture.note_error = None


def test_notes_empty_model_uses_effective_default_for_credential(albert_env, albert_on, monkeypatch):
    """Albert ON, modèle vide : la clé exigée est celle du défaut effectif (``resolve_default_llm_model``).

    Albert OFF, la règle historique s'applique (voir
    ``test_notes_off_empty_model_keeps_historical_openai_check``).
    """
    env = albert_env
    # Défaut OpenRouter : member_keys n'a qu'une clé OpenAI → erreur openrouter_api_key, aucune aide.
    monkeypatch.setenv("OPENROUTER_DEFAULT_MODEL", OPENROUTER_SLUG_DEFAULT)
    events, helpers = _notes(env, "member_keys", note_mode="extended")
    assert len(events) == 1 and _credential_error_event(events[0], "openrouter_api_key"), events
    assert helpers == []

    # Défaut Albert : clé Albert exigée et transmise, jamais la clé OpenAI.
    monkeypatch.setenv("OPENROUTER_DEFAULT_MODEL", ALBERT_NOTES_MODEL)
    events, helpers = _notes(env, "member_albert", note_mode="extended", model="")
    assert events[-1].get("type") == "complete", events
    assert len(helpers) == 2
    for helper in helpers:
        assert helper["albert"] == "member_albert.albert_api_key"
        assert helper["model"] in (None, "", ALBERT_NOTES_MODEL)
    events, helpers = _notes(env, "member_nokeys", note_mode="extended")
    assert len(events) == 1 and _credential_error_event(events[0], ALBERT_CREDENTIAL), events
    assert helpers == []

    # Défaut OpenAI (historique) : inchangé.
    monkeypatch.setenv("OPENROUTER_DEFAULT_MODEL", "gpt-4o-mini")
    events, helpers = _notes(env, "member_keys", note_mode="extended")
    assert events[-1].get("type") == "complete" and len(helpers) == 2
    assert all(h["openai"] == "member_keys.openai_api_key" for h in helpers)


def _record_dotenv_reads(monkeypatch):
    """Enveloppe les lectures du fichier ``.env`` du module des notes ; renvoie leur journal.

    ``find_dotenv`` (déjà dirigé par le harnais vers un fichier absent) et
    ``dotenv_values`` sont ceux qu'utilise ``resolve_default_llm_model`` : tout
    calcul du défaut web par ce module passe par eux.
    """
    reads = []
    real_find = llm_note_generator.find_dotenv
    real_values = llm_note_generator.dotenv_values

    def _find(*args, **kwargs):
        """Note la recherche du fichier puis délègue."""
        reads.append("find_dotenv")
        return real_find(*args, **kwargs)

    def _values(*args, **kwargs):
        """Note la lecture du fichier puis délègue."""
        reads.append("dotenv_values")
        return real_values(*args, **kwargs)

    monkeypatch.setattr(llm_note_generator, "find_dotenv", _find)
    monkeypatch.setattr(llm_note_generator, "dotenv_values", _values)
    return reads


@pytest.mark.parametrize("switch", ["0", None, "off"])
def test_notes_off_empty_model_keeps_historical_openai_check(albert_env, monkeypatch, switch):
    """Albert OFF, modèle vide, défaut OpenRouter (slug) : contrôle historique de la clé OpenAI.

    Aucun golden ne couvre ce cas (le défaut des goldens n'a pas de « / ») :
    un modèle vide exige la clé OpenAI, les aides reçoivent le modèle tel
    quel (vide ou absent) et la clé OpenAI, le ``.env`` n'est pas lu ; en
    miroir, une persona qui n'a qu'une clé OpenRouter reçoit l'événement
    ``openai_api_key``.
    """
    if switch is None:
        monkeypatch.delenv("ALBERT_ENABLED", raising=False)
    else:
        monkeypatch.setenv("ALBERT_ENABLED", switch)
    monkeypatch.setenv("OPENROUTER_DEFAULT_MODEL", OPENROUTER_SLUG_DEFAULT)
    env = albert_env
    reads = _record_dotenv_reads(monkeypatch)
    openai_error = {
        "type": "error",
        "message": CREDENTIAL_ERROR_MESSAGES["openai_api_key"],
        "credential_required": "openai_api_key",
    }
    for mode in sorted(NOTE_HELPERS):
        for model in ("", None):
            label = (mode, model)
            events, helpers = _notes(env, "member_keys", note_mode=mode, model=model)
            assert events[-1].get("type") == "complete", (label, events)
            assert not [e for e in events if e.get("type") == "error" or "credential_required" in e], (label, events)
            assert [h["fn"] for h in helpers] == [NOTE_HELPERS[mode]] * 2, label
            for helper in helpers:
                assert helper["model"] in (None, ""), (label, helper["model"])
                assert helper["openai"] == "member_keys.openai_api_key", label
                assert helper["openrouter"] is None and helper["albert"] is None, label

            events, helpers = _notes(env, "member_openrouter", note_mode=mode, model=model)
            assert events == [openai_error], (label, events)
            assert helpers == [], label
    assert reads == [], "défaut web relu dans le .env avec Albert OFF"

    # Modèle slug explicite : règle historique inchangée (clé OpenRouter exigée et transmise).
    events, helpers = _notes(env, "member_keys", note_mode="extended", model=OPENROUTER_SLUG_DEFAULT)
    assert len(events) == 1 and _credential_error_event(events[0], "openrouter_api_key"), events
    assert helpers == []
    events, helpers = _notes(env, "member_openrouter", note_mode="extended", model=OPENROUTER_SLUG_DEFAULT)
    assert events[-1].get("type") == "complete", events
    assert [h["openrouter"] for h in helpers] == ["member_openrouter.openrouter_api_key"] * 2
    assert [h["model"] for h in helpers] == [OPENROUTER_SLUG_DEFAULT] * 2

    # Contrôle du harnais : un calcul du défaut web passe bien par l'enveloppe.
    assert llm_note_generator.resolve_default_llm_model() == OPENROUTER_SLUG_DEFAULT
    assert "dotenv_values" in reads


# ======================================================================
# Albert ON : /upload_db
# ======================================================================
def test_upload_db_albert_input_order_and_args(albert_env, albert_on):
    """Entrée ``resolve_albert_input`` (avant le contrôle sparse), drapeaux CLI, clé, délai.

    Toutes les sessions appartiennent au projet Albert (propriétaire
    ``member_albert``, ``member_nokeys`` collaborateur) : aucun cas ne dépend
    d'un accès à une session étrangère.
    """
    env = albert_env
    expected_inputs = {
        "gsess-alb-chunks": "output_chunks.json",
        "gsess-alb-dense": "output_chunks_with_embeddings.json",
        SESSION: "output_chunks_with_embeddings_sparse.json",
    }
    for session, input_name in expected_inputs.items():
        resp, errors = _post(env, "member_albert", "/upload_db",
                             _upload_form(session, albert_create_collection="true"))
        assert resp.status_code == 200 and errors == [], (session, resp.text[:300])
        launches, _ = env.capture.take()
        assert len(launches) == 1, session
        cmd = launches[0]["argv"]
        assert launches[0]["script"] == "rad_vectordb.py"
        assert os.path.basename(_argv_value(cmd, "--input")) == input_name, session
        assert os.path.dirname(_argv_value(cmd, "--input")) == str(env.uploads / session)
        assert _argv_value(cmd, "--db") == "albert"
        assert _argv_value(cmd, "--albert-collection-name") == "Corpus RAGpy"
        assert _has_flag(cmd, "--albert-create-collection") and _has_flag(cmd, "--albert-ack-retention")
        assert not _has_flag(cmd, "--albert-collection-id")
        assert not [f for f in ("--index", "--namespace", "--class", "--tenant", "--collection") if _has_flag(cmd, f)]
        assert launches[0]["timeout"] == ALBERT_TIMEOUT_DEFAULT
        assert launches[0]["env"]["ALBERT_API_KEY"] == "member_albert.albert_api_key"

    # Identifiant de collection, sans création.
    resp, errors = _post(env, "member_albert", "/upload_db",
                         {"path": SESSION, "db_choice": "albert", "albert_collection_id": "4242",
                          "albert_gdpr_ack": "true"})
    assert resp.status_code == 200 and errors == []
    cmd = env.capture.take()[0][0]["argv"]
    assert _argv_value(cmd, "--albert-collection-id") == "4242"
    assert not _has_flag(cmd, "--albert-create-collection") and _has_flag(cmd, "--albert-ack-retention")

    # Nom commençant par « - » : jamais lu comme une option du script.
    resp, errors = _post(env, "member_albert", "/upload_db", _upload_form(SESSION, albert_collection_name="-x"))
    if resp.status_code == 200:
        cmd = env.capture.take()[0][0]["argv"]
        assert "-x" not in cmd and _argv_value(cmd, "--albert-collection-name") == "-x"
    else:
        assert resp.status_code == 400 and env.capture.take() == ([], [])

    # Aucune entrée dans la session : 400, aucun lancement.
    resp, errors = _post(env, "member_albert", "/upload_db", _upload_form("gsess-alb-empty"))
    assert resp.status_code == 400 and errors == []
    assert env.capture.take() == ([], [])

    # Clé Albert manquante (collaborateur non-admin, clé serveur présente) : 403 albert_api_key.
    resp, errors = _post(env, "member_nokeys", "/upload_db", _upload_form(SESSION))
    assert resp.status_code == 403 and errors == []
    assert resp.json()["credential_required"] == ALBERT_CREDENTIAL
    assert env.capture.take() == ([], [])



def test_upload_db_albert_foreign_session_403_no_launch(albert_env, albert_on):
    """Session enregistrée d'un projet inaccessible : 403 sans lancement, audit, manifeste ni écriture.

    ``gsess-full`` appartient au projet de ``member_nokeys`` (aucun membre) :
    ni ``member_albert`` ni ``member_keys`` (clés Albert valides l'un et
    l'autre) ne peuvent en envoyer les textes vers une collection Albert. Le
    refus n'est pas une erreur d'identifiant (pas de ``credential_required``).
    Contrôles : un collaborateur du projet Albert et l'administrateur passent.
    """
    env = albert_env
    foreign_dir = env.uploads / FOREIGN_SESSION
    files_before = sorted(os.listdir(foreign_dir))
    real_before = _real_manifest_files()
    forms = (
        _upload_form(FOREIGN_SESSION, albert_create_collection="true"),
        {"path": FOREIGN_SESSION, "db_choice": "albert", "albert_collection_id": "4242", "albert_gdpr_ack": "true"},
    )
    for persona in ("member_albert", "member_keys"):
        for form in forms:
            resp, errors = _post(env, persona, "/upload_db", form)
            assert resp.status_code == 403 and errors == [], (persona, resp.status_code, resp.text[:300])
            body = _json(resp)
            assert "credential_required" not in body, (persona, body)
            assert not _contains_secret(resp.text)
            assert env.capture.take() == ([], []), persona
        assert _manifest_copies(env.ragpy_root, env.users[persona].id) == [], persona
    assert sorted(os.listdir(foreign_dir)) == files_before
    env.db.expire_all()
    assert env.db.query(AuditLog).count() == 0
    assert _real_manifest_files() == real_before

    # Contrôles : collaborateur du projet Albert sur sa session, administrateur sur toute session.
    for persona, session in (("member_keys", SESSION), ("admin", FOREIGN_SESSION)):
        resp, errors = _post(env, persona, "/upload_db", _upload_form(session))
        assert resp.status_code == 200 and errors == [], (persona, resp.text[:300])
        launches, _ = env.capture.take()
        assert len(launches) == 1 and _is_albert_upload(launches[0]["argv"]), persona
        assert os.path.dirname(_argv_value(launches[0]["argv"], "--input")) == str(env.uploads / session)
    assert _real_manifest_files() == real_before

def test_upload_db_albert_requires_gdpr_ack(albert_env, albert_on):
    """Sans ``albert_gdpr_ack=true`` : 400 ``gdpr_ack_required``, sans lancement ni audit."""
    env = albert_env
    for ack in (None, "false", "", "0"):
        resp, errors = _post(env, "member_albert", "/upload_db", _upload_form(SESSION, albert_gdpr_ack=ack))
        assert resp.status_code == 400 and errors == [], ack
        assert "gdpr_ack_required" in json.dumps(resp.json()), ack
        assert env.capture.take() == ([], []), ack
    env.db.expire_all()
    assert env.db.query(AuditLog).count() == 0


def test_upload_db_albert_parses_existing_manifest_and_journal_with_spaces(albert_env, albert_on):
    """Lignes ``Skipped (existing)``, ``Albert manifest`` et ``Dedup journal`` lues en entier (espaces)."""
    env = albert_env
    session = "gsess albert space"
    directory = str(env.uploads / session)
    resp, errors = _post(env, "member_albert", "/upload_db", _upload_form(session))
    assert resp.status_code == 200 and errors == [], resp.text[:300]
    body = resp.json()
    assert body["status"] == "success"
    assert body["inserted_count"] == 3
    assert body["skipped_count"] == 1
    assert body.get("existing_count", body.get("skipped_existing")) == 2
    assert body["journal_path"] == os.path.join(directory, "dedup_journal.jsonl")
    assert os.path.join(directory, MANIFEST_NAME) in body.values()

    # Aucun chunk déjà présent : le compte est 0 (jamais absent ou happé ailleurs).
    env.capture.take()
    env.capture.stdout_for = lambda cmd: _albert_stdout(cmd, existing=0, with_journal=False) if _is_albert_upload(cmd) else None
    resp, errors = _post(env, "member_albert", "/upload_db", _upload_form(session))
    assert resp.status_code == 200 and errors == []
    body = resp.json()
    assert body.get("existing_count", body.get("skipped_existing", 0)) == 0
    assert "journal_path" not in body


def _manifest_copies(root, user_id):
    """Copies de manifeste sous ``<root>/data/albert_manifests/<user_id>/``."""
    return sorted(glob.glob(os.path.join(str(root), "data", "albert_manifests", str(user_id), "*.jsonl")))


def _real_manifest_files():
    """Fichiers présents sous ``data/albert_manifests`` du dépôt réel."""
    return set(glob.glob(os.path.join(REAL_MANIFEST_BASE, "**", "*"), recursive=True))


def test_upload_db_albert_audit_and_manifest_copy(albert_env, albert_on):
    """Audit ``ALBERT_COLLECTION_UPLOAD`` et copie du manifeste (même en échec) hors ``uploads/``."""
    env = albert_env
    before_real = _real_manifest_files()
    real_base_existed = os.path.isdir(REAL_MANIFEST_BASE)
    user = env.users["member_albert"]
    try:
        resp, errors = _post(env, "member_albert", "/upload_db",
                             _upload_form("gsess-alb-chunks", albert_create_collection="true"))
        assert resp.status_code == 200 and errors == [], resp.text[:300]
        leaked = sorted(_real_manifest_files() - before_real)
        assert leaked == [], "copie du manifeste écrite dans le dépôt réel (chemin non dérivé de RAGPY_DIR)"
        copies = _manifest_copies(env.ragpy_root, user.id)
        assert len(copies) == 1, copies
        copy_path = copies[0]
        assert not os.path.realpath(copy_path).startswith(os.path.realpath(str(env.uploads)) + os.sep)
        assert os.path.basename(copy_path).startswith("gsess-alb-chunks")
        with open(copy_path, encoding="utf-8") as fh:
            assert [json.loads(line) for line in fh if line.strip()] == list(MANIFEST_ROWS)

        env.db.expire_all()
        rows = env.db.query(AuditLog).filter(AuditLog.action == "ALBERT_COLLECTION_UPLOAD").all()
        assert len(rows) == 1
        row = rows[0]
        assert row.user_id == user.id
        assert row.resource_type == "albert_collection"
        details = row.details or {}
        assert {"collection_id", "collection_name", "session", "gdpr_ack"} <= set(details)
        assert details["gdpr_ack"] is True
        assert details["collection_name"] == "Corpus RAGpy"
        assert details["session"] == "gsess-alb-chunks"
        assert not _contains_secret(json.dumps(details))

        # Échec du script après une tranche : la copie (partielle) est faite quand même.
        env.capture.take()
        env.capture.returncode = 1
        resp, errors = _post(env, "member_albert", "/upload_db", _upload_form("gsess-alb-dense"))
        assert resp.status_code == 500 and errors == []
        assert not _contains_secret(resp.text)
        assert len(_manifest_copies(env.ragpy_root, user.id)) == 2
    finally:
        for path in sorted(_real_manifest_files() - before_real, reverse=True):
            if os.path.isdir(path):
                os.rmdir(path)
            else:
                os.remove(path)
        if not real_base_existed and os.path.isdir(REAL_MANIFEST_BASE) and not os.listdir(REAL_MANIFEST_BASE):
            os.rmdir(REAL_MANIFEST_BASE)


def test_upload_db_albert_account_error_exit_2(albert_env, albert_on):
    """Sortie 2 du connecteur (erreur de compte ou de quota Albert) : ``credential_required``."""
    env = albert_env
    env.capture.returncode = 2
    resp, errors = _post(env, "member_albert", "/upload_db", _upload_form(SESSION))
    assert resp.status_code == 500 and errors == []
    body = resp.json()
    assert body.get("credential_required") == ALBERT_CREDENTIAL
    assert not _contains_secret(resp.text)
    env.capture.returncode = 1
    resp, errors = _post(env, "member_albert", "/upload_db", _upload_form(SESSION))
    assert resp.status_code == 500 and "credential_required" not in resp.json()


def test_upload_db_other_dbs_unchanged(golden_app, monkeypatch):
    """Albert ON : pinecone, weaviate et qdrant rendent exactement G7 (champs Albert ignorés)."""
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    env = golden_app
    doc = golden._read_golden("g7_upload_db.json")
    mismatches = []
    for case in doc["cases"]:
        form = dict(case["request"]["form"])
        if form.get("db_choice") not in HISTORICAL_DB_CHOICES:
            continue
        form.update(ALBERT_UPLOAD_FIELDS)
        out = env.norm(golden._post_case(env, case["case"], case["persona"], "/upload_db", form))
        expected = {k: v for k, v in case.items() if k != "request"}
        observed = {k: v for k, v in out.items() if k != "request"}
        if observed != expected:
            mismatches.append(f"{case['case']}: {json.dumps(observed, ensure_ascii=False)[:2000]}")
    assert not mismatches, "\n".join(mismatches)


def test_upload_db_other_dbs_keep_first_token_journal_regex(albert_env, albert_on):
    """Les 3 connecteurs historiques gardent ``^Dedup journal:\\s*(\\S+)`` et n'ont aucune clé Albert."""
    env = albert_env

    def _spaced_stdout(cmd):
        """Bloc Result d'un connecteur historique dont le journal contient une espace."""
        if golden._script_name(cmd) != "rad_vectordb.py" or _is_albert_upload(cmd):
            return None
        return ("\n=== Result ===\nStatus: success\nMessage: ok\nInserted: 3\nSkipped (dedup): 1\n"
                "Dedup journal: /data/dir with space/dedup_journal.jsonl\n"
                "Skipped (existing): 9\nAlbert manifest: /data/dir with space/albert_manifest.jsonl\n")

    env.capture.stdout_for = _spaced_stdout
    resp, errors = _post(env, "admin", "/upload_db",
                         {"path": "gsess-full", "db_choice": "pinecone", "pinecone_index_name": "idx"})
    assert resp.status_code == 200 and errors == []
    body = resp.json()
    assert body == {
        "status": "success",
        "message": "Uploaded to pinecone",
        "inserted_count": 3,
        "skipped_count": 1,
        "journal_path": "/data/dir",
    }


# ======================================================================
# Statut de session : champs d'espace vectoriel exposés seulement s'ils existent
# ======================================================================
def test_pipeline_files_expose_embedding_space_only_when_present(albert_env):
    """``embedding_*`` des chunks hors espace par défaut exposés ; fichiers par défaut inchangés (G7)."""
    env = albert_env
    url = "/api/pipeline/sessions/gsess-full/files"
    default = env.client.get(url, headers=env.headers["member_nokeys"]).json()
    for stage in ("dense_embedding", "sparse_embedding"):
        assert not [k for k in default["files"][stage] if k.startswith("embedding_")], stage

    space = {"embedding_provider": "albert", "embedding_model": "bge-m3", "embedding_dim": 1024}
    folder = env.uploads / "gsess-full"
    for kind in ("dense", "sparse"):
        chunks = [dict(chunk, **space) for chunk in golden._chunks(kind)]
        golden._write_json(str(folder / ARTEFACT_FILES[kind]), chunks)
    body = env.client.get(url, headers=env.headers["member_nokeys"]).json()
    for stage in ("dense_embedding", "sparse_embedding"):
        fields = {k: v for k, v in body["files"][stage].items() if k.startswith("embedding_")}
        assert fields == space, stage
        assert body["files"][stage]["chunk_count"] == 3


# ======================================================================
# Albert ON : citations
# ======================================================================
def _citation_run(env, persona, route, model, n_citations=2):
    """Crée une session de citations et lance ``route`` : (statut, événements ou corps)."""
    pid, sid = golden._make_citation_session(env, persona, model, n_citations=n_citations)
    url = f"/api/projects/{pid}/{route}"
    resp, errors = _post(env, persona, url, {"session_id": sid, "batch_size": 5})
    assert errors == [], route
    payload = _json(resp) if route.endswith("_bg") else _parse_events(resp.text)
    return resp.status_code, payload


def test_citations_three_sites_forward_key(albert_env, albert_on):
    """Filtrage SSE, import progressif et tâche de fond : la clé Albert de l'utilisateur est transmise."""
    env = albert_env
    status, events = _citation_run(env, "member_albert", "filter_citations_sse", ALBERT_CHAT_MODEL)
    assert status == 200 and events[-1].get("type") == "complete", events
    status, events = _citation_run(env, "member_albert", "batch_import_citations_sse", ALBERT_CHAT_MODEL)
    assert status == 200 and events[-1].get("type") == "complete", events
    status, body = _citation_run(env, "member_albert", "filter_citations_bg", ALBERT_CHAT_MODEL)
    assert status == 200 and "task_id" in body, body
    assert env.capture.bg_errors == []
    _, helpers = env.capture.take()
    by_fn = {}
    for helper in helpers:
        by_fn.setdefault(helper["fn"], []).append(helper)
    assert len(by_fn["process_citations_parallel"]) == 2  # SSE + tâche de fond
    assert len(by_fn["filter_citation_with_llm"]) == 2    # import progressif, 2 citations
    for helper in helpers:
        assert helper["albert"] == "member_albert.albert_api_key", helper["fn"]
        assert helper["model"] == ALBERT_CHAT_MODEL, helper["fn"]

    # Sans clé Albert : SSE → événement credential_required ; tâche de fond → 400 ; aucune aide.
    for route in ("filter_citations_sse", "batch_import_citations_sse"):
        status, events = _citation_run(env, "member_mistral", route, ALBERT_CHAT_MODEL)
        assert status == 200 and len(events) == 1, (route, events)
        assert _credential_error_event(events[0], ALBERT_CREDENTIAL), (route, events)
    status, body = _citation_run(env, "member_mistral", "filter_citations_bg", ALBERT_CHAT_MODEL)
    assert status == 400, body
    assert env.capture.take() == ([], [])


def test_citations_albert_no_outer_semaphore(albert_env, albert_on):
    """Import progressif ``albert/…`` : aucun sémaphore tenu par la route ; chemin historique inchangé."""
    env = albert_env
    semaphore = env.capture.semaphore
    status, events = _citation_run(env, "member_albert", "batch_import_citations_sse", ALBERT_CHAT_MODEL)
    assert status == 200 and events[-1].get("type") == "complete", events
    _, helpers = env.capture.take()
    filters = [h for h in helpers if h["fn"] == "filter_citation_with_llm"]
    assert len(filters) == 2
    assert [h["outer_semaphore_held"] for h in filters] == [False, False]
    assert semaphore.acquired == 0

    # Chemin OpenAI : le sémaphore externe reste tenu autour de chaque appel (code actuel).
    status, events = _citation_run(env, "member_keys", "batch_import_citations_sse", "gpt-4o-mini")
    assert status == 200 and events[-1].get("type") == "complete", events
    _, helpers = env.capture.take()
    filters = [h for h in helpers if h["fn"] == "filter_citation_with_llm"]
    assert [h["outer_semaphore_held"] for h in filters] == [True, True]
    assert semaphore.acquired == 2 and semaphore.held == 0


def test_citations_sse_account_error_event(albert_env, albert_on):
    """Erreur de compte Albert pendant le filtrage : événement ``credential_required`` puis arrêt."""
    env = albert_env
    env.capture.citation_error = AlbertAuthError(reason="account_expired", status=403)
    status, events = _citation_run(env, "member_albert", "filter_citations_sse", ALBERT_CHAT_MODEL)
    assert status == 200
    assert _credential_error_event(events[-1], ALBERT_CREDENTIAL), events
    assert not [e for e in events if e.get("type") in ("complete", "progress")], events
    assert not _contains_secret(json.dumps(events))
    env.capture.citation_error = None

    env.capture.filter_error = AlbertQuotaExhausted(status=429)
    status, events = _citation_run(env, "member_albert", "batch_import_citations_sse", ALBERT_CHAT_MODEL)
    assert status == 200
    assert _credential_error_event(events[-1], ALBERT_CREDENTIAL), events
    assert not [e for e in events if e.get("type") in ("complete", "batch_complete")], events
    _, helpers = env.capture.take()
    assert [h["fn"] for h in helpers if h["fn"] == "filter_citation_with_llm"] == ["filter_citation_with_llm"]
    env.capture.filter_error = None


# ======================================================================
# Celery
# ======================================================================
CELERY_TASK_MODULES = ("app.tasks.extraction", "app.tasks.chunking", "app.tasks.embeddings", "app.tasks.vectordb")


def _celery_tasks():
    """Toutes les tâches Celery des modules du pipeline (anciennes et nouvelles)."""
    found = {}
    for module_name in CELERY_TASK_MODULES:
        module = importlib.import_module(module_name)
        for value in vars(module).values():
            if isinstance(value, CeleryTask):
                found[value.name] = value
    return found


@pytest.fixture
def celery_albert(world, monkeypatch):
    """Monde Celery complété : membre à clé Albert, sessions d'envoi, toutes les tâches enregistrées.

    ``queued`` reçoit ``(tâche, kwargs)`` par soumission (``delay`` ou
    ``apply_async``) et ``options`` les options de soumission correspondantes
    (``{}`` pour ``delay``), dans le même ordre.
    """
    db = world.session_factory()
    try:
        user = User(
            email="member.albert@celery-albert.test", hashed_password="x", roles=["USER"], is_active=True,
            is_verified=True, api_credentials=encrypt_credentials({"albert_api_key": MEMBER_ALBERT_ONLY_KEY}),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        project = Project(name="Project albert", owner_id=user.id)
        db.add(project)
        db.commit()
        db.refresh(project)
        for folder, kinds in (("csess-alb-chunks", ("plain",)), ("csess-alb-full", ("plain", "dense", "sparse")),
                              ("csess-alb-fresh", None)):
            if kinds is None:  # export Zotero seul (extraction)
                celery_harness._write_session(os.path.join(world.uploads, folder), "fresh")
            else:
                _write_session(os.path.join(world.uploads, folder), kinds)
            row = PipelineSession(project_id=project.id, session_folder=folder, source_type="zip")
            db.add(row)
            db.commit()
            db.refresh(row)
            world.sessions[folder] = row.id
        world.users["member_albert"] = user.id
        world.headers["member_albert"] = {"Authorization": "Bearer " + create_access_token(subject=str(user.id))}
    finally:
        db.close()

    queued = []
    options = []
    tasks = _celery_tasks()

    def _delay_for(task):
        """``.delay`` de ``task`` : note la tâche et ses arguments nommés."""

        def _delay(*args, **kwargs):
            """Enregistre la mise en file (jamais de broker)."""
            assert not args, "tasks are always queued with keyword arguments"
            queued.append((task, dict(kwargs)))
            options.append({})
            return SimpleNamespace(id=f"task-albert-{len(queued)}")

        return _delay

    def _apply_async_for(task):
        """``.apply_async`` de ``task`` : note la tâche, ses arguments nommés et ses options."""

        def _apply_async(args=None, kwargs=None, **opts):
            """Enregistre la mise en file et les options (limites de temps comprises)."""
            assert not args, "tasks are always queued with keyword arguments"
            queued.append((task, dict(kwargs or {})))
            options.append(dict(opts))
            return SimpleNamespace(id=f"task-albert-{len(queued)}")

        return _apply_async

    def _update_state(*args, **kwargs):
        """État Celery ignoré."""

    def _retry(*args, **kwargs):
        """Note une demande de réessai (jamais attendue sur la branche albert)."""
        world.retries.append(type(kwargs.get("exc")).__name__)
        return RuntimeError("retry requested")

    for task in tasks.values():
        monkeypatch.setattr(task, "delay", _delay_for(task))
        monkeypatch.setattr(task, "apply_async", _apply_async_for(task))
        monkeypatch.setattr(task, "update_state", _update_state)
        monkeypatch.setattr(task, "retry", _retry)

    script_calls = []

    def _albert_run_script(cmd, env, *, on_progress=None, timeout=None, **kwargs):
        """Remplace ``runner.run_script`` : note argv, env réduit et délai ; sortie albert."""
        cmd = [str(c) for c in cmd]
        script_calls.append({"argv": cmd, "env": _env_facts(env), "timeout": timeout})
        if _is_albert_upload(cmd):
            _write_manifest(os.path.join(os.path.dirname(_argv_value(cmd, "--input")), MANIFEST_NAME))
            return runner.ScriptResult(0, _albert_stdout(cmd), "")
        celery_harness._simulate_outputs(cmd)
        return runner.ScriptResult(0, celery_harness._fake_stdout(cmd), "")

    monkeypatch.setattr(runner, "run_script", _albert_run_script)
    # Copies de manifeste : jamais dans le dépôt réel.
    manifest_dir = os.path.join(os.path.dirname(world.uploads), "celery_manifests")
    for name in dir(runner):
        value = getattr(runner, name)
        if "MANIFEST" in name.upper() and isinstance(value, str) and value.startswith(golden._RAGPY_ROOT):
            monkeypatch.setattr(runner, name, manifest_dir)
    return SimpleNamespace(world=world, queued=queued, options=options, script_calls=script_calls,
                           manifest_dir=manifest_dir)


def _celery_upload(ctx, persona, folder, **extra):
    """Soumet ``/api/celery/upload_vectordb`` pour ``folder``."""
    data = {"path": folder, "session_id": str(ctx.world.sessions[folder]), "db_choice": "albert"}
    data.update({k: v for k, v in extra.items() if v is not None})
    return ctx.world.client.post("/api/celery/upload_vectordb", data=data, headers=ctx.world.headers[persona])


def test_celery_albert_gated_and_never_autoretried(celery_albert, monkeypatch):
    """Branche albert Celery : 400 OFF, acquittement et clé exigés, jamais réessayée automatiquement."""
    ctx = celery_albert
    ack = {"albert_collection_name": "Corpus RAGpy", "albert_gdpr_ack": "true"}

    # OFF : réponses actuelles (contrôle sparse puis liste blanche), rien en file.
    monkeypatch.setenv("ALBERT_ENABLED", "0")
    resp = _celery_upload(ctx, "member_albert", "csess-alb-full", **ack)
    assert resp.status_code == 400 and "Invalid db_choice" in resp.text
    resp = _celery_upload(ctx, "member_albert", "csess-alb-chunks", **ack)
    assert resp.status_code == 400 and "Embeddings file not found" in resp.text
    assert ctx.queued == []

    monkeypatch.setenv("ALBERT_ENABLED", "1")
    # Sans acquittement : 400 gdpr_ack_required.
    for value in (None, "false"):
        resp = _celery_upload(ctx, "member_albert", "csess-alb-full", albert_collection_name="Corpus RAGpy",
                              albert_gdpr_ack=value)
        assert resp.status_code == 400 and "gdpr_ack_required" in resp.text, value
    # Sans clé Albert personnelle (clé du worker présente) : 403.
    resp = ctx.world.client.post(
        "/api/celery/upload_vectordb",
        data=dict({"path": "csess-n", "session_id": str(ctx.world.sessions["csess-n"]), "db_choice": "albert"}, **ack),
        headers=ctx.world.headers["member_nokeys"],
    )
    assert resp.status_code == 403 and resp.json()["credential_required"] == ALBERT_CREDENTIAL
    assert ctx.queued == []

    db = ctx.world.session_factory()
    try:
        assert db.query(AuditLog).filter(AuditLog.action == "ALBERT_COLLECTION_UPLOAD").count() == 0
    finally:
        db.close()

    # Soumission valide : aucun secret en file, argv albert, clé de l'utilisateur, délai Albert.
    resp = _celery_upload(ctx, "member_albert", "csess-alb-full", **ack)
    assert resp.status_code == 200, resp.text[:300]
    assert len(ctx.queued) == 1
    db = ctx.world.session_factory()
    try:
        rows = db.query(AuditLog).filter(AuditLog.action == "ALBERT_COLLECTION_UPLOAD").all()
        assert len(rows) == 1 and rows[0].user_id == ctx.world.users["member_albert"]
        assert (rows[0].details or {}).get("gdpr_ack") is True
    finally:
        db.close()
    task, kwargs = ctx.queued[-1]
    blob = json.dumps(kwargs, sort_keys=True, default=str)
    assert not _contains_secret(blob)
    assert kwargs.get("user_id") == ctx.world.users["member_albert"]
    assert kwargs.get("db_choice") == "albert"
    result = task.run(**kwargs)
    assert result["status"] == "success"
    call = ctx.script_calls[-1]
    assert _argv_value(call["argv"], "--db") == "albert"
    assert _has_flag(call["argv"], "--albert-ack-retention")
    assert _argv_value(call["argv"], "--albert-collection-name") == "Corpus RAGpy"
    assert call["env"]["ALBERT_API_KEY"] == "member_albert.albert_api_key"
    assert call["timeout"] == ALBERT_TIMEOUT_DEFAULT

    # Jamais réessayée : ni une erreur d'infrastructure, ni un échec du script.
    retries_before = list(ctx.world.retries)

    def _infra(cmd, env, **kw):
        """Échec transitoire de lancement."""
        raise runner.TaskInfrastructureError("EAGAIN")

    def _failed(cmd, env, **kw):
        """Sortie non nulle du connecteur."""
        return runner.ScriptResult(1, "=== Result ===\nStatus: error\n", "")

    for fake in (_infra, _failed):
        monkeypatch.setattr(runner, "run_script", fake)
        resp = _celery_upload(ctx, "member_albert", "csess-alb-full", **ack)
        assert resp.status_code == 200
        task, kwargs = ctx.queued[-1]
        with pytest.raises(Exception) as excinfo:
            task.run(**kwargs)
        assert not isinstance(excinfo.value, RuntimeError) or "retry requested" not in str(excinfo.value)
        assert ctx.world.retries == retries_before, fake.__name__


def test_celery_albert_accepts_output_chunks(celery_albert, monkeypatch):
    """Albert ON : une session sans fichier sparse est acceptée (``resolve_albert_input``)."""
    ctx = celery_albert
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    resp = _celery_upload(ctx, "member_albert", "csess-alb-chunks",
                          albert_collection_name="Corpus RAGpy", albert_gdpr_ack="true")
    assert resp.status_code == 200, resp.text[:300]
    task, kwargs = ctx.queued[-1]
    assert os.path.basename(kwargs["input_file"]) == "output_chunks.json"
    result = task.run(**kwargs)
    call = ctx.script_calls[-1]
    assert os.path.basename(_argv_value(call["argv"], "--input")) == "output_chunks.json"
    assert _argv_value(call["argv"], "--db") == "albert"
    assert result.get("existing_count") == 2
    # Manifeste copié hors uploads/ (répertoire redirigé), même contenu.
    copies = sorted(glob.glob(os.path.join(ctx.manifest_dir, str(ctx.world.users["member_albert"]), "*.jsonl")))
    assert len(copies) == 1, copies
    with open(copies[0], encoding="utf-8") as fh:
        assert [json.loads(line) for line in fh if line.strip()] == list(MANIFEST_ROWS)

    # Aucune entrée d'envoi : 400, rien en file.
    queued_before = len(ctx.queued)
    empty = os.path.join(ctx.world.uploads, "csess-alb-chunks")
    os.remove(os.path.join(empty, "output_chunks.json"))
    resp = _celery_upload(ctx, "member_albert", "csess-alb-chunks",
                          albert_collection_name="Corpus RAGpy", albert_gdpr_ack="true")
    assert resp.status_code == 400
    assert len(ctx.queued) == queued_before


def test_celery_albert_chunking_and_dense_selection(celery_albert, monkeypatch):
    """Celery chunking et dense : sélection Albert gardée, clé exigée, env et délai de la tâche."""
    ctx = celery_albert
    world = ctx.world

    def _submit(endpoint, persona, folder, **extra):
        """Soumet une étape Celery pour ``folder``."""
        data = {"path": folder, "session_id": str(world.sessions[folder])}
        data.update(extra)
        return world.client.post(endpoint, data=data, headers=world.headers[persona])

    # OFF : albert/… et embedding_provider=albert refusés (400), rien en file.
    monkeypatch.setenv("ALBERT_ENABLED", "0")
    resp = _submit("/api/celery/initial_chunking", "member_albert", "csess-alb-full", model=ALBERT_CHAT_MODEL)
    assert resp.status_code == 400
    resp = _submit("/api/celery/dense_embedding", "member_albert", "csess-alb-full", embedding_provider="albert")
    assert resp.status_code == 400
    assert ctx.queued == []

    monkeypatch.setenv("ALBERT_ENABLED", "1")
    # Clé Albert exigée (clé du worker présente, jamais utilisée pour un non-admin).
    resp = _submit("/api/celery/initial_chunking", "member_nokeys", "csess-n", model=ALBERT_CHAT_MODEL)
    assert resp.status_code == 403 and resp.json()["credential_required"] == ALBERT_CREDENTIAL
    resp = _submit("/api/celery/dense_embedding", "member_nokeys", "csess-n", embedding_provider="albert")
    assert resp.status_code == 403 and resp.json()["credential_required"] == ALBERT_CREDENTIAL
    resp = _submit("/api/celery/dense_embedding", "member_albert", "csess-alb-full", embedding_provider="cohere")
    assert resp.status_code == 400
    assert ctx.queued == []

    resp = _submit("/api/celery/initial_chunking", "member_albert", "csess-alb-full", model=ALBERT_CHAT_MODEL)
    assert resp.status_code == 200, resp.text[:300]
    task, kwargs = ctx.queued[-1]
    assert not _contains_secret(json.dumps(kwargs, default=str))
    task.run(**kwargs)
    call = ctx.script_calls[-1]
    assert _argv_value(call["argv"], "--model") == ALBERT_CHAT_MODEL
    assert call["env"]["ALBERT_API_KEY"] == "member_albert.albert_api_key"
    assert call["timeout"] == ALBERT_TIMEOUT_DEFAULT

    resp = _submit("/api/celery/dense_embedding", "member_albert", "csess-alb-full", embedding_provider="albert")
    assert resp.status_code == 200, resp.text[:300]
    task, kwargs = ctx.queued[-1]
    task.run(**kwargs)
    call = ctx.script_calls[-1]
    assert _argv_value(call["argv"], "--phase") == "dense"
    assert call["env"]["EMBEDDING_PROVIDER"] == "albert"
    assert call["env"]["ALBERT_API_KEY"] == "member_albert.albert_api_key"
    assert call["timeout"] == ALBERT_TIMEOUT_DEFAULT
    assert world.retries == []


def _celery_submit(ctx, endpoint, persona, folder, **extra):
    """Soumet l'étape Celery ``endpoint`` pour ``folder`` en tant que ``persona``."""
    data = {"path": folder, "session_id": str(ctx.world.sessions[folder])}
    data.update({k: v for k, v in extra.items() if v is not None})
    return ctx.world.client.post(endpoint, data=data, headers=ctx.world.headers[persona])


# Soumissions Celery qui sélectionnent Albert : (étiquette, endpoint, dossier, champs).
CELERY_ALBERT_SUBMISSIONS = (
    ("extraction ocr albert", "/api/celery/process_dataframe", "csess-alb-fresh", {}),
    ("chunking albert", "/api/celery/initial_chunking", "csess-alb-full", {"model": ALBERT_CHAT_MODEL}),
    ("dense albert", "/api/celery/dense_embedding", "csess-alb-full", {"embedding_provider": "albert"}),
    ("upload albert", "/api/celery/upload_vectordb", "csess-alb-full",
     {"db_choice": "albert", "albert_collection_name": "Corpus RAGpy", "albert_gdpr_ack": "true"}),
)
# Soumissions historiques de member_keys (aucune clé Albert) : (étiquette, endpoint, dossier, champs).
CELERY_HISTORICAL_SUBMISSIONS = (
    ("extraction mistral", "/api/celery/process_dataframe", "csess-a-fresh", {}),
    ("chunking openai", "/api/celery/initial_chunking", "csess-a", {"model": "gpt-4o-mini"}),
    ("dense default", "/api/celery/dense_embedding", "csess-a", {}),
    ("dense openai", "/api/celery/dense_embedding", "csess-a", {"embedding_provider": "openai"}),
    ("upload pinecone", "/api/celery/upload_vectordb", "csess-a",
     {"db_choice": "pinecone", "pinecone_index_name": "idx-test"}),
)


@pytest.mark.parametrize("custom", [None, str(ALBERT_TIMEOUT_CUSTOM)])
def test_celery_albert_submissions_carry_time_limits(celery_albert, monkeypatch, custom):
    """Soumissions Celery Albert : ``soft_time_limit``/``time_limit`` couvrent ``ALBERT_SUBPROCESS_TIMEOUT``.

    Les limites globales de Celery (1 h douce, 2 h dure) tueraient une tâche
    Albert bien avant le délai du sous-processus (6 h par défaut) : chaque
    soumission qui sélectionne Albert (OCR actif, ``albert/…``,
    ``embedding_provider=albert``, ``db_choice=albert``) porte des limites
    au moins égales à ce délai, la limite dure au moins égale à la douce, et
    le délai du script lancé par la tâche tient dans la limite douce. Les
    soumissions historiques (Albert ON ou OFF) n'en portent aucune.
    """
    ctx = celery_albert
    expected = ALBERT_TIMEOUT_DEFAULT
    if custom is not None:
        monkeypatch.setenv("ALBERT_SUBPROCESS_TIMEOUT", custom)
        expected = int(custom)
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    monkeypatch.setenv("OCR_ENABLE_ALBERT", "1")

    for label, endpoint, folder, extra in CELERY_ALBERT_SUBMISSIONS:
        queued_before = len(ctx.queued)
        resp = _celery_submit(ctx, endpoint, "member_albert", folder, **extra)
        assert resp.status_code == 200, (label, resp.text[:300])
        assert len(ctx.queued) == queued_before + 1, label
        opts = ctx.options[-1]
        soft, hard = opts.get("soft_time_limit"), opts.get("time_limit")
        assert isinstance(soft, (int, float)) and isinstance(hard, (int, float)), (label, sorted(opts))
        assert soft >= expected and hard >= soft, (label, soft, hard, expected)
        task, kwargs = ctx.queued[-1]
        assert not _contains_secret(json.dumps(kwargs, sort_keys=True, default=str)), label
        task.run(**kwargs)
        call = ctx.script_calls[-1]
        assert call["timeout"] == expected and call["timeout"] <= soft, (label, call["timeout"], soft)
        assert call["env"]["ALBERT_API_KEY"] == "member_albert.albert_api_key", label

    for switch in ("1", "0"):
        monkeypatch.setenv("ALBERT_ENABLED", switch)
        for label, endpoint, folder, extra in CELERY_HISTORICAL_SUBMISSIONS:
            queued_before = len(ctx.queued)
            resp = _celery_submit(ctx, endpoint, "member_keys", folder, **extra)
            assert resp.status_code == 200, (switch, label, resp.text[:300])
            assert len(ctx.queued) == queued_before + 1, (switch, label)
            limits = [name for name in CELERY_TIME_LIMIT_OPTIONS if name in ctx.options[-1]]
            assert limits == [], (switch, label, limits)
    assert ctx.world.retries == []


def test_celery_albert_foreign_session_403_nothing_queued(celery_albert, monkeypatch):
    """Celery, Albert ON : une session d'un projet inaccessible est refusée (403), rien en file ni audité."""
    ctx = celery_albert
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    monkeypatch.setenv("OCR_ENABLE_ALBERT", "1")
    # csess-n : projet de member_nokeys ; member_albert n'en est pas membre.
    for label, endpoint, _folder, extra in CELERY_ALBERT_SUBMISSIONS:
        resp = _celery_submit(ctx, endpoint, "member_albert", "csess-n", **extra)
        assert resp.status_code == 403, (label, resp.status_code, resp.text[:300])
        assert "credential_required" not in resp.text, label
    assert ctx.queued == [] and ctx.script_calls == []
    db = ctx.world.session_factory()
    try:
        assert db.query(AuditLog).filter(AuditLog.action == "ALBERT_COLLECTION_UPLOAD").count() == 0
    finally:
        db.close()
