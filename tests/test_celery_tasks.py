"""Tests of the authenticated Celery executor (routes, runner and tasks).

No broker and no worker are needed: ``CELERY_ENABLED``,
``is_celery_available`` and every ``.delay`` are patched, the task owner
registry is a ``FakeRedis`` (``tests/albert_fakes.py``), and the tasks run
in-process through ``task.run(...)`` with ``update_state`` recorded and
``runner.run_script`` replaced, except in the tests that exercise the real
subprocess executor with a local ``python -c`` child.

Credentials are fake (``fake-...``) and assertions only compare booleans and
names, never print a credential value.

Run: pytest tests/test_celery_tasks.py -q
"""

import ast
import csv
import importlib
import json
import os
import re
import signal
import subprocess
import sys
import textwrap
import threading
import time
from types import SimpleNamespace

import pytest
from celery.exceptions import Ignore
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.credentials import CREDENTIAL_ENV_MAPPING, CredentialMissingError, encrypt_credentials
from app.core.security import create_access_token
from app.database.base import Base
from app.database.session import get_db
from app.middleware.auth import get_current_active_user, require_admin
from app.models import audit as _audit_models, background_task as _bg_models  # noqa: F401
from app.models import pipeline_session as _ps_models, project as _project_models  # noqa: F401
from app.models.pipeline_session import PipelineSession, SessionStatus
from app.models.project import Project
from app.models.user import User
import app.routes.celery_tasks as celery_routes
import app.routes.processing as processing_routes
from app.tasks import runner
from app.tasks.chunking import initial_chunking_task
from app.tasks.embeddings import dense_embedding_task, sparse_embedding_task
from app.tasks.extraction import process_dataframe_task
from app.tasks.vectordb import upload_to_vectordb_task
from tests.albert_fakes import FAKE_ALBERT_KEY, FakeRedis

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The real executor, captured before any fixture replaces it.
_ORIGINAL_RUN_SCRIPT = runner.run_script

# ----------------------------------------------------------------------
# Literal fixtures
# ----------------------------------------------------------------------
# What the worker process holds from the server .env (admin fallback).
FAKE_WORKER_ENV = {
    "OPENAI_API_KEY": "fake-openai-worker-0001",
    "OPENROUTER_API_KEY": "fake-openrouter-worker-0001",
    "MISTRAL_API_KEY": "fake-mistral-worker-0001",
    "PINECONE_API_KEY": "fake-pinecone-worker-0001",
    "WEAVIATE_URL": "fake-weaviate-url-worker-0001",
    "WEAVIATE_API_KEY": "fake-weaviate-worker-0001",
    "QDRANT_URL": "fake-qdrant-url-worker-0001",
    "QDRANT_API_KEY": "fake-qdrant-worker-0001",
    "ALBERT_API_KEY": FAKE_ALBERT_KEY,
}

PERSONAS = ("admin", "member_keys", "member_other", "member_nokeys")
PERSONA_ROLES = {
    "admin": ["USER", "ADMIN"],
    "member_keys": ["USER"],
    "member_other": ["USER"],
    "member_nokeys": ["USER"],
}
FAKE_DB_CREDENTIALS = {
    "admin": {},
    "member_keys": {
        "openai_api_key": "fake-openai-member-0001",
        "openrouter_api_key": "fake-openrouter-member-0001",
        "mistral_api_key": "fake-mistral-member-0001",
        "pinecone_api_key": "fake-pinecone-member-0001",
        "weaviate_url": "fake-weaviate-url-member-0001",
        "weaviate_api_key": "fake-weaviate-member-0001",
        "qdrant_url": "fake-qdrant-url-member-0001",
    },
    "member_other": {"openai_api_key": "fake-openai-other-0001"},
    "member_nokeys": {},
}

# Session folders: (owner persona, artefacts).
SESSIONS = {
    "csess-a": ("member_keys", "full"),
    "csess-a-fresh": ("member_keys", "fresh"),
    "csess-b": ("member_other", "full"),
    "csess-n": ("member_nokeys", "full"),
    "csess-n-fresh": ("member_nokeys", "fresh"),
}

TASKS = {
    "extraction": process_dataframe_task,
    "chunking": initial_chunking_task,
    "dense": dense_embedding_task,
    "sparse": sparse_embedding_task,
    "vectordb": upload_to_vectordb_task,
}

CELERY_ENDPOINTS = {
    "extraction": "/api/celery/process_dataframe",
    "chunking": "/api/celery/initial_chunking",
    "dense": "/api/celery/dense_embedding",
    "sparse": "/api/celery/sparse_embedding",
    "vectordb": "/api/celery/upload_vectordb",
}
HTTP_ENDPOINTS = {
    "extraction": "/process_dataframe",
    "chunking": "/initial_text_chunking",
    "dense": "/dense_embedding_generation",
    "sparse": "/sparse_embedding_generation",
    "vectordb": "/upload_db",
}

# (stage, extra form fields) exercised by the parity tests.
PARITY_CASES = (
    ("extraction", {}),
    ("chunking", {}),
    ("chunking", {"model": "gpt-4o-mini"}),
    ("chunking", {"model": "google/gemini-2.5-flash"}),
    ("dense", {}),
    ("sparse", {}),
    ("vectordb", {"db_choice": "pinecone", "pinecone_index_name": "idx-test", "pinecone_namespace": "ns-test"}),
    ("vectordb", {"db_choice": "pinecone", "pinecone_index_name": "idx-test"}),
    ("vectordb", {"db_choice": "weaviate", "weaviate_class_name": "Article", "weaviate_tenant_name": "tenant-a"}),
    ("vectordb", {"db_choice": "qdrant", "qdrant_collection_name": "coll-test"}),
)

VECTORDB_STDOUT = (
    "=== rad_vectordb.py ===\n"
    "Loading chunks (build 2026-09-26)\n"
    "\n=== Result ===\n"
    "Status: success_partial_data\n"
    "Message: 3 chunks inserted\n"
    "Inserted: 3\n"
    "Skipped (dedup): 2\n"
    "Dedup journal: /tmp/journal_dir/dedup_journal.jsonl\n"
)

CELERY_OWNER_TTL = 7 * 24 * 3600


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _write_json(path, payload):
    """Write ``payload`` as UTF-8 JSON at ``path``."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _chunks(kind):
    """Build two fake chunks (plain, dense or sparse)."""
    chunks = []
    for idx in (1, 2):
        chunk = {"id": f"c{idx}", "doc_id": "d", "chunk_index": idx, "total_chunks": 2, "text": f"T{idx}"}
        if kind in ("dense", "sparse"):
            chunk["embedding"] = [0.1, 0.2]
        if kind == "sparse":
            chunk["sparse_embedding"] = {"indices": [1], "values": [1.0]}
        chunks.append(chunk)
    return chunks


def _write_output_csv(path):
    """Write a two-row ``output.csv`` with OCR text."""
    rows = [
        ["itemKey", "title", "texteocr", "texteocr_provider"],
        ["ITEMA001", "Article A", "Texte A.", "mistral"],
        ["ITEMB002", "Article B", "Texte B.", "mistral"],
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)


def _write_session(folder, kind):
    """Write the artefacts of a fresh (Zotero JSON only) or full session."""
    os.makedirs(folder, exist_ok=True)
    _write_json(os.path.join(folder, "biblio.json"), [{"key": "ITEMA001", "title": "Article A"}])
    if kind == "full":
        _write_output_csv(os.path.join(folder, "output.csv"))
        _write_json(os.path.join(folder, "output_chunks.json"), _chunks("plain"))
        _write_json(os.path.join(folder, "output_chunks_with_embeddings.json"), _chunks("dense"))
        _write_json(os.path.join(folder, "output_chunks_with_embeddings_sparse.json"), _chunks("sparse"))


def _argv_value(cmd, flag):
    """Value following ``flag`` in ``cmd`` (None if absent)."""
    if flag in cmd:
        idx = cmd.index(flag)
        if idx + 1 < len(cmd):
            return cmd[idx + 1]
    return None


def _simulate_outputs(cmd):
    """Write the file the pipeline script launched by ``cmd`` would produce."""
    script = next((os.path.basename(c) for c in cmd if str(c).endswith(".py")), "")
    if script == "rad_dataframe.py":
        _write_output_csv(_argv_value(cmd, "--output"))
    elif script == "rad_chunk.py":
        names = {
            "initial": ("output_chunks.json", "plain"),
            "dense": ("output_chunks_with_embeddings.json", "dense"),
            "sparse": ("output_chunks_with_embeddings_sparse.json", "sparse"),
        }
        fname, kind = names[_argv_value(cmd, "--phase")]
        _write_json(os.path.join(_argv_value(cmd, "--output"), fname), _chunks(kind))


def _fake_stdout(cmd):
    """Plausible stdout of the script launched by ``cmd``."""
    if any(str(c).endswith("rad_vectordb.py") for c in cmd):
        return VECTORDB_STDOUT
    return "PROGRESS|row|1/1|done\n"


def _credential_names(env):
    """Credential env names present in ``env`` plus the dotenv deny-list (no values)."""
    names = sorted(n for n in set(CREDENTIAL_ENV_MAPPING.values()) if env.get(n))
    return {"present": names, "deny": env.get("RAGPY_DOTENV_DENY")}


def _all_secret_values():
    """Every fake credential value of the worker env and of the personas."""
    values = set(FAKE_WORKER_ENV.values())
    for creds in FAKE_DB_CREDENTIALS.values():
        values.update(creds.values())
    return values


def _dependency_calls(dependant):
    """Every dependency callable of a route (recursively)."""
    calls = []
    for dep in dependant.dependencies:
        calls.append(dep.call)
        calls.extend(_dependency_calls(dep))
    return calls


def _pid_alive(pid):
    """True while ``pid`` exists (a zombie counts as alive until reaped)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _python_child(code):
    """argv running ``code`` with the current interpreter."""
    return [sys.executable, "-c", textwrap.dedent(code)]


class _SafetySigterm:
    """Temporary SIGTERM handler that records instead of killing the test process."""

    def __init__(self):
        """Start with no signal received."""
        self.received = 0

    def __call__(self, signum, frame):
        """Record the signal."""
        self.received += 1


# ----------------------------------------------------------------------
# World fixture
# ----------------------------------------------------------------------
class CeleryWorld:
    """Test application, database, personas and recorders of one test."""

    def __init__(self, client, users, headers, sessions, uploads, session_factory, redis, delays, http_calls,
                 script_calls, states, retries):
        """Store the pieces built by the ``world`` fixture."""
        self.client = client
        self.users = users
        self.headers = headers
        self.sessions = sessions
        self.uploads = uploads
        self.session_factory = session_factory
        self.redis = redis
        self.delays = delays
        self.http_calls = http_calls
        self.script_calls = script_calls
        self.states = states
        self.retries = retries

    def celery_form(self, stage, folder, extra=None):
        """Form data of a Celery submission for ``folder``."""
        data = {"path": folder, "session_id": str(self.sessions[folder])}
        data.update(extra or {})
        return data

    def http_form(self, stage, folder, extra=None):
        """Form data of the matching HTTP route for ``folder``."""
        data = {"path": folder}
        data.update(extra or {})
        return data

    def submit(self, stage, persona, folder, extra=None):
        """POST a Celery submission as ``persona``."""
        return self.client.post(CELERY_ENDPOINTS[stage], data=self.celery_form(stage, folder, extra),
                                headers=self.headers[persona])

    def call_http(self, stage, persona, folder, extra=None):
        """POST the matching HTTP route as ``persona``."""
        return self.client.post(HTTP_ENDPOINTS[stage], data=self.http_form(stage, folder, extra),
                                headers=self.headers[persona])

    def run_last_delay(self):
        """Run in-process the task queued by the last ``.delay`` call."""
        name, kwargs = self.delays[-1]
        return TASKS[name].run(**kwargs)

    def session_status(self, folder):
        """Current status of the pipeline session ``folder``."""
        db = self.session_factory()
        try:
            row = db.query(PipelineSession).filter(PipelineSession.session_folder == folder).first()
            return row.status
        finally:
            db.close()


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Build the test app (Celery + HTTP pipeline routers) with its fakes."""
    for name in set(CREDENTIAL_ENV_MAPPING.values()) | {"RAGPY_DOTENV_DENY", "PINECONE_ENV"}:
        monkeypatch.delenv(name, raising=False)
    for name, value in FAKE_WORKER_ENV.items():
        monkeypatch.setenv(name, value)

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    for module in (celery_routes, processing_routes):
        monkeypatch.setattr(module, "UPLOAD_DIR", str(uploads))

    engine = create_engine(f"sqlite:///{tmp_path / 'celery.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(runner, "SessionLocal", session_factory)

    db = session_factory()
    users = {}
    for persona in PERSONAS:
        creds = FAKE_DB_CREDENTIALS[persona]
        user = User(
            email=f"{persona.replace('_', '.')}@celery.test",
            hashed_password="x",
            roles=list(PERSONA_ROLES[persona]),
            is_active=True,
            is_verified=True,
            api_credentials=encrypt_credentials(creds) if creds else None,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        users[persona] = user
    projects = {}
    for persona in ("member_keys", "member_other", "member_nokeys"):
        project = Project(name=f"Project {persona}", owner_id=users[persona].id)
        db.add(project)
        db.commit()
        db.refresh(project)
        projects[persona] = project
    sessions = {}
    for folder, (owner, kind) in SESSIONS.items():
        _write_session(str(uploads / folder), kind)
        row = PipelineSession(project_id=projects[owner].id, session_folder=folder, source_type="zip")
        db.add(row)
        db.commit()
        db.refresh(row)
        sessions[folder] = row.id
    user_ids = {persona: user.id for persona, user in users.items()}
    db.close()

    # Celery: enabled, broker reachable, nothing really queued.
    monkeypatch.setattr(celery_routes, "CELERY_ENABLED", True)
    monkeypatch.setattr(celery_routes, "is_celery_available", lambda: True)
    delays = []

    def _fake_delay_for(name):
        """Build the ``.delay`` stand-in of task ``name``."""

        def _fake_delay(*args, **kwargs):
            """Record the payload and return a fake async result."""
            assert not args, "tasks are always queued with keyword arguments"
            delays.append((name, dict(kwargs)))
            return SimpleNamespace(id=f"task-{name}-{len(delays)}")

        return _fake_delay

    for name, task in TASKS.items():
        monkeypatch.setattr(task, "delay", _fake_delay_for(name))

    # Task owner registry.
    redis = FakeRedis()
    monkeypatch.setattr(runner, "get_owner_store", lambda: redis)

    # In-process task execution: state updates and retries recorded.
    states = []
    retries = []

    def _fake_update_state(task_id=None, state=None, meta=None, **kwargs):
        """Record a Celery state update."""
        states.append({"state": state, "meta": meta})

    def _fake_retry(*args, **kwargs):
        """Record a retry request (never expected for deterministic errors)."""
        retries.append(type(kwargs.get("exc")).__name__)
        return RuntimeError("retry requested")

    for task in TASKS.values():
        monkeypatch.setattr(task, "update_state", _fake_update_state)
        monkeypatch.setattr(task, "retry", _fake_retry)

    script_calls = []

    def _fake_run_script(cmd, env, *, on_progress=None, timeout=None, **kwargs):
        """Stand in for ``runner.run_script``: record, simulate outputs, exit 0."""
        script_calls.append({"argv": list(cmd), "env": dict(env or {}), "timeout": timeout})
        _simulate_outputs(list(cmd))
        return runner.ScriptResult(0, _fake_stdout(cmd), "")

    monkeypatch.setattr(runner, "run_script", _fake_run_script)

    # HTTP routes: subprocess launcher recorded.
    http_calls = []

    async def _fake_tracked(cmd, session_folder, timeout=1800, env=None):
        """Stand in for ``processing.run_tracked_subprocess``."""
        http_calls.append({"argv": list(cmd), "env": dict(env or {}), "timeout": timeout})
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=_fake_stdout(cmd), stderr="")

    monkeypatch.setattr(processing_routes, "run_tracked_subprocess", _fake_tracked)

    app = FastAPI()
    app.include_router(celery_routes.router)
    app.include_router(processing_routes.router)

    def _override_get_db():
        """Yield a session of the test database."""
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override_get_db
    client = TestClient(app)
    headers = {
        persona: {"Authorization": "Bearer " + create_access_token(subject=str(uid))}
        for persona, uid in user_ids.items()
    }
    world = CeleryWorld(client, user_ids, headers, sessions, str(uploads), session_factory, redis, delays,
                        http_calls, script_calls, states, retries)
    yield world
    engine.dispose()


# ----------------------------------------------------------------------
# Routes: authentication and access
# ----------------------------------------------------------------------
def test_every_route_requires_auth(world):
    routes = [r for r in celery_routes.router.routes if hasattr(r, "dependant")]
    assert len(routes) == 9
    for route in routes:
        calls = _dependency_calls(route.dependant)
        assert get_current_active_user in calls or require_admin in calls, route.path
        for method in route.methods:
            path = route.path.replace("{task_id}", "task-x")
            if method == "POST":
                resp = world.client.post(path, data={"path": "csess-a", "session_id": "1", "db_choice": "pinecone"})
            else:
                resp = world.client.request(method, path)
            assert resp.status_code == 401, (method, path)
    assert world.delays == []


def test_status_and_workers_require_admin(world, monkeypatch):
    for path in ("/api/celery/status", "/api/celery/workers"):
        resp = world.client.get(path, headers=world.headers["member_keys"])
        assert resp.status_code == 403, path
    resp = world.client.get("/api/celery/status", headers=world.headers["admin"])
    assert resp.status_code == 200
    assert set(resp.json()) == {"enabled", "available", "broker_url"}


def test_foreign_session_denied(world):
    for stage in CELERY_ENDPOINTS:
        extra = {"db_choice": "pinecone", "pinecone_index_name": "idx"} if stage == "vectordb" else {}
        folder = "csess-a-fresh" if stage == "extraction" else "csess-a"
        resp = world.submit(stage, "member_other", folder, extra)
        assert resp.status_code == 403, stage
    # Unknown session folder: 404; path traversal never reaches the disk.
    resp = world.client.post(CELERY_ENDPOINTS["dense"], data={"path": "../csess-a", "session_id": "1"},
                             headers=world.headers["member_keys"])
    assert resp.status_code == 404
    # session_id of another session: refused.
    data = {"path": "csess-a", "session_id": str(world.sessions["csess-b"])}
    resp = world.client.post(CELERY_ENDPOINTS["dense"], data=data, headers=world.headers["member_keys"])
    assert resp.status_code == 400
    assert world.delays == []
    # The owner and an administrator are allowed.
    assert world.submit("dense", "member_keys", "csess-a").status_code == 200
    assert world.submit("dense", "admin", "csess-b").status_code == 200
    assert [d[1]["user_id"] for d in world.delays] == [world.users["member_keys"], world.users["admin"]]


def test_missing_credential_403_same_shape(world):
    cases = (
        ("extraction", "csess-n-fresh", {}),
        ("chunking", "csess-n", {}),
        ("chunking", "csess-n", {"model": "google/gemini-2.5-flash"}),
        ("dense", "csess-n", {}),
        ("vectordb", "csess-n", {"db_choice": "pinecone", "pinecone_index_name": "idx"}),
        ("vectordb", "csess-n", {"db_choice": "weaviate"}),
        ("vectordb", "csess-n", {"db_choice": "qdrant", "qdrant_collection_name": "coll"}),
    )
    for stage, folder, extra in cases:
        celery_resp = world.submit(stage, "member_nokeys", folder, extra)
        http_resp = world.call_http(stage, "member_nokeys", folder, extra)
        assert celery_resp.status_code == 403, (stage, extra)
        assert http_resp.status_code == 403, (stage, extra)
        assert celery_resp.json() == http_resp.json(), (stage, extra)
        assert set(celery_resp.json()) == {"error", "credential_required", "configure_url"}
    # The worker env holds admin keys: they never satisfy a non-admin check.
    assert world.delays == []
    assert world.http_calls == []


def test_delay_payload_has_no_secret(world):
    for stage, extra in PARITY_CASES:
        folder = "csess-a-fresh" if stage == "extraction" else "csess-a"
        resp = world.submit(stage, "member_keys", folder, extra)
        assert resp.status_code == 200, (stage, extra)
        assert set(resp.json()) == {"task_id", "status", "message"}
    assert len(world.delays) == len(PARITY_CASES)
    secrets = _all_secret_values()
    for name, kwargs in world.delays:
        blob = json.dumps(kwargs, sort_keys=True)
        leaked = [True for value in secrets if value in blob]
        assert not leaked, name
        assert kwargs["user_id"] == world.users["member_keys"], name
        assert not [k for k in kwargs if "key" in k.lower() or "secret" in k.lower() or "token" in k.lower()]


def test_route_records_owner_with_ttl(world):
    resp = world.submit("dense", "member_keys", "csess-a")
    task_id = resp.json()["task_id"]
    key = runner.owner_key(task_id)
    assert key == "ragpy:celery_owner:" + task_id
    assert world.redis.get(key) == str(world.users["member_keys"]).encode()
    assert world.redis.ttl(key) == CELERY_OWNER_TTL


def test_status_and_cancel_owner_or_admin(world, monkeypatch):
    monkeypatch.setattr(celery_routes, "get_task_status", lambda task_id: {"state": "PENDING"})
    revoked = []

    def _fake_revoke(task_id, terminate=False):
        """Record a revocation."""
        revoked.append((task_id, terminate))
        return {"success": True, "message": "revoked"}

    monkeypatch.setattr(celery_routes, "revoke_task", _fake_revoke)
    task_id = world.submit("dense", "member_keys", "csess-a").json()["task_id"]
    status_path = f"/api/celery/task/{task_id}/status"
    cancel_path = f"/api/celery/task/{task_id}/cancel?terminate=true"

    assert world.client.get(status_path, headers=world.headers["member_keys"]).status_code == 200
    assert world.client.get(status_path, headers=world.headers["member_other"]).status_code == 403
    assert world.client.get(status_path, headers=world.headers["admin"]).status_code == 200
    assert world.client.post(cancel_path, headers=world.headers["member_other"]).status_code == 403
    assert revoked == []
    assert world.client.post(cancel_path, headers=world.headers["member_keys"]).status_code == 200
    assert world.client.post(cancel_path, headers=world.headers["admin"]).status_code == 200
    assert revoked == [(task_id, True), (task_id, True)]

    # Unknown owner: administrators only.
    unknown = "/api/celery/task/task-unknown/status"
    assert world.client.get(unknown, headers=world.headers["member_keys"]).status_code == 403
    assert world.client.get(unknown, headers=world.headers["admin"]).status_code == 200

    # Registry unavailable: administrators only.
    monkeypatch.setattr(runner, "get_owner_store", lambda: None)
    assert world.client.get(status_path, headers=world.headers["member_keys"]).status_code == 403
    assert world.client.get(status_path, headers=world.headers["admin"]).status_code == 200


# ----------------------------------------------------------------------
# Tasks: environment, argv, retries, results
# ----------------------------------------------------------------------
def test_task_env_is_per_user(world):
    member_openai = FAKE_DB_CREDENTIALS["member_keys"]["openai_api_key"]
    worker_openai = FAKE_WORKER_ENV["OPENAI_API_KEY"]

    # Non-admin with keys: its own keys, never the worker's.
    assert world.submit("dense", "member_keys", "csess-a").status_code == 200
    world.run_last_delay()
    env = world.script_calls[-1]["env"]
    uses_member_key = env.get("OPENAI_API_KEY") == member_openai
    assert uses_member_key
    has_albert_key = "ALBERT_API_KEY" in env
    has_worker_qdrant_key = "QDRANT_API_KEY" in env
    assert not has_albert_key
    assert not has_worker_qdrant_key
    deny = env["RAGPY_DOTENV_DENY"].split(",")
    assert "OPENAI_API_KEY" not in deny and "ALBERT_API_KEY" in deny and "QDRANT_API_KEY" in deny

    # Non-admin without keys, sparse (no requirement): nothing leaks.
    assert world.submit("sparse", "member_nokeys", "csess-n").status_code == 200
    world.run_last_delay()
    env = world.script_calls[-1]["env"]
    leaked = [name for name in CREDENTIAL_ENV_MAPPING.values() if env.get(name)]
    assert leaked == []
    assert "OPENAI_API_KEY" in env["RAGPY_DOTENV_DENY"].split(",")

    # Non-admin without keys, dense queued anyway (e.g. key removed after
    # submission): refused in the worker, no script launched.
    calls_before = len(world.script_calls)
    with pytest.raises(CredentialMissingError):
        dense_embedding_task.run(
            input_file=os.path.join(world.uploads, "csess-n", "output_chunks.json"),
            output_dir=os.path.join(world.uploads, "csess-n"),
            session_id=world.sessions["csess-n"],
            user_id=world.users["member_nokeys"],
        )
    assert len(world.script_calls) == calls_before
    assert world.retries == []

    # Administrator: worker (.env) fallback kept, no deny-list.
    assert world.submit("dense", "admin", "csess-b").status_code == 200
    world.run_last_delay()
    env = world.script_calls[-1]["env"]
    uses_worker_key = env.get("OPENAI_API_KEY") == worker_openai
    has_deny = "RAGPY_DOTENV_DENY" in env
    assert uses_worker_key
    assert not has_deny


def test_argv_matches_http_routes(world):
    for stage, extra in PARITY_CASES:
        folder = "csess-a-fresh" if stage == "extraction" else "csess-a"
        world.http_calls.clear()
        world.script_calls.clear()
        world.call_http(stage, "member_keys", folder, extra)
        assert len(world.http_calls) == 1, (stage, extra)
        resp = world.submit(stage, "member_keys", folder, extra)
        assert resp.status_code == 200, (stage, extra)
        result = world.run_last_delay()
        assert result["status"] == "success"
        assert len(world.script_calls) == 1, (stage, extra)
        http_call, task_call = world.http_calls[0], world.script_calls[0]
        assert task_call["argv"] == http_call["argv"], (stage, extra)
        assert task_call["timeout"] == http_call["timeout"], (stage, extra)
        assert _credential_names(task_call["env"]) == _credential_names(http_call["env"]), (stage, extra)
        if extra.get("db_choice") == "weaviate":
            assert "--class" in task_call["argv"]


def test_weaviate_argv_uses_class_flag():
    cmd = runner.vectordb_argv("/in.json", "weaviate", weaviate_class_name="Article", weaviate_tenant_name="t")
    assert cmd[-4:] == ["--class", "Article", "--tenant", "t"]
    assert cmd[0] == "python3" and cmd[1].endswith(os.path.join("scripts", "rad_vectordb.py"))


def test_task_results_keep_their_keys(world):
    expected = {
        "extraction": {"status", "row_count", "output", "duration_seconds"},
        "chunking": {"status", "chunk_count", "output", "duration_seconds"},
        "dense": {"status", "chunk_count", "output", "duration_seconds"},
        "sparse": {"status", "chunk_count", "output", "duration_seconds"},
        "vectordb": {"status", "db_status", "inserted_count", "database", "duration_seconds",
                     "skipped_count", "journal_path"},
    }
    for stage, keys in expected.items():
        folder = "csess-a-fresh" if stage == "extraction" else "csess-a"
        extra = {"db_choice": "pinecone", "pinecone_index_name": "idx"} if stage == "vectordb" else {}
        assert world.submit(stage, "member_keys", folder, extra).status_code == 200
        result = world.run_last_delay()
        assert set(result) == keys, stage
    assert result["db_status"] == "success_partial_data"
    assert result["inserted_count"] == 3 and result["skipped_count"] == 2
    assert result["journal_path"] == "/tmp/journal_dir/dedup_journal.jsonl"
    assert world.session_status("csess-a") == SessionStatus.COMPLETED


def test_nonzero_exit_not_retried(world, monkeypatch):
    member_key = FAKE_DB_CREDENTIALS["member_keys"]["openai_api_key"]

    def _failing_run_script(cmd, env, **kwargs):
        """Exit 1 with the user's key echoed on stderr."""
        return runner.ScriptResult(1, "partial output\n", "Traceback: boom with " + member_key + "\n")

    monkeypatch.setattr(runner, "run_script", _failing_run_script)
    for stage in TASKS:
        folder = "csess-a-fresh" if stage == "extraction" else "csess-a"
        extra = {"db_choice": "pinecone", "pinecone_index_name": "idx"} if stage == "vectordb" else {}
        assert world.submit(stage, "member_keys", folder, extra).status_code == 200
        with pytest.raises(runner.ScriptFailedError) as excinfo:
            world.run_last_delay()
        assert excinfo.value.returncode == 1
        key_in_message = member_key in str(excinfo.value)
        assert not key_in_message, stage
        assert world.session_status(folder) == SessionStatus.ERROR, stage
    assert world.retries == []

    # Deterministic errors are outside autoretry_for; infrastructure errors are in.
    for task in TASKS.values():
        assert tuple(task.autoretry_for) == runner.INFRASTRUCTURE_ERRORS
        for exc_class in (runner.ScriptFailedError, runner.ScriptTimeoutError, runner.ScriptRevokedError,
                          runner.TaskUserError, CredentialMissingError, ValueError):
            assert not issubclass(exc_class, tuple(task.autoretry_for))

    def _timeout_run_script(cmd, env, **kwargs):
        """Simulate a script timeout."""
        raise runner.ScriptTimeoutError("timed out", timeout=kwargs.get("timeout"))

    monkeypatch.setattr(runner, "run_script", _timeout_run_script)
    assert world.submit("sparse", "member_keys", "csess-a").status_code == 200
    with pytest.raises(runner.ScriptTimeoutError):
        world.run_last_delay()
    assert world.retries == []

    # Positive control: an infrastructure error does go through retry().
    def _infra_run_script(cmd, env, **kwargs):
        """Simulate a transient spawn failure."""
        raise runner.TaskInfrastructureError("EAGAIN")

    monkeypatch.setattr(runner, "run_script", _infra_run_script)
    assert world.submit("sparse", "member_keys", "csess-a").status_code == 200
    with pytest.raises(RuntimeError):
        world.run_last_delay()
    assert world.retries == ["TaskInfrastructureError"]


def test_revoked_task_is_ignored_not_failed(world, monkeypatch):
    def _revoked_run_script(cmd, env, **kwargs):
        """Simulate a revocation while the script runs."""
        raise runner.ScriptRevokedError("Tâche annulée")

    monkeypatch.setattr(runner, "run_script", _revoked_run_script)
    assert world.submit("dense", "member_keys", "csess-a").status_code == 200
    with pytest.raises(Ignore):
        world.run_last_delay()
    assert world.retries == []
    assert world.session_status("csess-a") == SessionStatus.ERROR


def test_task_without_user_is_refused(world):
    kwargs = dict(
        input_file=os.path.join(world.uploads, "csess-a", "output_chunks_with_embeddings.json"),
        output_dir=os.path.join(world.uploads, "csess-a"),
        session_id=world.sessions["csess-a"],
    )
    for user_id in (None, 999999, "not-an-id"):
        with pytest.raises(runner.TaskUserError):
            sparse_embedding_task.run(user_id=user_id, **kwargs)
    db = world.session_factory()
    try:
        user = db.query(User).filter(User.id == world.users["member_keys"]).first()
        user.is_active = False
        db.commit()
    finally:
        db.close()
    with pytest.raises(runner.TaskUserError):
        sparse_embedding_task.run(user_id=world.users["member_keys"], **kwargs)
    assert world.script_calls == []
    assert world.retries == []


def test_parse_vectordb_stdout_parity(world, monkeypatch):
    # Same literal patterns as the /upload_db route parser.
    route_patterns = set()
    for node in ast.walk(_find_function(processing_routes.__file__, "upload_db")):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "search"
                and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            route_patterns.add(node.args[0].value)
    runner_patterns = {runner.VECTORDB_INSERTED_PATTERN, runner.VECTORDB_SKIPPED_PATTERN,
                       runner.VECTORDB_JOURNAL_PATTERN}
    assert runner_patterns <= route_patterns

    # Same results as the route on tricky outputs.
    stdouts = (
        VECTORDB_STDOUT,
        "Loading 2026 chunks\n=== Result ===\nStatus: success\nInserted: 7\nSkipped (dedup): 0\n",
        "note: Inserted: 99 (not anchored)\n=== Result ===\nInserted:12\n",
        "no result block at all\n",
        "Dedup journal: /a/b c.jsonl\nInserted: 1\n",
    )
    for stdout in stdouts:
        async def _fake_tracked(cmd, session_folder, timeout=1800, env=None, _out=stdout):
            """Return the crafted stdout."""
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=_out, stderr="")

        monkeypatch.setattr(processing_routes, "run_tracked_subprocess", _fake_tracked)
        resp = world.call_http("vectordb", "member_keys", "csess-a", {"db_choice": "pinecone",
                                                                     "pinecone_index_name": "idx"})
        body = resp.json()
        parsed = runner.parse_vectordb_stdout(stdout)
        assert body.get("inserted_count") == parsed["inserted_count"], stdout
        assert body.get("skipped_count") == (parsed["skipped_count"] or None), stdout
        assert body.get("journal_path") == (parsed["journal_path"] or None), stdout


def _find_function(path, name):
    """AST node of the top-level function ``name`` of the module at ``path``."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


# ----------------------------------------------------------------------
# Runner: real subprocesses (local python child, no network)
# ----------------------------------------------------------------------
class _RecordingTask:
    """Minimal stand-in of a bound Celery task recording ``update_state``."""

    def __init__(self):
        """Start with no state update."""
        self.states = []

    def update_state(self, task_id=None, state=None, meta=None, **kwargs):
        """Record a state update."""
        self.states.append({"state": state, "meta": dict(meta or {})})


PROGRESS_CHILD = """
import sys
print("PROGRESS|init|2|Found 2 documents to process", flush=True)
print("PROGRESS|row|1/2|Doc one", flush=True)
print("some log line", flush=True)
sys.stderr.write("tqdm-like noise on stderr\\n")
print("PROGRESS|row|1/2|Doc one again (same percent)", flush=True)
print("\\x1b[32mPROGRESS|row|2/2|Doc two\\x1b[0m", flush=True)
print("done", flush=True)
"""


def test_progress_lines_update_state():
    task = _RecordingTask()
    result = _ORIGINAL_RUN_SCRIPT(
        _python_child(PROGRESS_CHILD),
        dict(os.environ),
        on_progress=runner.make_progress_reporter(task, "Processing document"),
        timeout=60,
    )
    assert result.returncode == 0
    assert "done" in result.stdout
    states = task.states
    assert [s["state"] for s in states] == ["PROGRESS"] * 3
    assert states[0]["meta"] == {"current": 0, "total": 2, "percent": 0,
                                 "item": "Found 2 documents to process", "status": "Found 2 documents to process"}
    assert states[1]["meta"]["status"] == "Processing document 1/2"
    assert states[1]["meta"]["percent"] == 50 and states[1]["meta"]["item"] == "Doc one"
    assert states[2]["meta"]["status"] == "Processing document 2/2"
    assert states[2]["meta"]["item"] == "Doc two"
    assert set(states[2]["meta"]) == {"current", "total", "percent", "item", "status"}
    assert "tqdm-like noise" in result.stderr


def test_progress_lines_on_stderr_are_parsed():
    task = _RecordingTask()
    code = "import sys; sys.stderr.write('PROGRESS|chunk|3/4|Chunk 3/4\\n'); sys.stderr.flush()"
    result = _ORIGINAL_RUN_SCRIPT(_python_child(code), dict(os.environ),
                               on_progress=runner.make_progress_reporter(task, "Generating embeddings"), timeout=60)
    assert result.returncode == 0
    assert [s["meta"]["status"] for s in task.states] == ["Generating embeddings 3/4"]


def test_progress_lines_update_state_through_task(world, monkeypatch):
    def _child_run_script(cmd, env, *, on_progress=None, timeout=None, **kwargs):
        """Run a local child printing PROGRESS lines instead of the real script."""
        _simulate_outputs(list(cmd))
        return _ORIGINAL_RUN_SCRIPT(_python_child(PROGRESS_CHILD), env, on_progress=on_progress, timeout=timeout)

    monkeypatch.setattr(runner, "run_script", _child_run_script)
    assert world.submit("dense", "member_keys", "csess-a").status_code == 200
    world.run_last_delay()
    statuses = [s["meta"]["status"] for s in world.states]
    assert statuses[0] == "Starting dense embedding generation"
    assert "Generating embeddings 1/2" in statuses and "Generating embeddings 2/2" in statuses
    assert all(s["state"] == "PROGRESS" for s in world.states)


def test_run_script_stdin_is_devnull():
    code = """
    import sys
    data = sys.stdin.read()
    print("stdin-bytes=%d" % len(data), flush=True)
    """
    result = _ORIGINAL_RUN_SCRIPT(_python_child(code), dict(os.environ), timeout=30)
    assert result.returncode == 0
    assert "stdin-bytes=0" in result.stdout


def test_run_script_nonzero_exit_returns_result():
    result = _ORIGINAL_RUN_SCRIPT(_python_child("import sys; sys.stderr.write('bad\\n'); sys.exit(3)"),
                                  dict(os.environ), timeout=30)
    assert result.returncode == 3 and "bad" in result.stderr
    with pytest.raises(runner.ScriptFailedError) as excinfo:
        runner.check_script_result(result, ["python3", "x.py"], {})
    assert excinfo.value.returncode == 3


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="process groups are POSIX only")
def test_run_script_timeout_kills_process_group(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    code = f"""
    import subprocess, sys, time
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    open({str(pid_file)!r}, "w").write(str(child.pid))
    time.sleep(60)
    """
    started = time.monotonic()
    with pytest.raises(runner.ScriptTimeoutError):
        _ORIGINAL_RUN_SCRIPT(_python_child(code), dict(os.environ), timeout=1.5, kill_grace=0.5)
    assert time.monotonic() - started < 20
    grandchild = int(pid_file.read_text())
    deadline = time.monotonic() + 10
    while _pid_alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _pid_alive(grandchild)


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="process groups are POSIX only")
def test_run_script_sigterm_revokes_and_restores_handler():
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signal handlers can only be installed from the main thread")
    safety = _SafetySigterm()
    previous = signal.signal(signal.SIGTERM, safety)
    try:
        timer = threading.Timer(1.0, os.kill, args=(os.getpid(), signal.SIGTERM))
        timer.start()
        started = time.monotonic()
        try:
            with pytest.raises(runner.ScriptRevokedError):
                _ORIGINAL_RUN_SCRIPT(_python_child("import time; time.sleep(60)"), dict(os.environ),
                                     timeout=60, kill_grace=0.5, handle_sigterm=True)
        finally:
            timer.cancel()
        assert time.monotonic() - started < 20
        assert safety.received == 0
        assert signal.getsignal(signal.SIGTERM) is safety
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_sigterm_left_alone_outside_pool_child():
    # In a worker main process (solo pool) SIGTERM is the warm shutdown
    # request: the running script must be allowed to finish.
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signal handlers can only be installed from the main thread")
    assert runner._in_pool_child() is False
    safety = _SafetySigterm()
    previous = signal.signal(signal.SIGTERM, safety)
    try:
        timer = threading.Timer(0.3, os.kill, args=(os.getpid(), signal.SIGTERM))
        timer.start()
        try:
            result = _ORIGINAL_RUN_SCRIPT(_python_child("import time; time.sleep(1.5); print('finished')"),
                                          dict(os.environ), timeout=60)
        finally:
            timer.cancel()
        assert result.returncode == 0 and "finished" in result.stdout
        assert safety.received == 1
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_redact_env_secrets():
    env = {"OPENAI_API_KEY": "fake-openai-member-0001", "OTHER": "plain", "CUSTOM_API_KEY": "fake-custom-0001"}
    text = "error fake-openai-member-0001 and fake-custom-0001 plain"
    cleaned = runner.redact_env_secrets(text, env)
    assert ("fake-openai-member-0001" in cleaned) is False
    assert ("fake-custom-0001" in cleaned) is False
    assert "plain" in cleaned


def test_owner_registry_without_store(monkeypatch):
    monkeypatch.setattr(runner, "get_owner_store", lambda: None)
    assert runner.record_task_owner("task-1", 5) is False
    assert runner.get_task_owner("task-1") is None
    store = FakeRedis()
    monkeypatch.setattr(runner, "get_owner_store", lambda: store)
    assert runner.record_task_owner("", 5) is False
    assert runner.record_task_owner("task-1", 5) is True
    assert runner.get_task_owner("task-1") == 5
    assert runner.get_task_owner("x" * 300) is None


# ----------------------------------------------------------------------
# Static checks
# ----------------------------------------------------------------------
# The pipeline task modules, their runner and the Celery routes (last).
CELERY_SOURCES = tuple(
    os.path.join(REPO_ROOT, "app", "tasks", name)
    for name in ("__init__.py", "runner.py", "extraction.py", "chunking.py", "embeddings.py", "vectordb.py")
) + (os.path.join(REPO_ROOT, "app", "routes", "celery_tasks.py"),)


def test_no_missing_symbol_imports():
    missing = []
    for path in CELERY_SOURCES:
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                continue
            module = importlib.import_module(node.module)
            for alias in node.names:
                if hasattr(module, alias.name):
                    continue
                try:
                    importlib.import_module(f"{node.module}.{alias.name}")
                except ImportError:
                    missing.append(f"{os.path.basename(path)}: {node.module}.{alias.name}")
    assert missing == []


def test_tasks_never_read_credentials_from_env():
    pattern = re.compile(r"run_(initial|dense|sparse)_phase|getenv[(].(PINECONE|WEAVIATE|QDRANT)_")
    env_read = re.compile(r"os\.(getenv|environ)")
    credential_names = set(CREDENTIAL_ENV_MAPPING.values())
    offenders = []
    for path in CELERY_SOURCES[:-1]:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        if pattern.search(text):
            offenders.append(os.path.basename(path))
        for line in text.splitlines():
            if env_read.search(line) and any(name in line for name in credential_names):
                offenders.append(os.path.basename(path) + ": " + line.strip()[:60])
    assert offenders == []
