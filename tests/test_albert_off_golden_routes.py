"""Golden references of the web routes with Albert absent (G7, G8, G9).

These tests capture the CURRENT behaviour of the FastAPI routes on the
baseline code, so that later lots can prove that the default path (Albert
OFF) stays byte-identical:

* G7: argv, mapped env (15 baseline names, fingerprints only), ``timeout``
  kwarg, 403/400 bodies and SSE events of the pipeline, notes, citations and
  clustering routes, plus the pipeline status JSON;
* G8: JSON of ``GET /users/me/credentials`` and ``GET /get_credentials``
  (key order kept, values fingerprinted);
* G9: sha256 of the normalised HTML of the pipeline (``index.html``),
  profile and project detail pages, for an admin and a non-admin, plus a
  per-line hash list so that a mismatch names the lines that moved.

Golden files live in ``tests/fixtures/albert/golden_off/routes/``. They are
rewritten only when ``RAGPY_UPDATE_GOLDEN=1``; otherwise every test compares
and fails with a unified diff.

Albert is OFF but present: ``ALBERT_ENABLED=0`` with a fake Albert key in
the server env, in the settings ``.env`` and in the database credentials of
two personas, so the goldens also prove that a key alone changes nothing.

Hygiene: real credentials are removed from the process env, a fake server
env (``fake-`` values only) is installed, dotenv discovery is pointed at an
absent file, every default-model source is pinned to the fake env default,
subprocess launchers and every LLM / Zotero / web helper reached in-process
are replaced by recording fakes (real process creation is blocked), and
secrets are only ever stored as ``sha256(value)[:12]``. Paths are normalised
to ``<RAGPY>`` and ``<TMP>`` (the fixture session folders keep their
deterministic names), timestamps and durations to ``<T>``, dates to
``<DATE>``; database rows get a fixed creation time.

Run: pytest tests/test_albert_off_golden_routes.py -q
"""

import ast
import asyncio
import csv
import difflib
import functools
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import dotenv
import dotenv.main
import pytest

# --- sys.path: make the repo root and scripts/ importable (repo pattern) ---
_THIS = os.path.dirname(os.path.abspath(__file__))
_RAGPY_ROOT = os.path.dirname(_THIS)
for _p in (_RAGPY_ROOT, os.path.join(_RAGPY_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.main import app  # noqa: E402
from app.database.base import Base  # noqa: E402
from app.database.session import get_db  # noqa: E402
from app.models import user as _user_models, project as _project_models  # noqa: E402,F401
from app.models import audit as _audit_models, pipeline_session as _ps_models  # noqa: E402,F401
from app.models import background_task as _bg_models  # noqa: E402,F401
from app.models.user import User  # noqa: E402
from app.models.project import Project  # noqa: E402
from app.models.pipeline_session import PipelineSession, SessionOwner  # noqa: E402
from app.core.security import create_access_token  # noqa: E402
from app.core.credentials import encrypt_credentials, mask_credential  # noqa: E402
import app.config as app_settings_module  # noqa: E402
import app.core.config as core_config  # noqa: E402
import app.core.credentials as credentials_core  # noqa: E402
import app.routes.processing as processing_routes  # noqa: E402
import app.routes.citations as citation_routes  # noqa: E402
import app.routes.pipeline as pipeline_routes  # noqa: E402
import app.routes.settings as settings_routes  # noqa: E402
import app.routes.pages as pages_routes  # noqa: E402
import app.utils.sse_helpers as sse_helpers  # noqa: E402
import app.utils.llm_note_generator as llm_note_generator  # noqa: E402
import app.utils.book_note_generator as book_note_generator  # noqa: E402
import app.utils.citation_filter as citation_filter  # noqa: E402
import app.utils.parallel_citation_processor as parallel_citation_processor  # noqa: E402
import app.utils.zotero_client as zotero_client  # noqa: E402
import scripts.rad_clustering as rad_clustering  # noqa: E402


# ======================================================================
# Literal lists (never derived from the code under test)
# ======================================================================
# The 15 credential env names of the baseline, in the baseline order.
BASELINE_ENV_NAMES = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENROUTER_DEFAULT_MODEL",
    "MISTRAL_API_KEY",
    "MISTRAL_OCR_MODEL",
    "MISTRAL_API_BASE_URL",
    "PINECONE_API_KEY",
    "PINECONE_ENV",
    "WEAVIATE_API_KEY",
    "WEAVIATE_URL",
    "QDRANT_API_KEY",
    "QDRANT_URL",
    "ZOTERO_API_KEY",
    "ZOTERO_USER_ID",
    "ZOTERO_GROUP_ID",
)

# The 6 OCRResult fields read by getattr in the golden harness (shared
# literal of the OFF goldens; the route goldens do not read OCRResult).
OCR_RESULT_FIELDS = ("text", "provider", "partial", "pages_done", "pages_total", "error")

# Other env names cleared before each test (every ALBERT_* name is cleared too).
EXTRA_CLEARED_ENV_NAMES = ("OCR_ENABLE_ALBERT", "EMBEDDING_PROVIDER", "RAGPY_DOTENV_DENY")

# Fake "server .env": what an admin falls back to. ZOTERO_GROUP_ID stays
# absent so the Zotero library type is "users", as in a typical install.
FAKE_SERVER_ENV = {
    "OPENAI_API_KEY": "fake-openai-0001",
    "OPENROUTER_API_KEY": "fake-openrouter-0001",
    "OPENROUTER_DEFAULT_MODEL": "fake-openrouter-model-0001",
    "MISTRAL_API_KEY": "fake-mistral-0001",
    "MISTRAL_OCR_MODEL": "fake-mistral-model-0001",
    "MISTRAL_API_BASE_URL": "fake-mistral-url-0001",
    "PINECONE_API_KEY": "fake-pinecone-0001",
    "PINECONE_ENV": "fake-pinecone-env-0001",
    "WEAVIATE_API_KEY": "fake-weaviate-0001",
    "WEAVIATE_URL": "fake-weaviate-url-0001",
    "QDRANT_API_KEY": "fake-qdrant-0001",
    "QDRANT_URL": "fake-qdrant-url-0001",
    "ZOTERO_API_KEY": "fake-zotero-0001",
    "ZOTERO_USER_ID": "fake-zotero-user-0001",
}

# Albert OFF but present (invariants 2 and 3): set after the ALBERT_* purge.
FAKE_ALBERT_KEY = "fake-albert-key-0001"
ALBERT_OFF_ENV = {"ALBERT_ENABLED": "0", "ALBERT_API_KEY": FAKE_ALBERT_KEY}

# The one web default model: env value, module pins and effective default
# all agree on it (it has no "/", so it routes to OpenAI).
FAKE_DEFAULT_MODEL = FAKE_SERVER_ENV["OPENROUTER_DEFAULT_MODEL"]
# Model values a helper may receive when the request left the model empty.
DEFAULT_MODEL_FORMS = (None, "", FAKE_DEFAULT_MODEL)

# Modules whose ``find_dotenv`` name (present or future) is neutralised.
DOTENV_DISCOVERY_MODULES = (
    dotenv, dotenv.main, app_settings_module, credentials_core, llm_note_generator, book_note_generator,
    citation_filter, parallel_citation_processor, processing_routes, citation_routes, pages_routes,
    settings_routes, pipeline_routes, sse_helpers,
)
# Modules holding an import-time copy of OPENROUTER_DEFAULT_MODEL.
DEFAULT_MODEL_PIN_MODULES = (pages_routes, citation_routes, citation_filter, parallel_citation_processor)

# Launchers that must never create a real process inside the route modules.
BLOCKED_ASYNCIO_NAMES = ("create_subprocess_exec", "create_subprocess_shell")
BLOCKED_SUBPROCESS_NAMES = ("Popen", "run", "call", "check_call", "check_output", "getoutput", "getstatusoutput")

# Fixed creation time of every database row (pages render dates).
FIXED_CREATED_AT = datetime(2025, 1, 15, 10, 0, 0)

# Personal (database) credentials per persona.
PERSONAS = ("admin", "member_keys", "member_nokeys")
PERSONA_EMAILS = {
    "admin": "admin@golden.test",
    "member_keys": "member.keys@golden.test",
    "member_nokeys": "member.nokeys@golden.test",
}
PERSONA_ROLES = {
    "admin": ["USER", "ADMIN"],
    "member_keys": ["USER"],
    "member_nokeys": ["USER"],
}
FAKE_DB_CREDENTIALS = {
    "admin": {"openai_api_key": "fake-openai-admin-db-0001", "albert_api_key": "fake-albert-key-admin-db-0001"},
    "member_keys": {
        "openai_api_key": "fake-openai-member-0001",
        "mistral_api_key": "fake-mistral-member-0001",
        "pinecone_api_key": "fake-pinecone-member-0001",
        "zotero_api_key": "fake-zotero-member-0001",
        "zotero_user_id": "fake-zotero-user-member-0001",
        "albert_api_key": "fake-albert-key-member-0001",
    },
    "member_nokeys": {},
}

# Keyword arguments of in-process helpers whose values are credentials.
SECRET_KWARG_NAMES = (
    "api_key",
    "openai_api_key",
    "openrouter_api_key",
    "zotero_api_key",
    "library_id",
)
# Keyword arguments carrying long texts (stored as length + fingerprint).
TEXT_KWARG_NAMES = ("text_content", "web_content", "note_html", "pdf_bytes")

# Helper kwargs frozen by the goldens (baseline names); other kwargs are
# ignored. ``model`` is recorded literally when the request (form or
# citation config) supplied one, and as ``<default>`` when it was left empty
# and the helper got None, "" or the one fake default (later lots may make
# the default explicit); any other value is recorded literally.
MODEL_KWARG = "model"
HELPER_KWARG_ALLOWLIST = {
    "build_note_html_async": (
        "metadata", "text_content", "model", "mode", "use_llm", "openai_api_key", "openrouter_api_key",
    ),
    "build_abstract_text_async": ("metadata", "text_content", "model", "openai_api_key", "openrouter_api_key"),
    "build_book_note_async": ("metadata", "text_content", "model", "openai_api_key", "openrouter_api_key"),
    "verify_api_key": ("api_key",),
    "check_note_exists": ("library_type", "library_id", "item_key", "sentinel", "api_key"),
    "create_child_note": (
        "library_type", "library_id", "item_key", "note_html", "tags", "api_key", "library_version",
    ),
    "update_item_abstract": ("library_type", "library_id", "item_key", "new_abstract", "api_key"),
    "process_citations_parallel": (
        "citations", "config", "openai_api_key", "openrouter_api_key", "batch_size",
    ),
    "fetch_citation_content": ("article_url", "fulltext_url", "timeout", "max_chars"),
    "filter_citation_with_llm": (
        "citation", "web_content", "web_content_source", "project_name", "project_description",
        "collection_name", "collection_description", "model", "openai_api_key", "openrouter_api_key",
    ),
    "get_or_create_collection": ("library_type", "library_id", "collection_name", "description", "api_key"),
    "fetch_collection_items": ("library_type", "library_id", "collection_key", "api_key"),
    "create_or_update_item": ("library_type", "library_id", "item_data", "api_key"),
    "download_pdf": ("fulltext_url", "timeout", "max_size_mb", "convert_html"),
    "upload_file_attachment": ("library_type", "library_id", "parent_item_key", "filename", "api_key"),
    "run_clustering_pipeline": (
        "embeddings_json_path", "output_dir", "session_name", "min_cluster_size", "aggregation_method",
    ),
}

# Lines a pipeline script could print; each route's own parser turns them
# into SSE events (words that trigger the error path are avoided).
SAMPLE_SUBPROCESS_LINES = (
    "INFO - Detected Zotero JSON format: direct array with 2 items",
    "PROGRESS|init|2|Found 2 documents to process",
    "PROGRESS|row|1/2|Processing: doc_a.pdf",
    "Processing Zotero items: 50%|#####     | 1/2 [00:01<00:01,  1.00it/s]",
    "Document #1 traité, 3 chunks produits.",
    "PROGRESS|row|2/2|Processing: doc_b.pdf",
)
SSE_COMPLETE_EVENT = 'data: {"type": "complete", "message": "Process completed successfully"}\n\n'

GOLDEN_DIR = Path(_THIS) / "fixtures" / "albert" / "golden_off" / "routes"
# Session folders are deterministic fixture names (gsess-*): never masked,
# so a route reading or writing the wrong session shows up in the diff.
TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
)
DURATION_RE = re.compile(r"\b\d+\.\d+ ?(?:s|sec|seconds)\b")
DATE_RE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")
MAX_SSE_EVENTS = 20

# SSE cases whose missing-credential branch crashes on the baseline (the
# error_generator closure reads ``e`` after the except block unbound it):
# status 200, no event, NameError. They live in a dedicated golden that
# accepts either that frozen crash or the single fixed error event.
KNOWN_DEFECT_GOLDEN = "g7_sse_credential_error_known_defect.json"
CHUNKING_SSE_DEFECT_CASES = (
    "member_keys:gemini_slug", "member_nokeys:default", "member_nokeys:gpt-4o-mini", "member_nokeys:gemini_slug",
)
DENSE_SSE_DEFECT_CASES = ("member_nokeys",)

# G9: per-line hashes (diagnostics only; the page sha256 is the verdict).
G9_LINES_GOLDEN = "g9_pages_html_lines.json"
LINE_HASH_CHARS = 8
MAX_DIAGNOSTIC_LINES = 60


# ======================================================================
# Helpers
# ======================================================================
def _fp(value):
    """Return the ``sha256(value)[:12]`` fingerprint of a secret, or None."""
    if value is None:
        return None
    return "fp:" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def _env_fp(env):
    """Restrict a subprocess env to the 15 baseline names, as fingerprints."""
    if env is None:
        return None
    return {name: (_fp(env[name]) if name in env else None) for name in BASELINE_ENV_NAMES}


def _jsonable(value):
    """Convert a captured value into deterministic JSON-compatible data."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"bytes_len": len(value)}
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    return "<object " + type(value).__name__ + ">"


def _describe_kwargs(kwargs, allowed):
    """Describe the ``allowed`` helper kwargs: secrets fingerprinted, texts summarised."""
    out = {}
    for name in allowed:
        if name not in kwargs:
            continue
        value = kwargs[name]
        if name in SECRET_KWARG_NAMES:
            out[name] = _fp(value) if value else value
        elif name in TEXT_KWARG_NAMES:
            raw = value if isinstance(value, (str, bytes)) else str(value)
            data = raw.encode("utf-8") if isinstance(raw, str) else raw
            out[name] = {"len": len(raw), "sha": "sha:" + hashlib.sha256(data).hexdigest()[:12]}
        elif name == "citations":
            out[name] = [getattr(c, "title", None) for c in value]
        else:
            out[name] = _jsonable(value)
    return out


def _bind(names, args, kwargs):
    """Map positional ``args`` onto ``names`` and merge with ``kwargs``."""
    bound = dict(zip(names, args))
    bound.update(kwargs)
    return bound


def _argv_value(cmd, flag):
    """Return the value following ``flag`` in an argv list, or None."""
    if flag in cmd:
        idx = cmd.index(flag)
        if idx + 1 < len(cmd):
            return cmd[idx + 1]
    return None


def _script_name(cmd):
    """Return the basename of the script launched by ``cmd``."""
    for item in cmd:
        if str(item).endswith(".py"):
            return os.path.basename(str(item))
    return ""


def _chunks(kind):
    """Build the 3 fake chunks of a session (plain, dense or sparse)."""
    chunks = []
    for idx, (title, text) in enumerate(
        (("Golden article A", "Texte A1"), ("Golden article A", "Texte A2"), ("Golden article B", "Texte B1")),
        start=1,
    ):
        chunk = {
            "id": f"chunk-{idx}",
            "doc_id": "doc-golden",
            "chunk_index": idx,
            "total_chunks": 3,
            "title": title,
            "text": text,
        }
        if kind in ("dense", "sparse"):
            chunk["embedding"] = [0.1, 0.2, 0.3]
        if kind == "sparse":
            chunk["sparse_embedding"] = {"indices": [1], "values": [1.0]}
        chunks.append(chunk)
    return chunks


def _write_json(path, payload):
    """Write ``payload`` as UTF-8 JSON at ``path``."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _write_output_csv(path):
    """Write the 2-row ``output.csv`` of a session (texteocr filled)."""
    rows = [
        ["itemKey", "title", "authors", "date", "texteocr", "texteocr_provider"],
        ["ITEMA001", "Golden article A", "Doe, J.", "2024", "Texte OCR du document A.", "mistral"],
        ["ITEMB002", "Golden article B", "Roe, R.", "2023", "Texte OCR du document B.", "mistral"],
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)


def _write_zotero_json(folder):
    """Write a minimal Zotero export next to nothing else (fresh session)."""
    _write_json(os.path.join(folder, "biblio.json"), [{"key": "ITEMA001", "title": "Golden article A"}])


def _write_full_session(folder):
    """Write every pipeline artefact of a completed session."""
    _write_output_csv(os.path.join(folder, "output.csv"))
    _write_json(os.path.join(folder, "output_chunks.json"), _chunks("plain"))
    _write_json(os.path.join(folder, "output_chunks_with_embeddings.json"), _chunks("dense"))
    _write_json(os.path.join(folder, "output_chunks_with_embeddings_sparse.json"), _chunks("sparse"))


def _simulate_script_outputs(cmd):
    """Write the files the launched script would have produced."""
    script = _script_name(cmd)
    if script == "rad_dataframe.py":
        out = _argv_value(cmd, "--output")
        if out:
            _write_output_csv(out)
    elif script == "rad_chunk.py":
        out_dir = _argv_value(cmd, "--output")
        phase = _argv_value(cmd, "--phase")
        names = {
            "initial": ("output_chunks.json", "plain"),
            "dense": ("output_chunks_with_embeddings.json", "dense"),
            "sparse": ("output_chunks_with_embeddings_sparse.json", "sparse"),
        }
        if out_dir and phase in names:
            fname, kind = names[phase]
            _write_json(os.path.join(out_dir, fname), _chunks(kind))
    elif script == "rad_clustering.py":
        out_dir = _argv_value(cmd, "--output")
        if out_dir:
            _write_json(
                os.path.join(out_dir, "clustering_results.json"),
                {"n_documents": 2, "n_clusters": 1, "n_noise": 0, "documents": []},
            )


def _fake_stdout(cmd):
    """Return a realistic stdout for the launched script."""
    if _script_name(cmd) == "rad_vectordb.py":
        journal_dir = os.path.dirname(_argv_value(cmd, "--input") or "")
        return (
            "Loading chunks (build 2026-09-26)\n"
            "Upserting 3 vectors in 1 batch\n"
            "\n=== Result ===\n"
            "Status: success\n"
            "Message: 3 chunks inserted\n"
            "Inserted: 3\n"
            "Skipped (dedup): 1\n"
            f"Dedup journal: {os.path.join(journal_dir, 'dedup_journal.jsonl')}\n"
        )
    return "PROGRESS|row|1/1|done\n"


def _prefix_length(pair):
    """Sort key: length of the path prefix of a (prefix, token) pair."""
    return len(pair[0])


class _Normaliser:
    """Replace machine-specific paths, timestamps, durations and dates."""

    def __init__(self, tmp_path):
        """Collect the path prefixes to replace, longest first."""
        pairs = []
        for real in {str(tmp_path), os.path.realpath(str(tmp_path))}:
            pairs.append((real, "<TMP>"))
        for real in {_RAGPY_ROOT, os.path.realpath(_RAGPY_ROOT)}:
            pairs.append((real, "<RAGPY>"))
        self.pairs = sorted(pairs, key=_prefix_length, reverse=True)

    def text(self, value):
        """Normalise one string."""
        for real, token in self.pairs:
            value = value.replace(real, token)
        value = TIMESTAMP_RE.sub("<T>", value)
        value = DURATION_RE.sub("<T>", value)
        return DATE_RE.sub("<DATE>", value)

    def __call__(self, obj):
        """Normalise every string (keys included) of a JSON-like object."""
        if isinstance(obj, dict):
            return {self.text(str(k)): self(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self(v) for v in obj]
        if isinstance(obj, str):
            return self.text(obj)
        return obj


class _Recorder:
    """Recording fakes for subprocess launchers and in-process helpers."""

    def __init__(self):
        """Start with an empty call log and no supplied model."""
        self.calls = []
        self.supplied_model = None

    def take(self):
        """Return the calls recorded since the last take and reset the log."""
        calls, self.calls = self.calls, []
        return calls

    def describe_model(self, value):
        """Describe a ``model`` kwarg against the model the request supplied."""
        if not self.supplied_model and value in DEFAULT_MODEL_FORMS:
            return "<default>"
        return value

    def record(self, fn, kwargs):
        """Append one helper call described by its allowlisted kwargs."""
        described = _describe_kwargs(kwargs, HELPER_KWARG_ALLOWLIST[fn])
        if MODEL_KWARG in described:
            described[MODEL_KWARG] = self.describe_model(kwargs[MODEL_KWARG])
        self.calls.append({"fn": fn, "kwargs": described})

    def blocked_launch(self, qualname, *args, **kwargs):
        """Record then refuse a real process launch (guard, never expected)."""
        self.calls.append({"fn": "BLOCKED " + qualname})
        raise AssertionError("golden harness: real process launch attempted via " + qualname)

    async def fake_tracked(self, *args, **kwargs):
        """Stand in for ``processing.run_tracked_subprocess``."""
        bound = _bind(("cmd", "session_folder", "timeout", "env"), args, kwargs)
        cmd = list(bound.get("cmd") or [])
        self.calls.append({
            "fn": "run_tracked_subprocess",
            "argv": cmd,
            "session_folder": bound.get("session_folder"),
            "timeout": bound.get("timeout", "<default>"),
            "env": _env_fp(bound.get("env")),
        })
        _simulate_script_outputs(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=_fake_stdout(cmd), stderr="")

    async def fake_sse(self, cmd, progress_parser, **kwargs):
        """Stand in for ``sse_helpers.run_subprocess_with_sse``."""
        self.calls.append({
            "fn": "run_subprocess_with_sse",
            "argv": list(cmd),
            "session_folder": kwargs.get("session_folder"),
            "timeout": kwargs.get("timeout", "<default>"),
            "error_keywords": _jsonable(kwargs.get("error_keywords", "<default>")),
            "env": _env_fp(kwargs.get("env")),
        })
        for line in SAMPLE_SUBPROCESS_LINES:
            event = progress_parser(line)
            if event:
                yield f"data: {json.dumps(event)}\n\n"
        _simulate_script_outputs(cmd)
        yield SSE_COMPLETE_EVENT

    # --- notes (generate_zotero_notes_sse) ---
    async def fake_note_html(self, *args, **kwargs):
        """Stand in for ``build_note_html_async``."""
        self.record("build_note_html_async", _bind(("metadata", "text_content"), args, kwargs))
        return ("<!-- ragpy-note:golden -->", "<p>Golden note</p>")

    async def fake_abstract(self, *args, **kwargs):
        """Stand in for ``build_abstract_text_async``."""
        self.record("build_abstract_text_async", _bind(("metadata", "text_content"), args, kwargs))
        return "Golden abstract."

    async def fake_book_note(self, *args, **kwargs):
        """Stand in for ``build_book_note_async``."""
        self.record("build_book_note_async", _bind(("metadata", "text_content"), args, kwargs))
        return ("<!-- ragpy-book:golden -->", "<p>Golden book note</p>")

    def fake_verify_api_key(self, *args, **kwargs):
        """Stand in for ``zotero_client.verify_api_key``."""
        self.record("verify_api_key", _bind(HELPER_KWARG_ALLOWLIST["verify_api_key"], args, kwargs))
        return {"key": "ok"}

    def fake_check_note_exists(self, *args, **kwargs):
        """Stand in for ``zotero_client.check_note_exists``."""
        self.record("check_note_exists", _bind(HELPER_KWARG_ALLOWLIST["check_note_exists"], args, kwargs))
        return False

    def fake_create_child_note(self, *args, **kwargs):
        """Stand in for ``zotero_client.create_child_note``."""
        self.record("create_child_note", kwargs)
        return {"success": True, "new_version": 7}

    def fake_update_item_abstract(self, *args, **kwargs):
        """Stand in for ``zotero_client.update_item_abstract``."""
        self.record("update_item_abstract", kwargs)
        return {"success": True, "new_abstract_length": 16}

    # --- citations ---
    async def fake_process_citations_parallel(self, *args, **kwargs):
        """Stand in for ``process_citations_parallel`` (async generator)."""
        bound = _bind(HELPER_KWARG_ALLOWLIST["process_citations_parallel"], args, kwargs)
        self.record("process_citations_parallel", bound)
        citations = list(bound.get("citations") or [])
        yield ("init", {"total": len(citations), "batch_size": bound.get("batch_size")})
        for idx, citation in enumerate(citations, start=1):
            citation_dict = citation.model_dump(mode="json")
            relevant = idx == 1
            yield ("progress", {
                "current": idx,
                "total": len(citations),
                "status": "relevant" if relevant else "skipped",
                "title": citation_dict.get("title", "")[:50],
                "filter_result": {
                    "zotero_item": {"itemType": "journalArticle", "title": citation_dict.get("title")},
                    "relevance_score": 0.9,
                    "relevance_reason": "golden reason",
                } if relevant else None,
                "citation": citation_dict,
                "web_source": "article_url",
                "error_message": None,
            })
        yield ("complete", {"relevant": 1, "skipped": max(len(citations) - 1, 0), "errors": 0})

    async def fake_fetch_citation_content(self, *args, **kwargs):
        """Stand in for ``fetch_citation_content``."""
        self.record("fetch_citation_content", kwargs)
        return ("Golden web content.", "article_url")

    async def fake_filter_citation_with_llm(self, *args, **kwargs):
        """Stand in for ``filter_citation_with_llm``."""
        self.record("filter_citation_with_llm", kwargs)
        return {
            "zotero_item": {"itemType": "journalArticle", "title": "Golden citation one", "DOI": "N/A"},
            "relevance_score": 0.9,
            "relevance_reason": "golden reason",
        }

    def fake_get_or_create_collection(self, *args, **kwargs):
        """Stand in for ``get_or_create_collection``."""
        self.record("get_or_create_collection", kwargs)
        return {"key": "COLLGOLD1"}

    def fake_fetch_collection_items(self, *args, **kwargs):
        """Stand in for ``fetch_collection_items``."""
        self.record("fetch_collection_items", kwargs)
        return []

    def fake_create_or_update_item(self, *args, **kwargs):
        """Stand in for ``create_or_update_item``."""
        self.record("create_or_update_item", kwargs)
        return {"success": True, "item_key": "ITEMGOLD1", "action": "created"}

    async def fake_download_pdf(self, *args, **kwargs):
        """Stand in for ``download_pdf`` (never expected: no fulltext_url)."""
        self.record("download_pdf", kwargs)
        return SimpleNamespace(success=False, pdf_bytes=None, error="golden: no download")

    def fake_upload_file_attachment(self, *args, **kwargs):
        """Stand in for ``upload_file_attachment`` (never expected)."""
        self.record("upload_file_attachment", kwargs)
        return {"success": False, "message": "golden: no upload"}

    async def fake_start_task(self, task_id, coroutine, db):
        """Stand in for ``background_task_manager.start_task``: never runs it."""
        self.calls.append({"fn": "background_task_manager.start_task", "task_id": task_id})
        coroutine.close()

    def fake_run_clustering_pipeline(self, *args, **kwargs):
        """Stand in for ``rad_clustering.run_clustering_pipeline``."""
        self.record("run_clustering_pipeline", kwargs)
        return {
            "n_documents": 2,
            "n_clusters": 1,
            "n_noise": 0,
            "noise_ratio": 0.0,
            "cluster_sizes": {"0": 2},
        }


def _guard_llm_clients(*args, **kwargs):
    """Fail loudly if a real LLM client were about to be built."""
    raise AssertionError("golden harness: real LLM client construction attempted")


class _GuardedModule:
    """Proxy of a module whose process-launching functions are blocked.

    Installed in place of the ``asyncio`` / ``subprocess`` names of the route
    modules only, so the rest of the process keeps the real modules.
    """

    def __init__(self, module, blocked, recorder):
        """Wrap ``module``; calling a name of ``blocked`` records and raises."""
        self._module = module
        self._blocked = frozenset(blocked)
        self._recorder = recorder

    def __getattr__(self, name):
        """Delegate to the wrapped module, except for the blocked launchers."""
        if name in self._blocked:
            return functools.partial(self._recorder.blocked_launch, self._module.__name__ + "." + name)
        return getattr(self._module, name)


def _dotenv_path_factory(absent_path):
    """Build a ``find_dotenv`` stand-in that always points at ``absent_path``."""

    def _find_absent_dotenv(*args, **kwargs):
        """Return the absent dotenv path (the host ``.env`` is never found)."""
        return absent_path

    return _find_absent_dotenv


def _leaf_errors(excs):
    """Name the exceptions (exception groups flattened) that escaped the app.

    Only type names are kept: messages differ between Python versions.
    """
    out = []
    stack = list(excs)
    while stack:
        exc = stack.pop(0)
        subs = getattr(exc, "exceptions", None)
        if subs:
            stack[0:0] = list(subs)
            continue
        out.append(type(exc).__name__)
    return out


def _fingerprint_legend():
    """Map each fake credential fingerprint to a readable label."""
    legend = {}
    for name, value in FAKE_SERVER_ENV.items():
        legend.setdefault(_fp(value), "server_env." + name)
    for persona in PERSONAS:
        for key, value in FAKE_DB_CREDENTIALS[persona].items():
            legend.setdefault(_fp(value), f"db.{persona}.{key}")
            legend.setdefault(_fp(mask_credential(value)), f"masked(db.{persona}.{key})")
    return legend


def _sse_events(text):
    """Split an SSE body into its ``data:`` payloads (first events only)."""
    events = []
    for block in text.split("\n\n"):
        block = block.strip("\n")
        if not block:
            continue
        events.append(block[len("data: "):] if block.startswith("data: ") else block)
    return events[:MAX_SSE_EVENTS]


def _body(resp):
    """Return the JSON body of a response, or its text."""
    try:
        return resp.json()
    except ValueError:
        return {"text": resp.text}


def _golden_text(payload):
    """Serialise a golden payload (key order kept, never sorted)."""
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _update_mode():
    """True when the goldens must be rewritten (``RAGPY_UPDATE_GOLDEN=1``)."""
    return os.environ.get("RAGPY_UPDATE_GOLDEN") == "1"


def _read_golden(name):
    """Return the parsed golden ``name`` or fail if it does not exist."""
    path = GOLDEN_DIR / name
    if not path.exists():
        pytest.fail(f"golden file missing: {name} (write it with RAGPY_UPDATE_GOLDEN=1 on the baseline)")
    return json.loads(path.read_text(encoding="utf-8"))


def _check_golden(name, payload, hint=""):
    """Write the golden (update mode) or compare it and fail with a diff.

    ``hint`` is appended to the failure message (extra diagnostics).
    """
    text = _golden_text(payload)
    path = GOLDEN_DIR / name
    if _update_mode():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return
    if not path.exists():
        pytest.fail(f"golden file missing: {name} (write it with RAGPY_UPDATE_GOLDEN=1 on the baseline)")
    expected = path.read_text(encoding="utf-8")
    if expected != text:
        diff = "".join(difflib.unified_diff(
            expected.splitlines(True), text.splitlines(True),
            fromfile=f"golden/{name}", tofile="current", n=3,
        ))
        pytest.fail(f"golden mismatch for {name}:\n{diff[:20000]}{hint}")


# ======================================================================
# Fixtures
# ======================================================================
@pytest.fixture(autouse=True)
def _golden_env_hygiene(monkeypatch):
    """Remove real credentials, install the fake server env, Albert OFF with a key."""
    for name in BASELINE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith("ALBERT_"):
            monkeypatch.delenv(name, raising=False)
    for name in EXTRA_CLEARED_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name, value in FAKE_SERVER_ENV.items():
        monkeypatch.setenv(name, value)
    for name, value in ALBERT_OFF_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("DEFAULT_MAX_WORKERS", "1")


@pytest.fixture
def force_split(monkeypatch):
    """Force _extract_text_with_mistral down the split path: tiny page cap.

    Copied from tests/test_ocr_providers.py for parity with the OCR golden
    harness; the route goldens do not request it.
    """
    from scripts import rad_dataframe as rad

    monkeypatch.setattr(rad, "MISTRAL_API_KEY", "fake-mistral-0001")
    monkeypatch.setattr(rad, "MISTRAL_MAX_PAGES", 2)
    monkeypatch.setattr(rad, "MISTRAL_SPLIT_PART_PAGES", 1)
    monkeypatch.setattr(rad, "MISTRAL_AUTO_SPLIT", True)
    monkeypatch.setattr(rad, "MISTRAL_AUTO_COMPRESS", False)


@pytest.fixture
def golden_app(tmp_path, monkeypatch):
    """Build the app under test: temp DB, 3 personas, sessions and fakes."""
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    for module in (core_config, processing_routes, citation_routes, pipeline_routes):
        monkeypatch.setattr(module, "UPLOAD_DIR", str(uploads))
    settings_home = tmp_path / "ragpy_home"
    settings_home.mkdir()
    monkeypatch.setattr(settings_routes, "RAGPY_DIR", str(settings_home))
    # Import-time copies of OPENROUTER_DEFAULT_MODEL depend on the host env:
    # pin them to the fake env default, so every default-model source (env,
    # module copies, a default read fresh later) agrees on one value.
    for module in DEFAULT_MODEL_PIN_MODULES:
        monkeypatch.setattr(module, "DEFAULT_LLM_MODEL", FAKE_DEFAULT_MODEL)
    # dotenv discovery must never reach the host .env (a default model or a
    # key read fresh from it would make the goldens host dependent): every
    # find_dotenv name, present or added later, points at an absent file.
    find_absent = _dotenv_path_factory(str(tmp_path / "absent.env"))
    for module in DOTENV_DISCOVERY_MODULES:
        monkeypatch.setattr(module, "find_dotenv", find_absent, raising=False)
    # cluster_documents inserts into sys.path: work on a copy.
    monkeypatch.setattr(sys, "path", list(sys.path))

    engine = create_engine(
        f"sqlite:///{tmp_path / 'golden.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(autocommit=False, autoflush=False, bind=engine)()

    users = {}
    for persona in PERSONAS:
        creds = FAKE_DB_CREDENTIALS[persona]
        user = User(
            email=PERSONA_EMAILS[persona],
            hashed_password="x",
            roles=list(PERSONA_ROLES[persona]),
            is_active=True,
            is_verified=True,
            api_credentials=encrypt_credentials(creds) if creds else None,
            created_at=FIXED_CREATED_AT,
            updated_at=FIXED_CREATED_AT,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        users[persona] = user
    assert users["admin"].is_admin and not users["member_keys"].is_admin

    for name in ("gsess-full", "gsess-empty") + tuple(
        "gsess-fresh-" + p.replace("_", "-") for p in PERSONAS
    ):
        (uploads / name).mkdir()
    _write_full_session(str(uploads / "gsess-full"))
    for persona in PERSONAS:
        _write_zotero_json(str(uploads / ("gsess-fresh-" + persona.replace("_", "-"))))

    project = Project(
        name="Golden Project",
        description="Golden description",
        owner_id=users["member_nokeys"].id,
        session_folder="gsess-full",
        created_at=FIXED_CREATED_AT,
        updated_at=FIXED_CREATED_AT,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    pipeline_session = PipelineSession(
        project_id=project.id, session_folder="gsess-full", original_filename="biblio.zip", source_type="zip",
        created_at=FIXED_CREATED_AT, updated_at=FIXED_CREATED_AT,
    )
    db.add(pipeline_session)
    # Each fresh folder stands for that persona's own upload outside any project:
    # /upload_zip records its uploader (SessionOwner, audit A02 of 2026-09-27).
    for persona in PERSONAS:
        db.add(SessionOwner(
            session_folder="gsess-fresh-" + persona.replace("_", "-"), user_id=users[persona].id,
            source_type="zip", original_filename="biblio.zip", created_at=FIXED_CREATED_AT,
        ))
    db.commit()

    recorder = _Recorder()
    monkeypatch.setattr(processing_routes, "run_tracked_subprocess", recorder.fake_tracked)
    monkeypatch.setattr(sse_helpers, "run_subprocess_with_sse", recorder.fake_sse)
    # Also cover a launcher import hoisted to module level later on.
    monkeypatch.setattr(processing_routes, "run_subprocess_with_sse", recorder.fake_sse, raising=False)
    # Real process creation from the route modules is refused (and recorded).
    for module in (processing_routes, sse_helpers):
        monkeypatch.setattr(module, "asyncio", _GuardedModule(asyncio, BLOCKED_ASYNCIO_NAMES, recorder), raising=False)
        monkeypatch.setattr(
            module, "subprocess", _GuardedModule(subprocess, BLOCKED_SUBPROCESS_NAMES, recorder), raising=False
        )

    def _yield_db():
        """Yield the shared test session (stands in for get_db())."""
        yield db

    monkeypatch.setattr(processing_routes, "get_db", _yield_db)
    monkeypatch.setattr(llm_note_generator, "build_note_html_async", recorder.fake_note_html)
    monkeypatch.setattr(llm_note_generator, "build_abstract_text_async", recorder.fake_abstract)
    monkeypatch.setattr(book_note_generator, "build_book_note_async", recorder.fake_book_note)
    monkeypatch.setattr(llm_note_generator, "_get_llm_clients", _guard_llm_clients)
    monkeypatch.setattr(citation_filter, "_get_llm_clients", _guard_llm_clients)
    monkeypatch.setattr(zotero_client, "verify_api_key", recorder.fake_verify_api_key)
    monkeypatch.setattr(zotero_client, "check_note_exists", recorder.fake_check_note_exists)
    monkeypatch.setattr(zotero_client, "create_child_note", recorder.fake_create_child_note)
    monkeypatch.setattr(zotero_client, "update_item_abstract", recorder.fake_update_item_abstract)
    monkeypatch.setattr(citation_routes, "process_citations_parallel", recorder.fake_process_citations_parallel)
    monkeypatch.setattr(citation_routes, "fetch_citation_content", recorder.fake_fetch_citation_content)
    monkeypatch.setattr(citation_routes, "filter_citation_with_llm", recorder.fake_filter_citation_with_llm)
    monkeypatch.setattr(citation_routes, "get_or_create_collection", recorder.fake_get_or_create_collection)
    monkeypatch.setattr(citation_routes, "fetch_collection_items", recorder.fake_fetch_collection_items)
    monkeypatch.setattr(citation_routes, "create_or_update_item", recorder.fake_create_or_update_item)
    monkeypatch.setattr(citation_routes, "download_pdf", recorder.fake_download_pdf)
    monkeypatch.setattr(citation_routes, "upload_file_attachment", recorder.fake_upload_file_attachment)
    monkeypatch.setattr(citation_routes.background_task_manager, "start_task", recorder.fake_start_task)
    monkeypatch.setattr(rad_clustering, "run_clustering_pipeline", recorder.fake_run_clustering_pipeline)

    headers = {
        persona: {"Authorization": "Bearer " + create_access_token(subject=str(user.id))}
        for persona, user in users.items()
    }
    server_errors = []

    async def _capturing_app(scope, receive, send):
        """Run the app and remember any exception escaping it.

        The client does not re-raise, so a stream that breaks after its
        headers were sent is observed as the browser sees it (status and
        partial body) while the exception itself goes into the golden.
        """
        try:
            await app(scope, receive, send)
        except BaseException as exc:
            server_errors.append(exc)
            raise

    def _override_get_db():
        """Return the shared test session (dependency override)."""
        return db

    app.dependency_overrides[get_db] = _override_get_db
    try:
        yield SimpleNamespace(
            client=TestClient(_capturing_app, raise_server_exceptions=False),
            server_errors=server_errors,
            db=db,
            users=users,
            headers=headers,
            recorder=recorder,
            norm=_Normaliser(tmp_path),
            uploads=uploads,
            settings_home=settings_home,
            project=project,
            citation_count=0,
        )
    finally:
        app.dependency_overrides.pop(get_db, None)
        db.close()
        engine.dispose()


# ======================================================================
# Case runners
# ======================================================================
def _take_server_errors(env):
    """Return and clear the exceptions that escaped the app."""
    errors = _leaf_errors(env.server_errors)
    del env.server_errors[:]
    return errors


def _post_case(env, case, persona, url, form, sse=False, supplied_model=None):
    """POST a form as ``persona`` and describe the response and the calls.

    ``supplied_model`` is the model the request carries outside the form
    (citation config); by default the form ``model`` field is used.
    """
    form = {k: v for k, v in form.items() if v is not None}
    env.recorder.supplied_model = supplied_model if supplied_model is not None else form.get("model")
    resp = env.client.post(url, data=form, headers=env.headers[persona])
    out = {"case": case, "persona": persona, "request": {"method": "POST", "path": url, "form": form}}
    out["status"] = resp.status_code
    if sse:
        out["content_type"] = resp.headers.get("content-type")
        out["events"] = _sse_events(resp.text)
    else:
        out["body"] = _body(resp)
    out["server_exceptions"] = _take_server_errors(env)
    out["calls"] = env.recorder.take()
    return out


def _get_json_case(env, case, persona, url):
    """GET a JSON route as ``persona`` and describe the response."""
    env.recorder.supplied_model = None
    resp = env.client.get(url, headers=env.headers[persona])
    out = {"case": case, "persona": persona, "request": {"method": "GET", "path": url}}
    out["status"] = resp.status_code
    out["body"] = _body(resp)
    out["server_exceptions"] = _take_server_errors(env)
    out["calls"] = env.recorder.take()
    return out


def _golden(env, name, route, cases, hint=""):
    """Normalise the collected cases and check them against a golden file."""
    cases = env.norm(cases)
    serialised = json.dumps(cases)
    legend = {fp: label for fp, label in _fingerprint_legend().items() if fp in serialised}
    _check_golden(name, {"route": route, "fingerprint_legend": legend, "cases": cases}, hint=hint)


def _fresh(persona):
    """Session folder holding only a Zotero export, for ``persona``."""
    return "gsess-fresh-" + persona.replace("_", "-")


def _make_citation_session(env, persona, model, n_citations=1):
    """Create a project, a pipeline session and its citation files."""
    env.citation_count += 1
    folder = f"gsess-cit-{env.citation_count}"
    path = env.uploads / folder
    path.mkdir()
    citations = [
        {
            "uid": f"GOLDEN:{i:04d}",
            "title": f"Golden citation {word}",
            "authors": ["A. Author"],
            "year": 2024,
            "article_url": f"https://example.org/golden-{i}",
            "doi": f"10.0000/golden.{i}",
        }
        for i, word in zip(range(1, n_citations + 1), ("one", "two", "three"))
    ]
    _write_json(str(path / "publishorperish.json"), citations)
    _write_json(str(path / "config.json"), {
        "project_name": "Golden citations",
        "project_description": "Golden citation filtering",
        "collection_name": "Golden collection",
        "collection_description": "Golden collection description",
        "model": model,
    })
    project = Project(
        name=f"Citations {env.citation_count}", owner_id=env.users[persona].id, session_folder=folder,
        created_at=FIXED_CREATED_AT, updated_at=FIXED_CREATED_AT,
    )
    env.db.add(project)
    env.db.commit()
    env.db.refresh(project)
    ps = PipelineSession(
        project_id=project.id, session_folder=folder, source_type="publishorperish",
        created_at=FIXED_CREATED_AT, updated_at=FIXED_CREATED_AT,
    )
    env.db.add(ps)
    env.db.commit()
    env.db.refresh(ps)
    return project.id, ps.id


CITATION_SCENARIOS = (
    ("admin_openai", "admin", "gpt-4o-mini"),
    ("admin_openrouter_slug", "admin", "google/gemini-2.5-flash"),
    ("nokeys", "member_nokeys", "gpt-4o-mini"),
    ("openrouter_slug_nokey_member_nokeys", "member_nokeys", "google/gemini-2.5-flash"),
    ("openrouter_slug_nokey_member_keys", "member_keys", "google/gemini-2.5-flash"),
)


# ======================================================================
# G7 — pipeline routes (subprocess launchers)
# ======================================================================
def test_g7_process_dataframe(golden_app):
    """G7: argv/env/timeout, 403 and 400 bodies of /process_dataframe."""
    env = golden_app
    cases = [_post_case(env, p, p, "/process_dataframe", {"path": _fresh(p)}) for p in PERSONAS]
    cases.append(_post_case(env, "skip_existing_csv", "member_nokeys", "/process_dataframe", {"path": "gsess-full"}))
    cases.append(_post_case(env, "missing_dir", "admin", "/process_dataframe", {"path": "gsess-missing"}))
    cases.append(_post_case(env, "no_zotero_json", "admin", "/process_dataframe", {"path": "gsess-empty"}))
    _golden(env, "g7_process_dataframe.json", "POST /process_dataframe", cases)


def test_g7_process_dataframe_sse(golden_app):
    """G7: SSE events and launcher call of /process_dataframe_sse."""
    env = golden_app
    url = "/process_dataframe_sse"
    cases = [_post_case(env, p, p, url, {"path": _fresh(p)}, sse=True) for p in PERSONAS]
    cases.append(_post_case(env, "missing_dir", "admin", url, {"path": "gsess-missing"}, sse=True))
    cases.append(_post_case(env, "no_zotero_json", "admin", url, {"path": "gsess-empty"}, sse=True))
    _golden(env, "g7_process_dataframe_sse.json", "POST " + url, cases)


CHUNK_MODELS = (("default", None), ("gpt-4o-mini", "gpt-4o-mini"), ("gemini_slug", "google/gemini-2.5-flash"))


def test_g7_initial_text_chunking(golden_app):
    """G7: /initial_text_chunking for the default, OpenAI and OpenRouter models."""
    env = golden_app
    url = "/initial_text_chunking"
    cases = [
        _post_case(env, f"{p}:{label}", p, url, {"path": "gsess-full", "model": model})
        for p in PERSONAS for label, model in CHUNK_MODELS
    ]
    cases.append(_post_case(env, "missing_dir", "admin", url, {"path": "gsess-missing"}))
    cases.append(_post_case(env, "missing_output_csv", "admin", url, {"path": "gsess-empty"}))
    _golden(env, "g7_initial_text_chunking.json", "POST " + url, cases)


def test_g7_initial_text_chunking_sse(golden_app):
    """G7: SSE twin of /initial_text_chunking (missing-key cases: known-defect golden)."""
    env = golden_app
    url = "/initial_text_chunking_sse"
    cases = [
        _post_case(env, f"{p}:{label}", p, url, {"path": "gsess-full", "model": model}, sse=True)
        for p in PERSONAS for label, model in CHUNK_MODELS
        if f"{p}:{label}" not in CHUNKING_SSE_DEFECT_CASES
    ]
    cases.append(_post_case(env, "missing_dir", "admin", url, {"path": "gsess-missing"}, sse=True))
    cases.append(_post_case(env, "missing_output_csv", "admin", url, {"path": "gsess-empty"}, sse=True))
    _golden(env, "g7_initial_text_chunking_sse.json", "POST " + url, cases)


def _same_outcome(case_a, case_b):
    """Return both cases without their label and request (outcome only)."""
    strip = ("case", "request")
    return (
        {k: v for k, v in case_a.items() if k not in strip},
        {k: v for k, v in case_b.items() if k not in strip},
    )


def test_g7_dense_embedding_generation(golden_app):
    """G7: /dense_embedding_generation (OpenAI key required; OFF ignores embedding_provider=openai)."""
    env = golden_app
    url = "/dense_embedding_generation"
    cases = [_post_case(env, p, p, url, {"path": "gsess-full"}) for p in PERSONAS]
    cases.append(_post_case(
        env, "admin:embedding_provider_openai", "admin", url, {"path": "gsess-full", "embedding_provider": "openai"}
    ))
    cases.append(_post_case(env, "missing_dir", "admin", url, {"path": "gsess-missing"}))
    cases.append(_post_case(env, "missing_chunks", "admin", url, {"path": "gsess-empty"}))
    plain, with_field = _same_outcome(cases[0], cases[len(PERSONAS)])
    assert with_field == plain, "embedding_provider=openai must be ignored while Albert is OFF"
    _golden(env, "g7_dense_embedding_generation.json", "POST " + url, cases)


def test_g7_dense_embedding_generation_sse(golden_app):
    """G7: SSE twin of /dense_embedding_generation (missing-key case: known-defect golden)."""
    env = golden_app
    url = "/dense_embedding_generation_sse"
    cases = [
        _post_case(env, p, p, url, {"path": "gsess-full"}, sse=True)
        for p in PERSONAS if p not in DENSE_SSE_DEFECT_CASES
    ]
    cases.append(_post_case(
        env, "admin:embedding_provider_openai", "admin", url,
        {"path": "gsess-full", "embedding_provider": "openai"}, sse=True,
    ))
    cases.append(_post_case(env, "missing_chunks", "admin", url, {"path": "gsess-empty"}, sse=True))
    plain, with_field = _same_outcome(cases[0], cases[-2])
    assert with_field == plain, "embedding_provider=openai must be ignored while Albert is OFF"
    _golden(env, "g7_dense_embedding_generation_sse.json", "POST " + url, cases)


def _known_defect_specs():
    """List the known-defect SSE cases: (case, persona, sse_url, json_twin_url, form)."""
    specs = []
    for p in PERSONAS:
        for label, model in CHUNK_MODELS:
            if f"{p}:{label}" in CHUNKING_SSE_DEFECT_CASES:
                form = {"path": "gsess-full"}
                if model is not None:
                    form["model"] = model
                specs.append((
                    f"initial_text_chunking_sse:{p}:{label}", p,
                    "/initial_text_chunking_sse", "/initial_text_chunking", form,
                ))
    for p in PERSONAS:
        if p in DENSE_SSE_DEFECT_CASES:
            specs.append((
                f"dense_embedding_generation_sse:{p}", p,
                "/dense_embedding_generation_sse", "/dense_embedding_generation", {"path": "gsess-full"},
            ))
    return specs


def _sse_outcome(out):
    """Keep the comparable part of an SSE case (status, events, errors, calls)."""
    return {
        "status": out["status"],
        "content_type": out["content_type"],
        "events": out["events"],
        "server_exceptions": out["server_exceptions"],
        "calls": out["calls"],
    }


def _parsed_events(outcome):
    """Return ``outcome`` with its events parsed as JSON, or None if one is not JSON."""
    try:
        events = [json.loads(e) for e in outcome["events"]]
    except ValueError:
        return None
    return dict(outcome, events=events)


def test_g7_sse_credential_error_known_defect(golden_app):
    """G7: missing-key SSE branches of chunking and dense (baseline crash, fix allowed).

    Accepted outcomes, nothing else: the frozen baseline crash (200, no
    event, NameError) or the single fixed event ``{type: error, message:
    <the JSON twin's 403 text>, credential_required: <key>}``. In both, no
    launcher is called and no env is built (``calls == []``).

    Lot 9 amendment (decision of 2026-09-27): a case that targets a
    registered session of another user's project mirrors the twin's
    session refusal instead (403, ``{type: error, message: <the twin's 403
    text>}``, no ``credential_required`` since the twin names none). For
    that case (``member_keys:gemini_slug``) both slots hold the refusal,
    although the slot is named ``frozen_crash`` and the file's ``route``
    label still says "missing key".

    Never apply update mode blindly to this golden: since the lot 7 fix the
    non-foreign cases no longer reproduce the baseline crash, so a
    regeneration would replace their ``frozen_crash`` baseline with the
    fixed event. The lot 9 file was recomposed by hand: the 4
    ``member_nokeys`` cases kept byte for byte, only the foreign case taken
    from the regeneration.
    """
    env = golden_app
    specs = _known_defect_specs()
    observed = [
        env.norm(_post_case(env, case, persona, sse_url, form, sse=True))
        for case, persona, sse_url, _twin, form in specs
    ]
    for out in observed:
        assert out["calls"] == [], f"{out['case']}: a launcher or helper was called: {out['calls']}"

    if _update_mode():
        golden_cases = []
        for (case, persona, _sse_url, twin_url, form), out in zip(specs, observed):
            twin = _post_case(env, case, persona, twin_url, form)
            assert twin["status"] == 403 and twin["calls"] == [], f"{case}: JSON twin is not a clean 403"
            fixed_event = {"type": "error", "message": twin["body"]["error"]}
            if "credential_required" in twin["body"]:
                fixed_event["credential_required"] = twin["body"]["credential_required"]
            golden_cases.append({
                "case": case,
                "persona": persona,
                "request": out["request"],
                "json_twin": twin_url,
                "accepted_outcomes": {
                    "frozen_crash": _sse_outcome(out),
                    "fixed_event": dict(_sse_outcome(out), events=[fixed_event], server_exceptions=[]),
                },
            })
        _check_golden(KNOWN_DEFECT_GOLDEN, {
            "route": "POST /initial_text_chunking_sse, POST /dense_embedding_generation_sse (missing key)",
            "policy": "frozen_crash or fixed_event (events compared as parsed JSON); calls must stay []",
            "cases": golden_cases,
        })
        return

    golden = _read_golden(KNOWN_DEFECT_GOLDEN)
    expected_ids = [(c["case"], c["persona"], c["request"]) for c in golden["cases"]]
    assert [(o["case"], o["persona"], o["request"]) for o in observed] == expected_ids, (
        f"{KNOWN_DEFECT_GOLDEN}: the known-defect case list changed"
    )
    failures = []
    for gcase, out in zip(golden["cases"], observed):
        accepted = gcase["accepted_outcomes"]
        outcome = _sse_outcome(out)
        if outcome == accepted["frozen_crash"] or _parsed_events(outcome) == accepted["fixed_event"]:
            continue
        failures.append(
            f"{gcase['case']}: outcome is neither the frozen crash nor the fixed event\n"
            f"observed: {json.dumps(outcome, ensure_ascii=False)}\n"
            f"accepted: {json.dumps(accepted, ensure_ascii=False)}"
        )
    assert not failures, "\n\n".join(failures)


def test_g7_sparse_embedding_generation(golden_app):
    """G7: /sparse_embedding_generation (no key required, env still filtered)."""
    env = golden_app
    url = "/sparse_embedding_generation"
    cases = [_post_case(env, p, p, url, {"path": "gsess-full"}) for p in PERSONAS]
    cases.append(_post_case(env, "missing_dir", "admin", url, {"path": "gsess-missing"}))
    cases.append(_post_case(env, "missing_dense", "admin", url, {"path": "gsess-empty"}))
    _golden(env, "g7_sparse_embedding_generation.json", "POST " + url, cases)


def test_g7_sparse_embedding_generation_sse(golden_app):
    """G7: SSE twin of /sparse_embedding_generation."""
    env = golden_app
    url = "/sparse_embedding_generation_sse"
    cases = [_post_case(env, p, p, url, {"path": "gsess-full"}, sse=True) for p in PERSONAS]
    cases.append(_post_case(env, "missing_dense", "admin", url, {"path": "gsess-empty"}, sse=True))
    _golden(env, "g7_sparse_embedding_generation_sse.json", "POST " + url, cases)


UPLOAD_DB_FORMS = (
    ("pinecone", {"db_choice": "pinecone", "pinecone_index_name": "idx-golden", "pinecone_namespace": "ns-golden"}),
    ("weaviate", {"db_choice": "weaviate", "weaviate_class_name": "Article", "weaviate_tenant_name": "tenant-golden"}),
    ("qdrant", {"db_choice": "qdrant", "qdrant_collection_name": "coll-golden"}),
    ("albert", {"db_choice": "albert"}),
)


def test_g7_upload_db(golden_app):
    """G7: /upload_db for pinecone, weaviate, qdrant and the unknown 'albert' choice."""
    env = golden_app
    url = "/upload_db"
    cases = [
        _post_case(env, f"{p}:{label}", p, url, dict({"path": "gsess-full"}, **form))
        for p in PERSONAS for label, form in UPLOAD_DB_FORMS
    ]
    cases.append(_post_case(env, "missing_dir", "admin", url, {"path": "gsess-missing", "db_choice": "pinecone"}))
    cases.append(_post_case(env, "missing_sparse", "admin", url, {"path": "gsess-empty", "db_choice": "pinecone"}))
    # OFF: the sparse-file check still comes before any albert handling.
    cases.append(_post_case(env, "missing_sparse_albert", "admin", url, {"path": "gsess-empty", "db_choice": "albert"}))
    _golden(env, "g7_upload_db.json", "POST " + url, cases)


# ======================================================================
# G7 — notes and citations (in-process helpers)
# ======================================================================
NOTES_CASES = (
    ("admin:extended:default_model", "admin", "gsess-full", "extended", None),
    ("admin:short:gemini_slug", "admin", "gsess-full", "short", "google/gemini-2.5-flash"),
    ("admin:book:default_model", "admin", "gsess-full", "book", None),
    ("admin:legacy_true_flag", "admin", "gsess-full", "true", "gpt-4o-mini"),
    ("member_keys:extended:gpt-4o-mini", "member_keys", "gsess-full", "extended", "gpt-4o-mini"),
    ("member_keys:extended:gemini_slug", "member_keys", "gsess-full", "extended", "google/gemini-2.5-flash"),
    ("member_nokeys:extended:default_model", "member_nokeys", "gsess-full", "extended", None),
    ("member_nokeys:extended:gemini_slug", "member_nokeys", "gsess-full", "extended", "google/gemini-2.5-flash"),
    ("admin:missing_output_csv", "admin", "gsess-empty", "extended", None),
)


def test_g7_generate_zotero_notes_sse(golden_app):
    """G7: notes SSE, keys forwarded to the note helpers and Zotero calls."""
    env = golden_app
    url = "/generate_zotero_notes_sse"
    cases = [
        _post_case(env, case, p, url, {"session": session, "note_mode": mode, "model": model}, sse=True)
        for case, p, session, mode, model in NOTES_CASES
    ]
    _golden(env, "g7_generate_zotero_notes_sse.json", "POST " + url, cases)


def test_g7_filter_citations_sse(golden_app):
    """G7: citation filtering SSE, with keys, without keys, OpenRouter slug without key."""
    env = golden_app
    cases = []
    for case, persona, model in CITATION_SCENARIOS:
        pid, sid = _make_citation_session(env, persona, model, n_citations=2)
        url = f"/api/projects/{pid}/filter_citations_sse"
        form = {"session_id": sid, "batch_size": 5}
        cases.append(_post_case(env, case, persona, url, form, sse=True, supplied_model=model))
    _golden(env, "g7_filter_citations_sse.json", "POST /api/projects/{project_id}/filter_citations_sse", cases)


def test_g7_batch_import_citations_sse(golden_app):
    """G7: progressive citation import SSE for the same scenarios."""
    env = golden_app
    cases = []
    for case, persona, model in CITATION_SCENARIOS:
        pid, sid = _make_citation_session(env, persona, model, n_citations=1)
        url = f"/api/projects/{pid}/batch_import_citations_sse"
        form = {"session_id": sid, "batch_size": 10}
        cases.append(_post_case(env, case, persona, url, form, sse=True, supplied_model=model))
    _golden(
        env, "g7_batch_import_citations_sse.json", "POST /api/projects/{project_id}/batch_import_citations_sse", cases
    )


def test_g7_filter_citations_background(golden_app):
    """G7: background citation filtering (400 on missing keys)."""
    env = golden_app
    cases = []
    for case, persona, model in CITATION_SCENARIOS:
        pid, sid = _make_citation_session(env, persona, model, n_citations=2)
        url = f"/api/projects/{pid}/filter_citations_bg"
        cases.append(_post_case(env, case, persona, url, {"session_id": sid, "batch_size": 5}, supplied_model=model))
    _golden(env, "g7_filter_citations_bg.json", "POST /api/projects/{project_id}/filter_citations_bg", cases)


# ======================================================================
# G7 — clustering and pipeline status
# ======================================================================
def test_g7_cluster_documents(golden_app):
    """G7: in-process clustering route and its 400 case."""
    env = golden_app
    url = "/cluster_documents"
    cases = [
        _post_case(env, "admin:min_cluster_size", "admin", url,
                   {"session_folder": "gsess-full", "session_name": "Golden", "min_cluster_size": 3}),
        _post_case(env, "member_nokeys:auto", "member_nokeys", url,
                   {"session_folder": "gsess-full", "session_name": "Golden", "aggregation": "max"}),
        _post_case(env, "missing_embeddings", "admin", url,
                   {"session_folder": "gsess-empty", "session_name": "Golden"}),
    ]
    _golden(env, "g7_cluster_documents.json", "POST " + url, cases)


def test_g7_cluster_documents_sse(golden_app):
    """G7: clustering SSE (launcher call, parser output, results event)."""
    env = golden_app
    url = "/cluster_documents_sse"
    cases = [
        _post_case(env, "admin:min_cluster_size", "admin", url,
                   {"session_folder": "gsess-full", "session_name": "Golden", "min_cluster_size": 3}, sse=True),
        _post_case(env, "member_nokeys:auto", "member_nokeys", url,
                   {"session_folder": "gsess-full", "session_name": "Golden", "aggregation": "max"}, sse=True),
        _post_case(env, "missing_embeddings", "admin", url,
                   {"session_folder": "gsess-empty", "session_name": "Golden"}, sse=True),
    ]
    _golden(env, "g7_cluster_documents_sse.json", "POST " + url, cases)


def test_g7_pipeline_session_files(golden_app):
    """G7: pipeline status JSON of a session (owner, admin, foreign member)."""
    env = golden_app
    url = "/api/pipeline/sessions/gsess-full/files"
    cases = []
    for case, persona in (("owner", "member_nokeys"), ("admin_not_member", "admin"), ("foreign_member", "member_keys")):
        out = _get_json_case(env, case, persona, url)
        upload = out["body"].get("files", {}).get("upload") if isinstance(out["body"], dict) else None
        if upload and isinstance(upload.get("files"), list):
            # os.listdir order is filesystem dependent: sorted for the golden.
            upload["files"] = sorted(upload["files"])
        cases.append(out)
    _golden(env, "g7_pipeline_session_files.json", "GET /api/pipeline/sessions/{session_folder}/files", cases)


# ======================================================================
# G8 — credentials JSON (key order kept, values fingerprinted)
# ======================================================================
def test_g8_users_me_credentials(golden_app):
    """G8: masked personal credentials JSON, key order kept."""
    env = golden_app
    cases = []
    for persona in PERSONAS:
        out = _get_json_case(env, persona, persona, "/users/me/credentials")
        body = out["body"]
        if isinstance(body, dict):
            for value in body.values():
                if isinstance(value, dict) and value.get("masked"):
                    value["masked"] = _fp(value["masked"])
        out["key_order"] = list(body) if isinstance(body, dict) else None
        cases.append(out)
    _golden(env, "g8_users_me_credentials.json", "GET /users/me/credentials", cases)


def test_g8_get_credentials(golden_app):
    """G8: admin settings form JSON (settings file + database overlay), key order kept."""
    env = golden_app
    cases = [_get_json_case(env, "admin:no_env_file", "admin", "/get_credentials")]
    lines = ["# fake settings file for the golden harness"]
    lines += [f"{name}={value}" for name, value in FAKE_SERVER_ENV.items() if name != "OPENROUTER_DEFAULT_MODEL"]
    lines += [f"{name}={value}" for name, value in ALBERT_OFF_ENV.items()]
    lines += ["OTHER_SETTING=fake-other-0001", ""]
    (env.settings_home / ".env").write_text("\n".join(lines), encoding="utf-8")
    cases.append(_get_json_case(env, "admin:env_file_and_db_overlay", "admin", "/get_credentials"))
    cases.append(_get_json_case(env, "member_keys:forbidden", "member_keys", "/get_credentials"))
    for out in cases:
        body = out["body"]
        if out["status"] == 200 and isinstance(body, dict):
            out["key_order"] = list(body)
            out["body"] = {k: (_fp(v) if v else v) for k, v in body.items()}
    _golden(env, "g8_get_credentials.json", "GET /get_credentials", cases)


# ======================================================================
# G9 — rendered HTML (sha256 of the normalised page + per-line hashes)
# ======================================================================
def _line_hashes(html):
    """Short sha256 of each line of a normalised page (split on newlines)."""
    return [hashlib.sha256(line.encode("utf-8")).hexdigest()[:LINE_HASH_CHARS] for line in html.split("\n")]


def _encode_line_pages(ordered):
    """Encode per-page line hash lists, near-duplicate pages as edits of a full one.

    ``ordered`` is a list of ``(page_sha, hashes)`` in case order. A page is
    stored in full (``{"lines": "h h ..."}``) unless it is cheaper as
    ``{"base": <full page sha>, "edits": [[i1, i2, "h h ..."], ...]}``, each
    edit replacing ``base[i1:i2]``. Deterministic for a given case order.
    """
    encoded = {}
    full = []
    for sha, hashes in ordered:
        if sha in encoded:
            continue
        best = None
        for base_sha in full:
            base = encoded[base_sha]["lines"].split()
            ops = difflib.SequenceMatcher(None, base, hashes, autojunk=False).get_opcodes()
            edits = [[i1, i2, " ".join(hashes[j1:j2])] for tag, i1, i2, j1, j2 in ops if tag != "equal"]
            cost = sum(j2 - j1 + 1 for tag, i1, i2, j1, j2 in ops if tag != "equal")
            if best is None or cost < best[0]:
                best = (cost, base_sha, edits)
        if best is not None and best[0] * 2 < len(hashes):
            encoded[sha] = {"base": best[1], "edits": best[2]}
        else:
            encoded[sha] = {"lines": " ".join(hashes)}
            full.append(sha)
    return encoded


def _decode_line_page(pages, sha):
    """Rebuild the line hash list of page ``sha`` (empty if unknown)."""
    entry = pages.get(sha)
    if entry is None:
        return []
    if "lines" in entry:
        return entry["lines"].split()
    base = _decode_line_page(pages, entry["base"])
    out, pos = [], 0
    for i1, i2, replacement in entry["edits"]:
        out.extend(base[pos:i1])
        out.extend(replacement.split())
        pos = i2
    out.extend(base[pos:])
    return out


def _g9_line_diagnostics(cases, htmls, line_pages):
    """Explain page hash mismatches line by line (golden vs current lines).

    Returns a text naming, for every page whose sha256 changed, the golden
    line ranges that were replaced or deleted and the current lines (with
    their content) that were inserted or changed.
    """
    try:
        expected_cases = {c["case"]: c for c in _read_golden("g9_pages_html.json")["cases"]}
    except (OSError, ValueError, KeyError):
        return ""
    out = []
    for case in cases:
        expected = expected_cases.get(case["case"])
        if not expected or expected.get("sha256") == case["sha256"]:
            continue
        golden_lines = _decode_line_page(line_pages, expected.get("sha256"))
        current_html = htmls[case["case"]].split("\n")
        matcher = difflib.SequenceMatcher(None, golden_lines, _line_hashes(htmls[case["case"]]), autojunk=False)
        out.append(f"\n--- {case['case']}: page changed (golden {len(golden_lines)} lines, now {len(current_html)})")
        shown = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            out.append(f"  {tag}: golden lines {i1 + 1}-{i2} -> current lines {j1 + 1}-{j2}")
            for idx in range(j1, j2):
                if shown >= MAX_DIAGNOSTIC_LINES:
                    break
                out.append(f"    +{idx + 1}: {current_html[idx][:200]}")
                shown += 1
    return "\n".join(out)


def test_g9_pages_html(golden_app):
    """G9: sha256 of the normalised pipeline, profile and project detail pages."""
    env = golden_app
    pid = env.project.id
    pages = (
        ("index.html", "/pipeline"),
        ("index.html:project", f"/pipeline?project={pid}&session=gsess-full"),
        ("user/profile.html", "/profile"),
        ("user/project_detail.html", f"/project/{pid}"),
    )
    cases = []
    htmls = {}
    ordered_hashes = []
    for template, url in pages:
        for persona in ("admin", "member_nokeys"):
            resp = env.client.get(url, headers=env.headers[persona], follow_redirects=False)
            html = env.norm.text(resp.text)
            sha = hashlib.sha256(html.encode("utf-8")).hexdigest()
            case = f"{template}:{persona}"
            htmls[case] = html
            ordered_hashes.append((sha, _line_hashes(html)))
            cases.append({
                "case": case,
                "persona": persona,
                "request": {"method": "GET", "path": url},
                "status": resp.status_code,
                "content_type": resp.headers.get("content-type"),
                "bytes": len(html.encode("utf-8")),
                "lines": html.count("\n") + 1,
                "sha256": sha,
                "server_exceptions": _take_server_errors(env),
            })
    assert env.recorder.take() == []
    line_pages = _encode_line_pages(ordered_hashes)
    for sha, hashes in ordered_hashes:
        assert _decode_line_page(line_pages, sha) == hashes, "G9 line encoding does not round-trip"
    hint = ""
    if not _update_mode():
        hint = _g9_line_diagnostics(cases, htmls, _read_golden(G9_LINES_GOLDEN)["pages"])
    _golden(env, "g9_pages_html.json", "GET pages (index, profile, project detail)", cases, hint=hint)
    _check_golden(G9_LINES_GOLDEN, {
        "note": (
            f"per-line sha256[:{LINE_HASH_CHARS}] of each normalised page, keyed by page sha256; "
            "diagnostics only (the page sha256 in g9_pages_html.json is the verdict)"
        ),
        "pages": {sha: line_pages[sha] for sha in sorted(line_pages)},
    })


# ======================================================================
# Meta tests on the harness itself
# ======================================================================
def test_golden_harness_uses_literal_lists():
    """The harness reads env names and OCR fields from literal lists only."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    forbidden_attr = "_as" + "dict"
    mapping_name = "CREDENTIAL_ENV_" + "MAPPING"
    literals = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr != forbidden_attr, "OCRResult must be read field by field (literal list)"
            base = node.value
            base_name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
            assert not (base_name == mapping_name and node.attr in ("values", "keys", "items")), (
                "the env names must come from the literal list, not from the mapping"
            )
        if isinstance(node, ast.Name):
            assert node.id != mapping_name, "the credential mapping must not be used by the harness"
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("BASELINE_ENV_NAMES", "OCR_RESULT_FIELDS"):
                literals[node.targets[0].id] = ast.literal_eval(node.value)
    env_names = literals["BASELINE_ENV_NAMES"]
    assert len(env_names) == 15 and len(set(env_names)) == 15
    assert all(isinstance(n, str) and n.isupper() for n in env_names)
    assert literals["OCR_RESULT_FIELDS"] == ("text", "provider", "partial", "pages_done", "pages_total", "error")
    assert tuple(BASELINE_ENV_NAMES) == env_names


def test_golden_harness_uses_fake_credentials():
    """Only fake credentials are installed and none leaks into the goldens."""
    # Every value installed by the harness is a declared fake.
    fakes = list(FAKE_SERVER_ENV.values()) + [FAKE_ALBERT_KEY]
    for creds in FAKE_DB_CREDENTIALS.values():
        fakes.extend(creds.values())
    assert fakes and all(v.startswith("fake-") for v in fakes)
    assert set(FAKE_SERVER_ENV) <= set(BASELINE_ENV_NAMES)
    # The autouse fixture is active here: only fake values; the only Albert
    # names are the OFF switch and a fake key (Albert OFF but present).
    for name in BASELINE_ENV_NAMES:
        assert os.environ.get(name, "fake-").startswith("fake-"), name
    assert sorted(n for n in os.environ if n.startswith("ALBERT_")) == ["ALBERT_API_KEY", "ALBERT_ENABLED"]
    assert os.environ["ALBERT_ENABLED"] == "0"
    assert os.environ["ALBERT_API_KEY"].startswith("fake-")
    for name in EXTRA_CLEARED_ENV_NAMES:
        assert name not in os.environ, name

    files = sorted(p for p in GOLDEN_DIR.glob("*") if p.is_file())
    assert files, "no golden file found"
    key_chars = "[A-Za-z0-9_" + "-]"
    patterns = [
        re.compile("s" + "k-" + key_chars + "{20,}"),
        re.compile("pc" + "sk_"),
        re.compile("Bea" + "rer "),
    ]
    user_path_markers = ("/" + "Users/", "/" + "private/", "/var/" + "folders", "/" + "tmp/")
    for path in files:
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            assert not pattern.search(text), f"{path.name}: key-like pattern found"
        for value in fakes:
            assert value not in text, f"{path.name}: raw credential value (must be fingerprinted)"
        for marker in user_path_markers:
            assert marker not in text, f"{path.name}: machine path not normalised"
