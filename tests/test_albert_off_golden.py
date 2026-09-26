"""Golden references of the baseline pipeline behaviour, Albert absent (OFF path).

Each ``test_g*`` below replays a baseline code path with mocked providers and
compares its observable output (text, OCRResult fields, HTTP requests, CSV and
JSON bytes, call kwargs, cache keys, stdout, logs, subprocess env) with a golden
file stored in ``tests/fixtures/albert/golden_off/scripts/``. Later lots must
keep these outputs byte-identical while Albert is OFF (sprint invariant 1,
decision 17).

Rules of the harness:

* goldens are rewritten ONLY when ``RAGPY_UPDATE_GOLDEN=1``; otherwise any
  difference fails with a unified diff;
* the autouse fixture removes the 15 baseline credential names (literal list),
  every ``ALBERT_*`` / ``RECODE_*`` / ``DEDUP_*`` name, ``OCR_ENABLE_ALBERT``,
  ``EMBEDDING_PROVIDER`` and ``RAGPY_DOTENV_DENY``; tests then set only
  ``fake-`` values, plus ``ALBERT_ENABLED=0`` where a fake Albert key is present
  while Albert is OFF;
* secret values are stored as ``sha256(value)[:12]`` fingerprints;
* machine paths become ``<RAGPY>``, ``<TMP>``, ``<SYSTMP>``, ``<HOME>``, and
  measured durations or timestamps ``<T>``;
* ``OCRResult`` is read field by field (literal list), never unpacked;
* spies and doubles accept extra arguments and bind only the names they know,
  so a new optional parameter never breaks a golden whose output is unchanged.

Regenerate (baseline code only)::

    RAGPY_UPDATE_GOLDEN=1 .venv/bin/python -m pytest tests/test_albert_off_golden.py -q -p no:cacheprovider
"""

import ast
import csv
import difflib
import hashlib
import json
import logging
import os
import re
import runpy
import subprocess
import sys
import tempfile
import time
import types
from types import SimpleNamespace

import pytest
import requests
import fitz  # type: ignore[import-not-found]

# --- sys.path: repo root and scripts/ importable (repo pattern) -------------
_THIS = os.path.dirname(os.path.abspath(__file__))
RAGPY_ROOT = os.path.dirname(_THIS)
for _p in (RAGPY_ROOT, os.path.join(RAGPY_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GOLDEN_DIR = os.path.join(_THIS, "fixtures", "albert", "golden_off", "scripts")
UPDATE_ENV = "RAGPY_UPDATE_GOLDEN"

# The 15 credential env names of the baseline, as a LITERAL list (never derived
# from the credential mapping, which later lots extend).
BASELINE_ENV_NAMES = [
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
]

# The 6 OCRResult fields, read one by one with getattr (LITERAL list).
OCR_RESULT_FIELDS = ["text", "provider", "partial", "pages_done", "pages_total", "error"]

# Names cleared besides the 15 above and the prefixed families.
EXTRA_CLEARED_NAMES = ["OCR_ENABLE_ALBERT", "EMBEDDING_PROVIDER", "RAGPY_DOTENV_DENY"]
CLEARED_PREFIXES = ("ALBERT_", "RECODE_", "DEDUP_")

# Fake values only ('fake-' prefix), one per baseline name.
FAKE_ENV = {
    "OPENAI_API_KEY": "fake-openai-0001",
    "OPENROUTER_API_KEY": "fake-openrouter-0001",
    "OPENROUTER_DEFAULT_MODEL": "fake-openrouter-model-0001",
    "MISTRAL_API_KEY": "fake-mistral-0001",
    "MISTRAL_OCR_MODEL": "fake-mistral-ocr-model-0001",
    "MISTRAL_API_BASE_URL": "fake-mistral-base-url-0001",
    "PINECONE_API_KEY": "fake-pinecone-0001",
    "PINECONE_ENV": "fake-pinecone-env-0001",
    "WEAVIATE_API_KEY": "fake-weaviate-0001",
    "WEAVIATE_URL": "fake-weaviate-url-0001",
    "QDRANT_API_KEY": "fake-qdrant-0001",
    "QDRANT_URL": "fake-qdrant-url-0001",
    "ZOTERO_API_KEY": "fake-zotero-0001",
    "ZOTERO_USER_ID": "fake-zotero-user-0001",
    "ZOTERO_GROUP_ID": "fake-zotero-group-0001",
}

# Sprint convention for the Albert key, and the ONLY two non-baseline names a
# golden may set: a fake key present while Albert stays OFF.
FAKE_ALBERT_KEY = "fake-albert-key-0001"
ALBERT_OFF_ENV = {"ALBERT_ENABLED": "0", "ALBERT_API_KEY": FAKE_ALBERT_KEY}

# rad_chunk prompts with input() when OPENAI_API_KEY is absent at import: a fake
# key is set for these imports only, then removed so that it never shadows the
# load_dotenv() of a module collected later.
_IMPORT_KEY_SET = "OPENAI_API_KEY" not in os.environ
if _IMPORT_KEY_SET:
    os.environ["OPENAI_API_KEY"] = FAKE_ENV["OPENAI_API_KEY"]
try:
    # Imported at collection time on purpose: these modules call load_dotenv()
    # at import, which must happen BEFORE the autouse fixture cleans the env.
    from scripts import rad_dataframe as rad  # noqa: E402
    import rad_chunk as rc  # noqa: E402
    import pandas as pd  # noqa: E402
    import app.models  # noqa: E402,F401  (registers every mapper used by User)
    import app.models.pipeline_session  # noqa: E402,F401
    from app.models.user import User  # noqa: E402
    from app.core import credentials as creds  # noqa: E402
    from app.utils import llm_note_generator as lng  # noqa: E402
    from app.utils import citation_filter as cfilter  # noqa: E402
finally:
    if _IMPORT_KEY_SET and os.environ.get("OPENAI_API_KEY") == FAKE_ENV["OPENAI_API_KEY"]:
        del os.environ["OPENAI_API_KEY"]

# Real OCR functions, captured once before any spy is installed.
_REAL_RAD = {
    name: getattr(rad, name)
    for name in (
        "_extract_text_with_mistral",
        "_mistral_upload_and_ocr",
        "_mistral_upload_and_ocr_once",
        "_extract_text_with_legacy_pdf",
    )
}

FIXED_DOC_ID = 424242424242
_TEMP_NAME_RE = re.compile(r"(ragpy_(?:part_p\d+-\d+|compressed|localocr)_)[a-z0-9_]+(\.pdf|\.md)")
_EMBED_PERF_RE = re.compile(r"en \d+(?:\.\d+)?s \(\d+(?:\.\d+)? emb/s\)")
_TIMESTAMP_KEYS = ("timestamp", "last_updated")


# ======================================================================
# Hygiene
# ======================================================================
@pytest.fixture(autouse=True)
def _golden_env(monkeypatch):
    """Remove every credential/Albert/cache variable so each golden starts clean."""
    for name in BASELINE_ENV_NAMES + EXTRA_CLEARED_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith(CLEARED_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    yield


def _set_fake_env(monkeypatch, *names):
    """Set the given baseline names to their ``fake-`` value from ``FAKE_ENV``."""
    for name in names:
        monkeypatch.setenv(name, FAKE_ENV[name])


def _set_albert_off_env(monkeypatch):
    """Set a fake Albert key while Albert stays disabled (``ALBERT_OFF_ENV``)."""
    for name in ALBERT_OFF_ENV:
        monkeypatch.setenv(name, ALBERT_OFF_ENV[name])


def _bind_known(args, kwargs, names):
    """Map positional ``args`` onto ``names`` and keep the known keyword ones only.

    Extra positional or keyword arguments are ignored on purpose: a later lot
    may add optional parameters without changing any OFF output.
    """
    bound = dict(zip(names, args))
    for name in names:
        if name in kwargs:
            bound[name] = kwargs[name]
    return bound


# ======================================================================
# Golden comparison and normalisation helpers
# ======================================================================
def _updating():
    """True when goldens must be (re)written instead of compared."""
    return os.environ.get(UPDATE_ENV) == "1"


def _golden_path(name):
    """Absolute path of the golden file ``name``."""
    return os.path.join(GOLDEN_DIR, name)


def _fail_with_diff(name, expected, actual):
    """Fail the test with a readable unified diff between golden and actual text."""
    diff = list(difflib.unified_diff(
        expected.splitlines(True), actual.splitlines(True),
        fromfile=f"golden/{name}", tofile="actuel", n=2,
    ))
    shown = "".join(diff[:300])
    more = "" if len(diff) <= 300 else f"\n... ({len(diff) - 300} lignes de diff en plus)"
    pytest.fail(
        f"Sortie différente du golden {name} (régénération interdite après le L0) :\n{shown}{more}",
        pytrace=False,
    )


def _check_golden_bytes(name, data):
    """Compare ``data`` (bytes) with golden ``name``; rewrite it in update mode."""
    path = _golden_path(name)
    if _updating():
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        return
    if not os.path.exists(path):
        pytest.fail(f"Golden absent : {name} (générer sur la baseline avec {UPDATE_ENV}=1)", pytrace=False)
    with open(path, "rb") as fh:
        expected = fh.read()
    if expected != data:
        _fail_with_diff(
            name,
            expected.decode("utf-8", errors="replace"),
            data.decode("utf-8", errors="replace"),
        )


def _dump(obj):
    """Serialise ``obj`` as stable, human-readable JSON text (insertion order kept)."""
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


def _check_golden_json(name, obj):
    """Compare the JSON rendering of ``obj`` with golden ``name``."""
    _check_golden_bytes(name, _dump(obj).encode("utf-8"))


def _path_variants(path):
    """Return the raw and resolved spellings of ``path``, longest first."""
    raw = str(path).rstrip("/")
    variants = {raw, os.path.realpath(raw)}
    return sorted((v for v in variants if v), key=len, reverse=True)


def _normalise_text(text, tmp_path=None):
    """Replace machine paths and random temp-file names by stable tokens."""
    if not isinstance(text, str):
        return text
    pairs = [(p, "<RAGPY>") for p in _path_variants(RAGPY_ROOT)]
    if tmp_path is not None:
        pairs += [(p, "<TMP>") for p in _path_variants(tmp_path)]
    pairs += [(p, "<SYSTMP>") for p in _path_variants(tempfile.gettempdir())]
    pairs += [(p, "<HOME>") for p in _path_variants(os.path.expanduser("~"))]
    for old, new in pairs:
        text = text.replace(old, new)
    return _TEMP_NAME_RE.sub(r"\1<RND>\2", text)


def _normalise_obj(obj, tmp_path=None):
    """Apply ``_normalise_text`` recursively to strings inside lists and dicts."""
    if isinstance(obj, str):
        return _normalise_text(obj, tmp_path)
    if isinstance(obj, list):
        return [_normalise_obj(item, tmp_path) for item in obj]
    if isinstance(obj, tuple):
        return [_normalise_obj(item, tmp_path) for item in obj]
    if isinstance(obj, dict):
        return {key: _normalise_obj(value, tmp_path) for key, value in obj.items()}
    return obj


def _mask_timestamps(obj):
    """Replace the values of timestamp-like keys (``_TIMESTAMP_KEYS``) by ``<T>``."""
    if isinstance(obj, list):
        return [_mask_timestamps(item) for item in obj]
    if isinstance(obj, dict):
        return {key: ("<T>" if key in _TIMESTAMP_KEYS else _mask_timestamps(value))
                for key, value in obj.items()}
    return obj


def _fingerprint(value):
    """Short sha256 fingerprint of a (possibly secret) value; never the value."""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def _freeze(value):
    """Deep JSON copy of a call argument (detaches it from later mutation)."""
    return json.loads(json.dumps(value, ensure_ascii=False))


def _sorted_kwargs(kwargs):
    """Kwargs with top-level keys sorted (call-site order is not a contract)."""
    return {key: _freeze(kwargs[key]) for key in sorted(kwargs)}


def _resp(content, finish="stop"):
    """Fake chat-completion response (``content`` + ``finish_reason``)."""
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason=finish)])


def _fake_vector(text):
    """Deterministic 8-dim vector derived from ``text`` (exact binary fractions)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [byte / 256 for byte in digest[:8]]


def _stdout_lines(text, tmp_path=None):
    """Split captured stdout into normalised lines (paths and measured durations)."""
    text = _normalise_text(text, tmp_path)
    text = _EMBED_PERF_RE.sub("en <T>s (<T> emb/s)", text)
    return text.splitlines()


class _RandomProxy:
    """Stand-in for the ``random`` module: fixed ``randint``/``uniform``, rest delegated."""

    def __init__(self, randint_value=FIXED_DOC_ID, uniform_value=0.0):
        """Store the fixed values returned by ``randint`` and ``uniform``."""
        self._randint_value = randint_value
        self._uniform_value = uniform_value

    def randint(self, a, b):
        """Return the fixed integer (doc_id) whatever the bounds."""
        return self._randint_value

    def uniform(self, a, b):
        """Return the fixed jitter (0.0 by default)."""
        return self._uniform_value

    def __getattr__(self, name):
        """Delegate any other attribute to the real ``random`` module."""
        import random as _random
        return getattr(_random, name)


class _TimeProxy:
    """Stand-in for the ``time`` module whose ``sleep`` only records the delay."""

    def __init__(self, sleeps):
        """Keep the list that receives every requested sleep duration."""
        self._sleeps = sleeps

    def sleep(self, seconds):
        """Record ``seconds`` instead of sleeping."""
        self._sleeps.append(seconds)

    def __getattr__(self, name):
        """Delegate any other attribute to the real ``time`` module."""
        return getattr(time, name)


class _FakeLLMClient:
    """OpenAI-compatible double recording ``chat.completions.create`` and
    ``embeddings.create`` calls in a shared list."""

    def __init__(self, label, calls, content_factory=None):
        """Name the client (``openai`` / ``openrouter``) and bind its call log."""
        self.label = label
        self.calls = calls
        self._content_factory = content_factory
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat_create))
        self.embeddings = SimpleNamespace(create=self._embeddings_create)

    def _chat_create(self, **kwargs):
        """Record the kwargs and answer with a deterministic content."""
        self.calls.append({"client": self.label, "endpoint": "chat.completions.create",
                           "kwargs": _sorted_kwargs(kwargs)})
        if self._content_factory is not None:
            return _resp(self._content_factory(self.label, kwargs))
        user_text = kwargs["messages"][-1]["content"]
        digest = hashlib.sha256(user_text.encode("utf-8")).hexdigest()[:10]
        return _resp(f"[{self.label}:{kwargs['model']}] Texte recodé {digest}.")

    def _embeddings_create(self, **kwargs):
        """Record the kwargs and return one fixed 8-dim vector per input text."""
        self.calls.append({"client": self.label, "endpoint": "embeddings.create",
                           "kwargs": _sorted_kwargs(kwargs)})
        return SimpleNamespace(data=[SimpleNamespace(embedding=_fake_vector(t)) for t in kwargs["input"]])


# ======================================================================
# G1 / G13 — OCR chain harness (scripts/rad_dataframe.py)
# ======================================================================
# Baseline defaults of the OCR module constants, pinned so that a local .env
# can never change a golden.
OCR_BASELINE_CONSTANTS = [
    ("MISTRAL_API_KEY", None),
    ("MISTRAL_API_BASE_URL", "https://api.mistral.ai"),
    ("MISTRAL_OCR_MODEL", "mistral-ocr-latest"),
    ("MISTRAL_OCR_TIMEOUT", 300),
    ("MISTRAL_DELETE_UPLOADED_FILE", True),
    ("MISTRAL_OCR_RETRIES", 4),
    ("MISTRAL_OCR_RETRY_BACKOFF", 3.0),
    ("MISTRAL_OCR_RETRY_MAX_BACKOFF", 60.0),
    ("OCR_ENABLE_OPENAI_FALLBACK", False),
    ("MISTRAL_MAX_UPLOAD_MB", 45.0),
    ("MISTRAL_AUTO_COMPRESS", True),
    ("MISTRAL_AUTO_SPLIT", True),
    ("MISTRAL_SPLIT_PART_MB", 30.0),
    ("MISTRAL_MAX_PAGES", 950),
    ("MISTRAL_SPLIT_PART_PAGES", 500),
    ("OPENAI_OCR_MAX_PAGES", 10),
    ("OCR_MIN_CHARS_PER_PAGE", 500),
    ("OCR_ENABLE_LOCAL_FALLBACK", True),
    ("LOCAL_OCR_ENGINE", "docling"),
    ("PDF_EXTRACTION_WORKERS", 1),
]


def _build_pdf(tmp_path, name, pages, words_per_page=100, word="terme"):
    """Generate a deterministic text PDF (at least 50 words per page, so the
    legacy extractor never falls back to Tesseract)."""
    doc = fitz.open()
    for page_no in range(1, pages + 1):
        page = doc.new_page(width=595, height=842)
        words = [f"{word}{page_no}x{i}" for i in range(words_per_page)]
        for row, start in enumerate(range(0, len(words), 10)):
            page.insert_text((40, 60 + 16 * row), " ".join(words[start:start + 10]), fontsize=9)
    path = tmp_path / f"{name}.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


def _http_error(status, text):
    """Build a ``requests.HTTPError`` carrying a response of the given status."""
    resp = SimpleNamespace(status_code=status, text=text, headers={})
    err = requests.HTTPError(f"HTTP {status}")
    err.response = resp
    return err


def _part_key(pdf_path):
    """``p<A>-<B>`` for a split part, else the file basename."""
    base = os.path.basename(str(pdf_path))
    match = re.match(r"ragpy_part_p(\d+)-(\d+)_", base)
    return f"p{match.group(1)}-{match.group(2)}" if match else base


def _ok(text):
    """Scripted Mistral outcome: return ``text``."""
    def outcome(pdf_path):
        """Return the scripted markdown."""
        return text
    return outcome


def _http_failure(status, body):
    """Scripted Mistral outcome: raise the real classification of an HTTP error."""
    def outcome(pdf_path):
        """Raise the exception ``_classify_mistral_http_error`` maps ``status`` to."""
        raise rad._classify_mistral_http_error(_http_error(status, body), pdf_path)
    return outcome


class _FakeHttpResponse:
    """Canned HTTP answer served by the Mistral HTTP double."""

    def __init__(self, status=200, body=None, text=None, headers=None):
        """Store the canned status, JSON body, raw text and headers."""
        self.status_code = status
        self._body = body
        self.text = text if text is not None else json.dumps(body, ensure_ascii=False)
        self.headers = dict(headers or {})

    def json(self):
        """Return a fresh deep copy of the canned JSON body."""
        return json.loads(json.dumps(self._body))

    def raise_for_status(self):
        """Raise ``requests.HTTPError`` (response attached) for 4xx and 5xx statuses."""
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error (golden)", response=self)

    def describe(self):
        """JSON view of the canned answer (status, body or text, headers)."""
        view = {"status": self.status_code}
        if self._body is not None:
            view["body"] = _freeze(self._body)
        else:
            view["text"] = self.text
        if self.headers:
            view["headers"] = dict(sorted(self.headers.items()))
        return view


def _describe_scripted(item):
    """JSON view of one scripted item (canned response or raised exception)."""
    if isinstance(item, BaseException):
        return {"raises": f"{type(item).__name__}: {item}"}
    return item.describe()


def _describe_http_kwargs(kwargs):
    """Normalised view of the keyword arguments of one HTTP call.

    The Authorization header is stored as a fingerprint, uploaded files as
    ``[normalised name, content type]``; every other argument is kept as is.
    """
    view = {}
    for name in sorted(kwargs):
        value = kwargs[name]
        if name == "headers":
            view[name] = {
                key: ("sha12:" + _fingerprint(val) if key.lower() == "authorization" else val)
                for key, val in sorted((value or {}).items())
            }
        elif name == "files":
            view[name] = {
                field: [_normalise_text(spec[0]), spec[2] if len(spec) > 2 else None]
                for field, spec in sorted(value.items())
            }
        else:
            view[name] = _freeze(value)
    return view


class _FakeMistralSession:
    """``requests.Session`` double forwarding every call to ``_FakeMistralHttp``."""

    def __init__(self, http, number):
        """Bind the shared HTTP double and this session's ordinal number."""
        self._http = http
        self.number = number

    def __enter__(self):
        """Enter the ``with requests.Session()`` block."""
        return self

    def __exit__(self, *exc_info):
        """Leave the block without swallowing exceptions."""
        return False

    def close(self):
        """Close the session (no-op)."""

    def post(self, url, *args, **kwargs):
        """Serve a scripted answer to a POST request."""
        return self._http.dispatch(self.number, "POST", url, kwargs)

    def delete(self, url, *args, **kwargs):
        """Serve a scripted answer to a DELETE request."""
        return self._http.dispatch(self.number, "DELETE", url, kwargs)

    def get(self, url, *args, **kwargs):
        """Serve a scripted answer to a GET request (none is expected)."""
        return self._http.dispatch(self.number, "GET", url, kwargs)


class _FakeMistralHttp:
    """Recording HTTP double for the real ``_mistral_upload_and_ocr_once``.

    Three FIFO queues answer ``POST /v1/files`` (upload), ``POST /v1/ocr`` and
    ``DELETE /v1/files/<id>``. An empty upload queue answers with an automatic
    file id, an empty delete queue with a success; any other unscripted call
    raises. Every call is logged with its normalised arguments and answer.
    """

    def __init__(self, uploads=None, ocr=None, deletes=None):
        """Store the scripted answers of each endpoint."""
        self.queues = {"uploads": list(uploads or []), "ocr": list(ocr or []),
                       "deletes": list(deletes or [])}
        self.requests = []
        self.sessions = 0
        self._auto_ids = 0

    def session_factory(self, *args, **kwargs):
        """Return a new recording session (what ``requests.Session()`` yields)."""
        self.sessions += 1
        return _FakeMistralSession(self, self.sessions)

    def _route(self, method, url):
        """Name of the queue answering ``method url`` (None when unknown)."""
        if method == "POST" and url.endswith("/v1/files"):
            return "uploads"
        if method == "POST" and url.endswith("/v1/ocr"):
            return "ocr"
        if method == "DELETE" and "/v1/files/" in url:
            return "deletes"
        return None

    def _default(self, queue, url):
        """Automatic answer when a queue is empty (upload and delete only)."""
        if queue == "uploads":
            self._auto_ids += 1
            return _FakeHttpResponse(200, {"id": f"file-golden-{self._auto_ids:02d}",
                                           "object": "file", "purpose": "ocr"})
        if queue == "deletes":
            return _FakeHttpResponse(200, {"id": url.rsplit("/", 1)[-1], "object": "file",
                                           "deleted": True})
        return AssertionError(f"golden: requête HTTP non scriptée {url}")

    def dispatch(self, session, method, url, kwargs):
        """Log one HTTP call, then return (or raise) its scripted answer."""
        queue = self._route(method, url)
        pending = self.queues.get(queue) if queue else None
        item = pending.pop(0) if pending else self._default(queue, url)
        entry = {"session": session, "method": method, "url": url,
                 "kwargs": _describe_http_kwargs(kwargs), "answer": _describe_scripted(item)}
        self.requests.append(entry)
        if isinstance(item, BaseException):
            raise item
        return item

    def report(self):
        """JSON view of the request log (each entry carries its scripted answer)
        and of the scripted answers left unconsumed."""
        return {
            "sessions": self.sessions,
            "requests": self.requests,
            "unconsumed": {key: [_describe_scripted(i) for i in items] for key, items in self.queues.items()},
        }


class _OcrHarness:
    """Spies on the OCR provider chain; the single Mistral attempt is either
    scripted (``outcomes``) or real with a recording HTTP double (``use_http``)."""

    def __init__(self, monkeypatch, tmp_path, norm_root=None):
        """Pin the module constants, install the spies and the time/random proxies."""
        self.monkeypatch = monkeypatch
        self.tmp_path = tmp_path
        self.norm_root = norm_root if norm_root is not None else tmp_path
        self.calls = []
        self.sleeps = []
        self.outcomes = {}
        self.legacy_override = None
        self.http = None
        for name, value in OCR_BASELINE_CONSTANTS:
            monkeypatch.setattr(rad, name, value)
        monkeypatch.setattr(rad, "_extract_text_with_mistral", self._extract_text_with_mistral)
        monkeypatch.setattr(rad, "_mistral_upload_and_ocr", self._mistral_upload_and_ocr)
        monkeypatch.setattr(rad, "_mistral_upload_and_ocr_once", self._mistral_upload_and_ocr_once)
        monkeypatch.setattr(rad, "_extract_text_with_openai", self._extract_text_with_openai)
        monkeypatch.setattr(rad, "_local_ocr_available", self._local_ocr_available)
        monkeypatch.setattr(rad, "_extract_text_with_local", self._extract_text_with_local)
        monkeypatch.setattr(rad, "_extract_text_with_legacy_pdf", self._extract_text_with_legacy_pdf)
        monkeypatch.setattr(rad, "time", _TimeProxy(self.sleeps))
        monkeypatch.setattr(rad, "random", _RandomProxy())

    def use_http(self, http):
        """Run the real single Mistral attempt against the HTTP double ``http``."""
        self.http = http
        self.monkeypatch.setattr(requests, "Session", http.session_factory)

    def norm(self, value):
        """Normalise a path or message for the golden."""
        return _normalise_text(str(value), self.norm_root)

    def _record(self, name, *args):
        """Append one provider call (normalised arguments) to the call log."""
        self.calls.append([name] + [self.norm(a) if isinstance(a, str) else a for a in args])

    def _extract_text_with_mistral(self, *args, **kwargs):
        """Spy around the real Mistral orchestration (fast path / split)."""
        bound = _bind_known(args, kwargs, ("pdf_path", "max_pages"))
        self._record("_extract_text_with_mistral", bound.get("pdf_path"), bound.get("max_pages"))
        return _REAL_RAD["_extract_text_with_mistral"](*args, **kwargs)

    def _mistral_upload_and_ocr(self, *args, **kwargs):
        """Spy around the real Mistral retry wrapper."""
        bound = _bind_known(args, kwargs, ("pdf_path", "max_pages"))
        self._record("_mistral_upload_and_ocr", bound.get("pdf_path"), bound.get("max_pages"))
        return _REAL_RAD["_mistral_upload_and_ocr"](*args, **kwargs)

    def _mistral_upload_and_ocr_once(self, *args, **kwargs):
        """Single Mistral attempt: real one over the HTTP double, or scripted."""
        bound = _bind_known(args, kwargs, ("pdf_path", "max_pages"))
        pdf_path = bound.get("pdf_path")
        self._record("_mistral_upload_and_ocr_once", pdf_path, bound.get("max_pages"))
        if self.http is not None:
            return _REAL_RAD["_mistral_upload_and_ocr_once"](*args, **kwargs)
        outcome = self.outcomes.get(_part_key(pdf_path))
        if outcome is None:
            raise rad.OCRExtractionError(f"golden: aucune réponse scriptée pour {_part_key(pdf_path)}")
        return outcome(pdf_path)

    def _extract_text_with_openai(self, *args, **kwargs):
        """OpenAI vision must stay unused on the OFF path; record and fail softly."""
        bound = _bind_known(args, kwargs, ("pdf_path", "api_key", "max_pages"))
        self._record("_extract_text_with_openai", bound.get("pdf_path"), bound.get("max_pages"))
        raise rad.OCRExtractionError("golden: OpenAI vision inattendu")

    def _local_ocr_available(self, *args, **kwargs):
        """Local OCR engine reported as not installed (baseline test machine)."""
        self._record("_local_ocr_available")
        return False

    def _extract_text_with_local(self, *args, **kwargs):
        """Local OCR must not run when unavailable; record and fail softly."""
        bound = _bind_known(args, kwargs, ("pdf_path", "max_pages"))
        self._record("_extract_text_with_local", bound.get("pdf_path"), bound.get("max_pages"))
        raise rad.OCRExtractionError("golden: OCR local inattendu")

    def _extract_text_with_legacy_pdf(self, *args, **kwargs):
        """Spy around the real PyMuPDF extractor (or a scripted override)."""
        bound = _bind_known(args, kwargs, ("pdf_path", "max_pages"))
        self._record("_extract_text_with_legacy_pdf", bound.get("pdf_path"), bound.get("max_pages"))
        if self.legacy_override is not None:
            return self.legacy_override
        return _REAL_RAD["_extract_text_with_legacy_pdf"](*args, **kwargs)

    def run(self, pdf_path, scenario, max_pages=None):
        """Run ``extract_text_with_ocr_retry(return_details=True)`` and capture it."""
        result = None
        error = None
        try:
            if max_pages is None:
                result = rad.extract_text_with_ocr_retry(pdf_path, return_details=True)
            else:
                result = rad.extract_text_with_ocr_retry(pdf_path, max_pages=max_pages, return_details=True)
        except Exception as exc:  # noqa: BLE001 — the failure itself is the golden
            error = {"type": type(exc).__name__, "message": self.norm(exc)}
        fields = None
        if result is not None:
            fields = {field: getattr(result, field) for field in OCR_RESULT_FIELDS}
            fields = _normalise_obj(fields, self.norm_root)
        return {
            "scenario": scenario,
            "result_type": type(result).__name__ if result is not None else None,
            "result": fields,
            "exception": error,
            "calls": self.calls,
            "sleeps": self.sleeps,
        }


@pytest.fixture
def ocr_harness(monkeypatch, tmp_path):
    """OCR chain harness with pinned baseline constants and provider spies."""
    return _OcrHarness(monkeypatch, tmp_path)


def _apply_force_split(monkeypatch):
    """Fake Mistral key and a tiny page cap: 3-page PDFs split into 1-page parts."""
    monkeypatch.setattr(rad, "MISTRAL_API_KEY", FAKE_ENV["MISTRAL_API_KEY"])
    monkeypatch.setattr(rad, "MISTRAL_MAX_PAGES", 2)
    monkeypatch.setattr(rad, "MISTRAL_SPLIT_PART_PAGES", 1)
    monkeypatch.setattr(rad, "MISTRAL_AUTO_SPLIT", True)
    monkeypatch.setattr(rad, "MISTRAL_AUTO_COMPRESS", False)


@pytest.fixture
def force_split(monkeypatch, ocr_harness):
    """Force _extract_text_with_mistral down the split path: tiny page cap.

    Copied from ``tests/test_ocr_providers.py`` (``force_split``); the key is
    the sprint's fake Mistral value, and the harness constants are pinned first.
    """
    _apply_force_split(monkeypatch)


def _scenario_legacy_no_keys(h):
    """No Mistral key, no OpenAI key: PyMuPDF legacy on a dense 2-page PDF."""
    pdf = _build_pdf(h.tmp_path, "article_dense", 2)
    return h.run(pdf, "legacy_no_keys")


def _scenario_legacy_sparse_openai_key(h, monkeypatch):
    """No Mistral key, OpenAI key set (vision OFF): sparse legacy flagged partial."""
    _set_fake_env(monkeypatch, "OPENAI_API_KEY")
    pdf = _build_pdf(h.tmp_path, "scan_sparse", 2, words_per_page=55, word="m")
    return h.run(pdf, "legacy_sparse_openai_key_set")


def _scenario_mistral_fast_path(h, monkeypatch):
    """Mistral key set, small PDF: single upload+OCR attempt succeeds."""
    monkeypatch.setattr(rad, "MISTRAL_API_KEY", FAKE_ENV["MISTRAL_API_KEY"])
    pdf = _build_pdf(h.tmp_path, "article_mistral", 2)
    h.outcomes["article_mistral.pdf"] = _ok("<!-- Page 1 -->\n# Titre\nAlpha\n\n<!-- Page 2 -->\nBeta")
    return h.run(pdf, "mistral_fast_path")


def _scenario_mistral_split_salvage(h):
    """Split into 3 one-page parts; part 2 stays transient (503) until retries
    are exhausted, parts 1 and 3 succeed: salvaged, flagged partial."""
    pdf = _build_pdf(h.tmp_path, "book_split", 3)
    h.outcomes["p1-1"] = _ok("<!-- Page 1 -->\nAAA")
    h.outcomes["p2-2"] = _http_failure(503, "Service Unavailable")
    h.outcomes["p3-3"] = _ok("<!-- Page 1 -->\nCCC")
    return h.run(pdf, "mistral_split_salvage")


# ----------------------------------------------------------------------
# G1 — scripted single Mistral attempt (chain order, split/salvage)
# ----------------------------------------------------------------------
def test_g1_legacy_no_mistral_key(ocr_harness):
    _check_golden_json("g1_legacy_no_mistral_key.json", _scenario_legacy_no_keys(ocr_harness))


def test_g1_legacy_sparse_flagged(ocr_harness, monkeypatch):
    capture = _scenario_legacy_sparse_openai_key(ocr_harness, monkeypatch)
    _check_golden_json("g1_legacy_sparse_flagged.json", capture)


def test_g1_mistral_fast_path(ocr_harness, monkeypatch):
    _check_golden_json("g1_mistral_fast_path.json", _scenario_mistral_fast_path(ocr_harness, monkeypatch))


def test_g1_mistral_split_salvage(ocr_harness, force_split):
    _check_golden_json("g1_mistral_split_salvage.json", _scenario_mistral_split_salvage(ocr_harness))


def test_g1_mistral_auth_mid_split(ocr_harness, force_split):
    h = ocr_harness
    pdf = _build_pdf(h.tmp_path, "book_auth", 3)
    h.outcomes["p1-1"] = _ok("<!-- Page 1 -->\nAAA")
    h.outcomes["p2-2"] = _http_failure(401, "Unauthorized")
    h.outcomes["p3-3"] = _ok("<!-- Page 1 -->\nCCC")
    _check_golden_json("g1_mistral_auth_mid_split.json", h.run(pdf, "mistral_auth_mid_split"))


def test_g1_all_providers_fail(ocr_harness, force_split, monkeypatch):
    h = ocr_harness
    _set_fake_env(monkeypatch, "OPENAI_API_KEY")
    pdf = _build_pdf(h.tmp_path, "book_unreadable", 3)
    for key in ("p1-1", "p2-2", "p3-3"):
        h.outcomes[key] = _http_failure(422, "Unprocessable Entity")
    h.legacy_override = ""
    capture = h.run(pdf, "all_providers_fail")
    assert capture["exception"] is not None
    _check_golden_json("g1_all_providers_fail.json", capture)


# ----------------------------------------------------------------------
# G1 / G13 — real single Mistral attempt over a recording HTTP double
# (payload, pages[] parser, delete-after-OCR, empty-answer WARNING)
# ----------------------------------------------------------------------
def _upload_ok(file_id):
    """Canned successful ``/v1/files`` answer carrying ``file_id``."""
    return _FakeHttpResponse(200, {"id": file_id, "object": "file", "purpose": "ocr"})


def _ocr_ok(body):
    """Canned successful ``/v1/ocr`` answer with JSON ``body``."""
    return _FakeHttpResponse(200, body)


def _http_scenario_specs():
    """Fresh catalogue of the HTTP-level Mistral scenarios.

    Keys of a spec: ``pages`` (PDF page count), ``max_pages``, ``split`` (use the
    force-split settings), ``uploads`` / ``ocr`` / ``deletes`` (scripted answers).
    """
    part_503 = [_FakeHttpResponse(503, text="upstream unavailable") for _ in range(5)]
    return {
        "pages_index_markdown": {
            "pages": 2,
            "uploads": [_upload_ok("file-golden-a")],
            "ocr": [_ocr_ok({
                "pages": [
                    {"index": 0, "markdown": "# Titre\n\nAlpha  ", "images": [],
                     "dimensions": {"dpi": 200, "height": 2200, "width": 1700}},
                    {"index": 1, "markdown": "\nBeta\n", "images": []},
                ],
                "model": "mistral-ocr-2505",
                "usage_info": {"pages_processed": 2, "doc_size_bytes": 1234},
            })],
        },
        "pages_mixed_shapes": {
            "pages": 3,
            "uploads": [_FakeHttpResponse(200, {"file_id": "file-golden-b", "object": "file"})],
            "ocr": [_ocr_ok({"pages": [
                {"index": 0, "markdown": "  Alpha  "},
                {"index": 1, "text": "Texte seul"},
                {"index": 2, "markdown": "   ", "text": "Repli sur text"},
                {"index": 3, "markdown": "  \n "},
                "pas un dict",
                None,
                {"markdown": "Sans index"},
                {"index": "8", "markdown": "Index en chaîne"},
                {"index": 9, "markdown": 123, "text": "Markdown non textuel"},
                {"index": True, "markdown": "Index booléen"},
            ]})],
        },
        "pages_blank_top_level_markdown": {
            "pages": 2,
            "uploads": [_FakeHttpResponse(200, {"data": {"id": "file-golden-c"}})],
            "ocr": [_ocr_ok({
                "pages": [{"index": 0, "markdown": "   "}, {"index": 1}],
                "markdown": "  Markdown global\n\nsuite  ",
                "text": "ignoré",
            })],
        },
        "top_level_text_no_pages": {
            "pages": 2,
            "ocr": [_ocr_ok({"pages": [], "markdown": "   ", "text": "  Texte global sans pages  "})],
        },
        "output_array_shape": {
            "pages": 2,
            "ocr": [_ocr_ok({"pages": "pas une liste", "output": [
                {"content": "Bloc contenu"},
                "pas un dict",
                {"markdown": "   ", "text": "Bloc texte"},
                {"markdown": "Bloc md", "text": "ignoré"},
                {"content": 42},
                {"text": "  "},
            ]})],
        },
        "empty_payload_falls_back_to_legacy": {
            "pages": 2,
            "ocr": [_ocr_ok({})],
        },
        "non_dict_payload_falls_back_to_legacy": {
            "pages": 2,
            "ocr": [_ocr_ok([{"markdown": "liste ignorée"}])],
        },
        "upload_without_id_falls_back_to_legacy": {
            "pages": 2,
            "uploads": [_FakeHttpResponse(200, {"object": "file", "purpose": "ocr"})],
        },
        "max_pages_fast_path": {
            "pages": 3,
            "max_pages": 2,
            "ocr": [_ocr_ok({"pages": [{"index": 0, "markdown": "Page un"},
                                       {"index": 1, "markdown": "Page deux"}]})],
        },
        "ocr_http_422_falls_back_to_legacy": {
            "pages": 2,
            "ocr": [_FakeHttpResponse(
                422, text='{"detail":[{"type":"extra_forbidden","loc":["body","page_ranges"]}]}')],
        },
        "ocr_http_401_falls_back_to_legacy": {
            "pages": 2,
            "ocr": [_FakeHttpResponse(401, text='{"message":"Unauthorized"}')],
        },
        "upload_503_then_retry_success": {
            "pages": 2,
            "uploads": [_FakeHttpResponse(503, text="upstream unavailable"), _upload_ok("file-golden-k")],
            "ocr": [_ocr_ok({"pages": [{"index": 0, "markdown": "Après reprise"}]})],
        },
        "ocr_429_retry_after_then_success": {
            "pages": 2,
            "ocr": [
                _FakeHttpResponse(429, text="rate limited", headers={"Retry-After": "7"}),
                _ocr_ok({"pages": [{"index": 0, "markdown": "Après Retry-After"}]}),
            ],
        },
        "delete_failure_is_ignored": {
            "pages": 2,
            "ocr": [_ocr_ok({"pages": [{"index": 0, "markdown": "Suppression en échec"}]})],
            "deletes": [requests.ConnectionError("golden: suppression refusée")],
        },
        "split_salvage_http": {
            "pages": 3,
            "split": True,
            "ocr": [_ocr_ok({"pages": [{"index": 0, "markdown": "Part un"}]})]
            + part_503
            + [_ocr_ok({"pages": [{"index": 0, "markdown": "Part trois"}]})],
        },
        "split_max_pages_http": {
            "pages": 3,
            "split": True,
            "max_pages": 2,
            "ocr": [_ocr_ok({"pages": [{"index": 0, "markdown": "Part un"}]}),
                    _ocr_ok({"pages": [{"index": 0, "markdown": "Part deux"}]})],
        },
    }


HTTP_SCENARIO_NAMES = sorted(_http_scenario_specs())


def _log_records(caplog, tmp_path, start=0):
    """INFO+ records (from index ``start``) as [logger, level, normalised message]."""
    return [
        [record.name, record.levelname, _normalise_text(record.getMessage(), tmp_path)]
        for record in caplog.records[start:]
        if record.levelno >= logging.INFO
    ]


def _run_http_scenario(monkeypatch, tmp_path, name, caplog=None):
    """Run HTTP scenario ``name`` through the real single Mistral attempt.

    Returns ``(capture, logs)``; ``logs`` is None when ``caplog`` is None.
    """
    spec = _http_scenario_specs()[name]
    work = tmp_path / name
    work.mkdir()
    h = _OcrHarness(monkeypatch, work, norm_root=tmp_path)
    monkeypatch.setattr(rad, "MISTRAL_API_KEY", FAKE_ENV["MISTRAL_API_KEY"])
    if spec.get("split"):
        _apply_force_split(monkeypatch)
    http = _FakeMistralHttp(spec.get("uploads"), spec.get("ocr"), spec.get("deletes"))
    h.use_http(http)
    pdf = _build_pdf(work, "document", spec["pages"])
    start = len(caplog.records) if caplog is not None else 0
    capture = h.run(pdf, name, max_pages=spec.get("max_pages"))
    capture["max_pages"] = spec.get("max_pages")
    capture["http"] = _normalise_obj(http.report(), tmp_path)
    logs = _log_records(caplog, tmp_path, start) if caplog is not None else None
    return capture, logs


@pytest.mark.parametrize("name", HTTP_SCENARIO_NAMES)
def test_g1_mistral_http(name, monkeypatch, tmp_path):
    capture, _logs = _run_http_scenario(monkeypatch, tmp_path, name)
    _check_golden_json(f"g1_mistral_http_{name}.json", capture)


# ----------------------------------------------------------------------
# G1 — rad_dataframe CSV and errors.json (load_zotero_to_dataframe_incremental)
# ----------------------------------------------------------------------
def _zotero_items():
    """Zotero export: a Mistral success, a split-salvage partial, an unsupported file."""
    return [
        {"key": "ITEMOK01", "itemType": "journalArticle", "title": "Article complet",
         "abstractNote": "Résumé de l'article.", "date": "2021-05-01",
         "url": "https://example.org/article", "DOI": "10.1234/golden.a",
         "creators": [{"lastName": "Dupont", "firstName": "Jeanne"},
                      {"firstName": "Paul", "lastName": "Martin"}],
         "attachments": [{"path": "files/article_ok.pdf", "title": "Full Text PDF"}]},
        {"key": "ITEMSPL2", "itemType": "book", "title": "Livre découpé en parts",
         "abstractNote": "", "date": "2019", "url": "", "DOI": "",
         "creators": [{"lastName": "Lahire", "firstName": ""}],
         "attachments": [{"path": "files/book_split.pdf", "title": "PDF"}]},
        {"key": "ITEMDOC3", "itemType": "report", "title": "Pièce jointe non supportée",
         "creators": [],
         "attachments": [{"path": "files/notes.docx", "title": "Notes"}]},
    ]


def test_g1_zotero_incremental_csv_and_errors(ocr_harness, force_split, tmp_path, capsys, caplog):
    caplog.set_level(logging.INFO)
    h = ocr_harness
    base = tmp_path / "zotero"
    files = base / "files"
    files.mkdir(parents=True)
    _build_pdf(files, "article_ok", 2)
    _build_pdf(files, "book_split", 3)
    h.outcomes["article_ok.pdf"] = _ok("<!-- Page 1 -->\n# Article\nAlpha\n\n<!-- Page 2 -->\nBeta")
    h.outcomes["p1-1"] = _ok("<!-- Page 1 -->\nAAA")
    h.outcomes["p2-2"] = _http_failure(503, "Service Unavailable")
    h.outcomes["p3-3"] = _ok("<!-- Page 1 -->\nCCC")
    json_path = base / "export.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"items": _zotero_items()}, fh, ensure_ascii=False, indent=2)
    output_csv = base / "output.csv"
    capsys.readouterr()
    start = len(caplog.records)
    df = rad.load_zotero_to_dataframe_incremental(str(json_path), str(base), str(output_csv))
    stdout = _stdout_lines(capsys.readouterr().out, tmp_path)
    # Exact CSV bytes (header order, quoting, CRLF row ends) kept as one JSON
    # string, so no line-ending conversion can alter the golden file itself.
    csv_text = output_csv.read_bytes().decode("utf-8")
    csv_body = _normalise_text(csv_text[1:] if csv_text.startswith("﻿") else csv_text, tmp_path)
    with open(rad.get_errors_file_path(str(output_csv)), encoding="utf-8") as fh:
        errors_json = _mask_timestamps(_normalise_obj(json.load(fh), tmp_path))
    with open(rad.get_progress_file_path(str(output_csv)), encoding="utf-8") as fh:
        progress = json.load(fh)
    progress["processed_keys"] = sorted(progress["processed_keys"])  # written from a set
    _check_golden_json("g1_zotero_incremental.json", {
        "csv_starts_with_bom": csv_text.startswith("﻿"),
        "csv_header": next(csv.reader([csv_body.split("\r\n", 1)[0]])),
        "csv_text": csv_body,
        "dataframe_columns": list(df.columns),
        "dataframe_rows": len(df),
        "errors_json": errors_json,
        "progress_json": _mask_timestamps(progress),
        "stdout_lines": stdout,
        "calls": h.calls,
        "sleeps": h.sleeps,
        "logs": _log_records(caplog, tmp_path, start),
    })


# ----------------------------------------------------------------------
# G13 — OCR logs (INFO and above)
# ----------------------------------------------------------------------
def test_g13_ocr_logs_mistral_success(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    _capture, logs = _run_http_scenario(monkeypatch, tmp_path, "pages_index_markdown", caplog)
    _check_golden_json("g13_ocr_logs_mistral_success.json", logs)


def test_g13_ocr_logs_mistral_http(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    out = {}
    for name in HTTP_SCENARIO_NAMES:
        _capture, logs = _run_http_scenario(monkeypatch, tmp_path, name, caplog)
        out[name] = logs
    _check_golden_json("g13_ocr_logs_mistral_http.json", out)


def test_g13_ocr_logs_mistral_split_salvage(ocr_harness, force_split, caplog):
    caplog.set_level(logging.INFO)
    _scenario_mistral_split_salvage(ocr_harness)
    _check_golden_json("g13_ocr_logs_mistral_split_salvage.json", _log_records(caplog, ocr_harness.tmp_path))


def test_g13_ocr_logs_legacy(ocr_harness, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    _scenario_legacy_no_keys(ocr_harness)
    first = _log_records(caplog, ocr_harness.tmp_path)
    caplog.clear()
    ocr_harness.calls.clear()
    _scenario_legacy_sparse_openai_key(ocr_harness, monkeypatch)
    second = _log_records(caplog, ocr_harness.tmp_path)
    _check_golden_json("g13_ocr_logs_legacy.json", {
        "legacy_no_keys": first,
        "legacy_sparse_openai_key_set": second,
    })


# ======================================================================
# G2 / G3 / G4 / G12 — rad_chunk
# ======================================================================
def _golden_ocr_text():
    """Deterministic OCR-like text long enough to give two token chunks."""
    paragraphs = []
    for i in range(1, 71):
        paragraphs.append(
            f"Paragraphe {i}. Le langage ordinaire structure l'expérience sociale ; "
            f"l'énoncé numéro {i} renvoie à un contexte d'usage précis et situé."
        )
    return "\n\n".join(paragraphs)


def _golden_row(tmp_path, provider="legacy", text=None):
    """First row of a fixed CSV, read the way rad_chunk's CLI reads it."""
    csv_path = tmp_path / "golden_input.csv"
    header = ["filename", "title", "authors", "date", "year", "doi", "url", "path",
              "itemKey", "notes", "texteocr", "texteocr_provider"]
    row = ["golden.pdf", "Le langage ordinaire", "Dupont, Jeanne; Martin, Paul", "2021-05-01",
           2021, "10.1234/golden.2021", "https://example.org/golden", "/data/golden/golden.pdf",
           "ABCD1234", "", _golden_ocr_text() if text is None else text, provider]
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerow(row)
    df = pd.read_csv(str(csv_path))
    return df, next(df.iterrows())[1]


@pytest.fixture
def chunk_env(monkeypatch):
    """rad_chunk with pinned workers/batches, fixed doc_id and recorded clients."""
    monkeypatch.setattr(rc, "DEFAULT_MAX_WORKERS", 1)
    monkeypatch.setattr(rc, "DEFAULT_DOC_WORKERS", 3)
    monkeypatch.setattr(rc, "DEFAULT_BATCH_SIZE_GPT", 5)
    monkeypatch.setattr(rc, "DEFAULT_EMBEDDING_BATCH_SIZE", 2)
    monkeypatch.setattr(rc, "random", _RandomProxy())
    calls = []
    openai_client = _FakeLLMClient("openai", calls)
    openrouter_client = _FakeLLMClient("openrouter", calls)
    monkeypatch.setattr(rc, "client", openai_client)
    monkeypatch.setattr(rc, "openrouter_client", openrouter_client)
    return SimpleNamespace(calls=calls, openai=openai_client, openrouter=openrouter_client)


G2_CASES = [
    ("openai", "gpt-4o-mini", True),
    ("openrouter", "google/gemini-2.5-flash", True),
    ("openrouter_missing_fallback", "google/gemini-2.5-flash", False),
]
G2_SKIP_PROVIDERS = ["mistral", "csv"]


@pytest.mark.parametrize("case,model,with_openrouter", G2_CASES, ids=[c[0] for c in G2_CASES])
def test_g2_process_document_chunks(case, model, with_openrouter, chunk_env, tmp_path, monkeypatch):
    if not with_openrouter:
        monkeypatch.setattr(rc, "openrouter_client", None)
    _df, row = _golden_row(tmp_path)
    json_file = tmp_path / "output_chunks.json"
    rc.process_document_chunks(row, json_file=str(json_file), model=model)
    _check_golden_bytes(f"g2_chunks_{case}.json", json_file.read_bytes())
    _check_golden_json(f"g2_create_kwargs_{case}.json", chunk_env.calls)


@pytest.mark.parametrize("provider", G2_SKIP_PROVIDERS)
def test_g2_process_document_chunks_skip_recode(provider, chunk_env, tmp_path):
    _df, row = _golden_row(tmp_path, provider=provider)
    json_file = tmp_path / "output_chunks.json"
    rc.process_document_chunks(row, json_file=str(json_file), model="gpt-4o-mini")
    assert chunk_env.calls == []
    _check_golden_bytes(f"g2_chunks_skip_{provider}.json", json_file.read_bytes())


def _write_chunks_file(tmp_path):
    """Write a fixed ``output_chunks.json`` (4 chunks, 2 documents)."""
    chunks = []
    for doc_id, title in (("111111111111", "Doc A"), ("222222222222", "Doc B")):
        for idx in (1, 2):
            chunks.append({
                "id": f"{doc_id}_{idx}",
                "doc_id": doc_id,
                "chunk_index": idx,
                "total_chunks": 2,
                "text": f"{title} — chunk {idx} : texte recodé stable pour l'embedding dense.",
                "title": title,
                "texteocr_provider": "mistral",
            })
    path = tmp_path / "output_chunks.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)
    return str(path)


def test_g3_generate_and_save_embeddings(chunk_env, tmp_path):
    input_path = _write_chunks_file(tmp_path)
    output_path = str(tmp_path / "output_chunks_with_embeddings.json")
    returned = rc.generate_and_save_embeddings(input_path, output_path)
    assert returned == output_path
    with open(output_path, "rb") as fh:
        _check_golden_bytes("g3_dense_output.json", fh.read())
    _check_golden_json("g3_embeddings_create_kwargs.json", chunk_env.calls)


G4_MODELS = ["gpt-4o-mini", "google/gemini-2.5-flash", "openai/gpt-oss-120b"]
G4_RAW = ("Chapitre 3 — Le langage ordinaire et ses usages sociaux. Ce passage, issu "
          "d'un OCR brut, sert de clé de cache stable pour la matrice golden des clés.")
G4_RECODE_KEY_ARGS = ("content_hash", "model", "provider", "prompt_version", "decode_params_json")
G4_EMBED_KEY_ARGS = ("text", "embed_model", "embed_params_json")


def test_g4_recode_and_embed_keys(chunk_env, tmp_path, monkeypatch):
    cache_mod = rc.rad_recode_cache
    real_recode_key = cache_mod.recode_key
    real_embed_key = cache_mod.embed_key
    key_calls = []

    def recode_key_spy(*args, **kwargs):
        """Record the known arguments and the value of the real recode_key."""
        value = real_recode_key(*args, **kwargs)
        bound = _bind_known(args, kwargs, G4_RECODE_KEY_ARGS)
        key_calls.append(("recode", {"content_hash": bound.get("content_hash"),
                                     "model": bound.get("model"),
                                     "provider": bound.get("provider"),
                                     "prompt_version": bound.get("prompt_version"),
                                     "decode_params": bound.get("decode_params_json")}, value))
        return value

    def embed_key_spy(*args, **kwargs):
        """Record the known arguments and the value of the real embed_key."""
        value = real_embed_key(*args, **kwargs)
        bound = _bind_known(args, kwargs, G4_EMBED_KEY_ARGS)
        key_calls.append(("embed", {"text_sha12": _fingerprint(bound.get("text")),
                                    "embed_model": bound.get("embed_model"),
                                    "embed_params": bound.get("embed_params_json")}, value))
        return value

    monkeypatch.setattr(cache_mod, "recode_key", recode_key_spy)
    monkeypatch.setattr(cache_mod, "embed_key", embed_key_spy)
    cells = []
    cell_no = 0
    for model in G4_MODELS:
        for prefer_openai in (0, 1):
            for harden in (0, 1):
                cell_no += 1
                key_calls.clear()
                chunk_env.calls.clear()
                cell_dir = tmp_path / f"cell_{cell_no:02d}"
                cell_dir.mkdir()
                monkeypatch.setenv("RECODE_CACHE_ENABLED", "1")
                monkeypatch.setenv("RECODE_EMBED_CACHE_ENABLED", "1")
                monkeypatch.setenv("RECODE_CACHE_PATH", str(cell_dir / "cache.sqlite"))
                monkeypatch.setenv("RECODE_PREFER_OPENAI", str(prefer_openai))
                monkeypatch.setenv("RECODE_HARDEN_ENABLED", str(harden))
                _df, row = _golden_row(cell_dir, text=G4_RAW)
                chunks_json = cell_dir / "output_chunks.json"
                rc.process_document_chunks(row, json_file=str(chunks_json), model=model)
                rc.generate_and_save_embeddings(str(chunks_json), str(cell_dir / "output_dense.json"))
                with open(chunks_json, encoding="utf-8") as fh:
                    statuses = [chunk.get("recode_status") for chunk in json.load(fh)]
                recode = [c for c in key_calls if c[0] == "recode"]
                embed = [c for c in key_calls if c[0] == "embed"]
                chat = [c for c in chunk_env.calls if c["endpoint"] == "chat.completions.create"]
                cells.append({
                    "model": model,
                    "prefer_openai": prefer_openai,
                    "harden": harden,
                    "key_model": recode[0][1]["model"],
                    "provider_label": recode[0][1]["provider"],
                    "recode_key": recode[0][2],
                    "recode_key_args": recode[0][1],
                    "routed_client": chat[0]["client"] if chat else None,
                    "sent_model": chat[0]["kwargs"]["model"] if chat else None,
                    "statuses": statuses,
                    "embed_key": embed[0][2],
                    "embed_key_args": embed[0][1],
                })
    _check_golden_json("g4_recode_embed_keys.json", cells)


def test_g12_rad_chunk_stdout(chunk_env, tmp_path, capsys, monkeypatch):
    df, row = _golden_row(tmp_path)
    capsys.readouterr()
    rc.process_document_chunks(row, json_file=str(tmp_path / "direct_chunks.json"), model="gpt-4o-mini")
    direct = _stdout_lines(capsys.readouterr().out, tmp_path)
    rc.process_all_documents(df, json_file=str(tmp_path / "output_chunks.json"), model="google/gemini-2.5-flash")
    all_docs = _stdout_lines(capsys.readouterr().out, tmp_path)
    input_path = _write_chunks_file(tmp_path)
    rc.generate_and_save_embeddings(input_path, str(tmp_path / "output_chunks_with_embeddings.json"))
    dense = _stdout_lines(capsys.readouterr().out, tmp_path)
    skip_dir = tmp_path / "skip"
    skip_dir.mkdir()
    _skip_df, skip_row = _golden_row(skip_dir, provider="mistral")
    rc.process_document_chunks(skip_row, json_file=str(skip_dir / "skip_chunks.json"), model="gpt-4o-mini")
    skipped = _stdout_lines(capsys.readouterr().out, tmp_path)
    monkeypatch.setattr(rc, "openrouter_client", None)
    rc.process_document_chunks(row, json_file=str(tmp_path / "fallback_chunks.json"),
                               model="google/gemini-2.5-flash")
    fallback = _stdout_lines(capsys.readouterr().out, tmp_path)
    _check_golden_json("g12_rad_chunk_stdout.json", {
        "process_document_chunks_gpt-4o-mini": direct,
        "process_all_documents_google/gemini-2.5-flash": all_docs,
        "generate_and_save_embeddings": dense,
        "process_document_chunks_skip_recode_mistral": skipped,
        "process_document_chunks_google/gemini-2.5-flash_without_openrouter_client": fallback,
    })


# ======================================================================
# G5 — rad_vectordb CLI '=== Result ===' block (Pinecone, Weaviate, Qdrant mocked)
# ======================================================================
class _FakePineconeIndex:
    """Pinecone index double; ``upsert`` succeeds or always fails."""

    def __init__(self, fail_upsert):
        """Choose whether every upsert raises."""
        self.fail_upsert = fail_upsert

    def upsert(self, **kwargs):
        """Accept (or reject) a batch of vectors."""
        if self.fail_upsert:
            raise RuntimeError("golden: upsert refusé")
        return {"upserted_count": len(kwargs.get("vectors", []))}


def _fake_pinecone_module(metric, fail_upsert):
    """Build a stand-in ``pinecone`` module exposing a scripted ``Pinecone`` class."""
    index = _FakePineconeIndex(fail_upsert)

    class _FakePinecone:
        """Pinecone client double with a single index ``golden-index``."""

        def __init__(self, api_key=None, **kwargs):
            """Accept the (fake) API key without storing it."""

        def list_indexes(self):
            """List the single available index."""
            return SimpleNamespace(indexes=[SimpleNamespace(name="golden-index")])

        def Index(self, name, *args, **kwargs):
            """Return the shared index double."""
            return index

        def describe_index(self, name, *args, **kwargs):
            """Describe the index: metric and dimension (8, as the test vectors)."""
            return SimpleNamespace(metric=metric, dimension=8)

    module = types.ModuleType("pinecone")
    module.Pinecone = _FakePinecone
    return module


class _FakeWeaviateCollection:
    """Weaviate collection double: tenants API, ``with_tenant`` and ``insert_many``."""

    def __init__(self, existing_tenants, failed_indexes):
        """Store the existing tenants and the batch indexes that must fail."""
        self._tenants = list(existing_tenants)
        self._failed = list(failed_indexes)
        self.tenants = SimpleNamespace(get=self._tenants_get, create=self._tenants_create)
        self.data = SimpleNamespace(insert_many=self._insert_many)

    def _tenants_get(self, *args, **kwargs):
        """Existing tenants as a ``{name: tenant}`` mapping."""
        return {name: SimpleNamespace(name=name) for name in self._tenants}

    def _tenants_create(self, name, *args, **kwargs):
        """Create a tenant."""
        self._tenants.append(name)

    def with_tenant(self, name, *args, **kwargs):
        """Return the collection bound to a tenant (the same double)."""
        return self

    def _insert_many(self, objects, *args, **kwargs):
        """Batch result: ``has_errors`` and ``errors`` for the scripted indexes."""
        errors = {i: SimpleNamespace(message="golden: objet refusé")
                  for i in self._failed if i < len(objects)}
        return SimpleNamespace(has_errors=bool(errors), errors=errors)


class _FakeWeaviateClient:
    """Weaviate client double (readiness, ``collections.get``, ``close``)."""

    def __init__(self, collection, ready):
        """Bind the collection double and the readiness answer."""
        self._ready = ready
        self.collections = SimpleNamespace(get=lambda *args, **kwargs: collection)

    def is_ready(self):
        """Report whether the server is ready."""
        return self._ready

    def close(self):
        """Close the client (no-op)."""


def _fake_weaviate_modules(existing_tenants, ready=True, failed_indexes=()):
    """Stand-in ``weaviate`` package (and the submodules the connector imports)."""
    collection = _FakeWeaviateCollection(existing_tenants, failed_indexes)
    root = types.ModuleType("weaviate")
    classes = types.ModuleType("weaviate.classes")
    init = types.ModuleType("weaviate.classes.init")
    query = types.ModuleType("weaviate.classes.query")
    data = types.ModuleType("weaviate.classes.data")
    init.Auth = SimpleNamespace(api_key=lambda *args, **kwargs: SimpleNamespace(kind="api_key"))
    query.Filter = SimpleNamespace()
    query.MetadataQuery = SimpleNamespace()
    data.DataObject = lambda properties=None, uuid=None, vector=None, **kwargs: SimpleNamespace(
        properties=properties, uuid=uuid, vector=vector)
    classes.init, classes.query, classes.data = init, query, data
    root.classes = classes
    root.connect_to_weaviate_cloud = lambda *args, **kwargs: _FakeWeaviateClient(collection, ready)
    return {"weaviate": root, "weaviate.classes": classes, "weaviate.classes.init": init,
            "weaviate.classes.query": query, "weaviate.classes.data": data}


class _FakeQdrantClient:
    """Qdrant client double: collection lookup/creation and scripted upserts."""

    def __init__(self, collection_exists, upsert_answers):
        """Store whether the collection exists and the successive upsert answers."""
        self._exists = collection_exists
        self._answers = list(upsert_answers)

    def get_collections(self, *args, **kwargs):
        """List collections (connection check)."""
        return SimpleNamespace(collections=[])

    def get_collection(self, *args, **kwargs):
        """Return the collection, or raise when it does not exist yet."""
        if not self._exists:
            raise RuntimeError("golden: collection absente")
        return SimpleNamespace(status="green")

    def create_collection(self, *args, **kwargs):
        """Create the collection."""
        self._exists = True

    def upsert(self, *args, **kwargs):
        """Return (or raise) the next scripted upsert answer."""
        answer = self._answers.pop(0) if self._answers else "completed"
        if isinstance(answer, BaseException):
            raise answer
        return SimpleNamespace(status=answer)

    def close(self):
        """Close the client (no-op)."""


def _fake_qdrant_modules(collection_exists, upsert_answers):
    """Stand-in ``qdrant_client`` package with the ``models`` names the connector uses."""
    client = _FakeQdrantClient(collection_exists, upsert_answers)
    root = types.ModuleType("qdrant_client")
    models = types.ModuleType("qdrant_client.models")
    models.PointStruct = lambda id=None, vector=None, payload=None, **kwargs: SimpleNamespace(
        id=id, vector=vector, payload=payload)
    models.UpdateStatus = SimpleNamespace(COMPLETED="completed")
    models.VectorParams = lambda size=None, distance=None, **kwargs: SimpleNamespace(size=size, distance=distance)
    models.Distance = SimpleNamespace(COSINE="Cosine")
    models.Filter = lambda *args, **kwargs: SimpleNamespace()
    models.FieldCondition = lambda *args, **kwargs: SimpleNamespace()
    models.MatchValue = lambda *args, **kwargs: SimpleNamespace()
    root.models = models
    root.QdrantClient = lambda *args, **kwargs: client
    return {"qdrant_client": root, "qdrant_client.models": models}


def _vectordb_chunks(with_missing_embedding):
    """Chunks with dense + sparse vectors (optionally one without embedding)."""
    chunks = [
        {"id": "424242424242_1", "doc_id": "424242424242", "chunk_index": 1, "total_chunks": 2,
         "text": "alpha", "title": "Doc golden", "embedding": [0.1] * 8,
         "sparse_embedding": {"indices": [3, 7], "values": [0.5, 0.25]}},
        {"id": "424242424242_2", "doc_id": "424242424242", "chunk_index": 2, "total_chunks": 2,
         "text": "beta", "title": "Doc golden", "embedding": [0.2] * 8,
         "sparse_embedding": {"indices": [1], "values": [1.0]}},
        {"id": "515151515151_1", "doc_id": "515151515151", "chunk_index": 1, "total_chunks": 1,
         "text": "gamma", "title": "Autre doc", "embedding": [0.3] * 8},
    ]
    if with_missing_embedding:
        chunks.append({"id": "515151515151_2", "doc_id": "515151515151", "chunk_index": 2,
                       "total_chunks": 2, "text": "delta", "title": "Autre doc"})
    return chunks


def _run_vectordb_cli(monkeypatch, capsys, tmp_path, case, missing, db_args):
    """Run ``scripts/rad_vectordb.py`` as ``__main__`` and keep its Result block."""
    input_path = tmp_path / f"{case}_sparse.json"
    with open(input_path, "w", encoding="utf-8") as fh:
        json.dump(_vectordb_chunks(missing), fh)
    argv = ["rad_vectordb.py", "--input", str(input_path)] + list(db_args)
    monkeypatch.setattr(sys, "argv", argv)
    capsys.readouterr()
    exit_code = None
    try:
        runpy.run_path(os.path.join(RAGPY_ROOT, "scripts", "rad_vectordb.py"), run_name="__main__")
    except SystemExit as exc:
        exit_code = exc.code
    lines = _stdout_lines(capsys.readouterr().out, tmp_path)
    start = lines.index("=== Result ===")
    return {"case": case, "exit_code": exit_code, "result_block": lines[start:]}


@pytest.fixture
def vectordb_cli_env(monkeypatch):
    """Fake credentials of the 3 databases, batch sizes pinned, no real sleep."""
    _set_fake_env(monkeypatch, "PINECONE_API_KEY", "WEAVIATE_URL", "WEAVIATE_API_KEY",
                  "QDRANT_URL", "QDRANT_API_KEY")
    monkeypatch.setenv("PINECONE_BATCH_SIZE", "100")
    monkeypatch.setenv("WEAVIATE_BATCH_SIZE", "100")
    monkeypatch.setenv("QDRANT_BATCH_SIZE", "100")
    monkeypatch.setattr(time, "sleep", lambda *_a: None)
    monkeypatch.setitem(sys.modules, "pinecone", _fake_pinecone_module("dotproduct", False))


G5_PINECONE_CASES = [
    ("success_namespace", "dotproduct", False, False, "golden-ns"),
    ("success_partial_data", "cosine", False, True, None),
    ("partial_error", "dotproduct", True, False, None),
]


def test_g5_vectordb_pinecone_result_block(vectordb_cli_env, tmp_path, monkeypatch, capsys):
    results = []
    for case, metric, fail_upsert, missing, namespace in G5_PINECONE_CASES:
        monkeypatch.setitem(sys.modules, "pinecone", _fake_pinecone_module(metric, fail_upsert))
        db_args = ["--db", "pinecone", "--index", "golden-index"]
        if namespace:
            db_args += ["--namespace", namespace]
        results.append(_run_vectordb_cli(monkeypatch, capsys, tmp_path, case, missing, db_args))
    _check_golden_json("g5_vectordb_pinecone_result_block.json", results)


G5_WEAVIATE_CASES = [
    ("success_existing_tenant", ["golden-tenant"], True, (), False),
    ("new_tenant_partial_data", [], True, (), True),
    ("batch_errors", ["golden-tenant"], True, (1,), False),
    ("not_ready", ["golden-tenant"], False, (), False),
]


def test_g5_vectordb_weaviate_result_block(vectordb_cli_env, tmp_path, monkeypatch, capsys):
    results = []
    for case, tenants, ready, failed, missing in G5_WEAVIATE_CASES:
        for name, module in _fake_weaviate_modules(tenants, ready, failed).items():
            monkeypatch.setitem(sys.modules, name, module)
        db_args = ["--db", "weaviate", "--class-name", "GoldenArticle", "--tenant", "golden-tenant"]
        results.append(_run_vectordb_cli(monkeypatch, capsys, tmp_path, case, missing, db_args))
    _check_golden_json("g5_vectordb_weaviate_result_block.json", results)


G5_QDRANT_CASES = [
    ("success_existing_collection", True, ["completed"], False),
    ("created_collection_partial_data", False, ["completed"], True),
    ("upsert_retry_then_success", True, [RuntimeError("golden: upsert refusé"), "completed"], False),
    ("upsert_not_completed", True, ["acknowledged"], False),
]


def test_g5_vectordb_qdrant_result_block(vectordb_cli_env, tmp_path, monkeypatch, capsys):
    results = []
    for case, exists, answers, missing in G5_QDRANT_CASES:
        for name, module in _fake_qdrant_modules(exists, answers).items():
            monkeypatch.setitem(sys.modules, name, module)
        db_args = ["--db", "qdrant", "--collection", "golden-collection"]
        results.append(_run_vectordb_cli(monkeypatch, capsys, tmp_path, case, missing, db_args))
    _check_golden_json("g5_vectordb_qdrant_result_block.json", results)


# ======================================================================
# G6 — build_subprocess_env (app.core.credentials), 15 literal names
# ======================================================================
def _user(is_admin, personal):
    """Transient User with ADMIN (or USER) role and encrypted personal credentials."""
    roles = ["USER", "ADMIN"] if is_admin else ["USER"]
    return User(
        email="admin@example.test" if is_admin else "user@example.test",
        hashed_password="x",
        roles=roles,
        is_active=True,
        is_verified=True,
        api_credentials=creds.encrypt_credentials(personal) if personal else None,
    )


def _env_view(env, personal):
    """Restrict ``env`` to the 15 literal names: presence, fingerprint and origin."""
    personal_values = set(personal.values())
    view = {}
    for name in BASELINE_ENV_NAMES:
        if name not in env:
            view[name] = None
            continue
        value = env[name]
        if name in os.environ and os.environ[name] == value:
            origin = "env"
        elif value in personal_values:
            origin = "user"
        else:
            origin = "derived"
        view[name] = {"sha12": _fingerprint(value), "origin": origin}
    return view


def test_g6_build_subprocess_env(monkeypatch):
    admin_personal = {
        "openai_api_key": "fake-openai-admin-0002",
        "mistral_url": "https://api.mistral.ai/v1/ocr",
        "qdrant_url": "fake-qdrant-url-admin-0002",
        "albert_api_key": FAKE_ALBERT_KEY,
    }
    user_personal = {
        "openai_api_key": "fake-openai-user-0003",
        "zotero_api_key": "fake-zotero-user-0003",
        "zotero_user_id": "fake-zotero-user-id-0003",
        "albert_api_key": FAKE_ALBERT_KEY,
    }
    albert_only = {"albert_api_key": FAKE_ALBERT_KEY}
    cases = [
        ("admin_no_personal", True, {}, None),
        ("admin_with_personal", True, admin_personal, None),
        ("admin_required_present", True, {}, ["openai_api_key", "mistral_api_key"]),
        ("admin_albert_key_only", True, albert_only, None),
        ("nonadmin_no_personal", False, {}, None),
        ("nonadmin_with_personal", False, user_personal, None),
        ("nonadmin_albert_key_only", False, albert_only, None),
        ("nonadmin_required_missing", False, user_personal, ["openai_api_key", "mistral_api_key"]),
    ]
    _set_fake_env(monkeypatch, *BASELINE_ENV_NAMES)
    _set_albert_off_env(monkeypatch)
    out = []
    for case, is_admin, personal, required in cases:
        user = _user(is_admin, personal)
        entry = {"case": case, "is_admin": is_admin, "required_keys": required}
        try:
            env = creds.build_subprocess_env(user, required_keys=required)
            entry["env"] = _env_view(env, personal)
            entry["error"] = None
        except creds.CredentialMissingError as exc:
            entry["env"] = None
            entry["error"] = {"type": type(exc).__name__, "credential_key": exc.credential_key,
                              "is_admin": exc.is_admin, "message": str(exc)}
        out.append(entry)
    _check_golden_json("g6_build_subprocess_env.json", out)


# ======================================================================
# G10 — stdout of `import rad_chunk` in a subprocess
# ======================================================================
G10_CASES = [
    ("g10_rad_chunk_import_stdout.json", ()),
    ("g10_rad_chunk_import_stdout_openrouter.json", ("OPENROUTER_API_KEY",)),
]


@pytest.mark.parametrize("golden_name,fake_names", G10_CASES, ids=["no_openrouter_key", "openrouter_key"])
def test_g10_rad_chunk_import_stdout(golden_name, fake_names, tmp_path):
    env = {name: value for name, value in os.environ.items() if not name.startswith(CLEARED_PREFIXES)}
    # An EMPTY value means "no key"; the subprocess runs from tmp_path (python -c
    # makes find_dotenv search the cwd), so no local .env can inject a value.
    for name in BASELINE_ENV_NAMES:
        env[name] = ""
    env["OPENAI_API_KEY"] = FAKE_ENV["OPENAI_API_KEY"]
    for name in fake_names:
        env[name] = FAKE_ENV[name]
    for name in ALBERT_OFF_ENV:
        env[name] = ALBERT_OFF_ENV[name]
    env["OCR_ENABLE_ALBERT"] = "0"
    env["EMBEDDING_PROVIDER"] = ""
    code = f"import sys; sys.path.insert(0, {os.path.join(RAGPY_ROOT, 'scripts')!r}); import rad_chunk"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(tmp_path), env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=300,
    )
    _check_golden_json(golden_name, {
        "returncode": proc.returncode,
        "stdout_lines": _stdout_lines(proc.stdout, tmp_path),
    })


# ======================================================================
# G11 — kwargs of the web LLM calls (notes, citation filter)
# ======================================================================
def _llm_clients_double(calls, key_log, with_openrouter=True):
    """Replacement for ``_get_llm_clients`` returning recording clients."""
    def fake_get_llm_clients(*args, **extra):
        """Log which credentials were passed (booleans only) and return doubles."""
        bound = _bind_known(args, extra, ("openai_api_key", "openrouter_api_key"))
        key_log.append({"openai_api_key_passed": bound.get("openai_api_key") is not None,
                        "openrouter_api_key_passed": bound.get("openrouter_api_key") is not None})

        def content(label, kwargs):
            """Deterministic HTML answer naming the client and the model."""
            return f"<p>[{label}:{kwargs['model']}] Réponse golden.</p>"

        openai_client = _FakeLLMClient("openai", calls, content)
        openrouter_client = _FakeLLMClient("openrouter", calls, content) if with_openrouter else None
        return openai_client, openrouter_client, "gpt-4o-mini"
    return fake_get_llm_clients


def test_g11_llm_call_kwargs(monkeypatch):
    prompt = "PROMPT GOLDEN G11 : produire une fiche de lecture sur « Le langage ordinaire »."
    note_cases = [
        ("short_openai", "short", "gpt-4o-mini", True),
        ("extended_openrouter", "extended", "google/gemini-2.5-flash", True),
        ("extended_default_model", "extended", None, True),
        ("short_openrouter_missing_fallback", "short", "google/gemini-2.5-flash", False),
    ]
    filter_cases = [
        ("filter_openai", "gpt-4o-mini", True),
        ("filter_openrouter", "google/gemini-2.5-flash", True),
        ("filter_openrouter_missing", "google/gemini-2.5-flash", False),
    ]
    out = []
    for case, mode, model, with_or in note_cases:
        calls, key_log = [], []
        monkeypatch.setattr(lng, "_get_llm_clients", _llm_clients_double(calls, key_log, with_or))
        entry = {"function": "_generate_with_llm", "case": case, "mode": mode, "model": model}
        try:
            entry["returned"] = lng._generate_with_llm(
                prompt, model=model, mode=mode,
                openai_api_key=FAKE_ENV["OPENAI_API_KEY"], openrouter_api_key=FAKE_ENV["OPENROUTER_API_KEY"],
            )
            entry["error"] = None
        except Exception as exc:  # noqa: BLE001 — the failure itself is the golden
            entry["returned"] = None
            entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
        entry["credentials_passed"] = key_log
        entry["calls"] = calls
        out.append(entry)
    for case, model, with_or in filter_cases:
        calls, key_log = [], []
        monkeypatch.setattr(cfilter, "_get_llm_clients", _llm_clients_double(calls, key_log, with_or))
        entry = {"function": "_call_llm_api", "case": case, "model": model}
        try:
            entry["returned"] = cfilter._call_llm_api(
                prompt, model=model,
                openai_api_key=FAKE_ENV["OPENAI_API_KEY"], openrouter_api_key=FAKE_ENV["OPENROUTER_API_KEY"],
            )
            entry["error"] = None
        except Exception as exc:  # noqa: BLE001 — the failure itself is the golden
            entry["returned"] = None
            entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
        entry["credentials_passed"] = key_log
        entry["calls"] = calls
        out.append(entry)
    _check_golden_json("g11_llm_call_kwargs.json", out)


# ======================================================================
# Harness self-checks
# ======================================================================
def test_golden_harness_uses_literal_lists():
    with open(os.path.abspath(__file__), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    forbidden_attr = "_as" + "dict"
    mapping_name = "CREDENTIAL_ENV" + "_MAPPING"
    offenders = []
    literal_lists = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr == forbidden_attr:
                offenders.append(f"{forbidden_attr} (ligne {node.lineno})")
            owner = node.value
            owner_name = owner.id if isinstance(owner, ast.Name) else getattr(owner, "attr", None)
            if owner_name == mapping_name and node.attr in ("values", "keys", "items"):
                offenders.append(f"{mapping_name}.{node.attr} (ligne {node.lineno})")
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            if target in ("BASELINE_ENV_NAMES", "OCR_RESULT_FIELDS"):
                value = node.value
                is_literal = isinstance(value, ast.List) and all(
                    isinstance(elt, ast.Constant) and isinstance(elt.value, str) for elt in value.elts)
                literal_lists[target] = [elt.value for elt in value.elts] if is_literal else None
    assert offenders == []
    assert literal_lists.get("OCR_RESULT_FIELDS") == [
        "text", "provider", "partial", "pages_done", "pages_total", "error"]
    names = literal_lists.get("BASELINE_ENV_NAMES")
    assert names is not None and len(names) == 15 and len(set(names)) == 15


def _is_fake_value_expr(node):
    """True for ``FAKE_ENV[...]`` / ``ALBERT_OFF_ENV[...]`` or the ``FAKE_ALBERT_KEY`` name."""
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        return node.value.id in ("FAKE_ENV", "ALBERT_OFF_ENV")
    return isinstance(node, ast.Name) and node.id == "FAKE_ALBERT_KEY"


def _is_credential_name(node):
    """True unless ``node`` is a string literal outside the credential/Albert names."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value in BASELINE_ENV_NAMES or node.value.startswith("ALBERT_")
    return True  # a computed name is treated as a credential name


def _env_writes_offenders(tree):
    """Env writes (``setenv`` calls, ``env[...] =``) of a credential name whose value
    is neither a fake constant (``FAKE_ENV``/``ALBERT_OFF_ENV``) nor the empty string."""
    offenders = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setenv" and len(node.args) >= 2):
            name_node, value_node = node.args[0], node.args[1]
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Subscript)
              and isinstance(node.targets[0].value, ast.Name) and node.targets[0].value.id == "env"):
            name_node, value_node = node.targets[0].slice, node.value
        else:
            continue
        if not _is_credential_name(name_node):
            continue
        is_empty = isinstance(value_node, ast.Constant) and value_node.value == ""
        if not (_is_fake_value_expr(value_node) or is_empty):
            offenders.append(f"ligne {node.lineno}")
    return offenders


def test_golden_harness_uses_fake_credentials():
    patterns = [
        re.compile("s" + "k-" + "[A-Za-z0-9_" + "-]{20,}"),
        re.compile("pc" + "sk_"),
        re.compile("Bea" + "rer "),
    ]
    raw_secret_values = [FAKE_ENV[name] for name in BASELINE_ENV_NAMES if name.endswith("_API_KEY")]
    raw_secret_values.append(FAKE_ALBERT_KEY)
    home = os.path.expanduser("~")
    home_user = os.path.basename(home.rstrip("/"))
    assert os.path.isdir(GOLDEN_DIR) and os.listdir(GOLDEN_DIR)
    hits = []
    for root, _dirs, files in os.walk(GOLDEN_DIR):
        for name in sorted(files):
            with open(os.path.join(root, name), encoding="utf-8", errors="replace") as fh:
                content = fh.read()
            for pattern in patterns:
                if pattern.search(content):
                    hits.append(f"{name}: motif de clé")
            if any(value in content for value in raw_secret_values):
                hits.append(f"{name}: valeur de clé factice non empreinte")
            if home in content or (len(home_user) >= 4 and home_user in content):
                hits.append(f"{name}: chemin personnel")
            if RAGPY_ROOT in content:
                hits.append(f"{name}: chemin du dépôt non normalisé")
    assert hits == []
    # Fake values only; exactly two Albert names, a fake key with Albert OFF.
    assert all(value.startswith("fake-") for value in FAKE_ENV.values())
    assert sorted(FAKE_ENV) == sorted(BASELINE_ENV_NAMES)
    assert FAKE_ALBERT_KEY == "fake-albert-key-0001"
    assert ALBERT_OFF_ENV == {"ALBERT_ENABLED": "0", "ALBERT_API_KEY": FAKE_ALBERT_KEY}
    with open(os.path.abspath(__file__), encoding="utf-8") as fh:
        assert _env_writes_offenders(ast.parse(fh.read())) == []
    for name in BASELINE_ENV_NAMES + EXTRA_CLEARED_NAMES:
        value = os.environ.get(name)
        assert value is None or value.startswith("fake-"), name
    assert not [name for name in os.environ if name.startswith(CLEARED_PREFIXES)]
