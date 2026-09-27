"""
Celery Task Runner
==================

Shared executor of the Celery pipeline tasks. A Celery task runs the SAME
command-line script as the matching HTTP route of ``app/routes/processing.py``
(same argv, ``--class`` included, same timeout) in a subprocess whose
environment is built by ``build_subprocess_env`` for the submitting user,
loaded from the database by ``user_id``. No credential ever travels through
the broker or the result backend: a task message only carries paths, options
and the user id.

Components:
    - ``load_user``: the submitting user, loaded by id (inactive or unknown
      users are refused).
    - ``build_task_env``: per-stage credential requirements (same keys as the
      HTTP routes) on top of ``build_subprocess_env``; also used by the Celery
      routes at submission time, so a missing key gives the same 403.
    - ``*_argv``: argv builders identical to the HTTP routes.
    - ``run_script``: ``subprocess.Popen`` with ``stdin=DEVNULL`` in its own
      process group; ``PROGRESS|...`` lines are parsed with
      ``parse_multilevel_progress`` and handed to ``on_progress`` from the
      calling thread; the whole process group is killed on timeout, on
      revocation (SIGTERM received by a prefork pool child) and on any
      exception raised while waiting (for instance Celery's soft time limit).
    - ``parse_vectordb_stdout``: the anchored regexes of the ``/upload_db``
      route parser for the counts; the ``Dedup journal:`` line naming
      ``dedup_journal.jsonl`` is taken whole (paths with spaces).
    - Task owner registry: ``ragpy:celery_owner:<task_id>`` in Redis (TTL 7
      days); without Redis, only administrators may read or cancel a task.

Retry policy: only ``TaskInfrastructureError`` (database unavailable, process
spawn failure) is listed in ``INFRASTRUCTURE_ERRORS``, the ``autoretry_for``
of every pipeline task. A non-zero exit, a timeout, a missing credential or
a revocation is deterministic and never retried.

Albert (DINUM, opt-in, ``ALBERT_ENABLED=1``), same rules as the HTTP routes:
    - chunking keys from ``resolve_llm_provider`` (``albert/`` model while
      Albert is disabled: ``AlbertDisabledError``, a ``ValueError``);
    - extraction: third attempt with ``albert_api_key`` when the Albert OCR
      link is active for the user;
    - dense: ``EMBEDDING_PROVIDER`` resolved like the route and set in the
      environment explicitly;
    - vectordb: the ``albert`` target (``--albert-*`` flags), never retried;
    - ``task_timeout``: ``ALBERT_SUBPROCESS_TIMEOUT`` only when the task
      selects Albert, the historical timeout otherwise;
    - ``albert_time_limits``: Celery time limits passed at submission
      (``apply_async``) when the task selects Albert, above that timeout.
"""
import errno
import logging
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence

from app.core.config import RAGPY_DIR
from app.core.credentials import (
    CREDENTIAL_ENV_MAPPING,
    CredentialMissingError,
    albert_enabled,
    build_subprocess_env,
    get_credential_or_env,
)
from app.database.session import SessionLocal
from app.models.user import User
from app.utils.sse_helpers import parse_multilevel_progress

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Constants (mirrors of the HTTP routes of app/routes/processing.py)
# ----------------------------------------------------------------------
PYTHON_EXECUTABLE = "python3"

EXTRACTION_TIMEOUT = 1800
CHUNKING_TIMEOUT = 1800
DENSE_TIMEOUT = 1800
SPARSE_TIMEOUT = 1800
VECTORDB_TIMEOUT = 3600

DEFAULT_CHUNKING_MODEL = "gpt-4o-mini"

STAGE_EXTRACTION = "extraction"
STAGE_CHUNKING = "chunking"
STAGE_DENSE = "dense"
STAGE_SPARSE = "sparse"
STAGE_VECTORDB = "vectordb"
STAGES = (STAGE_EXTRACTION, STAGE_CHUNKING, STAGE_DENSE, STAGE_SPARSE, STAGE_VECTORDB)

VECTORDB_REQUIRED_KEYS = {
    "pinecone": ["pinecone_api_key"],
    "weaviate": ["weaviate_url"],  # API key optional for some setups
    "qdrant": ["qdrant_url"],      # API key optional for local instances
}
VECTORDB_CHOICES = tuple(VECTORDB_REQUIRED_KEYS)

# Anchored parsers of the rad_vectordb.py "=== Result ===" block: the SAME
# patterns as the /upload_db route parser (app/routes/processing.py).
VECTORDB_INSERTED_PATTERN = r'^Inserted:\s*(\d+)'
VECTORDB_SKIPPED_PATTERN = r'^Skipped \(dedup\):\s*(\d+)'
VECTORDB_JOURNAL_PATTERN = r'^Dedup journal:\s*(\S+)'
# The journal line as a whole (Celery only): rad_vectordb.py prints the
# absolute path of the journal, whose file name is always
# rad_dedup.JOURNAL_FILENAME; such a line is taken whole, so a session under
# a folder with spaces keeps its full path, as the in-process call did on main.
VECTORDB_JOURNAL_LINE_PATTERN = r'^Dedup journal:\s*(.+)$'
DEDUP_JOURNAL_FILENAME = "dedup_journal.jsonl"

# Albert target (the albert branch of the /upload_db route): extra lines, and
# the whole "Dedup journal:" line (paths with spaces) for this target only.
ALBERT_DB_CHOICE = "albert"
ALBERT_CREDENTIAL_KEY = "albert_api_key"
ALBERT_EXISTING_PATTERN = r'^Skipped \(existing\):\s*(\d+)'
ALBERT_MANIFEST_PATTERN = r'^Albert manifest:\s*(.+)$'
ALBERT_JOURNAL_PATTERN = r'^Dedup journal:\s*(.+)$'
# Copies of the upload manifests, out of uploads/ (same place as the route).
ALBERT_MANIFEST_DIR = os.path.join(RAGPY_DIR, "data", "albert_manifests")
_SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SAFE_LABEL_MAX_CHARS = 120

# Seconds added to ALBERT_SUBPROCESS_TIMEOUT for the Celery time limits of a
# task that selects Albert (soft: + margin, hard: + 2 * margin).
ALBERT_TIME_LIMIT_MARGIN = 300

# Dense embedding providers (form field / server EMBEDDING_PROVIDER).
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"
EMBEDDING_PROVIDERS = ("openai", "albert")

# Task owner registry.
OWNER_KEY_PREFIX = "ragpy:celery_owner:"
OWNER_TTL_SECONDS = 7 * 24 * 3600
_MAX_TASK_ID_LENGTH = 255
_REDIS_SCHEMES = ("redis://", "rediss://", "unix://")

# Subprocess handling.
DEFAULT_POLL_INTERVAL = 0.2
DEFAULT_KILL_GRACE = 5.0
_READER_JOIN_MARGIN = 2.0
_DETAIL_TAIL_CHARS = 1000
_MIN_SECRET_LENGTH = 8
_ANSI_ESCAPE_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
_TRANSIENT_SPAWN_ERRNOS = frozenset({errno.EAGAIN, errno.ENOMEM, errno.EMFILE, errno.ENFILE})


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------
class TaskError(Exception):
    """Deterministic task failure: never retried by Celery.

    Every subclass keeps a single message in ``args`` so that the result
    backend can serialise and rebuild it.
    """


class TaskUserError(TaskError):
    """The submitting user is unknown or no longer active."""


class ScriptFailedError(TaskError):
    """The pipeline script failed (non-zero exit, missing output, launch error).

    Attributes:
        returncode: Exit code of the script, or None when it did not run.
    """

    def __init__(self, message: str, *, returncode: Optional[int] = None):
        """Store the (already redacted) message and the exit code."""
        super().__init__(message)
        self.returncode = returncode


class ScriptTimeoutError(TaskError):
    """The pipeline script exceeded its timeout; its process group was killed.

    Attributes:
        timeout: The timeout in seconds.
    """

    def __init__(self, message: str, *, timeout: Optional[float] = None):
        """Store the message and the timeout."""
        super().__init__(message)
        self.timeout = timeout


class ScriptRevokedError(TaskError):
    """The task was revoked (SIGTERM) while its script ran; the group was killed."""


class TaskInfrastructureError(Exception):
    """Transient infrastructure failure (database unavailable, process spawn).

    The only error class retried by the pipeline tasks.
    """


INFRASTRUCTURE_ERRORS = (TaskInfrastructureError,)


class ScriptResult(NamedTuple):
    """Outcome of ``run_script``.

    Attributes:
        returncode: Exit code of the script.
        stdout: Full decoded standard output.
        stderr: Full decoded standard error.
    """

    returncode: int
    stdout: str
    stderr: str


# ----------------------------------------------------------------------
# User and environment
# ----------------------------------------------------------------------
def load_user(user_id: Any, *, session_factory: Optional[Callable[[], Any]] = None) -> User:
    """Load the submitting user by id, detached from its database session.

    Args:
        user_id: Id of the user who submitted the task.
        session_factory: Session factory (default: ``SessionLocal`` of this
            module, looked up at call time so tests can patch it).

    Returns:
        The ``User`` instance (columns loaded, safe to read after close).

    Raises:
        TaskUserError: No user id, unknown user, or inactive account.
        TaskInfrastructureError: The database could not be queried.
    """
    if user_id is None or isinstance(user_id, bool):
        raise TaskUserError("Tâche sans utilisateur : user_id manquant.")
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        raise TaskUserError("Tâche sans utilisateur valide : user_id invalide.") from None

    factory = session_factory or SessionLocal
    try:
        from sqlalchemy.exc import OperationalError
    except ImportError:  # pragma: no cover - sqlalchemy is a hard dependency
        OperationalError = ()  # type: ignore[assignment]

    db = factory()
    try:
        user = db.query(User).filter(User.id == uid).first()
        if user is not None:
            db.expunge(user)
    except OperationalError as exc:
        raise TaskInfrastructureError(f"Base de données indisponible : {type(exc).__name__}") from exc
    finally:
        db.close()

    if user is None:
        raise TaskUserError(f"Utilisateur {uid} introuvable.")
    if not user.is_active or getattr(user, "is_pending_approval", False):
        raise TaskUserError(f"Utilisateur {uid} inactif : tâche refusée.")
    return user


def _albert_modules() -> SimpleNamespace:
    """Import the stdlib Albert helpers lazily (configuration, errors, resolver).

    Returns:
        A namespace with ``AlbertConfig``, ``FIELD_ENV_NAMES``,
        ``AlbertDisabledError``, ``resolve_llm_provider`` and ``PROVIDER_ALBERT``.
    """
    from scripts.rad_albert.config import AlbertConfig, FIELD_ENV_NAMES
    from scripts.rad_albert.errors import AlbertDisabledError
    from scripts.rad_providers import PROVIDER_ALBERT, resolve_llm_provider

    return SimpleNamespace(
        AlbertConfig=AlbertConfig,
        FIELD_ENV_NAMES=FIELD_ENV_NAMES,
        AlbertDisabledError=AlbertDisabledError,
        resolve_llm_provider=resolve_llm_provider,
        PROVIDER_ALBERT=PROVIDER_ALBERT,
    )


def albert_setting(field_name: str) -> Any:
    """Read one ``AlbertConfig`` field from its environment variable only.

    Same rule as the HTTP routes: the configuration is built from a mapping
    holding that single variable (parser and default of
    ``AlbertConfig.from_env``), so nothing else is read or validated.

    Args:
        field_name: Name of an ``AlbertConfig`` field.

    Returns:
        The parsed value of the field.
    """
    modules = _albert_modules()
    env_name = modules.FIELD_ENV_NAMES[field_name]
    raw = os.environ.get(env_name)
    mapping = {} if raw is None else {env_name: raw}
    return getattr(modules.AlbertConfig.from_env(mapping), field_name)


def task_timeout(default: float, albert_selected: bool) -> float:
    """Timeout of a pipeline script (same rule as ``processing._subprocess_timeout``).

    Args:
        default: The historical timeout of the stage (``*_TIMEOUT``).
        albert_selected: True when the task selects Albert.

    Returns:
        ``ALBERT_SUBPROCESS_TIMEOUT`` when Albert is selected, else ``default``.
    """
    if not albert_selected:
        return default
    return albert_setting("subprocess_timeout")


def albert_time_limits() -> Dict[str, int]:
    """Celery time limits of a task that selects Albert (``apply_async`` options).

    The global limits of ``app/celery_app.py`` (soft 3600 s, hard 7200 s)
    would stop an Albert task long before its script timeout
    (``ALBERT_SUBPROCESS_TIMEOUT``, 21600 s by default). The soft limit is
    that timeout plus ``ALBERT_TIME_LIMIT_MARGIN``, so ``run_script``'s own
    timeout (process group killed, ``ScriptTimeoutError``) always fires
    first; the hard limit adds the margin once more.

    Returns:
        ``{'soft_time_limit': timeout + margin, 'time_limit': timeout + 2 * margin}``.
    """
    timeout = int(albert_setting("subprocess_timeout"))
    return {
        "soft_time_limit": timeout + ALBERT_TIME_LIMIT_MARGIN,
        "time_limit": timeout + 2 * ALBERT_TIME_LIMIT_MARGIN,
    }


def resolve_chat_model(model: Optional[str]) -> Any:
    """Resolve the chat provider of a recoding model (single resolver).

    Args:
        model: Recoding model (empty means the default model).

    Returns:
        ``ProviderResolution(provider, wire_model, credential_key)``.

    Raises:
        AlbertDisabledError: ``albert/…`` while Albert is disabled.
        ValueError: ``albert/`` without a model identifier.
    """
    return _albert_modules().resolve_llm_provider(model or DEFAULT_CHUNKING_MODEL, albert_enabled=albert_enabled())


def chunking_selects_albert(model: Optional[str]) -> bool:
    """True when the recoding model selects Albert (``albert/<id>``, Albert ON).

    Args:
        model: Recoding model.

    Returns:
        True for an Albert resolution; False otherwise (refused models included).
    """
    try:
        return resolve_chat_model(model).provider == _albert_modules().PROVIDER_ALBERT
    except ValueError:
        return False


def chunking_required_keys(model: Optional[str]) -> List[str]:
    """Credential keys required to recode with ``model`` (HTTP route rule).

    The single resolver decides: ``albert/<id>`` → ``albert_api_key`` (Albert
    ON), a model containing ``/`` → OpenRouter, any other one → OpenAI.

    Args:
        model: Recoding model (empty means the default model).

    Returns:
        A one-element list of credential keys.

    Raises:
        AlbertDisabledError: ``albert/…`` while Albert is disabled.
        ValueError: ``albert/`` without a model identifier.
    """
    return [resolve_chat_model(model).credential_key]


def resolve_embedding_provider(value: Optional[str]) -> Optional[str]:
    """Resolve the dense embedding provider (same rule as the HTTP route).

    Albert disabled: ignored (None), except ``albert`` which is refused.
    Albert enabled: the given value, else the server ``EMBEDDING_PROVIDER``,
    else ``openai``; the result must be ``openai`` or ``albert``.

    Args:
        value: Requested provider (form field), or None.

    Returns:
        ``'openai'`` or ``'albert'`` while Albert is enabled, else None.

    Raises:
        AlbertDisabledError: ``albert`` requested while Albert is disabled.
        ValueError: Unknown provider while Albert is enabled.
    """
    albert_disabled_error = _albert_modules().AlbertDisabledError
    requested = (value or "").strip()
    if not albert_enabled():
        if requested.lower() == ALBERT_DB_CHOICE:
            raise albert_disabled_error(
                "embedding_provider=albert exige Albert, désactivé sur ce serveur (ALBERT_ENABLED=1 requis).",
                model="embedding_provider=albert",
            )
        return None
    raw = requested or (os.environ.get(EMBEDDING_PROVIDER_ENV) or "").strip()
    name = raw.lower() or EMBEDDING_PROVIDERS[0]
    if name not in EMBEDDING_PROVIDERS:
        raise ValueError(
            f"embedding_provider={raw!r} inconnu : valeurs admises « openai » ou « albert »."
        )
    return name


def ocr_albert_active(user: User) -> bool:
    """True when the Albert OCR link runs for ``user`` (Albert ON, OCR on, key).

    Args:
        user: The submitting user.

    Returns:
        True when ``ALBERT_ENABLED``, ``OCR_ENABLE_ALBERT`` and an Albert key
        available to the user are all set (nothing is read while Albert is OFF).
    """
    if not albert_enabled():
        return False
    if not albert_setting("ocr_enabled"):
        return False
    return bool(get_credential_or_env(user, ALBERT_CREDENTIAL_KEY))


def _require_albert_enabled(what: str) -> None:
    """Raise ``AlbertDisabledError`` when Albert is disabled on this worker.

    Args:
        what: The Albert selection, quoted in the message.

    Raises:
        AlbertDisabledError: Albert is disabled.
    """
    if not albert_enabled():
        raise _albert_modules().AlbertDisabledError(model=what)


def build_task_env(
    user: User,
    stage: str,
    *,
    model: Optional[str] = None,
    db_choice: Optional[str] = None,
    embedding_provider: Optional[str] = None,
) -> Dict[str, str]:
    """Build the subprocess environment of a pipeline stage for ``user``.

    Credential requirements are those of the HTTP routes:

    - extraction: ``mistral_api_key``, else ``openai_api_key``, else (Albert
      OCR link active for the user) ``albert_api_key``; when all are
      missing, ``CredentialMissingError('mistral_api_key')``;
    - chunking: ``chunking_required_keys(model)``;
    - dense: ``openai_api_key``, or ``albert_api_key`` for
      ``embedding_provider='albert'``; a resolved provider is also written to
      ``EMBEDDING_PROVIDER`` (None: environment untouched, historical path);
    - sparse: none (isolation only);
    - vectordb: ``VECTORDB_REQUIRED_KEYS[db_choice]``, or ``albert_api_key``
      for the ``albert`` target (Albert ON only).

    Args:
        user: The submitting user.
        stage: One of ``STAGES``.
        model: Recoding model (chunking only).
        db_choice: Target database (vectordb only).
        embedding_provider: Resolved dense provider (dense only, None = default).

    Returns:
        The environment returned by ``build_subprocess_env``.

    Raises:
        CredentialMissingError: A required credential is missing.
        AlbertDisabledError: Albert selected while it is disabled.
        ValueError: Unknown stage, database or provider.
    """
    if stage == STAGE_EXTRACTION:
        try:
            return build_subprocess_env(user, required_keys=["mistral_api_key"])
        except CredentialMissingError:
            try:
                return build_subprocess_env(user, required_keys=["openai_api_key"])
            except CredentialMissingError:
                if ocr_albert_active(user):
                    try:
                        return build_subprocess_env(user, required_keys=[ALBERT_CREDENTIAL_KEY])
                    except CredentialMissingError:
                        pass
                raise CredentialMissingError(
                    credential_key="mistral_api_key",
                    is_admin=bool(getattr(user, "is_admin", False)),
                ) from None
    if stage == STAGE_CHUNKING:
        return build_subprocess_env(user, required_keys=chunking_required_keys(model))
    if stage == STAGE_DENSE:
        if embedding_provider is None:
            return build_subprocess_env(user, required_keys=["openai_api_key"])
        if embedding_provider not in EMBEDDING_PROVIDERS:
            raise ValueError(f"Unknown embedding provider: {embedding_provider}")
        if embedding_provider == ALBERT_DB_CHOICE:
            _require_albert_enabled("embedding_provider=albert")
            env = build_subprocess_env(user, required_keys=[ALBERT_CREDENTIAL_KEY])
        else:
            env = build_subprocess_env(user, required_keys=["openai_api_key"])
        env[EMBEDDING_PROVIDER_ENV] = embedding_provider
        return env
    if stage == STAGE_SPARSE:
        return build_subprocess_env(user)
    if stage == STAGE_VECTORDB:
        if db_choice == ALBERT_DB_CHOICE:
            _require_albert_enabled("db_choice=albert")
            return build_subprocess_env(user, required_keys=[ALBERT_CREDENTIAL_KEY])
        if db_choice not in VECTORDB_REQUIRED_KEYS:
            raise ValueError(f"Unknown database type: {db_choice}")
        return build_subprocess_env(user, required_keys=list(VECTORDB_REQUIRED_KEYS[db_choice]))
    raise ValueError(f"Unknown pipeline stage: {stage}")


# ----------------------------------------------------------------------
# argv builders (identical to the HTTP routes)
# ----------------------------------------------------------------------
def script_path(name: str) -> str:
    """Absolute path of a pipeline script, built like the HTTP routes."""
    return os.path.join(RAGPY_DIR, "scripts", name)


def extraction_argv(json_path: str, base_dir: str, output_path: str) -> List[str]:
    """argv of ``rad_dataframe.py`` (route ``POST /process_dataframe``)."""
    return [
        PYTHON_EXECUTABLE, os.path.abspath(script_path("rad_dataframe.py")),
        "--json", json_path,
        "--dir", base_dir,
        "--output", output_path,
    ]


def chunking_argv(input_csv: str, output_dir: str, model: Optional[str]) -> List[str]:
    """argv of ``rad_chunk.py --phase initial`` (route ``POST /initial_text_chunking``)."""
    return [
        PYTHON_EXECUTABLE, script_path("rad_chunk.py"),
        "--input", input_csv,
        "--output", output_dir,
        "--phase", "initial",
        "--model", model or DEFAULT_CHUNKING_MODEL,
    ]


def dense_argv(input_file: str, output_dir: str) -> List[str]:
    """argv of ``rad_chunk.py --phase dense`` (route ``POST /dense_embedding_generation``)."""
    return [
        PYTHON_EXECUTABLE, script_path("rad_chunk.py"),
        "--input", input_file,
        "--output", output_dir,
        "--phase", "dense",
    ]


def sparse_argv(input_file: str, output_dir: str) -> List[str]:
    """argv of ``rad_chunk.py --phase sparse`` (route ``POST /sparse_embedding_generation``)."""
    return [
        PYTHON_EXECUTABLE, script_path("rad_chunk.py"),
        "--input", input_file,
        "--output", output_dir,
        "--phase", "sparse",
    ]


def vectordb_argv(
    input_file: str,
    db_choice: str,
    *,
    pinecone_index_name: Optional[str] = None,
    pinecone_namespace: Optional[str] = None,
    weaviate_class_name: Optional[str] = None,
    weaviate_tenant_name: Optional[str] = None,
    qdrant_collection_name: Optional[str] = None,
    albert_collection_id: Optional[int] = None,
    albert_collection_name: Optional[str] = None,
    albert_create_collection: bool = False,
    albert_ack_retention: bool = False,
) -> List[str]:
    """argv of ``rad_vectordb.py`` (route ``POST /upload_db``, ``--class`` included).

    For the ``albert`` target, the flags are those of the route: the id as
    ``--albert-collection-id ID``, the name as ``--albert-collection-name=NAME``
    (a name starting with ``-`` is never read as an option), then
    ``--albert-create-collection`` and ``--albert-ack-retention``.
    """
    cmd = [PYTHON_EXECUTABLE, script_path("rad_vectordb.py"), "--input", input_file, "--db", db_choice]
    if db_choice == "pinecone":
        if pinecone_index_name:
            cmd.extend(["--index", pinecone_index_name])
        if pinecone_namespace:
            cmd.extend(["--namespace", pinecone_namespace])
    elif db_choice == "weaviate":
        if weaviate_class_name:
            cmd.extend(["--class", weaviate_class_name])
        if weaviate_tenant_name:
            cmd.extend(["--tenant", weaviate_tenant_name])
    elif db_choice == "qdrant":
        if qdrant_collection_name:
            cmd.extend(["--collection", qdrant_collection_name])
    elif db_choice == ALBERT_DB_CHOICE:
        if albert_collection_id is not None:
            cmd.extend(["--albert-collection-id", str(int(albert_collection_id))])
        if albert_collection_name:
            cmd.append(f"--albert-collection-name={albert_collection_name}")
        if albert_create_collection:
            cmd.append("--albert-create-collection")
        if albert_ack_retention:
            cmd.append("--albert-ack-retention")
    return cmd


# ----------------------------------------------------------------------
# Subprocess execution
# ----------------------------------------------------------------------
def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    """Send ``sig`` to the process group led by ``proc`` (errors ignored)."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, sig)
        else:  # pragma: no cover - non-POSIX platforms
            proc.send_signal(sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _kill_process_group(proc: subprocess.Popen, grace: float) -> None:
    """Terminate the whole process group of ``proc``: SIGTERM, then SIGKILL.

    Args:
        proc: The script process (leader of its own process group).
        grace: Seconds to wait for the leader after SIGTERM.
    """
    _signal_group(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # Children that ignored SIGTERM (or outlived the leader) are killed too.
    _signal_group(proc, getattr(signal, "SIGKILL", signal.SIGTERM))
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:  # pragma: no cover - SIGKILL cannot be ignored
        logger.warning("Script process %s did not exit after SIGKILL", proc.pid)


def _pump_stream(stream: Any, sink: List[str], events: "queue.Queue[Dict[str, Any]]") -> None:
    """Read ``stream`` line by line into ``sink``; queue parsed PROGRESS events.

    Runs in a reader thread; ``on_progress`` itself is always called from the
    thread that runs ``run_script`` (Celery's request context is thread-local).
    """
    try:
        for raw in iter(stream.readline, b""):
            text = raw.decode("utf-8", errors="replace")
            sink.append(text)
            line = _ANSI_ESCAPE_RE.sub("", text).strip()
            if line.startswith("PROGRESS|"):
                event = parse_multilevel_progress(line)
                if event:
                    events.put(event)
    except (OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except (OSError, ValueError):
            pass


def _deliver(events: "queue.Queue[Dict[str, Any]]", on_progress: Optional[Callable[[Dict[str, Any]], None]],
             first: Optional[Dict[str, Any]] = None) -> None:
    """Hand ``first`` and every queued event to ``on_progress`` (errors are logged, not raised)."""
    pending = [] if first is None else [first]
    while True:
        try:
            pending.append(events.get_nowait())
        except queue.Empty:
            break
    if on_progress is None:
        return
    for event in pending:
        try:
            on_progress(event)
        except Exception as exc:  # progress is advisory: never fail the script for it
            logger.debug("Progress callback failed: %s", exc)


_NOT_INSTALLED = object()


def _in_pool_child() -> bool:
    """True inside a Celery prefork pool child (billiard process not named ``MainProcess``).

    Only there does SIGTERM mean "revoke this task" (``terminate=True``) or a
    cold shutdown; in the worker main process (``solo`` pool) SIGTERM is the
    warm shutdown request, which must keep letting the running task finish.
    """
    try:
        from billiard.process import current_process
    except ImportError:  # pragma: no cover - billiard ships with celery
        return False
    try:
        return current_process().name != "MainProcess"
    except Exception:  # pragma: no cover - defensive
        return False


def _install_revoke_handler(proc: subprocess.Popen) -> Any:
    """Install a temporary SIGTERM handler that kills the script's process group.

    Celery delivers SIGTERM to the worker child when a running task is
    revoked with ``terminate=True``. The script runs in its own session, so
    it would otherwise outlive the task. Only possible on the main thread.

    Returns:
        The previous handler, or ``_NOT_INSTALLED`` when nothing was installed.
    """
    if threading.current_thread() is not threading.main_thread():
        return _NOT_INSTALLED
    previous = signal.getsignal(signal.SIGTERM)
    if previous is None:  # handler not set from Python: restore the default one
        previous = signal.SIG_DFL

    def _on_sigterm(signum: int, frame: Any) -> None:
        """Kill the script's group and interrupt the wait with ScriptRevokedError."""
        _signal_group(proc, signal.SIGTERM)
        raise ScriptRevokedError("Tâche annulée : le script a été arrêté.")

    signal.signal(signal.SIGTERM, _on_sigterm)
    return previous


def _restore_revoke_handler(previous: Any) -> None:
    """Restore the SIGTERM handler saved by ``_install_revoke_handler``."""
    if previous is _NOT_INSTALLED:
        return
    try:
        signal.signal(signal.SIGTERM, previous)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        pass


def run_script(
    cmd: Sequence[str],
    env: Optional[Dict[str, str]],
    *,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    timeout: Optional[float] = None,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    kill_grace: float = DEFAULT_KILL_GRACE,
    handle_sigterm: Optional[bool] = None,
) -> ScriptResult:
    """Run a pipeline script to completion in its own process group.

    ``stdin`` is ``DEVNULL`` (a script can never block on a prompt), stdout
    and stderr are read concurrently, and every ``PROGRESS|level|x/y|msg``
    line is parsed by ``parse_multilevel_progress`` and passed to
    ``on_progress`` from the calling thread.

    Args:
        cmd: argv of the script.
        env: Subprocess environment (``build_task_env``).
        on_progress: Callback receiving each parsed progress event.
        timeout: Seconds before the process group is killed (None: no limit).
        poll_interval: Seconds between two checks of the process state.
        kill_grace: Seconds granted after SIGTERM before SIGKILL.
        handle_sigterm: Treat SIGTERM as a revocation while the script runs
            (kill the group, raise ``ScriptRevokedError``). None (default):
            only inside a Celery prefork pool child (``_in_pool_child``).

    Returns:
        ``ScriptResult(returncode, stdout, stderr)``, whatever the exit code.

    Raises:
        ScriptTimeoutError: The timeout expired (group killed).
        ScriptRevokedError: SIGTERM received while waiting (group killed).
        ScriptFailedError: The script could not be launched.
        TaskInfrastructureError: Transient spawn failure (EAGAIN, ENOMEM...).
    """
    argv = list(cmd)
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
    except OSError as exc:
        if exc.errno in _TRANSIENT_SPAWN_ERRNOS:
            raise TaskInfrastructureError(f"Lancement du script impossible : {exc.strerror}") from exc
        raise ScriptFailedError(f"Lancement du script impossible : {exc.strerror or exc}") from exc

    script = _script_name(argv)
    logger.info("Started script %s (PID %s)", script, proc.pid)
    stdout_parts: List[str] = []
    stderr_parts: List[str] = []
    events: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    readers = [
        threading.Thread(target=_pump_stream, args=(proc.stdout, stdout_parts, events), daemon=True),
        threading.Thread(target=_pump_stream, args=(proc.stderr, stderr_parts, events), daemon=True),
    ]
    for reader in readers:
        reader.start()

    deadline = None if timeout is None else time.monotonic() + float(timeout)
    if handle_sigterm is None:
        handle_sigterm = _in_pool_child()
    previous_handler = _install_revoke_handler(proc) if handle_sigterm else _NOT_INSTALLED
    try:
        while True:
            _deliver(events, on_progress)
            if proc.poll() is not None:
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise ScriptTimeoutError(
                    f"Script {script} timed out after {timeout} s; process group killed.",
                    timeout=timeout,
                )
            try:
                event = events.get(timeout=poll_interval)
            except queue.Empty:
                continue
            _deliver(events, on_progress, first=event)
    except BaseException:
        _restore_revoke_handler(previous_handler)
        previous_handler = _NOT_INSTALLED
        _kill_process_group(proc, kill_grace)
        raise
    finally:
        _restore_revoke_handler(previous_handler)
        for reader in readers:
            reader.join(timeout=kill_grace + _READER_JOIN_MARGIN)

    _deliver(events, on_progress)
    logger.info("Script %s finished with code %s", script, proc.returncode)
    return ScriptResult(proc.returncode, "".join(stdout_parts), "".join(stderr_parts))


def _script_name(cmd: Sequence[str]) -> str:
    """Basename of the ``.py`` script launched by ``cmd`` (or of ``cmd[0]``)."""
    for item in cmd:
        if str(item).endswith(".py"):
            return os.path.basename(str(item))
    return os.path.basename(str(cmd[0])) if cmd else ""


def redact_env_secrets(text: str, env: Optional[Dict[str, str]]) -> str:
    """Mask every credential value of ``env`` found in ``text``.

    Script output ends up in the result backend (task failure message) and in
    the session ``error_message``: a credential printed by a script must
    never be stored there.

    Args:
        text: Text to clean.
        env: Environment given to the script (``None``: nothing to mask).

    Returns:
        ``text`` with each credential value replaced by ``***``.
    """
    if not text or not env:
        return text
    names = set(CREDENTIAL_ENV_MAPPING.values())
    names.update(name for name in env if name.endswith("_API_KEY"))
    for name in sorted(names):
        value = env.get(name)
        if value and len(value) >= _MIN_SECRET_LENGTH and value in text:
            text = text.replace(value, "***")
    return text


def check_script_result(result: ScriptResult, cmd: Sequence[str], env: Optional[Dict[str, str]],
                        *, prefer_stdout: bool = False) -> None:
    """Raise ``ScriptFailedError`` when the script exited with a non-zero code.

    Args:
        result: Outcome of ``run_script``.
        cmd: argv of the script (for the message).
        env: Environment of the script (secrets masked in the details).
        prefer_stdout: Take the details from the stdout tail first
            (``rad_vectordb.py`` prints its diagnostics on stdout).

    Raises:
        ScriptFailedError: Non-zero exit code (never retried).
    """
    if result.returncode == 0:
        return
    stdout_tail = (result.stdout or "")[-_DETAIL_TAIL_CHARS:].strip()
    stderr_tail = (result.stderr or "")[-_DETAIL_TAIL_CHARS:].strip()
    if prefer_stdout:
        details = stdout_tail or stderr_tail
    else:
        details = stderr_tail or stdout_tail
    details = redact_env_secrets(details, env)
    message = f"Script {_script_name(cmd)} failed with code {result.returncode}."
    if details:
        message = f"{message} {details}"
    raise ScriptFailedError(message, returncode=result.returncode)


def parse_vectordb_stdout(stdout: Optional[str]) -> Dict[str, Any]:
    """Parse the ``=== Result ===`` block of ``rad_vectordb.py``.

    Same anchored patterns (``re.MULTILINE``) as the ``/upload_db`` route for
    the counts. The ``Dedup journal:`` line is read by
    ``_parse_journal_path``: whole when it names the rad_vectordb journal
    (paths with spaces kept), first token otherwise (as the route).

    Args:
        stdout: Standard output of the script.

    Returns:
        ``{"inserted_count": int|None, "skipped_count": int|None,
        "journal_path": str|None}`` (None when the line is absent).
    """
    text = stdout or ""
    parsed: Dict[str, Any] = {"inserted_count": None, "skipped_count": None, "journal_path": None}
    m = re.search(VECTORDB_INSERTED_PATTERN, text, re.MULTILINE)
    if m:
        parsed["inserted_count"] = int(m.group(1))
    m = re.search(VECTORDB_SKIPPED_PATTERN, text, re.MULTILINE)
    if m:
        parsed["skipped_count"] = int(m.group(1))
    parsed["journal_path"] = _parse_journal_path(text)
    return parsed


def _parse_journal_path(text: str) -> Optional[str]:
    """Path printed on the ``Dedup journal:`` line of ``rad_vectordb.py``, or None.

    rad_vectordb.py prints the absolute path of ``dedup_journal.jsonl``
    (``os.path.abspath`` of the session folder, which may contain spaces, as
    under ``Google Drive``). A line ending with that file name is taken
    whole, so the Celery task returns the same path as the in-process call
    it replaced; any other line keeps the route's first-token reading.

    Args:
        text: Standard output of the script.

    Returns:
        The journal path, or None when the line is absent.
    """
    m = re.search(VECTORDB_JOURNAL_LINE_PATTERN, text, re.MULTILINE)
    if m:
        whole = m.group(1).strip()
        if os.path.basename(whole) == DEDUP_JOURNAL_FILENAME:
            return whole
    m = re.search(VECTORDB_JOURNAL_PATTERN, text, re.MULTILINE)
    return m.group(1) if m else None


def parse_albert_stdout(stdout: Optional[str]) -> Dict[str, Any]:
    """Parse the ``=== Result ===`` block of ``rad_vectordb.py --db albert``.

    Same anchored patterns as the albert branch of the ``/upload_db`` route:
    those of ``parse_vectordb_stdout`` for the counts, the whole
    ``Dedup journal:`` line (paths with spaces), ``Skipped (existing)`` and
    the ``Albert manifest:`` line.

    Args:
        stdout: Standard output of the script.

    Returns:
        ``parse_vectordb_stdout`` keys plus ``existing_count`` (int|None) and
        ``manifest_path`` (str|None).
    """
    text = stdout or ""
    parsed = parse_vectordb_stdout(text)
    m = re.search(ALBERT_JOURNAL_PATTERN, text, re.MULTILINE)
    parsed["journal_path"] = (m.group(1).strip() or None) if m else None
    m = re.search(ALBERT_EXISTING_PATTERN, text, re.MULTILINE)
    parsed["existing_count"] = int(m.group(1)) if m else None
    m = re.search(ALBERT_MANIFEST_PATTERN, text, re.MULTILINE)
    parsed["manifest_path"] = (m.group(1).strip() or None) if m else None
    return parsed


def _safe_label(text: Any) -> str:
    """Turn a session path into a file name fragment (no separator, no leading dot).

    Args:
        text: The session folder (path or name).

    Returns:
        A non-empty label made of ``[A-Za-z0-9._-]`` characters.
    """
    label = _SAFE_LABEL_RE.sub("_", str(text or "")).strip("._")
    return label[:_SAFE_LABEL_MAX_CHARS] or "session"


def albert_manifest_source(input_file: str) -> str:
    """Manifest written by ``rad_vectordb.py --db albert`` next to its input.

    Args:
        input_file: The upload input file.

    Returns:
        ``<input dir>/albert_manifest.jsonl``.
    """
    from scripts.rad_albert.collections import manifest_path_for

    return manifest_path_for(os.path.dirname(os.path.abspath(input_file)))


def archive_albert_manifest(manifest_path: Optional[str], user_id: Any, session_label: str) -> Optional[str]:
    """Copy an Albert upload manifest out of ``uploads/`` (even a partial one).

    Same destination as the HTTP route:
    ``ALBERT_MANIFEST_DIR/<user_id>/<session>-<ts>.jsonl``. Never raises.

    Args:
        manifest_path: Manifest written next to the upload input.
        user_id: Id of the uploading user.
        session_label: Session folder (or name) of the upload.

    Returns:
        The path of the copy, or None when there was nothing to copy.
    """
    try:
        if not manifest_path or not os.path.isfile(manifest_path):
            return None
        target_dir = os.path.join(ALBERT_MANIFEST_DIR, str(int(user_id)))
        os.makedirs(target_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = os.path.join(target_dir, f"{_safe_label(session_label)}-{stamp}.jsonl")
        shutil.copyfile(manifest_path, target)
        logger.info("Albert manifest archived for user %s: %s", user_id, target)
        return target
    except Exception as exc:
        logger.error("Could not archive the Albert manifest: %s", type(exc).__name__)
        return None


# ----------------------------------------------------------------------
# Celery state reporting
# ----------------------------------------------------------------------
def report_state(task: Any, meta: Dict[str, Any]) -> None:
    """Publish a ``PROGRESS`` state for ``task`` (failures are logged, not raised)."""
    try:
        task.update_state(state="PROGRESS", meta=meta)
    except Exception as exc:
        logger.debug("update_state failed: %s", exc)


def make_progress_reporter(task: Any, status_label: str) -> Callable[[Dict[str, Any]], None]:
    """Build an ``on_progress`` callback publishing parsed events as Celery state.

    The meta keys are those read by ``app.celery_app.get_task_status``
    (``current``, ``total``, ``percent``, ``item``, ``status``). A progress
    event is published only when its level or percentage changes, or when it
    reaches its total, so a script printing one line per chunk does not
    flood the result backend.

    Args:
        task: The bound Celery task.
        status_label: Prefix of the human-readable status (``"<label> x/y"``).

    Returns:
        The callback.
    """
    last: Dict[str, Any] = {}

    def _on_progress(event: Dict[str, Any]) -> None:
        """Publish one parsed ``PROGRESS`` event."""
        if event.get("type") == "init":
            total = int(event.get("total") or 0)
            message = str(event.get("message") or "")
            last.clear()
            report_state(task, {
                "current": 0,
                "total": total,
                "percent": 0,
                "item": message[:100],
                "status": message or f"{status_label} 0/{total}",
            })
            return
        current = int(event.get("current") or 0)
        total = int(event.get("total") or 0)
        percent = int(event.get("percent") or 0)
        key = (event.get("level"), percent)
        if last.get("key") == key and current != total:
            return
        last["key"] = key
        report_state(task, {
            "current": current,
            "total": total,
            "percent": percent,
            "item": str(event.get("message") or "")[:100],
            "status": f"{status_label} {current}/{total}",
        })

    return _on_progress


# ----------------------------------------------------------------------
# Task owner registry (Redis)
# ----------------------------------------------------------------------
_OWNER_STORE: Any = None
_OWNER_STORE_LOCK = threading.Lock()


def owner_key(task_id: str) -> str:
    """Redis key holding the owner of ``task_id``."""
    return f"{OWNER_KEY_PREFIX}{task_id}"


def _valid_task_id(task_id: Any) -> bool:
    """True for a non-empty task id of reasonable length."""
    return isinstance(task_id, str) and 0 < len(task_id) <= _MAX_TASK_ID_LENGTH


def get_owner_store() -> Any:
    """Return the Redis client of the owner registry, or None when unavailable.

    The client targets ``CELERY_BROKER_URL`` (a Redis URL in every supported
    deployment) and is created once per process, lazily.
    """
    global _OWNER_STORE
    if _OWNER_STORE is not None:
        return _OWNER_STORE
    with _OWNER_STORE_LOCK:
        if _OWNER_STORE is None:
            try:
                from app.celery_app import CELERY_BROKER_URL
                if not str(CELERY_BROKER_URL).startswith(_REDIS_SCHEMES):
                    return None
                import redis
                _OWNER_STORE = redis.Redis.from_url(
                    CELERY_BROKER_URL, socket_connect_timeout=2, socket_timeout=2
                )
            except Exception as exc:
                logger.warning("Task owner registry unavailable: %s", type(exc).__name__)
                return None
    return _OWNER_STORE


def record_task_owner(task_id: str, user_id: int) -> bool:
    """Record ``user_id`` as the owner of ``task_id`` for ``OWNER_TTL_SECONDS``.

    Returns:
        True when the owner was stored, False otherwise (no Redis, error).
    """
    if not _valid_task_id(task_id):
        return False
    store = get_owner_store()
    if store is None:
        return False
    try:
        store.set(owner_key(task_id), str(int(user_id)), ex=OWNER_TTL_SECONDS)
        return True
    except Exception as exc:
        logger.warning("Could not record the owner of task %s: %s", task_id, type(exc).__name__)
        return False


def get_task_owner(task_id: str) -> Optional[int]:
    """Return the user id recorded as owner of ``task_id``, or None if unknown."""
    if not _valid_task_id(task_id):
        return None
    store = get_owner_store()
    if store is None:
        return None
    try:
        raw = store.get(owner_key(task_id))
    except Exception as exc:
        logger.warning("Could not read the owner of task %s: %s", task_id, type(exc).__name__)
        return None
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None
