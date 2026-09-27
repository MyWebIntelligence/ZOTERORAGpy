"""
Job Control
===========

Admission of the processing jobs and exclusion per session (audit A12,
2026-09-27), shared by every uvicorn worker and the Celery worker container.

* Exclusion: a job takes the lock of its session and group (``pipeline`` for
  the four stages, the vector upload and the stage-file uploads, ``notes``
  for the Zotero notes, ``clustering`` for the clustering routes). A second
  job of the same group on the same session is refused (409) instead of
  writing the same files concurrently. The canonical session folder is the
  key (``_session_key`` of ``app/routes/processing.py``).
* Admission: a job also takes one slot among ``MAX_ACTIVE_JOBS`` (whole
  server, default 8) and one among ``MAX_ACTIVE_JOBS_PER_USER`` (default 3);
  without a free slot the request is refused (429). ``0`` disables a limit.

Locks are ``fcntl.flock`` locks on small files of ``RAGPY_LOCK_DIR`` (default
``data/locks``, a volume shared by the web and worker containers): they
coordinate processes, are released by the kernel if a process dies, and
never leave anything in the session folders. Without ``fcntl`` (Windows) an
in-process lock is used instead (single process guarantees only).

Rates of provider requests and tokens are bounded elsewhere: the Albert
limiter (``ALBERT_LIMITER_BACKEND=redis`` to share it between processes)
and ``MAX_CONCURRENT_LLM_CALLS`` for the in-process LLM calls.
"""
import hashlib
import logging
import os
import threading
from typing import Dict, List, Optional, Tuple

try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

from app.core.config import RAGPY_DIR

logger = logging.getLogger(__name__)

LOCK_DIR_ENV = "RAGPY_LOCK_DIR"
DEFAULT_LOCK_DIR = os.path.join(RAGPY_DIR, "data", "locks")

GROUP_PIPELINE = "pipeline"
GROUP_NOTES = "notes"
GROUP_CLUSTERING = "clustering"
GROUP_LABELS = {
    GROUP_PIPELINE: "étapes du pipeline",
    GROUP_NOTES: "génération des notes",
    GROUP_CLUSTERING: "clustering",
}

DEFAULT_MAX_ACTIVE_JOBS = 8
DEFAULT_MAX_ACTIVE_JOBS_PER_USER = 3

SESSION_BUSY_MESSAGE = (
    "Un traitement est déjà en cours sur cette session ({label}) : attendez sa fin "
    "ou arrêtez-le avant d'en lancer un autre."
)
TOO_MANY_JOBS_MESSAGE = (
    "Trop de traitements en cours ({scope}) : réessayez quand l'un d'eux sera terminé."
)

# Fallback without fcntl: in-process locks keyed by lock file path.
_LOCAL_LOCKS: Dict[str, threading.Lock] = {}
_LOCAL_LOCKS_GUARD = threading.Lock()


def lock_dir() -> str:
    """Directory of the lock files (``RAGPY_LOCK_DIR`` or ``data/locks``), created if needed."""
    path = os.getenv(LOCK_DIR_ENV) or DEFAULT_LOCK_DIR
    os.makedirs(path, exist_ok=True)
    return path


def _limit(env_name: str, default: int) -> int:
    """Integer limit from ``env_name`` (``0`` or negative: no limit; invalid: ``default``)."""
    raw = os.getenv(env_name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"Invalid {env_name}={raw!r}; using {default}")
        return default


class _FileLock:
    """Exclusive, non-blocking lock on one file (``flock``), or an in-process lock without ``fcntl``."""

    def __init__(self, path: str):
        """Remember the lock file path; nothing is opened yet."""
        self.path = path
        self._handle = None
        self._local: Optional[threading.Lock] = None

    def try_acquire(self) -> bool:
        """Take the lock without waiting; False when another holder has it."""
        if fcntl is None:  # pragma: no cover - Windows
            with _LOCAL_LOCKS_GUARD:
                lock = _LOCAL_LOCKS.setdefault(self.path, threading.Lock())
            if lock.acquire(blocking=False):
                self._local = lock
                return True
            return False
        handle = open(self.path, "a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, PermissionError):
            handle.close()
            return False
        except BaseException:
            handle.close()
            raise
        self._handle = handle
        return True

    def release(self) -> None:
        """Release the lock (idempotent)."""
        if self._local is not None:
            self._local.release()
            self._local = None
        if self._handle is not None:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._handle.close()
                self._handle = None


class JobHold:
    """One holder of a ``JobTicket`` (the request, or a process it started); ``release`` is idempotent."""

    def __init__(self, ticket: "JobTicket"):
        """Attach to ``ticket`` (already counted by it)."""
        self._ticket = ticket
        self._released = False
        self._guard = threading.Lock()

    def release(self) -> None:
        """Drop this hold once; the ticket unlocks when no hold is left."""
        with self._guard:
            if self._released:
                return
            self._released = True
        self._ticket._drop()


class JobTicket:
    """Locks held by one job: its session lock and its admission slots.

    The locks stay taken while any holder keeps them: the request that took
    the ticket (``release``) and every process started for the job
    (``hold()``), so a script that outlives its HTTP stream (client
    disconnect) keeps the session locked until it really ends.
    """

    def __init__(self, locks: List[_FileLock], label: str):
        """Keep the acquired locks; the request is the first holder."""
        self._locks = locks
        self._guard = threading.Lock()
        self._holders = 1
        self._request_hold: Optional[JobHold] = None
        self.label = label

    def hold(self) -> Optional[JobHold]:
        """Add a holder (a started process); None when the ticket is already released."""
        with self._guard:
            if not self._locks:
                return None
            self._holders += 1
        return JobHold(self)

    def release(self) -> None:
        """Release the request's hold (idempotent, thread-safe)."""
        with self._guard:
            if self._request_hold is None:
                self._request_hold = JobHold(self)
            hold = self._request_hold
        hold.release()

    def _drop(self) -> None:
        """Remove one holder; unlock every lock when none is left."""
        with self._guard:
            self._holders -= 1
            if self._holders > 0:
                return
            locks, self._locks = self._locks, []
        for lock in reversed(locks):
            try:
                lock.release()
            except Exception as exc:  # never fail a request on unlock
                logger.warning(f"Job lock release failed for {lock.path}: {exc}")


def _session_lock_path(session_key: str, group: str) -> str:
    """Lock file of ``(session, group)``: hashed name, nothing in the session folder."""
    digest = hashlib.sha256(str(session_key).encode("utf-8")).hexdigest()[:32]
    return os.path.join(lock_dir(), f"session-{digest}-{group}.lock")


def _take_slot(prefix: str, size: int) -> Optional[_FileLock]:
    """First free slot among ``size`` slot files ``<prefix>-<i>.lock``, or None."""
    directory = lock_dir()
    for index in range(size):
        slot = _FileLock(os.path.join(directory, f"{prefix}-{index}.lock"))
        if slot.try_acquire():
            return slot
    return None


def acquire_job(
    session_key: str,
    group: str,
    user_id: Optional[int],
    *,
    admission: bool = True,
) -> Tuple[Optional[JobTicket], Optional[Tuple[int, str]]]:
    """
    Take the session lock of a job and, with ``admission``, its admission slots.

    Args:
        session_key: Canonical session folder.
        group: ``GROUP_PIPELINE``, ``GROUP_NOTES`` or ``GROUP_CLUSTERING``.
        user_id: The submitting user (per-user slots), or None.
        admission: Also take a global and a per-user slot (False for short
            exclusive operations such as a stage-file upload).

    Returns:
        ``(ticket, None)`` when the job may start (release the ticket when it
        ends), else ``(None, (409 | 429, message))``.
    """
    label = GROUP_LABELS.get(group, group)
    session_lock = _FileLock(_session_lock_path(session_key, group))
    if not session_lock.try_acquire():
        logger.info(f"Job refused: session '{session_key}' busy ({group})")
        return None, (409, SESSION_BUSY_MESSAGE.format(label=label))
    locks = [session_lock]
    if admission:
        max_jobs = _limit("MAX_ACTIVE_JOBS", DEFAULT_MAX_ACTIVE_JOBS)
        if max_jobs > 0:
            slot = _take_slot("jobs-global", max_jobs)
            if slot is None:
                JobTicket(locks, label).release()
                logger.warning(f"Job refused: {max_jobs} active jobs on the server")
                return None, (429, TOO_MANY_JOBS_MESSAGE.format(scope=f"{max_jobs} sur le serveur"))
            locks.append(slot)
        per_user = _limit("MAX_ACTIVE_JOBS_PER_USER", DEFAULT_MAX_ACTIVE_JOBS_PER_USER)
        if per_user > 0 and user_id is not None:
            slot = _take_slot(f"jobs-user-{int(user_id)}", per_user)
            if slot is None:
                JobTicket(locks, label).release()
                logger.warning(f"Job refused: {per_user} active jobs for user {user_id}")
                return None, (429, TOO_MANY_JOBS_MESSAGE.format(scope=f"{per_user} pour votre compte"))
            locks.append(slot)
    return JobTicket(locks, label), None


def session_busy(session_key: str, group: str) -> bool:
    """
    Tell whether a job of ``group`` holds the lock of ``session_key`` right now.

    Args:
        session_key: Canonical session folder.
        group: Lock group.

    Returns:
        True when the lock is held by a running job.
    """
    lock = _FileLock(_session_lock_path(session_key, group))
    if lock.try_acquire():
        lock.release()
        return False
    return True
