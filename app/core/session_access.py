"""
Session Access Control
======================

Central authorization of the session folders under ``uploads/`` (audit
A01/A02 of 2026-09-27). Every route that reads, writes, launches, uploads
into or stops a session folder calls ``session_refusal`` with the action it
performs, before touching the folder:

* ``READ``: inspect a session (list its files, read its status);
* ``WRITE``: run a pipeline stage, replace an artifact, change its status;
* ``STOP``: stop the processes of a session.

Rules, evaluated on the folder resolved strictly under ``UPLOAD_DIR`` (400
otherwise), on every spelling of it and on the related folders (a folder
above or below a recorded one belongs to that session):

1. administrators: every action;
2. session of a project (``PipelineSession`` row whose project exists):
   ``READ`` for the owner and every member, ``WRITE`` and ``STOP`` for the
   owner and the collaborators only (a viewer is refused);
3. upload outside any project (``SessionOwner`` row, written by
   ``/upload_zip`` and ``/upload_csv``): its uploader only;
4. folder without any such row (uploaded before this control, or whose
   project was deleted): administrators only; the user uploads the file
   again.

``canonical_session_folder`` gives the single session identifier shared by
the files, the database rows, the process registry and the stage locks: the
folder relative to ``UPLOAD_DIR`` as listed on disk, whatever the spelling
sent by the client (``sess/``, ``./sess``, another case or Unicode form on a
case- or normalisation-insensitive filesystem).

Every function takes the ``upload_dir`` of the calling route module, so a
test that redirects one module's ``UPLOAD_DIR`` keeps working.
"""
import logging
import os
import unicodedata
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.pipeline_session import PipelineSession, SessionOwner
from app.models.project import Project, ProjectRole

logger = logging.getLogger(__name__)


class SessionAction(str, Enum):
    """What a request does with a session folder."""
    READ = "read"
    WRITE = "write"
    STOP = "stop"


# Refusal messages (JSON body ``{"error": <message>}``, or one SSE error event).
SESSION_PATH_INVALID_MESSAGE = "Chemin de session invalide : le dossier doit se trouver sous uploads/."
SESSION_ACCESS_DENIED_MESSAGE = (
    "Accès non autorisé à cette session : elle appartient à un projet auquel vous n'avez pas accès."
)
SESSION_READ_ONLY_MESSAGE = (
    "Accès en lecture seule à cette session : seuls le propriétaire et les collaborateurs "
    "du projet peuvent la modifier ou arrêter ses traitements."
)
SESSION_OWNER_DENIED_MESSAGE = "Accès non autorisé à cette session : elle appartient à un autre utilisateur."
SESSION_UNOWNED_MESSAGE = (
    "Session sans propriétaire enregistré (import antérieur au contrôle d'accès, ou projet supprimé) : "
    "réservée aux administrateurs. Importez de nouveau le fichier pour continuer."
)

_EDIT_ROLES = (ProjectRole.OWNER.value, ProjectRole.COLLABORATOR.value)


def session_folder_under(upload_dir: str, path: str) -> Optional[str]:
    """
    Folder of ``path`` relative to ``upload_dir``, when it lies strictly under it.

    Symbolic links are followed on both sides, so a link that leaves the
    upload directory is outside.

    Args:
        upload_dir: The uploads directory of the calling module.
        path: Folder sent by the client, relative to ``upload_dir`` (an
            absolute path is accepted and checked the same way).

    Returns:
        The folder relative to ``upload_dir`` (e.g. ``"sess"`` for ``"sess/"``
        or ``"./sess"``), or None when it is outside ``upload_dir``, is
        ``upload_dir`` itself or cannot be resolved (embedded NUL byte,
        different drives).
    """
    upload_root = os.path.realpath(upload_dir)
    try:
        target = os.path.realpath(os.path.join(upload_dir, path))
        inside = target != upload_root and os.path.commonpath([upload_root, target]) == upload_root
    except (TypeError, ValueError):  # embedded NUL byte, different drives (Windows), not a string
        return None
    return os.path.relpath(target, upload_root) if inside else None


def listed_name(parent: str, name: str) -> str:
    """
    Name under which the entry ``parent/name`` is listed by its parent directory.

    On a case- or normalisation-insensitive filesystem (macOS APFS, Docker
    Desktop bind mounts of ``uploads/``), ``SESS`` or the NFD form of an
    accented name opens the directory listed as ``sess`` or in NFC, while
    ``os.path.realpath`` keeps the caller's spelling.

    Args:
        parent: Existing directory (a component of a resolved path).
        name: One path component under ``parent``.

    Returns:
        ``name`` when it is listed as such, when ``parent/name`` does not
        exist or when ``parent`` cannot be listed; otherwise the listed name
        of the same file (``os.path.samefile``).
    """
    try:
        with os.scandir(parent) as listing:
            entries = list(listing)
    except OSError:
        return name
    if any(entry.name == name for entry in entries):
        return name
    candidate = os.path.join(parent, name)
    if not os.path.exists(candidate):
        return name
    for entry in entries:
        try:
            if os.path.samefile(entry.path, candidate):
                return entry.name
        except OSError:
            continue
    return name


def _listed_folder(upload_root: str, folder: str) -> str:
    """Spelling of ``folder`` as listed on disk, component by component (``listed_name``)."""
    current = upload_root
    listed_parts = []
    for part in folder.split(os.sep):
        listed = listed_name(current, part)
        listed_parts.append(listed)
        current = os.path.join(current, listed)
    return os.sep.join(listed_parts)


def session_folder_spellings(upload_dir: str, folder: str) -> List[str]:
    """
    Spellings under which the resolved session folder ``folder`` may be recorded.

    ``folder`` itself; its spelling as listed on disk (``listed_name``); and
    the NFC and NFD forms of the latter when they open the same directory
    (normalisation-insensitive filesystems). On a case- and byte-exact
    filesystem (Linux ext4) the result is ``[folder]``.

    Args:
        upload_dir: The uploads directory of the calling module.
        folder: Folder relative to ``upload_dir``, from ``session_folder_under``.

    Returns:
        The distinct spellings, ``folder`` first.
    """
    upload_root = os.path.realpath(upload_dir)
    listed = _listed_folder(upload_root, folder)
    spellings = [folder]
    if listed not in spellings:
        spellings.append(listed)
    target = os.path.join(upload_root, folder)
    for form in ("NFC", "NFD"):
        variant = unicodedata.normalize(form, listed)
        if variant in spellings:
            continue
        try:
            if os.path.samefile(os.path.join(upload_root, variant), target):
                spellings.append(variant)
        except (OSError, ValueError):
            continue
    return spellings


def canonical_session_folder(path: str, upload_dir: str) -> Optional[str]:
    """
    Canonical identifier of a session folder: relative to ``upload_dir``, as listed on disk.

    ``sess``, ``sess/``, ``./sess``, ``other/../sess`` and (on an insensitive
    filesystem) ``SESS`` all give ``sess``. The process registry, the stage
    locks and the ownership rows use this value.

    Args:
        path: Folder sent by the client, relative to ``upload_dir``.
        upload_dir: The uploads directory of the calling module.

    Returns:
        The canonical folder, or None when ``path`` is not strictly under
        ``upload_dir``.
    """
    folder = session_folder_under(upload_dir, path)
    if folder is None:
        return None
    return _listed_folder(os.path.realpath(upload_dir), folder)


def _related_rows(db: Session, model, path: str, spellings: Iterable[str]) -> list:
    """
    Rows of ``model`` (``PipelineSession`` or ``SessionOwner``) recorded on a folder related to the request.

    Related folders: every spelling of the folder, every folder above it
    (the upload directory of a ZIP whose single root folder was recorded
    contains that session) and every folder below it (a subfolder of a
    recorded session belongs to it), plus the raw value sent by the client.
    Paths are compared component by component: ``sess-2`` is unrelated to
    ``sess``.

    Args:
        db: Database session.
        model: ``PipelineSession`` or ``SessionOwner`` (both have ``session_folder``).
        path: Folder as sent by the client.
        spellings: Spellings of the resolved folder (``session_folder_spellings``).

    Returns:
        The distinct related rows.
    """
    spellings = list(spellings)
    same_or_above = {path}
    for spelling in spellings:
        parts = spelling.split(os.sep)
        same_or_above.update(os.sep.join(parts[:depth]) for depth in range(1, len(parts) + 1))
    rows = db.query(model).filter(model.session_folder.in_(sorted(same_or_above))).all()
    prefixes = [spelling + os.sep for spelling in spellings]
    below = db.query(model).filter(or_(*[
        model.session_folder.startswith(prefix, autoescape=True) for prefix in prefixes
    ])).all()
    # LIKE may ignore the case (SQLite): keep the exact component prefixes only.
    below = [row for row in below if any(row.session_folder.startswith(prefix) for prefix in prefixes)]
    unique: Dict[int, object] = {}
    for row in rows + below:
        unique.setdefault(row.id, row)
    return list(unique.values())


def _project_refusal(project: Project, user, action: SessionAction) -> Optional[str]:
    """
    Refusal message of ``action`` on a session of ``project`` for ``user``, or None.

    Args:
        project: The project of a related ``PipelineSession`` row.
        user: The authenticated non-administrator user.
        action: The requested action.

    Returns:
        ``SESSION_ACCESS_DENIED_MESSAGE`` (not a member),
        ``SESSION_READ_ONLY_MESSAGE`` (viewer asking to write or stop), or
        None (allowed).
    """
    role = project.get_user_role(user.id)
    if role is None:
        return SESSION_ACCESS_DENIED_MESSAGE
    if action != SessionAction.READ and role not in _EDIT_ROLES:
        return SESSION_READ_ONLY_MESSAGE
    return None


def session_refusal(
    db: Session,
    path: str,
    user,
    action: SessionAction,
    *,
    upload_dir: str,
) -> Optional[Tuple[int, str]]:
    """
    Check ``action`` by ``user`` on the session folder ``path``, before any work on it.

    Args:
        db: Database session.
        path: Session folder as sent by the client (relative to ``upload_dir``).
        user: The authenticated user.
        action: ``SessionAction.READ``, ``WRITE`` or ``STOP``.
        upload_dir: The uploads directory of the calling module.

    Returns:
        None when the request may go on; ``(400, message)`` when the folder
        does not resolve strictly under ``upload_dir``; ``(403, message)``
        when the rules of the module refuse the action.
    """
    folder = session_folder_under(upload_dir, path)
    user_id = getattr(user, "id", None)
    if folder is None:
        logger.warning(f"Session folder refused for user {user_id}: outside uploads/ ({path!r})")
        return 400, SESSION_PATH_INVALID_MESSAGE
    if getattr(user, "is_admin", False):
        return None
    spellings = session_folder_spellings(upload_dir, folder)
    granted = False
    projects: Dict[int, Optional[Project]] = {}
    for row in _related_rows(db, PipelineSession, path, spellings):
        if row.project_id not in projects:
            projects[row.project_id] = db.query(Project).filter(Project.id == row.project_id).first()
        project = projects[row.project_id]
        if project is None:
            continue  # row of a deleted project: grants nothing
        message = _project_refusal(project, user, action)
        if message is not None:
            logger.warning(
                f"Session folder refused for user {user_id}: {action.value} on a session of project "
                f"{project.id} ({path!r})"
            )
            return 403, message
        granted = True
    for row in _related_rows(db, SessionOwner, path, spellings):
        if row.user_id != user_id:
            logger.warning(f"Session folder refused for user {user_id}: owned by another user ({path!r})")
            return 403, SESSION_OWNER_DENIED_MESSAGE
        granted = True
    if not granted:
        logger.warning(f"Session folder refused for user {user_id}: no recorded owner ({path!r})")
        return 403, SESSION_UNOWNED_MESSAGE
    return None


def record_session_owner(
    db: Session,
    session_folder: str,
    user_id: int,
    *,
    source_type: Optional[str] = None,
    original_filename: Optional[str] = None,
) -> SessionOwner:
    """
    Record ``user_id`` as the owner of an upload made outside any project.

    Args:
        db: Database session.
        session_folder: Canonical folder of the upload, relative to uploads/.
        user_id: The uploader.
        source_type: ``"zip"`` or ``"csv"``.
        original_filename: Name of the uploaded file, kept as metadata only
            (truncated to 255 characters).

    Returns:
        The committed ``SessionOwner`` row.

    Raises:
        Exception: Any database error, after a rollback; the caller removes
            the upload (an upload is never left without an owner).
    """
    row = SessionOwner(
        session_folder=session_folder,
        user_id=user_id,
        source_type=source_type,
        original_filename=(original_filename or "")[:255] or None,
    )
    try:
        db.add(row)
        db.commit()
        db.refresh(row)
    except Exception:
        db.rollback()
        raise
    return row
