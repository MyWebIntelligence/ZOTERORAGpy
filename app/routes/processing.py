"""
Processing Routes
=================

This module handles the execution of background processing tasks for the RAG pipeline.
It triggers scripts for data ingestion, chunking, and embedding generation, often
using subprocesses to run standalone scripts.

Key Features:
- Dataframe Processing: Converts raw data (JSON/CSV) into standardized formats.
- Chunking: Splits text into manageable chunks for embedding.
- Embedding Generation: Generates dense and sparse embeddings.
- Vector DB Upload: Uploads processed embeddings to vector databases.
- Process Management: Supports stopping running scripts and tracking progress via SSE.

Security:
- All processing endpoints require authentication.
- Credentials are retrieved from user's personal settings (encrypted in DB).
- Non-admin users cannot access .env credentials.
- Subprocess environments are built with user-specific credentials.

Albert (DINUM, opt-in, ``ALBERT_ENABLED=1``):
- The chat provider of a model is decided by ``resolve_llm_provider`` (the
  ``albert/`` prefix first, then the historical ``provider/model`` rule); an
  ``albert/`` model while Albert is disabled gives a 400 (or one SSE error
  event) and no subprocess is started.
- The dense phase takes an ``embedding_provider`` form field (ignored while
  Albert is disabled, except ``albert`` which gives a 400).
- ``/upload_db`` gains the ``albert`` target (collections, retention
  acknowledgement, audit entry, manifest archived out of ``uploads/``); its
  extra fields are read from the raw form so the OpenAPI schema never
  mentions them.
- ``_subprocess_timeout`` extends the subprocess timeout to
  ``ALBERT_SUBPROCESS_TIMEOUT`` only when the request selects Albert.
- While Albert is disabled every response, argv, environment and timeout is
  the historical one.

Session folders (decision of 2026-09-27, lot 9, extended by the audit A02):
- Every pipeline route that works on a session folder checks it first,
  before reading, writing or launching anything
  (``_pipeline_session_refusal``, rules of ``app.core.session_access``): a
  folder that does not resolve strictly under ``UPLOAD_DIR`` gives a 400;
  a session the non-admin user may not modify gives a 403 (a session of a
  project the user does not belong to or may only read, an upload of
  another user, a folder without any owner record). A single SSE error
  event carries the same status on the SSE routes, a JSON body on
  ``/cluster_documents_sse`` whose page handler reads JSON on a refusal
  status. The folder is matched whatever its spelling (``sess/``,
  ``./sess``, another case or Unicode form on a case- or
  normalisation-insensitive filesystem), and so are the folders under a
  recorded session and the folders that contain one.
- Processes are registered, and stopped, under the canonical folder
  (``_session_key``): ``/stop_all_scripts`` checks the stop right on the
  same rules and reaches the processes whatever the spelling it receives.
- Web usage ledger: when a request resolves its chat model to Albert, one
  ``UsageLedger`` is passed to the note / citation helpers and appended to
  ``<session>/albert_usage.jsonl`` at the end of the job, only when Albert
  was called (``ALBERT_USAGE_LOG=0`` keeps the summary log line only).
"""
import os
import re
import shutil
import subprocess
import logging
import json
import pandas as pd
import asyncio
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from fastapi import APIRouter, Form, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from sqlalchemy.orm import Session

from app.core.config import APP_DIR, RAGPY_DIR, UPLOAD_DIR
from app.services import job_control
from app.services.process_manager import process_manager
from app.middleware.auth import get_current_active_user
from app.models.user import User
from app.core.albert_policy import (
    apply_chat_policy,
    apply_embedding_policy,
    check_configuration as check_policy_configuration,
    is_policy_error,
    policy_error_body,
    restrict_subprocess_env,
    strict_policy,
)
from app.core.credentials import (
    build_subprocess_env,
    get_credential_or_env,
    get_credential_error_message,
    albert_enabled,
    CredentialMissingError
)
from app.core import session_access
from app.core.session_access import SessionAction
from app.database.session import get_db
from app.models.project import Project

try:
    from scripts.rad_albert.config import AlbertConfig, FIELD_ENV_NAMES
    from scripts.rad_albert.errors import AlbertDisabledError
    from scripts.rad_providers import PROVIDER_ALBERT, PROVIDER_OPENROUTER, is_compat, resolve_llm_provider
    from scripts.rad_settings.chat import routed_model_for, server_values
    from scripts.rad_settings.models import ServiceConfigError, resolve_service
except ImportError:  # scripts/ itself on sys.path (CLI import pattern)
    from rad_albert.config import AlbertConfig, FIELD_ENV_NAMES
    from rad_albert.errors import AlbertDisabledError
    from rad_providers import PROVIDER_ALBERT, PROVIDER_OPENROUTER, is_compat, resolve_llm_provider
    from rad_settings.chat import routed_model_for, server_values
    from rad_settings.models import ServiceConfigError, resolve_service

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Albert (DINUM) helpers — inert while Albert is disabled
# =============================================================================
ALBERT_DB_CHOICE = "albert"
ALBERT_CREDENTIAL_KEY = "albert_api_key"
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"
EMBEDDING_PROVIDERS = ("openai", "albert")
CREDENTIALS_CONFIGURE_URL = "/settings/credentials"
# Copies of the upload manifests, out of uploads/: <dir>/<user_id>/<session>-<ts>.jsonl
ALBERT_MANIFEST_DIR = os.path.join(RAGPY_DIR, "data", "albert_manifests")
ALBERT_UPLOAD_AUDIT_ACTION = "ALBERT_COLLECTION_UPLOAD"
ALBERT_COLLECTION_RESOURCE = "albert_collection"
# Raw form fields of the albert target of /upload_db (never declared as Form
# parameters, so the OpenAPI schema stays free of them).
ALBERT_FORM_COLLECTION_ID = "albert_collection_id"
ALBERT_FORM_COLLECTION_NAME = "albert_collection_name"
ALBERT_FORM_CREATE_COLLECTION = "albert_create_collection"
ALBERT_FORM_GDPR_ACK = "albert_gdpr_ack"
_ALBERT_NAME_MAX_CHARS = 255
_TRUE_FORM_VALUES = ("1", "true", "yes", "on")
_SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SAFE_LABEL_MAX_CHARS = 120
# Anchored lines of the pipeline scripts' output: the Result block header and
# the "Albert abort: kind=<kind> reason=<reason>[ credential_required=<key>]" line.
_RESULT_BLOCK_RE = re.compile(r"^=== Result ===\s*$", re.MULTILINE)
_ALBERT_ABORT_RE = re.compile(
    r"^Albert abort: kind=(\w+) reason=(\w+)(?: credential_required=(\w+))?\s*$", re.MULTILINE
)
# Refusals of the session check of the pipeline routes (``_pipeline_session_refusal``):
# JSON body ``{"error": <message>}``, or one SSE event ``{"type": "error", "message": <message>}``.
# The messages and the rules live in ``app.core.session_access`` (audit A02).
SESSION_PATH_INVALID_MESSAGE = session_access.SESSION_PATH_INVALID_MESSAGE
SESSION_ACCESS_DENIED_MESSAGE = session_access.SESSION_ACCESS_DENIED_MESSAGE
ALBERT_GDPR_MESSAGE = (
    "Confirmation requise avant l'envoi vers Albert : les textes des chunks sont "
    "conservés par la DINUM (sous-traitant, art. 28 RGPD) dans une collection privée "
    "jusqu'à leur suppression ; vérifier les droits d'auteur et l'absence de données "
    "personnelles non nécessaires."
)


def _albert_setting(field_name: str) -> Any:
    """
    Read one ``AlbertConfig`` field from its environment variable only.

    The configuration is built from a mapping holding that single variable,
    so the parser and the default are those of ``AlbertConfig.from_env`` and
    nothing else is read or validated (an invalid ``ALBERT_BASE_URL`` never
    raises here).

    Args:
        field_name: Name of an ``AlbertConfig`` field (``FIELD_ENV_NAMES`` key).

    Returns:
        The parsed value of the field.
    """
    env_name = FIELD_ENV_NAMES[field_name]
    raw = os.environ.get(env_name)
    mapping = {} if raw is None else {env_name: raw}
    return getattr(AlbertConfig.from_env(mapping), field_name)


def _subprocess_timeout(default: int, albert_selected: bool) -> int:
    """
    Timeout of a pipeline subprocess (decision 19 of the Albert sprint).

    Args:
        default: The historical literal timeout of the call site (1800 or 3600).
        albert_selected: True when the request selects Albert (``albert/``
            model, ``embedding_provider=albert``, ``db_choice=albert`` or the
            active Albert OCR link).

    Returns:
        ``ALBERT_SUBPROCESS_TIMEOUT`` when Albert is selected, else ``default``
        unchanged.
    """
    if not albert_selected:
        return default
    return _albert_setting("subprocess_timeout")


def _resolve_chat_model(model: Optional[str]):
    """
    Resolve the chat provider of a model with the single resolver.

    Any model without the ``albert/`` prefix is routed exactly as before
    (``provider/model`` → OpenRouter, else OpenAI).

    Args:
        model: Model name as entered.

    Returns:
        ``ProviderResolution(provider, wire_model, credential_key)``.

    Raises:
        AlbertDisabledError: ``albert/…`` while Albert is disabled.
        ValueError: ``albert/`` without a model identifier.
    """
    return resolve_llm_provider(model, albert_enabled=albert_enabled())


def _explicit_ocr_target(user_values=None):
    """Couple OCR déclaré, ou ``None`` (chaîne historique) : ``app/services/ocr_target.py``."""
    from app.services.ocr_target import explicit_ocr_target

    return explicit_ocr_target(user_values)


def _explicit_embedding_target(server=None, model=None):
    """Couple des embeddings déclaré, ou ``None`` (règle historique) : ``app/services/embedding_target.py``."""
    from app.services.embedding_target import explicit_embedding_target

    return explicit_embedding_target(server, model)


def _explicit_embedding_env(current_user: User, choice) -> Dict[str, str]:
    """Environnement de la phase dense pour un couple déclaré : ``app/services/embedding_target.py``."""
    from app.services.embedding_target import explicit_embedding_env

    return explicit_embedding_env(current_user, choice)


def _explicit_ocr_uses_albert(choice) -> bool:
    """Moteur OCR principal ou repli sur Albert : ``app/services/ocr_target.py``."""
    from app.services.ocr_target import explicit_ocr_uses_albert

    return explicit_ocr_uses_albert(choice)


def _user_values(db: Session, user: User) -> Dict[str, str]:
    """Choix personnels de serveur et de modèle de ``user`` (lot L9)."""
    from app.services.user_settings import get_user_settings

    return get_user_settings(db, user)


def _explicit_ocr_env(current_user: User, choice) -> Dict[str, str]:
    """Environnement du script d'OCR explicite : ``app/services/ocr_target.py``."""
    from app.services.ocr_target import explicit_ocr_env

    return explicit_ocr_env(current_user, choice)


def _ocr_albert_active(user: User) -> bool:
    """
    Tell whether the Albert OCR link runs for ``user`` (Albert ON, OCR on, key).

    Evaluated lazily: while Albert is disabled neither the OCR switch nor the
    user's credentials are read.

    Args:
        user: The authenticated user.

    Returns:
        True when ``ALBERT_ENABLED``, ``OCR_ENABLE_ALBERT`` and an Albert key
        available to the user (personal, or ``.env`` for admins) are all set.
    """
    if not albert_enabled():
        return False
    if not _albert_setting("ocr_enabled"):
        return False
    return bool(get_credential_or_env(user, ALBERT_CREDENTIAL_KEY))


def _resolve_embedding_provider(value: Optional[str]) -> Optional[str]:
    """
    Resolve the dense embedding provider of a request.

    Albert disabled: the field is ignored (None: historical behaviour, the
    environment is left untouched), except ``albert`` which is refused.
    Albert enabled: the form value, else the server ``EMBEDDING_PROVIDER``,
    else ``openai``; the result must be ``openai`` or ``albert``.

    Args:
        value: The ``embedding_provider`` form field (None or empty if absent).

    Returns:
        ``'openai'`` or ``'albert'`` while Albert is enabled, else None.

    Raises:
        AlbertDisabledError: ``albert`` requested while Albert is disabled.
        ValueError: Unknown provider while Albert is enabled.
    """
    requested = (value or "").strip()
    if not albert_enabled():
        check_policy_configuration()
        if requested.lower() == ALBERT_DB_CHOICE:
            raise AlbertDisabledError(
                "embedding_provider=albert exige Albert, désactivé sur ce serveur (ALBERT_ENABLED=1 requis).",
                model="embedding_provider=albert",
            )
        return None
    if strict_policy():
        # albert_only: bge-m3 only (empty -> albert, whatever EMBEDDING_PROVIDER says).
        return apply_embedding_policy(requested)
    raw = requested or (os.environ.get(EMBEDDING_PROVIDER_ENV) or "").strip()
    name = raw.lower() or EMBEDDING_PROVIDERS[0]
    if name not in EMBEDDING_PROVIDERS:
        raise ValueError(
            f"embedding_provider={raw!r} inconnu : valeurs admises « openai » ou « albert »."
        )
    return name


def _embedding_required_keys(provider: Optional[str]) -> list:
    """
    Credential keys of the dense phase for a resolved embedding provider.

    Args:
        provider: Result of ``_resolve_embedding_provider``.

    Returns:
        ``['albert_api_key']`` for Albert, else ``['openai_api_key']``.
    """
    if provider == ALBERT_DB_CHOICE:
        return [ALBERT_CREDENTIAL_KEY]
    return ["openai_api_key"]


def _sse_event(payload: Dict[str, Any]) -> str:
    """
    Format one Server-Sent Event carrying ``payload`` as JSON.

    Args:
        payload: The event body.

    Returns:
        The ``data: {...}`` block, blank line included.
    """
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_error_response(payload: Dict[str, Any], status_code: int = 200) -> StreamingResponse:
    """
    Build an SSE response made of a single event (no subprocess is started).

    The payload is serialised when the response is built, so the stream never
    depends on a variable of an enclosing ``except`` block.

    Args:
        payload: The event body (``type``, ``message`` and optional fields).
        status_code: HTTP status of the response (200, the historical SSE
            error style, unless a refusal status is given).

    Returns:
        The ``text/event-stream`` response.
    """
    event = _sse_event(payload)

    async def _single_event():
        """Yield the prepared event once."""
        yield event

    return StreamingResponse(_single_event(), media_type="text/event-stream", status_code=status_code)


def _credential_error_payload(error: CredentialMissingError) -> Dict[str, str]:
    """
    SSE error payload of a missing credential, equal to the JSON routes' 403 fields.

    Args:
        error: The error raised by ``build_subprocess_env``.

    Returns:
        ``{type: error, message, credential_required}``.
    """
    return {"type": "error", "message": str(error), "credential_required": error.credential_key}


def form_flag(value: Any) -> bool:
    """
    Interpret a raw form value as a boolean flag (``true``, ``1``, ``yes``, ``on``).

    Args:
        value: The raw form value (str, None or an upload).

    Returns:
        True for an affirmative string, False otherwise.
    """
    return isinstance(value, str) and value.strip().lower() in _TRUE_FORM_VALUES


def _safe_label(text: Any) -> str:
    """
    Turn a session path into a file name fragment (no separator, no leading dot).

    Args:
        text: The session folder (relative path).

    Returns:
        A non-empty label made of ``[A-Za-z0-9._-]`` characters.
    """
    label = _SAFE_LABEL_RE.sub("_", str(text or "")).strip("._")
    return label[:_SAFE_LABEL_MAX_CHARS] or "session"


def _archive_albert_manifest(manifest_path: Optional[str], user_id: Any, session: str) -> Optional[str]:
    """
    Copy an Albert upload manifest out of ``uploads/`` (even a partial one).

    The copy goes to ``ALBERT_MANIFEST_DIR/<user_id>/<session>-<ts>.jsonl``.
    Never raises: a failure is logged and the upload response is unchanged.

    Args:
        manifest_path: Manifest written next to the upload input.
        user_id: Id of the uploading user.
        session: Session folder of the upload.

    Returns:
        The path of the copy, or None when there was nothing to copy.
    """
    try:
        if not manifest_path or not os.path.isfile(manifest_path):
            return None
        target_dir = os.path.join(ALBERT_MANIFEST_DIR, str(int(user_id)))
        os.makedirs(target_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = os.path.join(target_dir, f"{_safe_label(session)}-{stamp}.jsonl")
        shutil.copyfile(manifest_path, target)
        logger.info(f"Albert manifest archived for user {user_id}: {target}")
        return target
    except Exception as exc:
        logger.error(f"Could not archive the Albert manifest of session '{session}': {type(exc).__name__}: {exc}")
        return None


def albert_collection_target(form: Any) -> Tuple[Optional[Dict[str, Any]], Optional[JSONResponse]]:
    """
    Validate the collection fields of an albert upload (raw form values).

    Args:
        form: The request form (``albert_collection_id``,
            ``albert_collection_name``, ``albert_create_collection``).

    Returns:
        ``(target, None)`` with ``collection_id`` (int or None),
        ``collection_name`` (str or None) and ``create_collection`` (bool),
        or ``(None, response)`` with a 400 response.
    """
    raw_id = form.get(ALBERT_FORM_COLLECTION_ID)
    raw_name = form.get(ALBERT_FORM_COLLECTION_NAME)
    raw_id = raw_id.strip() if isinstance(raw_id, str) else ""
    name = raw_name.strip() if isinstance(raw_name, str) else ""
    collection_id = None
    if raw_id:
        if not raw_id.isdigit() or int(raw_id) <= 0:
            return None, JSONResponse(status_code=400, content={
                "error": "Identifiant de collection Albert invalide (entier positif attendu)."
            })
        collection_id = int(raw_id)
    if name and (len(name) > _ALBERT_NAME_MAX_CHARS or any(ord(ch) < 32 or ord(ch) == 127 for ch in name)):
        return None, JSONResponse(status_code=400, content={
            "error": "Nom de collection Albert invalide (255 caractères au plus, sans caractère de contrôle)."
        })
    if collection_id is None and not name:
        return None, JSONResponse(status_code=400, content={
            "error": "Collection Albert requise : identifiant ou nom exact."
        })
    return {
        "collection_id": collection_id,
        "collection_name": name or None,
        "create_collection": form_flag(form.get(ALBERT_FORM_CREATE_COLLECTION)),
    }, None


def _albert_vectordb_flags(target: Dict[str, Any]) -> list:
    """
    CLI flags of ``rad_vectordb.py --db albert`` for a validated target.

    The name is passed as ``--albert-collection-name=<name>`` so that a name
    starting with ``-`` is never read as an option.

    Args:
        target: Result of ``albert_collection_target``.

    Returns:
        The flags, ``--albert-ack-retention`` last.
    """
    flags = []
    if target.get("collection_id") is not None:
        flags.extend(["--albert-collection-id", str(target["collection_id"])])
    if target.get("collection_name"):
        flags.append(f"--albert-collection-name={target['collection_name']}")
    if target.get("create_collection"):
        flags.append("--albert-create-collection")
    flags.append("--albert-ack-retention")
    return flags


def _session_folder_under_uploads(absolute_path: str) -> Optional[str]:
    """
    Normalised session folder of ``absolute_path``, when it lies strictly under ``UPLOAD_DIR``.

    Thin wrapper of ``app.core.session_access.session_folder_under`` bound to
    this module's ``UPLOAD_DIR``.

    Args:
        absolute_path: ``UPLOAD_DIR`` joined with the folder sent by the client.

    Returns:
        The folder relative to ``UPLOAD_DIR``, or None when it is outside.
    """
    return session_access.session_folder_under(UPLOAD_DIR, absolute_path)


def _session_folder_spellings(folder: str) -> List[str]:
    """
    Spellings under which the resolved session folder ``folder`` may be recorded.

    Thin wrapper of ``app.core.session_access.session_folder_spellings`` bound
    to this module's ``UPLOAD_DIR``.

    Args:
        folder: Folder relative to ``UPLOAD_DIR``, from ``_session_folder_under_uploads``.

    Returns:
        The distinct spellings, ``folder`` first.
    """
    return session_access.session_folder_spellings(UPLOAD_DIR, folder)


def _session_key(path: str) -> str:
    """
    Canonical identifier of the session folder ``path`` (process registry, stage locks).

    ``sess``, ``sess/`` and ``./sess`` give the same key, so a stop request
    reaches the processes whatever the spelling used to launch them.

    Args:
        path: Session folder as sent by the client (already checked).

    Returns:
        The folder relative to ``UPLOAD_DIR`` as listed on disk, or ``path``
        itself when it cannot be resolved.
    """
    return session_access.canonical_session_folder(path, UPLOAD_DIR) or path


def _session_refusal_for(
    db: Session,
    path: str,
    absolute_path: str,
    user: User,
    action: SessionAction = SessionAction.WRITE,
    *,
    upload_dir: Optional[str] = None,
) -> Optional[Tuple[int, str]]:
    """
    Check a session folder before any work on it (shared by the pipeline, upload and Albert routes).

    Delegates to ``app.core.session_access.session_refusal`` (audit A02):
    administrators always pass; a session of a project needs the membership
    (``READ``) or the edit right (``WRITE``, ``STOP``: owner or
    collaborator, never a viewer); an upload made outside any project is
    reserved to its uploader; a folder without any owner record is reserved
    to the administrators.

    Args:
        db: Database session.
        path: Session folder as sent by the client (relative to uploads/).
        absolute_path: ``UPLOAD_DIR`` joined with ``path`` (the folder the
            route works on; checked for confinement).
        user: The authenticated user.
        action: What the route does with the folder (``WRITE`` by default:
            every pipeline route writes into the session or pushes its data).
        upload_dir: Uploads directory of the calling module (this module's
            ``UPLOAD_DIR`` by default).

    Returns:
        None when the request may go on; ``(400, message)`` when the folder
        does not resolve strictly under the uploads directory; ``(403,
        message)`` when the rules refuse the action.
    """
    root = UPLOAD_DIR if upload_dir is None else upload_dir
    if session_access.session_folder_under(root, absolute_path) is None:
        logger.warning(f"Session folder refused for user {user.id}: outside uploads/ ('{path}')")
        return 400, SESSION_PATH_INVALID_MESSAGE
    return session_access.session_refusal(db, path, user, action, upload_dir=root)


def _albert_session_refusal(db: Session, path: str, absolute_path: str, user: User) -> Optional[JSONResponse]:
    """
    Refuse a session whose texts may not be sent to Albert by ``user``.

    Defence in depth for the Albert branch of ``/upload_db``: the route runs
    ``_pipeline_session_refusal`` on the same folder first, so this check
    refuses nothing there today; it keeps the branch from ever running
    unchecked (another caller, or a symbolic link swapped between the two
    checks). Same lookup (``_session_refusal_for``) and same JSON bodies as
    the pipeline routes.

    Args:
        db: Database session.
        path: Session folder as sent by the client.
        absolute_path: ``UPLOAD_DIR`` joined with ``path`` (absolute).
        user: The authenticated user.

    Returns:
        A 400 response when the folder is outside ``UPLOAD_DIR`` (or is
        ``UPLOAD_DIR`` itself), a 403 response for a session the user may
        not modify, else None.
    """
    refusal = _session_refusal_for(db, path, absolute_path, user)
    return _session_refusal_json(refusal) if refusal is not None else None


def _pipeline_session_refusal(
    db: Session, path: str, user: User, action: SessionAction = SessionAction.WRITE
) -> Optional[Tuple[int, str]]:
    """
    Check the session folder of a pipeline request, before any work on it.

    Decision of 2026-09-27 (lot 9), extended by the audit A02 of the same
    day: the pipeline routes write into the session (outputs, logs, state
    files) or push its data, so they need the write right (owner or
    collaborator of the project, uploader of a session outside any project,
    administrator). Another spelling of a folder (``sess/``, ``./sess``,
    ``other/../sess``, another case or Unicode form on an insensitive
    filesystem), a folder under it and a folder containing it get the same
    answer.

    Args:
        db: Database session.
        path: Session folder as sent by the client (relative to uploads/).
        user: The authenticated user.
        action: ``SessionAction.WRITE`` (default) or another action.

    Returns:
        None when the request may go on; ``(400, message)`` when the folder
        does not resolve strictly under ``UPLOAD_DIR``; ``(403, message)``
        when the rules refuse the action.
    """
    return _session_refusal_for(db, path, os.path.join(UPLOAD_DIR, path), user, action)


def _acquire_session_job(path: str, group: str, user: User):
    """
    Take the job ticket of a processing request on the session ``path`` (audit A12).

    Exclusion per session and group (``job_control``) and admission slots
    (server, user): a second job of the same group on the same session gives
    ``(409, message)``, no free slot ``(429, message)``.

    Args:
        path: Session folder as sent by the client (already checked).
        group: ``job_control.GROUP_PIPELINE``, ``GROUP_NOTES`` or ``GROUP_CLUSTERING``.
        user: The authenticated user.

    Returns:
        ``(ticket, None)`` or ``(None, (status, message))``.
    """
    return job_control.acquire_job(_session_key(path), group, getattr(user, "id", None))


def _release_after_stream(response: Any, ticket: Any) -> Any:
    """
    Hold ``ticket`` until a streamed response ends; release it now otherwise.

    The stream is wrapped so that its end, an error or a client disconnect
    releases the ticket; a background task releases it too when the stream
    never starts (idempotent release).

    Args:
        response: Response returned by the route body.
        ticket: ``job_control.JobTicket`` of the request.

    Returns:
        The same response.
    """
    if not isinstance(response, StreamingResponse):
        ticket.release()
        return response
    body = response.body_iterator

    async def _guarded_body():
        """Stream the original body, then release the ticket."""
        try:
            async for chunk in body:
                yield chunk
        finally:
            ticket.release()

    response.body_iterator = _guarded_body()
    previous = response.background

    async def _release_then_previous():
        """Release the ticket, then run the response's own background task."""
        ticket.release()
        if previous is not None:
            await previous()

    response.background = BackgroundTask(_release_then_previous)
    return response


def _json_list_count(path: str) -> int:
    """
    Number of items of a JSON list file (0 when the file holds another type).

    Called through ``asyncio.to_thread`` by the routes (audit A11): an
    embeddings file can weigh hundreds of MB, never parsed on the event loop.

    Args:
        path: JSON file.

    Returns:
        The list length, or 0.

    A list is counted element by element (``app/utils/json_stream.py``) : a
    multi-GB embeddings file is never loaded whole (the web process was
    killed by the container memory limit).

    Raises:
        OSError, ValueError: Unreadable file or invalid JSON (callers keep
            their historical handling).
    """
    from app.utils.json_stream import count_json_array

    with open(path, 'r', encoding='utf-8') as f:
        head = f.read(64).lstrip()
    if head.startswith('['):
        return count_json_array(path)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return len(data) if isinstance(data, list) else 0


def _session_refusal_json(refusal: Tuple[int, str]) -> JSONResponse:
    """
    JSON response of a session refusal (``{"error": <message>}``, the routes' error style).

    Args:
        refusal: ``(status, message)`` from ``_pipeline_session_refusal``.

    Returns:
        The 400 or 403 JSON response.
    """
    status, message = refusal
    return JSONResponse(status_code=status, content={"error": message})


def _session_refusal_sse(refusal: Tuple[int, str]) -> StreamingResponse:
    """
    SSE response of a session refusal: one error event, with the refusal status.

    Args:
        refusal: ``(status, message)`` from ``_pipeline_session_refusal``.

    Returns:
        The 400 or 403 ``text/event-stream`` response carrying
        ``{"type": "error", "message": <message>}``.
    """
    status, message = refusal
    return _sse_error_response({"type": "error", "message": message}, status_code=status)


def new_albert_usage_ledger() -> Any:
    """
    Create the usage ledger of one web request that selected Albert.

    Returns:
        A new ``scripts.rad_albert.usage.UsageLedger`` (imported here, so a
        request that does not select Albert never loads it).
    """
    try:
        from scripts.rad_albert.usage import UsageLedger
    except ImportError:  # scripts/ itself on sys.path (CLI import pattern)
        from rad_albert.usage import UsageLedger
    return UsageLedger()


def flush_albert_usage_ledger(ledger: Any, session_dir: str, context: str) -> Optional[str]:
    """
    Append the Albert usage of a web job to ``<session_dir>/albert_usage.jsonl``.

    Same contract as the pipeline scripts: nothing at all unless the ledger
    recorded an Albert call; then the records not yet written are appended
    (unless ``ALBERT_USAGE_LOG=0``) and the ledger's summary line is logged.
    Never raises: a write error is logged and the job's outcome is unchanged.

    Args:
        ledger: The request's ``UsageLedger``, or None (Albert not selected).
        session_dir: Absolute session folder.
        context: Short label of the job for the log line (``notes``, ``citations``…).

    Returns:
        The path written, or None when nothing was written.
    """
    if ledger is None or not ledger.called:
        return None
    written = None
    try:
        from scripts.rad_albert.usage import USAGE_FILENAME
    except ImportError:  # scripts/ itself on sys.path (CLI import pattern)
        from rad_albert.usage import USAGE_FILENAME
    try:
        if _albert_setting("usage_log"):
            written = ledger.write_jsonl(os.path.join(session_dir, USAGE_FILENAME))
    except Exception as exc:
        logger.warning(f"Albert usage ledger of {context} not written: {type(exc).__name__}: {exc}")
    logger.info(f"Albert usage ({context}): {ledger.summary_line()}")
    return written


def _albert_upload_credential_required(returncode: int, stdout: str) -> Optional[str]:
    """
    Credential named by a failed ``rad_vectordb.py --db albert`` run, or None.

    The script exits 2 for an Albert account or quota error (decision 22),
    but also for an invalid Albert configuration, Albert disabled in the
    subprocess environment or a usage error: only the account case sends
    the user to the key settings.

    Rule (exit code 2 only):

    * an ``Albert abort: kind=<kind> reason=<reason>`` line (anchored, the
      last one wins): the key only for ``kind=account`` (its
      ``credential_required`` value, ``albert_api_key`` by default);
    * no such line: the key only when the ``=== Result ===`` block was
      printed, since the script's single exit 2 after that block is the
      account stop (every other exit 2 happens before it).

    Args:
        returncode: Exit code of the script.
        stdout: Standard output of the script.

    Returns:
        The credential key to report, or None.
    """
    if returncode != 2:
        return None
    text = stdout or ""
    aborts = list(_ALBERT_ABORT_RE.finditer(text))
    if aborts:
        kind, _reason, credential = aborts[-1].groups()
        return (credential or ALBERT_CREDENTIAL_KEY) if kind == "account" else None
    return ALBERT_CREDENTIAL_KEY if _RESULT_BLOCK_RE.search(text) else None


async def run_tracked_subprocess(
    cmd: list,
    session_folder: str,
    timeout: int = 1800,
    env: Optional[Dict[str, str]] = None
) -> subprocess.CompletedProcess:
    """
    Lance un subprocess ASYNC avec tracking PID pour permettre l'arrêt par session.

    IMPORTANT: Cette fonction est async pour ne pas bloquer le serveur,
    permettant ainsi de traiter les requêtes de stop en parallèle.

    Cycle de vie (audit A08) : le script tourne dans son propre groupe de
    processus ; un dépassement du délai arrête tout le groupe (SIGTERM puis
    SIGKILL), et toute annulation de la requête (arrêt du serveur, tâche
    annulée) l'arrête aussi, sans attendre (``stop_process_group_later``) :
    aucun script ne survit sans PID enregistré.

    Args:
        cmd: Commande à exécuter (liste d'arguments).
        session_folder: Identifiant de la session pour le tracking.
        timeout: Timeout en secondes (défaut: 30 min).
        env: Dictionnaire d'environnement pour le subprocess. Si None, utilise
             l'environnement actuel. Utilisez build_subprocess_env() pour créer
             un environnement sécurisé avec les credentials utilisateur.

    Returns:
        subprocess.CompletedProcess avec stdout, stderr, returncode.

    Raises:
        subprocess.TimeoutExpired: Délai dépassé (groupe de processus arrêté).
    """
    from app.utils.sse_helpers import stop_process_group_later, terminate_process_group

    # Utiliser asyncio.create_subprocess_exec pour ne pas bloquer
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,  # never inherit the server's stdin
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,  # Pass custom environment if provided
        start_new_session=True,  # own process group: a stop reaches the children
    )

    # Enregistrer le PID pour permettre l'arrêt
    process_manager.register(session_folder, process.pid)
    logger.info(f"Started async process PID {process.pid} for session '{session_folder}'")

    handed_over = False
    try:
        # Attendre avec timeout sans bloquer le serveur
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout
        )
        stdout = stdout_bytes.decode('utf-8', errors='replace') if stdout_bytes else ''
        stderr = stderr_bytes.decode('utf-8', errors='replace') if stderr_bytes else ''

        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr
        )
    except asyncio.TimeoutError:
        await terminate_process_group(process)
        raise subprocess.TimeoutExpired(cmd, timeout)
    finally:
        if process.returncode is None:
            # Cancelled while the script runs: stop its group without awaiting;
            # the reaper unregisters the PID once the group is gone.
            stop_process_group_later(process, session_folder)
            handed_over = True
        if not handed_over:
            # Toujours désenregistrer le PID à la fin
            process_manager.unregister(session_folder, process.pid)
        logger.info(f"Process PID {process.pid} finished for session '{session_folder}'")


@router.post("/stop_all_scripts")
async def stop_all_scripts(
    session: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Arrête les processus de traitement d'une session spécifique.

    IMPORTANT: Le paramètre session est OBLIGATOIRE pour garantir l'isolation
    entre utilisateurs. Chaque utilisateur ne peut arrêter que les processus
    de sa propre session.

    Args:
        session: Identifiant de la session (chemin relatif du dossier session)

    Returns:
        JSONResponse avec le statut de l'opération
    \f
    La route exige un utilisateur authentifié (``current_user``) et le droit
    d'arrêt sur la session (``SessionAction.STOP``, mêmes règles que les
    routes du pipeline, audit A02) : 400 hors de ``uploads/``, 403 pour une
    session d'un projet dont il n'est pas membre, dont il n'est que lecteur,
    pour l'import d'un autre utilisateur ou pour un dossier sans
    propriétaire enregistré (administrateurs exceptés). Le registre des
    processus est interrogé avec l'identifiant canonique du dossier
    (``_session_key``) : ``./sess`` ou ``sess/`` désignent ``sess``, pour
    le contrôle comme pour l'arrêt.
    """
    if not session or not session.strip():
        logger.warning("stop_all_scripts called without session parameter")
        return JSONResponse(status_code=400, content={
            "error": "Session parameter is required",
            "details": "You must specify which session's processes to stop."
        })

    session = session.strip()

    refusal = _pipeline_session_refusal(db, session, current_user, SessionAction.STOP)
    if refusal is not None:
        logger.warning(f"stop_all_scripts: user {current_user.id} refused for session '{session}' ({refusal[0]})")
        return _session_refusal_json(refusal)

    session_key = _session_key(session)

    # Valider que le dossier session existe
    session_path = os.path.join(UPLOAD_DIR, session_key)
    if not os.path.isdir(session_path):
        logger.warning(f"stop_all_scripts: session not found: {session}")
        return JSONResponse(status_code=404, content={
            "error": f"Session not found: {session}",
            "details": "The specified session directory does not exist."
        })

    try:
        # Utiliser le ProcessManager pour arrêter uniquement les processus de cette session
        # Exécuter dans un thread pool pour ne pas bloquer pendant l'attente SIGTERM → SIGKILL
        result = await asyncio.to_thread(process_manager.stop_session, session_key)
        logger.info(f"stop_all_scripts result for session '{session_key}': {result}")
        return JSONResponse(result)

    except Exception as e:
        logger.error(f"Exception in stop_all_scripts for session '{session}': {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={
            "error": "Failed to execute stop command.",
            "details": str(e)
        })


@router.post("/process_dataframe")
async def process_dataframe(
    path: str = Form(...),
    force_fresh: bool = Form(False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Processes a Zotero JSON file to extract text content and create a CSV file.

    This endpoint handles the initial data processing step. It finds a Zotero
    JSON file in the specified session directory, extracts text from the associated
    documents (performing OCR if necessary), and saves the result as 'output.csv'.

    It includes logic to handle pre-existing CSV files:
    - If 'output.csv' exists and all rows have 'texteocr' content, OCR is skipped.
    - If 'output.csv' exists but some rows have empty 'texteocr' content, those
      rows are removed before proceeding.
    - If force_fresh=True, deletes existing CSV and progress files to start fresh.

    Args:
        path (str): The relative path to the session directory under the main upload
                    directory. This path is provided as form data.
        force_fresh (bool): If True, deletes existing output files and starts from scratch.

    Returns:
        JSONResponse: A JSON response containing the path to the created CSV file
                      and a preview of its first few rows. On error, returns a
                      JSON object with an error message.
    \f
    The session folder is checked first (``_pipeline_session_refusal``, with
    the database session ``db`` and ``current_user``): 400 outside uploads/,
    403 for a registered session of an inaccessible project.
    """
    refusal = _pipeline_session_refusal(db, path, current_user)
    if refusal is not None:
        return _session_refusal_json(refusal)
    ticket, busy = _acquire_session_job(path, job_control.GROUP_PIPELINE, current_user)
    if busy is not None:
        return _session_refusal_json(busy)
    try:
        return await _process_dataframe_impl(path=path, force_fresh=force_fresh, db=db, current_user=current_user)
    finally:
        ticket.release()


async def _process_dataframe_impl(
    path: str = Form(...),
    force_fresh: bool = Form(False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Body of ``/process_dataframe``: the route has checked the session and holds its job ticket (audit A12)."""
    refusal = _pipeline_session_refusal(db, path, current_user)
    if refusal is not None:
        return _session_refusal_json(refusal)

    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    logger.info(f"Received relative path: '{path}', resolved to absolute: '{absolute_processing_path}', force_fresh: {force_fresh}")

    # Find first JSON in directory
    try:
        if not os.path.isdir(absolute_processing_path):
            logger.error(f"Processing directory does not exist: {absolute_processing_path}")
            return JSONResponse(status_code=400, content={"error": f"Processing directory not found: {path}"})

        out_csv = os.path.join(absolute_processing_path, 'output.csv')
        progress_file = os.path.join(absolute_processing_path, 'output.progress.json')

        # Force fresh start if requested - delete existing files
        if force_fresh:
            logger.info("Force fresh mode: clearing existing output files")
            for file_to_remove in [out_csv, progress_file]:
                if os.path.exists(file_to_remove):
                    try:
                        os.remove(file_to_remove)
                        logger.info(f"Removed: {file_to_remove}")
                    except Exception as e:
                        logger.warning(f"Failed to remove {file_to_remove}: {e}")

        # Check if output.csv already exists with texteocr content
        # Determine OCR mode: "full", "skip", or "csv_cleanup"
        ocr_mode = "full"  # Default: run full OCR via rad_dataframe.py
        existing_df = None
        removed_rows_info = []  # Info about removed rows for user feedback

        if os.path.exists(out_csv):
            try:
                existing_df = await asyncio.to_thread(pd.read_csv, out_csv, dtype=str, keep_default_na=False)
                if 'texteocr' in existing_df.columns:
                    # Check which rows have empty texteocr
                    empty_mask = existing_df['texteocr'].str.strip().str.len() == 0
                    empty_count = empty_mask.sum()
                    total_count = len(existing_df)
                    non_empty_count = total_count - empty_count

                    if empty_count == 0:
                        # All rows have texteocr → skip OCR entirely
                        ocr_mode = "skip"
                        logger.info(
                            f"output.csv contains 'texteocr' for all {total_count} rows. "
                            f"Skipping OCR extraction."
                        )
                    elif non_empty_count > 0:
                        # Some rows have texteocr, some don't → remove empty rows
                        ocr_mode = "csv_cleanup"

                        # Collect info about rows to be removed
                        empty_rows = existing_df[empty_mask]
                        for idx, row in empty_rows.iterrows():
                            # Try to get identifying info (title, id, or row index)
                            row_id = row.get('title', row.get('id', f"Row {idx + 1}"))
                            removed_rows_info.append(str(row_id))

                        # Remove empty rows
                        existing_df = existing_df[~empty_mask].reset_index(drop=True)
                        await asyncio.to_thread(existing_df.to_csv, out_csv, index=False, encoding='utf-8-sig')

                        logger.info(
                            f"CSV cleanup: removed {empty_count} rows with empty 'texteocr'. "
                            f"Kept {non_empty_count} rows."
                        )
                    else:
                        # All texteocr empty → CSV is unusable
                        logger.error("CSV file has 'texteocr' column but all values are empty.")
                        return JSONResponse(status_code=400, content={
                            "error": "Your CSV file has no text content. All 'texteocr' values are empty. "
                                     "Please upload a CSV with text content in a column named 'texteocr', 'text', 'content', 'body', or 'description'."
                        })
            except Exception as e:
                logger.warning(f"Could not check existing output.csv: {e}. Proceeding with full OCR.")

        # Find JSON file only if full OCR is needed
        # Exclude pipeline-generated files (output_*.json, generated_*.json)
        json_path = None
        if ocr_mode == "full":
            excluded_prefixes = ('output_', 'output.', 'generated_')
            json_files = [f for f in os.listdir(absolute_processing_path)
                          if f.lower().endswith('.json') and not f.startswith(excluded_prefixes)]
            if not json_files:
                logger.error(f"No Zotero JSON file found in {absolute_processing_path} (excluding output_*.json)")
                return JSONResponse(status_code=400, content={
                    "error": "No Zotero JSON file found (excluding pipeline-generated output_*.json files)."
                })
            json_path = os.path.join(absolute_processing_path, json_files[0])
            logger.info(f"Processing dataframe with JSON: {json_path}, output: {out_csv}")

        if ocr_mode == "skip":
            logger.info("OCR skipped - using existing output.csv with complete texteocr content")
        elif ocr_mode == "csv_cleanup":
            logger.info(f"CSV cleanup completed - removed {len(removed_rows_info)} rows with empty content")
        else:
            # Run extraction script with improved error handling
            try:
                # OCR explicite (OCR_SERVER + OCR_MODEL, sprint « configuration unifiée »)
                try:
                    ocr_choice = _explicit_ocr_target(_user_values(db, current_user))
                except ServiceConfigError as e:
                    return JSONResponse(status_code=400, content={
                        "error": str(e), "model_not_configured": True, "variables": list(e.variables)})
                if ocr_choice is not None:
                    ocr_albert = _explicit_ocr_uses_albert(ocr_choice)
                    try:
                        subprocess_env = _explicit_ocr_env(current_user, ocr_choice)
                    except ValueError as e:
                        return JSONResponse(status_code=400, content=policy_error_body(e))
                    except CredentialMissingError as e:
                        return JSONResponse(status_code=403, content={
                            "error": str(e),
                            "credential_required": e.credential_key,
                            "configure_url": "/settings/credentials"
                        })
                else:
                    # Albert OCR link (Albert ON, OCR_ENABLE_ALBERT=1, key available)
                    ocr_albert = _ocr_albert_active(current_user)
                    # Build secure subprocess environment with user credentials
                    # rad_dataframe.py requires MISTRAL_API_KEY for OCR (or OPENAI_API_KEY as fallback)
                    try:
                        if strict_policy():
                            # albert_only: Albert OCR link (if active) then local OCR only.
                            check_policy_configuration()
                            subprocess_env = restrict_subprocess_env(build_subprocess_env(
                                current_user, required_keys=[ALBERT_CREDENTIAL_KEY] if ocr_albert else []
                            ))
                        else:
                            subprocess_env = build_subprocess_env(
                                current_user,
                                required_keys=["mistral_api_key"]  # Primary OCR provider
                            )
                    except ValueError as e:
                        return JSONResponse(status_code=400, content=policy_error_body(e))
                    except CredentialMissingError as e:
                        # Try OpenAI as fallback
                        try:
                            subprocess_env = build_subprocess_env(
                                current_user,
                                required_keys=["openai_api_key"]
                            )
                            logger.info(f"Using OpenAI fallback for OCR (user {current_user.email})")
                        except CredentialMissingError:
                            subprocess_env = None
                            if ocr_albert:
                                # Third attempt: the Albert OCR link alone
                                try:
                                    subprocess_env = build_subprocess_env(
                                        current_user,
                                        required_keys=[ALBERT_CREDENTIAL_KEY]
                                    )
                                    logger.info(f"Using the Albert OCR link only (user {current_user.email})")
                                except CredentialMissingError:
                                    subprocess_env = None
                            if subprocess_env is None:
                                logger.warning(f"User {current_user.email} missing OCR credentials")
                                return JSONResponse(status_code=403, content={
                                    "error": get_credential_error_message("mistral_api_key"),
                                    "credential_required": "mistral_api_key",
                                    "configure_url": "/settings/credentials"
                                })

                # Construct absolute path to the script using RAGPY_DIR
                project_scripts_dir = os.path.join(RAGPY_DIR, "scripts")
                script_path = os.path.join(project_scripts_dir, "rad_dataframe.py")
                script_path = os.path.abspath(script_path)

                logger.info(f"Executing rad_dataframe.py for user {current_user.email}")

                # Use tracked subprocess for session-aware process management
                result = await run_tracked_subprocess(
                    cmd=[
                        "python3", script_path,
                        "--json", json_path,
                        "--dir", absolute_processing_path,
                        "--output", out_csv
                    ],
                    session_folder=_session_key(path),
                    timeout=_subprocess_timeout(1800, ocr_albert),
                    env=subprocess_env  # Use user-specific credentials
                )

                # Manually check the return code and handle
                if result.returncode != 0:
                    logger.error(f"Extraction script failed with code {result.returncode}. stderr: {result.stderr}")
                    return JSONResponse(status_code=500, content={
                        "error": f"Extraction script failed with code {result.returncode}.",
                        "details": result.stderr,
                        "stdout": result.stdout[:500]
                    })
            except Exception as e:
                logger.error(f"An unexpected error occurred during dataframe processing: {str(e)}", exc_info=True)
                return JSONResponse(status_code=500, content={"error": "An unexpected error occurred.", "details": str(e)})

        # Load and preview CSV
        if not os.path.exists(out_csv):
            logger.error(f"Output CSV file not found after script execution: {out_csv}")
            return JSONResponse(status_code=500, content={"error": "Output CSV not found after script execution."})
    except Exception as e:
        logger.error(f"Error in process_dataframe: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Failed to process dataframe: {str(e)}"})
        
    try:
        # Try reading with escapechar and dtype=str, then adapt
        try:
            df = await asyncio.to_thread(pd.read_csv, out_csv, escapechar='\\', dtype=str, keep_default_na=False)
        except pd.errors.ParserError:
            logger.warning(f"Failed to parse CSV {out_csv} with escapechar='\\', dtype=str. Retrying without escapechar.")
            try:
                df = await asyncio.to_thread(pd.read_csv, out_csv, dtype=str, keep_default_na=False)
            except Exception as e_inner:
                logger.error(f"Failed to read CSV {out_csv} even with dtype=str and no escapechar: {str(e_inner)}")
                return JSONResponse(status_code=500, content={"error": "CSV parsing failed.", "details": str(e_inner)})
        except Exception as e_outer:
             logger.error(f"Failed to read CSV {out_csv} with escapechar='\\', dtype=str: {str(e_outer)}")
             return JSONResponse(status_code=500, content={"error": "CSV reading failed.", "details": str(e_outer)})

            
        if df.empty:
            logger.warning(f"CSV file {out_csv} is empty or contains no data after reading.")
            return JSONResponse(status_code=500, content={"error": "CSV file is empty or contains no data."})
            
        preview_df = df.head(5).fillna('') 
        preview = preview_df.to_dict(orient='records')
        
        # Final check for JSON serializability
        try:
            json.dumps(preview)
        except TypeError as te:
            logger.error(f"Preview data is not JSON serializable even after dtype=str and fillna: {str(te)}")
            safer_preview = []
            for record in preview:
                safer_record = {k: str(v) for k, v in record.items()}
                safer_preview.append(safer_record)
            preview = safer_preview
            
        response_data = {"csv": out_csv, "preview": preview}

        if removed_rows_info:
            response_data["warning"] = (
                f"{len(removed_rows_info)} row(s) with empty text content were removed from the CSV."
            )
            response_data["removed_rows"] = removed_rows_info

        return JSONResponse(response_data)
    except Exception as e:
        logger.error(f"General error in processing/previewing CSV {out_csv}: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "CSV read or preview failed.", "details": str(e)})


@router.post("/initial_text_chunking")
async def initial_text_chunking(
    path: str = Form(...),
    model: str = Form(None),
    server: str = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Generates initial text chunks from a CSV file.

    This endpoint takes the 'output.csv' file from a session directory, processes
    it using the 'rad_chunk.py' script, and generates a JSON file ('output_chunks.json')
    containing the text chunks.

    Args:
        path (str): The relative path to the session directory, which must contain
                    'output.csv'. Provided as form data.
        model (str, optional): The identifier for the language model to be used for
                               text recoding during the chunking process.
                               Defaults to "gpt-4o-mini" if not provided.

    Returns:
        JSONResponse: A success response with the path to the chunks file and the
                      number of chunks created, or an error response.
    \f
    The session folder is checked first (``_pipeline_session_refusal``, with
    the database session ``db`` and ``current_user``): 400 outside uploads/,
    403 for a registered session of an inaccessible project.
    """
    refusal = _pipeline_session_refusal(db, path, current_user)
    if refusal is not None:
        return _session_refusal_json(refusal)
    ticket, busy = _acquire_session_job(path, job_control.GROUP_PIPELINE, current_user)
    if busy is not None:
        return _session_refusal_json(busy)
    try:
        return await _initial_text_chunking_impl(path=path, model=model, server=server, db=db, current_user=current_user)
    finally:
        ticket.release()


async def _initial_text_chunking_impl(
    path: str = Form(...),
    model: str = Form(None),
    server: str = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Body of ``/initial_text_chunking``: the route has checked the session and holds its job ticket (audit A12)."""
    refusal = _pipeline_session_refusal(db, path, current_user)
    if refusal is not None:
        return _session_refusal_json(refusal)

    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    logger.info(f"Initial chunking requested for path: '{path}', resolved to: '{absolute_processing_path}'")
    
    if not os.path.isdir(absolute_processing_path):
        logger.error(f"Processing directory does not exist: {absolute_processing_path}")
        return JSONResponse(status_code=400, content={"error": f"Processing directory not found: {path}"})
    
    # Check if output.csv exists
    input_csv = os.path.join(absolute_processing_path, 'output.csv')
    if not os.path.exists(input_csv):
        logger.error(f"Input CSV not found: {input_csv}")
        return JSONResponse(status_code=400, content={
            "error": "output.csv not found. Please complete the extraction step first."
        })
    
    # Output file will be output_chunks.json
    output_chunks_file = os.path.join(absolute_processing_path, 'output_chunks.json')
    
    # Build command to run rad_chunk.py
    script_path = os.path.join(RAGPY_DIR, "scripts", "rad_chunk.py")
    if not os.path.exists(script_path):
        logger.error(f"Chunking script not found: {script_path}")
        return JSONResponse(status_code=500, content={
            "error": "Chunking script not found on server."
        })
    
    # Server + model of the recoding service (sprint « configuration unifiée »):
    # translated into the routed model; no-op in historical mode.
    try:
        routed = routed_model_for("recode", model, server, user_values=_user_values(db, current_user))
    except ServiceConfigError as e:
        return JSONResponse(status_code=400, content={
            "error": str(e), "model_not_configured": True, "variables": list(e.variables)})
    if routed is not None:
        model = routed.routed_model

    # Inference policy (ALBERT_DATA_POLICY=albert_only: Albert models only;
    # an empty model becomes the Albert default). Unchanged while compatible.
    try:
        model = apply_chat_policy(model, "recode")
    except ValueError as e:
        return JSONResponse(status_code=400, content=policy_error_body(e))

    # Set default model if not provided
    if not model:
        model = "gpt-4o-mini"
    
    logger.info(f"Running chunking script: {script_path}")
    logger.info(f"  Input CSV: {input_csv}")
    logger.info(f"  Output dir: {absolute_processing_path}")
    logger.info(f"  Model: {model}")

    # Determine required credentials based on model (single resolver:
    # 'provider/model' → OpenRouter, other names → OpenAI, see _resolve_chat_model)
    try:
        resolution = _resolve_chat_model(model)
    except ValueError as e:
        logger.warning(f"Chunking model refused for user {current_user.email}: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})
    required_keys = [resolution.credential_key]
    chat_albert = resolution.provider == PROVIDER_ALBERT

    try:
        # Build secure subprocess environment with user credentials
        try:
            subprocess_env = build_subprocess_env(current_user, required_keys=required_keys)
        except CredentialMissingError as e:
            logger.warning(f"User {current_user.email} missing credential: {e.credential_key}")
            return JSONResponse(status_code=403, content={
                "error": str(e),
                "credential_required": e.credential_key,
                "configure_url": "/settings/credentials"
            })
        restrict_subprocess_env(subprocess_env)

        # Run rad_chunk.py with phase=initial (tracked for session-aware stop)
        result = await run_tracked_subprocess(
            cmd=[
                "python3", script_path,
                "--input", input_csv,
                "--output", absolute_processing_path,
                "--phase", "initial",
                "--model", model
            ],
            session_folder=_session_key(path),
            timeout=_subprocess_timeout(1800, chat_albert),
            env=subprocess_env
        )

        if result.returncode != 0:
            logger.error(f"Chunking script failed with code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")
            return JSONResponse(status_code=500, content={
                "error": f"Chunking script failed with code {result.returncode}",
                "details": result.stderr,
                "stdout": result.stdout[:1000]
            })
        
        # Check if output file was created
        if not os.path.exists(output_chunks_file):
            logger.error(f"Output chunks file not created: {output_chunks_file}")
            return JSONResponse(status_code=500, content={
                "error": "Chunking completed but output file not found.",
                "details": result.stdout[:1000]
            })
        
        # Count chunks
        try:
            chunk_count = await asyncio.to_thread(_json_list_count, output_chunks_file)
        except Exception as e:
            logger.warning(f"Could not count chunks: {e}")
            chunk_count = 0
        
        logger.info(f"Chunking completed successfully. {chunk_count} chunks generated.")
        
        return JSONResponse({
            "status": "success",
            "file": output_chunks_file,
            "count": chunk_count,
            "message": f"Generated {chunk_count} chunks using model {model}"
        })
        
    except subprocess.TimeoutExpired:
        logger.error("Chunking script timed out after 30 minutes")
        return JSONResponse(status_code=500, content={
            "error": "Chunking process timed out (30 min limit).",
            "details": "The process took too long. Try processing fewer documents."
        })
    except Exception as e:
        logger.error(f"Unexpected error during chunking: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={
            "error": "An unexpected error occurred during chunking.",
            "details": str(e)
        })


@router.post("/process_dataframe_sse")
async def process_dataframe_sse(
    path: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Process dataframe with Server-Sent Events for real-time progress updates.
    Streams progress from rad_dataframe.py execution.
    \f
    The session folder is checked first (``_pipeline_session_refusal``, with
    the database session ``db`` and ``current_user``): one error event with
    status 400 outside uploads/, 403 for a registered session of an
    inaccessible project.
    """
    refusal = _pipeline_session_refusal(db, path, current_user)
    if refusal is not None:
        return _session_refusal_sse(refusal)
    ticket, busy = _acquire_session_job(path, job_control.GROUP_PIPELINE, current_user)
    if busy is not None:
        return _session_refusal_sse(busy)
    try:
        response = await _process_dataframe_sse_impl(path=path, db=db, current_user=current_user, job_ticket=ticket)
    except BaseException:
        ticket.release()
        raise
    return _release_after_stream(response, ticket)


async def _process_dataframe_sse_impl(
    path: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    job_ticket=None,
):
    """Body of ``/process_dataframe_sse``: the route has checked the session and holds its job ticket (audit A12)."""
    from app.utils.sse_helpers import (
        run_subprocess_with_sse, create_combined_parser,
        parse_tqdm_progress, parse_dataframe_logs, parse_multilevel_progress
    )

    refusal = _pipeline_session_refusal(db, path, current_user)
    if refusal is not None:
        return _session_refusal_sse(refusal)

    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    logger.info(f"SSE dataframe processing for path: '{path}', resolved to: '{absolute_processing_path}'")
    
    if not os.path.isdir(absolute_processing_path):
        async def error_generator():
            """Single SSE error event of an early refusal of this route."""
            yield f"data: {{\"type\": \"error\", \"message\": \"Processing directory not found: {path}\"}}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")
    
    # Find Zotero JSON file (exclude pipeline-generated output files)
    # Pipeline generates: output_chunks.json, output_chunks_with_embeddings.json, etc.
    excluded_prefixes = ('output_', 'output.', 'generated_')
    try:
        json_files = [f for f in os.listdir(absolute_processing_path)
                      if f.lower().endswith('.json') and not f.startswith(excluded_prefixes)]
    except Exception as e:
        # Event built here: the except block unbinds ``e`` when it ends.
        return _sse_error_response({"type": "error", "message": f"Failed to list directory: {e}"})

    if not json_files:
        async def error_generator():
            """Single SSE error event of an early refusal of this route."""
            yield f"data: {{\"type\": \"error\", \"message\": \"No Zotero JSON file found in directory (excluding output_*.json)\"}}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")
    
    json_path = os.path.join(absolute_processing_path, json_files[0])
    out_csv = os.path.join(absolute_processing_path, 'output.csv')

    # OCR explicite (OCR_SERVER + OCR_MODEL, sprint « configuration unifiée »)
    try:
        ocr_choice = _explicit_ocr_target(_user_values(db, current_user))
    except ServiceConfigError as e:
        return _sse_error_response({"type": "error", "message": str(e), "model_not_configured": True,
                                    "variables": list(e.variables)})
    if ocr_choice is not None:
        ocr_albert = _explicit_ocr_uses_albert(ocr_choice)
        try:
            subprocess_env = _explicit_ocr_env(current_user, ocr_choice)
        except ValueError as e:
            return _sse_error_response(dict(policy_error_body(e), type="error", message=str(e)))
        except CredentialMissingError as e:
            return _sse_error_response(_credential_error_payload(e))
    else:
        # Albert OCR link (Albert ON, OCR_ENABLE_ALBERT=1, key available)
        ocr_albert = _ocr_albert_active(current_user)

        # Build secure subprocess environment with user credentials
        # rad_dataframe.py requires MISTRAL_API_KEY for OCR (or OPENAI_API_KEY as fallback)
        try:
            if strict_policy():
                # albert_only: Albert OCR link (if active) then local OCR only.
                check_policy_configuration()
                subprocess_env = restrict_subprocess_env(build_subprocess_env(
                    current_user, required_keys=[ALBERT_CREDENTIAL_KEY] if ocr_albert else []
                ))
            else:
                subprocess_env = build_subprocess_env(
                    current_user,
                    required_keys=["mistral_api_key"]
                )
        except ValueError as e:
            return _sse_error_response(dict(policy_error_body(e), type="error", message=str(e)))
        except CredentialMissingError:
            # Try OpenAI as fallback
            try:
                subprocess_env = build_subprocess_env(
                    current_user,
                    required_keys=["openai_api_key"]
                )
                logger.info(f"Using OpenAI fallback for OCR (user {current_user.email})")
            except CredentialMissingError:
                subprocess_env = None
                if ocr_albert:
                    # Third attempt: the Albert OCR link alone
                    try:
                        subprocess_env = build_subprocess_env(
                            current_user,
                            required_keys=[ALBERT_CREDENTIAL_KEY]
                        )
                        logger.info(f"Using the Albert OCR link only (user {current_user.email})")
                    except CredentialMissingError:
                        subprocess_env = None
                if subprocess_env is None:
                    async def error_generator():
                        """Single SSE error event of an early refusal of this route."""
                        yield f"data: {{\"type\": \"error\", \"message\": \"{get_credential_error_message('mistral_api_key')}\", \"credential_required\": \"mistral_api_key\"}}\n\n"
                    return StreamingResponse(error_generator(), media_type="text/event-stream")

        # Build command
    script_path = os.path.join(RAGPY_DIR, "scripts", "rad_dataframe.py")
    cmd = ["python3", "-u", script_path, "--json", json_path, "--dir", absolute_processing_path, "--output", out_csv]

    logger.info(f"Executing for user {current_user.email}: {' '.join(cmd)}")

    # Use combined parser: prioritize structured PROGRESS logs, then tqdm, then custom logs
    parser = create_combined_parser(parse_multilevel_progress, parse_tqdm_progress, parse_dataframe_logs)

    timeout = _subprocess_timeout(1800, ocr_albert)

    # Wrap generator to add document count on complete
    async def sse_with_count():
        """Relay the script events, adding the output count to the completion event."""
        async for event in run_subprocess_with_sse(cmd, parser, session_folder=_session_key(path), timeout=timeout, env=subprocess_env, job_ticket=job_ticket):
            if '"type": "complete"' in event and '"message": "Process completed successfully"' in event:
                try:
                    if os.path.exists(out_csv):
                        df = await asyncio.to_thread(pd.read_csv, out_csv, dtype=str, keep_default_na=False)
                        count = len(df)
                        yield f'data: {{"type": "complete", "message": "Process completed successfully", "count": {count}}}\n\n'
                        continue
                except Exception:
                    pass
            yield event

    return StreamingResponse(sse_with_count(), media_type="text/event-stream")


@router.post("/dense_embedding_generation")
async def dense_embedding_generation(
    path: str = Form(...),
    embedding_provider: str = Form(None),
    server: str = Form(None),
    model: str = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Generate dense embeddings using rad_chunk.py with phase=dense.
    Requires OpenAI API key for embedding generation.
    \f
    The session folder is checked first (``_pipeline_session_refusal``, with
    the database session ``db`` and ``current_user``): 400 outside uploads/,
    403 for a registered session of an inaccessible project.
    """
    refusal = _pipeline_session_refusal(db, path, current_user)
    if refusal is not None:
        return _session_refusal_json(refusal)
    ticket, busy = _acquire_session_job(path, job_control.GROUP_PIPELINE, current_user)
    if busy is not None:
        return _session_refusal_json(busy)
    try:
        return await _dense_embedding_generation_impl(path=path, embedding_provider=embedding_provider, server=server,
                                                      model=model, db=db, current_user=current_user)
    finally:
        ticket.release()


async def _dense_embedding_generation_impl(
    path: str = Form(...),
    embedding_provider: str = Form(None),
    server: str = Form(None),
    model: str = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Body of ``/dense_embedding_generation``: the route has checked the session and holds its job ticket (audit A12)."""
    refusal = _pipeline_session_refusal(db, path, current_user)
    if refusal is not None:
        return _session_refusal_json(refusal)

    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    logger.info(f"Dense embedding generation for path: '{path}' by user {current_user.email}")
    
    if not os.path.isdir(absolute_processing_path):
        return JSONResponse(status_code=400, content={"error": f"Directory not found: {path}"})
    
    # Input should be output_chunks.json
    input_chunks = os.path.join(absolute_processing_path, 'output_chunks.json')
    if not os.path.exists(input_chunks):
        return JSONResponse(status_code=400, content={
            "error": "output_chunks.json not found. Please complete the chunking step first."
        })
    
    output_file = os.path.join(absolute_processing_path, 'output_chunks_with_embeddings.json')
    script_path = os.path.join(RAGPY_DIR, "scripts", "rad_chunk.py")

    # Couple EMBEDDING_SERVER + EMBEDDING_MODEL (sprint « configuration unifiée », lot L6)
    try:
        embedding_choice = _explicit_embedding_target(server, model)
    except ServiceConfigError as e:
        return JSONResponse(status_code=400, content={
            "error": str(e), "model_not_configured": True, "variables": list(e.variables)})
    if embedding_choice is not None:
        provider = ALBERT_DB_CHOICE if embedding_choice.server.key == ALBERT_DB_CHOICE else None
        try:
            subprocess_env = _explicit_embedding_env(current_user, embedding_choice)
        except CredentialMissingError as e:
            return JSONResponse(status_code=403, content={
                "error": str(e),
                "credential_required": e.credential_key,
                "configure_url": "/settings/credentials"
            })
        except ValueError as e:
            return JSONResponse(status_code=400, content=policy_error_body(e))
    else:
        # Embedding provider of the request (None: historical OpenAI behaviour)
        try:
            provider = _resolve_embedding_provider(embedding_provider)
        except ValueError as e:
            logger.warning(f"Embedding provider refused for user {current_user.email}: {e}")
            return JSONResponse(status_code=400, content={"error": str(e)})

        # Build secure subprocess environment with user credentials
        # Dense embedding generation requires OpenAI API key
        try:
            subprocess_env = build_subprocess_env(current_user, required_keys=_embedding_required_keys(provider))
            restrict_subprocess_env(subprocess_env)
        except CredentialMissingError as e:
            logger.warning(f"User {current_user.email} missing credential: {e.credential_key}")
            return JSONResponse(status_code=403, content={
                "error": str(e),
                "credential_required": e.credential_key,
                "configure_url": "/settings/credentials"
            })
        if provider is not None:
            subprocess_env[EMBEDDING_PROVIDER_ENV] = provider

    try:
        result = await run_tracked_subprocess(
            cmd=[
                "python3", script_path,
                "--input", input_chunks,
                "--output", absolute_processing_path,
                "--phase", "dense"
            ],
            session_folder=_session_key(path),
            timeout=_subprocess_timeout(1800, provider == ALBERT_DB_CHOICE),
            env=subprocess_env
        )

        if result.returncode != 0:
            logger.error(f"Dense embedding failed: {result.stderr}")
            return JSONResponse(status_code=500, content={
                "error": f"Dense embedding generation failed",
                "details": result.stderr[:1000]
            })
        
        if not os.path.exists(output_file):
            return JSONResponse(status_code=500, content={
                "error": "Output file not created"
            })
        
        # Count chunks
        try:
            count = await asyncio.to_thread(_json_list_count, output_file)
        except:
            count = 0
        
        return JSONResponse({
            "status": "success",
            "file": output_file,
            "count": count
        })
    except subprocess.TimeoutExpired:
        return JSONResponse(status_code=500, content={"error": "Process timed out"})
    except Exception as e:
        logger.error(f"Dense embedding error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/sparse_embedding_generation")
async def sparse_embedding_generation(
    path: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Generate sparse embeddings using rad_chunk.py with phase=sparse.
    Uses local spaCy model - no external API credentials required.
    \f
    The session folder is checked first (``_pipeline_session_refusal``, with
    the database session ``db`` and ``current_user``): 400 outside uploads/,
    403 for a registered session of an inaccessible project.
    """
    refusal = _pipeline_session_refusal(db, path, current_user)
    if refusal is not None:
        return _session_refusal_json(refusal)
    ticket, busy = _acquire_session_job(path, job_control.GROUP_PIPELINE, current_user)
    if busy is not None:
        return _session_refusal_json(busy)
    try:
        return await _sparse_embedding_generation_impl(path=path, db=db, current_user=current_user)
    finally:
        ticket.release()


async def _sparse_embedding_generation_impl(
    path: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Body of ``/sparse_embedding_generation``: the route has checked the session and holds its job ticket (audit A12)."""
    refusal = _pipeline_session_refusal(db, path, current_user)
    if refusal is not None:
        return _session_refusal_json(refusal)

    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    logger.info(f"Sparse embedding generation for path: '{path}' by user {current_user.email}")
    
    if not os.path.isdir(absolute_processing_path):
        return JSONResponse(status_code=400, content={"error": f"Directory not found: {path}"})
    
    # Input should be output_chunks_with_embeddings.json
    input_file = os.path.join(absolute_processing_path, 'output_chunks_with_embeddings.json')
    if not os.path.exists(input_file):
        return JSONResponse(status_code=400, content={
            "error": "output_chunks_with_embeddings.json not found. Please complete dense embedding first."
        })
    
    output_file = os.path.join(absolute_processing_path, 'output_chunks_with_embeddings_sparse.json')
    script_path = os.path.join(RAGPY_DIR, "scripts", "rad_chunk.py")

    # Build subprocess environment (no external credentials required for sparse/spaCy)
    # Still use build_subprocess_env for proper credential isolation
    subprocess_env = restrict_subprocess_env(build_subprocess_env(current_user))

    try:
        result = await run_tracked_subprocess(
            cmd=[
                "python3", script_path,
                "--input", input_file,
                "--output", absolute_processing_path,
                "--phase", "sparse"
            ],
            session_folder=_session_key(path),
            timeout=1800,
            env=subprocess_env
        )

        if result.returncode != 0:
            logger.error(f"Sparse embedding failed: {result.stderr}")
            return JSONResponse(status_code=500, content={
                "error": f"Sparse embedding generation failed",
                "details": result.stderr[:1000]
            })
        
        if not os.path.exists(output_file):
            return JSONResponse(status_code=500, content={
                "error": "Output file not created"
            })
        
        # Count chunks
        try:
            count = await asyncio.to_thread(_json_list_count, output_file)
        except:
            count = 0
        
        return JSONResponse({
            "status": "success",
            "file": output_file,
            "count": count
        })
    except subprocess.TimeoutExpired:
        return JSONResponse(status_code=500, content={"error": "Process timed out"})
    except Exception as e:
        logger.error(f"Sparse embedding error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


async def _upload_db_albert(
    request: Request,
    path: str,
    absolute_processing_path: str,
    db: Session,
    current_user: User
) -> JSONResponse:
    """
    Albert branch of ``/upload_db`` (private collection, embeddings computed server-side).

    Steps: session checked by ``_albert_session_refusal`` (400 outside
    uploads/, 403 for a registered session of a project the user cannot
    access; nothing is read or launched); input chosen by
    ``resolve_albert_input`` (sparse, then dense, then
    ``output_chunks.json``); ``albert_gdpr_ack`` must be true (400 with
    ``gdpr_ack_required`` otherwise); collection fields validated; the
    ``albert_api_key`` credential required (403 shape of the other targets);
    an audit entry ``ALBERT_COLLECTION_UPLOAD`` written before anything is
    sent (refused if it cannot be written); ``rad_vectordb.py --db albert``
    run with the Albert timeout; the manifest copied out of ``uploads/`` in a
    ``finally`` block, even after a failure.

    The ``Result`` block is parsed with the anchored patterns of the other
    targets, plus ``Skipped (existing)``, the manifest line and, for this
    target only, the whole ``Dedup journal:`` line (paths with spaces). A
    failure gives 500; ``credential_required`` is added only for an Albert
    account or quota stop (``_albert_upload_credential_required``).

    Args:
        request: The request (raw form fields ``albert_*``).
        path: Session folder, relative to uploads/.
        absolute_processing_path: Absolute session folder (exists).
        db: Database session (audit log).
        current_user: The authenticated user.

    Returns:
        The JSON response of the upload.
    """
    from app.models.audit import create_audit_log
    try:
        from scripts.rad_albert.collections import manifest_path_for, resolve_albert_input
    except ImportError:  # scripts/ itself on sys.path (CLI import pattern)
        from rad_albert.collections import manifest_path_for, resolve_albert_input

    # The texts leave for a third party: the folder must stay under uploads/
    # and must not be a registered session of a project the user cannot access.
    refusal = _albert_session_refusal(db, path, absolute_processing_path, current_user)
    if refusal is not None:
        return refusal

    input_file = resolve_albert_input(absolute_processing_path)
    if not input_file:
        return JSONResponse(status_code=400, content={
            "error": "Chunks file not found. Please complete the chunking step first."
        })

    form = await request.form()
    if not form_flag(form.get(ALBERT_FORM_GDPR_ACK)):
        return JSONResponse(status_code=400, content={
            "error": ALBERT_GDPR_MESSAGE,
            "gdpr_ack_required": True
        })

    target, error = albert_collection_target(form)
    if error is not None:
        return error

    script_path = os.path.join(RAGPY_DIR, "scripts", "rad_vectordb.py")
    if not os.path.exists(script_path):
        return JSONResponse(status_code=500, content={"error": "Vector DB script not found"})

    try:
        subprocess_env = restrict_subprocess_env(build_subprocess_env(current_user, required_keys=[ALBERT_CREDENTIAL_KEY]))
    except CredentialMissingError as e:
        logger.warning(f"User {current_user.email} missing credential: {e.credential_key}")
        return JSONResponse(status_code=403, content={
            "error": str(e),
            "credential_required": e.credential_key,
            "configure_url": CREDENTIALS_CONFIGURE_URL
        })

    cmd = ["python3", script_path, "--input", input_file, "--db", ALBERT_DB_CHOICE]
    cmd.extend(_albert_vectordb_flags(target))

    # Audit first: nothing leaves the server without a trace of the acknowledgement
    try:
        create_audit_log(
            db,
            action=ALBERT_UPLOAD_AUDIT_ACTION,
            user_id=current_user.id,
            resource_type=ALBERT_COLLECTION_RESOURCE,
            resource_id=target["collection_id"],
            details={
                "collection_id": target["collection_id"],
                "collection_name": target["collection_name"],
                "session": path,
                "gdpr_ack": True,
            },
        )
    except Exception as exc:
        logger.error(f"Albert upload refused, audit log unavailable: {type(exc).__name__}: {exc}")
        try:
            db.rollback()
        except Exception:
            pass
        return JSONResponse(status_code=500, content={
            "error": "Journal d'audit indisponible : envoi vers Albert refusé."
        })

    manifest_source = manifest_path_for(os.path.dirname(os.path.abspath(input_file)))
    # Sprint R2: only the manifest lines of THIS run are registered (append-only file).
    from app.services.albert_access import manifest_offset
    manifest_start = manifest_offset(manifest_source)
    try:
        result = await run_tracked_subprocess(
            cmd=cmd,
            session_folder=_session_key(path),
            timeout=_subprocess_timeout(3600, True),
            env=subprocess_env
        )

        stdout = result.stdout or ""
        if result.returncode != 0:
            stdout_tail = stdout[-2000:]
            stderr_tail = (result.stderr or "")[-2000:]
            details = stdout_tail.strip() or stderr_tail.strip()
            logger.error(f"Albert upload failed (rc={result.returncode}): {details}")
            body = {"error": "Vector DB upload failed", "details": details[:1000]}
            # Albert account or quota error only (decision 22): a configuration
            # or usage error (also exit 2) never points to the key settings.
            credential = _albert_upload_credential_required(result.returncode, stdout)
            if credential:
                body["credential_required"] = credential
            return JSONResponse(status_code=500, content=body)

        response = {"status": "success", "message": f"Uploaded to {ALBERT_DB_CHOICE}"}
        m = re.search(r'^Inserted:\s*(\d+)', stdout, re.MULTILINE)
        if m:
            response["inserted_count"] = int(m.group(1))
        m = re.search(r'^Skipped \(dedup\):\s*(\d+)', stdout, re.MULTILINE)
        if m and int(m.group(1)):
            response["skipped_count"] = int(m.group(1))
        m = re.search(r'^Dedup journal:\s*(.+)$', stdout, re.MULTILINE)
        if m and m.group(1).strip():
            response["journal_path"] = m.group(1).strip()
        m = re.search(r'^Skipped \(existing\):\s*(\d+)', stdout, re.MULTILINE)
        if m:
            response["existing_count"] = int(m.group(1))
        m = re.search(r'^Albert manifest:\s*(.+)$', stdout, re.MULTILINE)
        if m and m.group(1).strip():
            response["manifest_path"] = m.group(1).strip()
        return JSONResponse(response)
    except subprocess.TimeoutExpired:
        return JSONResponse(status_code=500, content={"error": "Upload timed out"})
    except Exception as e:
        logger.error(f"Albert upload error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        _archive_albert_manifest(manifest_source, current_user.id, path)
        await _register_albert_corpus(db, current_user, manifest_source, path, input_file, manifest_start)


async def _register_albert_corpus(db: Session, user: User, manifest_path: Optional[str], session: str,
                                  input_file: Optional[str], offset: int = 0) -> None:
    """
    Register the corpus and documents of an Albert upload in the local registry (never raises).

    Run after ``rad_vectordb.py --db albert``, success or failure: the
    successful slices written by this run (manifest lines after ``offset``,
    documents rolled back excluded) are registered under the uploader, so
    that the remote data can always be listed, searched and erased from
    RAGpy (``app/services/albert_access.py``). Earlier lines of the
    append-only session manifest (other runs, other users) are never
    re-registered. Nothing happens when the run wrote no slice.

    Args:
        db: Database session.
        user: The uploader.
        manifest_path: Manifest of the session (``albert_manifest.jsonl``).
        session: Session folder (relative to uploads/).
        input_file: Chunks file sent (bibliographic fields of the catalogue).
        offset: Manifest size taken before the run.
    """
    try:
        try:
            from scripts.rad_albert.config import AlbertConfig
        except ImportError:  # scripts/ itself on sys.path
            from rad_albert.config import AlbertConfig
        from app.services import albert_access

        if not manifest_path or not os.path.isfile(manifest_path):
            return
        api_key = get_credential_or_env(user, ALBERT_CREDENTIAL_KEY) or ""
        fingerprint = hashlib.sha256(api_key.strip().encode("utf-8")).hexdigest()[:12] if api_key else None
        corpora = await asyncio.to_thread(
            albert_access.register_manifest_run, db, user=user, manifest_path=manifest_path, offset=offset,
            base_url=AlbertConfig.from_env().base_url, key_fingerprint=fingerprint,
            session_folder=_session_key(session), chunks_path=input_file,
        )
        if not corpora:
            return
        logger.info(f"Albert corpus registry updated for user {user.id}: {[c.id for c in corpora]}")
    except Exception as exc:
        logger.error(f"Albert corpus registry not updated for session '{session}': {type(exc).__name__}: {exc}")


@router.post("/upload_db")
async def upload_db(
    request: Request,
    path: str = Form(...),
    db_choice: str = Form(...),
    pinecone_index_name: str = Form(None),
    pinecone_namespace: str = Form(None),
    weaviate_class_name: str = Form(None),
    weaviate_tenant_name: str = Form(None),
    qdrant_collection_name: str = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Upload embeddings to vector database using rad_vectordb.py.
    Requires credentials based on the selected database.
    \f
    Albert target (only while Albert is enabled, handled before the sparse
    file check): see ``_upload_db_albert``; its fields are read from the raw
    form. While Albert is disabled, ``db_choice=albert`` keeps the historical
    400 (after the sparse file check).

    The session folder is checked first, whatever the target
    (``_pipeline_session_refusal``): 400 outside uploads/, 403 for a
    registered session of an inaccessible project.
    """
    refusal = _pipeline_session_refusal(db, path, current_user)
    if refusal is not None:
        return _session_refusal_json(refusal)
    ticket, busy = _acquire_session_job(path, job_control.GROUP_PIPELINE, current_user)
    if busy is not None:
        return _session_refusal_json(busy)
    try:
        return await _upload_db_impl(request=request, path=path, db_choice=db_choice, pinecone_index_name=pinecone_index_name, pinecone_namespace=pinecone_namespace, weaviate_class_name=weaviate_class_name, weaviate_tenant_name=weaviate_tenant_name, qdrant_collection_name=qdrant_collection_name, db=db, current_user=current_user)
    finally:
        ticket.release()


async def _upload_db_impl(
    request: Request,
    path: str = Form(...),
    db_choice: str = Form(...),
    pinecone_index_name: str = Form(None),
    pinecone_namespace: str = Form(None),
    weaviate_class_name: str = Form(None),
    weaviate_tenant_name: str = Form(None),
    qdrant_collection_name: str = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Body of ``/upload_db``: the route has checked the session and holds its job ticket (audit A12)."""
    refusal = _pipeline_session_refusal(db, path, current_user)
    if refusal is not None:
        return _session_refusal_json(refusal)

    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    logger.info(f"Vector DB upload for path: '{path}', db: {db_choice}, user: {current_user.email}")
    
    if not os.path.isdir(absolute_processing_path):
        return JSONResponse(status_code=400, content={"error": f"Directory not found: {path}"})
    
    # Albert collections: own input resolution (no vector needed), before the
    # sparse file check, only while Albert is enabled
    if db_choice == ALBERT_DB_CHOICE and albert_enabled():
        return await _upload_db_albert(request, path, absolute_processing_path, db, current_user)

    # Input should be sparse embeddings file
    input_file = os.path.join(absolute_processing_path, 'output_chunks_with_embeddings_sparse.json')
    if not os.path.exists(input_file):
        return JSONResponse(status_code=400, content={
            "error": "Embeddings file not found. Please complete embedding generation first."
        })
    
    script_path = os.path.join(RAGPY_DIR, "scripts", "rad_vectordb.py")
    if not os.path.exists(script_path):
        return JSONResponse(status_code=500, content={"error": "Vector DB script not found"})

    # Determine required credentials based on db_choice
    if db_choice == "pinecone":
        required_keys = ["pinecone_api_key"]
    elif db_choice == "weaviate":
        required_keys = ["weaviate_url"]  # API key optional for some setups
    elif db_choice == "qdrant":
        required_keys = ["qdrant_url"]  # API key optional for local instances
    else:
        return JSONResponse(status_code=400, content={"error": f"Unknown database type: {db_choice}"})

    # Build secure subprocess environment with user credentials
    try:
        subprocess_env = restrict_subprocess_env(build_subprocess_env(current_user, required_keys=required_keys))
    except CredentialMissingError as e:
        logger.warning(f"User {current_user.email} missing credential: {e.credential_key}")
        return JSONResponse(status_code=403, content={
            "error": str(e),
            "credential_required": e.credential_key,
            "configure_url": "/settings/credentials"
        })

    # Build command based on db_choice
    cmd = ["python3", script_path, "--input", input_file, "--db", db_choice]

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

    try:
        result = await run_tracked_subprocess(
            cmd=cmd,
            session_folder=_session_key(path),
            timeout=3600,  # 1h for large uploads
            env=subprocess_env
        )

        if result.returncode != 0:
            # rad_vectordb.py écrit le vrai diagnostic (erreurs Pinecone, bloc
            # "=== Result ===") sur stdout ; stderr ne contient que la barre tqdm.
            # On privilégie donc le tail de stdout, avec repli sur stderr.
            stdout_tail = (result.stdout or "")[-2000:]
            stderr_tail = (result.stderr or "")[-2000:]
            details = stdout_tail.strip() or stderr_tail.strip()
            logger.error(f"Vector DB upload failed (rc={result.returncode}): {details}")
            return JSONResponse(status_code=500, content={
                "error": "Vector DB upload failed",
                "details": details[:1000]
            })
        
        # Parse the connector's canonical "=== Result ===" markers from stdout.
        # ANCHORED (^... + MULTILINE) : l'ancien re.search(r'(\d+)') non ancré
        # capturait la 1re suite de chiffres (ex. un "2026" de date/log) et corrompait
        # le compte ; un marqueur dédup l'aurait aussi avalé. rad_vectordb imprime ses
        # diagnostics sur stdout (stderr = barre tqdm). Seul le bloc Result émet ces
        # lignes en début de ligne.
        import re
        stdout = result.stdout or ""
        inserted_count = None
        skipped_count = None
        journal_path = None

        m = re.search(r'^Inserted:\s*(\d+)', stdout, re.MULTILINE)
        if m:
            inserted_count = int(m.group(1))
        m = re.search(r'^Skipped \(dedup\):\s*(\d+)', stdout, re.MULTILINE)
        if m:
            skipped_count = int(m.group(1))
        m = re.search(r'^Dedup journal:\s*(\S+)', stdout, re.MULTILINE)
        if m:
            journal_path = m.group(1)

        response = {
            "status": "success",
            "message": f"Uploaded to {db_choice}"
        }
        if inserted_count is not None:
            response["inserted_count"] = inserted_count
        if skipped_count:
            response["skipped_count"] = skipped_count
        if journal_path:
            response["journal_path"] = journal_path

        return JSONResponse(response)
    except subprocess.TimeoutExpired:
        return JSONResponse(status_code=500, content={"error": "Upload timed out (1h limit)"})
    except Exception as e:
        logger.error(f"DB upload error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/generate_zotero_notes_sse")
async def generate_zotero_notes_sse(
    session: str = Form(...),
    note_mode: str = Form("extended"),
    model: str = Form(None),
    server: str = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Generate Zotero notes with SSE progress updates.

    This implementation:
    1. Reads output.csv (contains texteocr for each document)
    2. Generates notes via LLM using build_note_html()
    3. Creates notes in Zotero via API (if credentials available)
    4. Streams real-time progress via SSE

    Args:
        session: Session folder name
        note_mode: Note generation mode. One of:
            - "extended": Full analysis [FICHE] (default)
            - "short": Quick summary for abstractNote field
            - "pedagogique": Pedagogical note [CLAIR] for L3 students
            - "evaluation": Peer review evaluation grid [EVAL]
        model: LLM model to use (e.g., "gpt-4o-mini", "google/gemini-2.5-flash")
        server: API address for this run (overrides ``LLM_NOTES_SERVER`` /
            ``LLM_BOOK_SERVER``, then ``LLM_DEFAULT_SERVER``); empty = configured
            server. Ignored while the ``.env`` has no ``LLM_DEFAULT_SERVER``
            (historical mode).

    Requires:
    - OpenAI API key (or OpenRouter for alternative models)
    - Zotero credentials (optional - for sync to Zotero library)
    \f
    The session folder is checked first (``_pipeline_session_refusal``, with
    the database session ``db`` and ``current_user``): one error event with
    status 400 outside uploads/, 403 for a registered session of an
    inaccessible project.

    When the effective model resolves to the sovereign provider, one usage
    ledger (``new_albert_usage_ledger``) is handed to the note builders
    (``albert_usage_ledger``) and appended to the session's usage journal
    at the end of the job (``flush_albert_usage_ledger``, only if a call was
    recorded); any other provider gets no ledger and no journal.
    """
    refusal = _pipeline_session_refusal(db, session, current_user)
    if refusal is not None:
        return _session_refusal_sse(refusal)
    ticket, busy = _acquire_session_job(session, job_control.GROUP_NOTES, current_user)
    if busy is not None:
        return _session_refusal_sse(busy)
    try:
        response = await _generate_zotero_notes_sse_impl(session=session, note_mode=note_mode, model=model, server=server, db=db, current_user=current_user)
    except BaseException:
        ticket.release()
        raise
    return _release_after_stream(response, ticket)


async def _generate_zotero_notes_sse_impl(
    session: str = Form(...),
    note_mode: str = Form("extended"),
    model: str = Form(None),
    server: str = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Body of ``/generate_zotero_notes_sse``: the route has checked the session and holds its job ticket (audit A12)."""
    from app.utils.llm_note_generator import (
        build_note_html_async, build_abstract_text_async, sentinel_in_html,
        TEMPLATE_MAP, NOTE_MODE_DISPLAY, ALBERT_ACCOUNT_ERRORS, resolve_default_llm_model
    )
    from app.utils.book_note_generator import build_book_note_async
    from app.utils.zotero_client import (
        verify_api_key, create_child_note, check_note_exists,
        update_item_abstract, ZoteroAPIError
    )

    refusal = _pipeline_session_refusal(db, session, current_user)
    if refusal is not None:
        return _session_refusal_sse(refusal)

    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, session))
    logger.info(f"Zotero notes generation for session: '{session}', mode: {note_mode}, model: {model}, user: {current_user.email}")

    # Validate and normalize note_mode
    # Support backward compatibility: "true"/"false" for extended_analysis
    if note_mode.lower() in ("true", "1", "yes"):
        note_mode = "extended"
    elif note_mode.lower() in ("false", "0", "no"):
        note_mode = "short"
    elif note_mode not in TEMPLATE_MAP:
        logger.warning(f"Unknown note_mode '{note_mode}', falling back to 'extended'")
        note_mode = "extended"

    # Determine if this is a "short" mode (updates abstractNote) vs HTML note modes
    use_short_mode = note_mode == "short"
    # Personal server + model choices (lot L9), read before the stream opens
    personal_values = _user_values(db, current_user)

    async def event_generator():
        """Stream the note generation events (init, progress, summary)."""
        # Usage ledger of this request: created only when Albert is selected
        albert_ledger = None
        try:
            # Check for output.csv (contains texteocr from pipeline)
            csv_path = os.path.join(absolute_processing_path, 'output.csv')
            if not os.path.exists(csv_path):
                yield f"data: {{\"type\": \"error\", \"message\": \"output.csv not found. Please complete the extraction step first.\"}}\n\n"
                return

            # Load CSV with documents
            try:
                df = await asyncio.to_thread(pd.read_csv, csv_path, dtype=str, keep_default_na=False)
            except Exception as e:
                yield f"data: {{\"type\": \"error\", \"message\": \"Failed to read CSV: {str(e)}\"}}\n\n"
                return

            if df.empty:
                yield f"data: {{\"type\": \"error\", \"message\": \"CSV file is empty.\"}}\n\n"
                return

            # Check required columns
            if 'texteocr' not in df.columns:
                yield f"data: {{\"type\": \"error\", \"message\": \"CSV missing 'texteocr' column.\"}}\n\n"
                return

            total_items = len(df)

            # Check LLM credentials (required for note generation)
            # Retrieve both API keys (one may be None)
            openai_key = get_credential_or_env(current_user, "openai_api_key")
            openrouter_key = get_credential_or_env(current_user, "openrouter_api_key")
            server_api_key = None

            # Server + model of the service (sprint « configuration unifiée »):
            # translated into the routed model the code below understands; no-op
            # in historical mode (no LLM_DEFAULT_SERVER in the .env).
            try:
                routed = routed_model_for("book" if note_mode == "book" else "notes", model, server,
                                         user_values=personal_values)
            except ServiceConfigError as e:
                yield _sse_event({"type": "error", "message": str(e), "model_not_configured": True,
                                  "variables": list(e.variables)})
                return
            job_model = routed.routed_model if routed is not None else model

            # Effective model: the form field; while Albert is enabled, an empty
            # field means the web default (read off the event loop). While Albert
            # is disabled an empty field is kept as is (no .env read): the
            # historical rule applies (empty -> OpenAI key) and the builders
            # resolve the default themselves, as before. The credential checked
            # is the one of the provider resolved by the single resolver.
            if job_model or not albert_enabled():
                effective_model = job_model
            else:
                effective_model = await asyncio.to_thread(resolve_default_llm_model)
            if strict_policy():
                # albert_only: the notes run on Albert only (empty or default
                # model -> Albert default of the role, other providers refused).
                try:
                    effective_model = apply_chat_policy(job_model, "notes")
                except ValueError as e:
                    yield _sse_event(dict(policy_error_body(e), type="error", message=str(e)))
                    return
            try:
                resolution = _resolve_chat_model(effective_model)
            except ValueError as e:
                yield _sse_event({"type": "error", "message": str(e)})
                return
            albert_key = None
            if resolution.provider == PROVIDER_ALBERT:
                albert_key = get_credential_or_env(current_user, ALBERT_CREDENTIAL_KEY)
                provider_key = albert_key
            elif resolution.provider == PROVIDER_OPENROUTER:
                provider_key = openrouter_key
            elif is_compat(resolution.provider):
                server_api_key = get_credential_or_env(current_user, resolution.credential_key)
                provider_key = server_api_key
            else:
                provider_key = openai_key
            if not provider_key:
                required = resolution.credential_key
                yield f"data: {{\"type\": \"error\", \"message\": \"{get_credential_error_message(required)}\", \"credential_required\": \"{required}\"}}\n\n"
                return

            # Albert only: one usage ledger for the whole job, handed to the
            # builders (no argument at all on the other providers)
            ledger_kwargs = {}
            if resolution.provider == PROVIDER_ALBERT:
                albert_ledger = new_albert_usage_ledger()
                ledger_kwargs["albert_usage_ledger"] = albert_ledger

            # Get Zotero credentials (optional - for sync to library)
            zotero_api_key = get_credential_or_env(current_user, "zotero_api_key") or ""
            zotero_user_id = get_credential_or_env(current_user, "zotero_user_id") or ""
            zotero_group_id = get_credential_or_env(current_user, "zotero_group_id") or ""

            # Determine library type and ID
            has_zotero_creds = bool(zotero_api_key)
            library_type = "groups" if zotero_group_id else "users"
            library_id = zotero_group_id if zotero_group_id else zotero_user_id

            if has_zotero_creds and library_id:
                try:
                    verify_api_key(zotero_api_key)
                    zotero_mode = "api"
                    logger.info(f"Zotero API verified. Library: {library_type}/{library_id}")
                except ZoteroAPIError as e:
                    logger.warning(f"Zotero API verification failed: {e}. Falling back to local-only mode.")
                    zotero_mode = "local"
            else:
                zotero_mode = "local"
                logger.info("No Zotero credentials. Notes will be generated locally only.")

            # Init event
            mode_msg = "with Zotero sync" if zotero_mode == "api" else "local only (no Zotero credentials)"
            yield f"data: {{\"type\": \"init\", \"total\": {total_items}, \"message\": \"Starting note generation ({mode_msg})...\"}}\n\n"

            # Load project description for {PROBLEMATIQUE} placeholder
            # The problematique combines project name and description
            problematique = "Non spécifiée"
            try:
                db = next(get_db())
                project = db.query(Project).filter(Project.session_folder == session).first()
                if project:
                    parts = []
                    if project.name:
                        parts.append(project.name)
                    if project.description:
                        parts.append(project.description)
                    if parts:
                        problematique = " — ".join(parts)
                    logger.info(f"Loaded project problematique: {problematique[:100]}...")
                else:
                    logger.warning(f"No project found for session '{session}', using default problematique")
            except Exception as e:
                logger.warning(f"Could not load project for problematique: {e}")

            # Counters for summary
            created = 0
            exists = 0
            skipped = 0
            errors = 0

            # Track library version for chaining (reduces 412 conflicts)
            current_library_version = None

            # Storage for generated notes (if local mode or for backup)
            generated_notes = []

            # Albert account or quota error that stopped the job (None otherwise)
            account_error = None

            # Process each document
            for idx, row in df.iterrows():
                doc_num = idx + 1
                title = str(row.get('title', f'Document {doc_num}'))[:100]
                safe_title = title.replace('"', '\\"').replace('\n', ' ')
                item_key = str(row.get('itemKey', ''))
                texteocr = str(row.get('texteocr', ''))

                # Skip if no text content
                if not texteocr.strip():
                    skipped += 1
                    yield f"data: {{\"type\": \"progress\", \"current\": {doc_num}, \"total\": {total_items}, \"item\": \"{safe_title}\", \"status\": \"skipped\", \"message\": \"Skipped (no text): {safe_title}\"}}\n\n"
                    continue

                try:
                    # Prepare metadata for note generation
                    metadata = {
                        "title": row.get('title', ''),
                        "authors": row.get('authors', ''),
                        "date": row.get('date', ''),
                        "abstract": row.get('abstract', ''),
                        "doi": row.get('doi', ''),
                        "url": row.get('url', ''),
                        "language": row.get('language', 'fr'),
                        "problematique": problematique,  # From project name + description
                    }

                    # Branch based on analysis mode
                    loop = asyncio.get_event_loop()
                    status = "created"

                    if not use_short_mode:
                        # HTML NOTE MODE (extended, pedagogique, evaluation, book)
                        # Uses global semaphore for concurrency control
                        if note_mode == "book":
                            # Books require a multi-phase pipeline (structure
                            # detection → per-chapter analysis → synthesis).
                            metadata["publisher"] = row.get("publisher", "") or row.get("publicationTitle", "")
                            metadata["itemType"] = row.get("itemType", "book")
                            metadata["numPages"] = row.get("numPages", "")
                            sentinel, note_html = await build_book_note_async(
                                metadata=metadata,
                                text_content=texteocr,
                                model=effective_model,
                                openai_api_key=openai_key,
                                openrouter_api_key=openrouter_key, server_api_key=server_api_key,
                                albert_api_key=albert_key,
                                **ledger_kwargs,
                            )
                        else:
                            sentinel, note_html = await build_note_html_async(
                                metadata=metadata,
                                text_content=texteocr,
                                model=effective_model,
                                use_llm=True,
                                mode=note_mode,
                                openai_api_key=openai_key,
                                openrouter_api_key=openrouter_key, server_api_key=server_api_key,
                                albert_api_key=albert_key,
                                **ledger_kwargs
                            )

                        # Store generated note
                        generated_notes.append({
                            "item_key": item_key,
                            "title": title,
                            "sentinel": sentinel,
                            "note_html": note_html,
                            "mode": note_mode
                        })

                        # If Zotero API mode, create child note
                        if zotero_mode == "api" and item_key:
                            try:
                                # Check if note already exists
                                note_exists = await loop.run_in_executor(
                                    None,
                                    lambda lt=library_type, lid=library_id, ik=item_key, s=sentinel, ak=zotero_api_key: check_note_exists(
                                        lt, lid, ik, s, ak
                                    )
                                )

                                if note_exists:
                                    exists += 1
                                    status = "exists"
                                else:
                                    # Create the child note with version chaining
                                    result = await loop.run_in_executor(
                                        None,
                                        lambda lt=library_type, lid=library_id, ik=item_key, nh=note_html, ak=zotero_api_key, lv=current_library_version: create_child_note(
                                            library_type=lt,
                                            library_id=lid,
                                            item_key=ik,
                                            note_html=nh,
                                            tags=["ragpy-generated"],
                                            api_key=ak,
                                            library_version=lv
                                        )
                                    )

                                    if result.get("success"):
                                        created += 1
                                        status = "created"
                                        # Update version for next call (reduces 412 conflicts)
                                        current_library_version = result.get("new_version")
                                    else:
                                        errors += 1
                                        status = "error"
                                        logger.warning(f"Failed to create child note for {item_key}: {result.get('message')}")
                            except ZoteroAPIError as e:
                                errors += 1
                                status = "error"
                                logger.error(f"Zotero API error for {item_key}: {e}")
                        else:
                            # Local mode - just count as created
                            created += 1

                    else:
                        # SHORT MODE: Generate plain text summary and update abstract
                        # Uses global semaphore for concurrency control
                        summary_text = await build_abstract_text_async(
                            metadata=metadata,
                            text_content=texteocr,
                            model=effective_model,
                            openai_api_key=openai_key,
                            openrouter_api_key=openrouter_key, server_api_key=server_api_key,
                            albert_api_key=albert_key,
                            **ledger_kwargs
                        )

                        # Store generated summary
                        generated_notes.append({
                            "item_key": item_key,
                            "title": title,
                            "summary": summary_text,
                            "mode": "short"
                        })

                        # If Zotero API mode, update abstract field
                        if zotero_mode == "api" and item_key:
                            try:
                                result = await loop.run_in_executor(
                                    None,
                                    lambda lt=library_type, lid=library_id, ik=item_key, summ=summary_text, ak=zotero_api_key: update_item_abstract(
                                        library_type=lt,
                                        library_id=lid,
                                        item_key=ik,
                                        new_abstract=summ,
                                        api_key=ak
                                    )
                                )

                                if result.get("success"):
                                    created += 1
                                    status = "created"
                                    logger.info(f"Updated abstract for {item_key} (length: {result.get('new_abstract_length', 'unknown')})")
                                else:
                                    errors += 1
                                    status = "error"
                                    logger.warning(f"Failed to update abstract for {item_key}: {result.get('message')}")
                            except ZoteroAPIError as e:
                                errors += 1
                                status = "error"
                                logger.error(f"Zotero API error updating abstract for {item_key}: {e}")
                        else:
                            # Local mode - just count as created
                            created += 1

                    yield f"data: {{\"type\": \"progress\", \"current\": {doc_num}, \"total\": {total_items}, \"item\": \"{safe_title}\", \"status\": \"{status}\", \"message\": \"Processed {doc_num}/{total_items}: {safe_title}\"}}\n\n"

                except ALBERT_ACCOUNT_ERRORS as e:
                    # Albert account or quota error: the whole job stops (decision 22)
                    account_error = e
                    logger.error(f"Albert account error on document {doc_num}, stopping the notes job: {e}")
                    break
                except Exception as e:
                    errors += 1
                    error_msg = str(e).replace('"', '\\"').replace('\n', ' ')[:100]
                    logger.error(f"Error processing document {doc_num}: {e}", exc_info=True)
                    yield f"data: {{\"type\": \"progress\", \"current\": {doc_num}, \"total\": {total_items}, \"item\": \"{safe_title}\", \"status\": \"error\", \"message\": \"Error: {error_msg}\"}}\n\n"

                # Small delay to prevent overwhelming the API
                await asyncio.sleep(0.1)

            # Save generated notes to file for backup/review (also before an Albert stop)
            if generated_notes:
                notes_file = os.path.join(absolute_processing_path, 'generated_notes.json')
                try:
                    with open(notes_file, 'w', encoding='utf-8') as f:
                        json.dump(generated_notes, f, ensure_ascii=False, indent=2)
                    logger.info(f"Saved {len(generated_notes)} notes to {notes_file}")
                except Exception as e:
                    logger.warning(f"Could not save notes file: {e}")

            if account_error is not None:
                yield _sse_event({
                    "type": "error",
                    "message": str(account_error),
                    "credential_required": getattr(account_error, "credential_required", ALBERT_CREDENTIAL_KEY),
                })
                return

            # Completion event with summary
            summary = {
                "created": created,
                "exists": exists,
                "skipped": skipped,
                "errors": errors,
                "mode": zotero_mode
            }
            yield f"data: {{\"type\": \"complete\", \"message\": \"Note generation completed\", \"summary\": {json.dumps(summary)}}}\n\n"

        except Exception as e:
            logger.error(f"Zotero notes SSE error: {e}", exc_info=True)
            error_msg = str(e).replace('"', '\\"').replace('\n', ' ')
            yield f"data: {{\"type\": \"error\", \"message\": \"{error_msg}\"}}\n\n"
        finally:
            # Albert usage of the job (also after a stop or a disconnection);
            # nothing when Albert was not selected or never called
            flush_albert_usage_ledger(albert_ledger, absolute_processing_path, "notes")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ============================================================================
# Optional SSE versions for better UX on long-running operations
# ============================================================================

@router.post("/initial_text_chunking_sse")
async def initial_text_chunking_sse(
    path: str = Form(...),
    model: str = Form(None),
    server: str = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    SSE version of initial_text_chunking for real-time progress updates.
    Uses multilevel progress parser for dual progress bars (documents + chunks).
    \f
    The session folder is checked first (``_pipeline_session_refusal``, with
    the database session ``db`` and ``current_user``): one error event with
    status 400 outside uploads/, 403 for a registered session of an
    inaccessible project.
    """
    refusal = _pipeline_session_refusal(db, path, current_user)
    if refusal is not None:
        return _session_refusal_sse(refusal)
    ticket, busy = _acquire_session_job(path, job_control.GROUP_PIPELINE, current_user)
    if busy is not None:
        return _session_refusal_sse(busy)
    try:
        response = await _initial_text_chunking_sse_impl(path=path, model=model, server=server, db=db, current_user=current_user, job_ticket=ticket)
    except BaseException:
        ticket.release()
        raise
    return _release_after_stream(response, ticket)


async def _initial_text_chunking_sse_impl(
    path: str = Form(...),
    model: str = Form(None),
    server: str = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    job_ticket=None,
):
    """Body of ``/initial_text_chunking_sse``: the route has checked the session and holds its job ticket (audit A12)."""
    from app.utils.sse_helpers import (
        run_subprocess_with_sse, create_combined_parser,
        parse_multilevel_progress, parse_tqdm_progress, parse_chunking_logs
    )

    refusal = _pipeline_session_refusal(db, path, current_user)
    if refusal is not None:
        return _session_refusal_sse(refusal)

    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    logger.info(f"SSE chunking for path: '{path}'")
    
    if not os.path.isdir(absolute_processing_path):
        async def error_generator():
            """Single SSE error event of an early refusal of this route."""
            yield f"data: {{\"type\": \"error\", \"message\": \"Directory not found: {path}\"}}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")
    
    input_csv = os.path.join(absolute_processing_path, 'output.csv')
    if not os.path.exists(input_csv):
        async def error_generator():
            """Single SSE error event of an early refusal of this route."""
            yield f"data: {{\"type\": \"error\", \"message\": \"output.csv not found. Complete extraction first.\"}}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")
    
    script_path = os.path.join(RAGPY_DIR, "scripts", "rad_chunk.py")
    # Server + model of the recoding service; no-op in historical mode.
    try:
        routed = routed_model_for("recode", model, server, user_values=_user_values(db, current_user))
    except ServiceConfigError as e:
        return _sse_error_response({"type": "error", "message": str(e), "model_not_configured": True,
                                    "variables": list(e.variables)})
    if routed is not None:
        model = routed.routed_model
    try:
        model = apply_chat_policy(model, "recode")
    except ValueError as e:
        return _sse_error_response(dict(policy_error_body(e), type="error", message=str(e)))
    model = model or "gpt-4o-mini"

    # Determine required credentials based on model (single resolver)
    try:
        resolution = _resolve_chat_model(model)
    except ValueError as e:
        return _sse_error_response({"type": "error", "message": str(e)})
    required_keys = [resolution.credential_key]
    chat_albert = resolution.provider == PROVIDER_ALBERT

    # Build secure subprocess environment with user credentials. The error
    # event is built here: the except block unbinds ``e`` when it ends.
    try:
        subprocess_env = build_subprocess_env(current_user, required_keys=required_keys)
    except CredentialMissingError as e:
        return _sse_error_response(_credential_error_payload(e))
    restrict_subprocess_env(subprocess_env)

    cmd = [
        "python3", "-u", script_path,  # -u for unbuffered output
        "--input", input_csv,
        "--output", absolute_processing_path,
        "--phase", "initial",
        "--model", model
    ]

    # Prioritize structured PROGRESS logs for multilevel progress display
    parser = create_combined_parser(parse_multilevel_progress, parse_tqdm_progress, parse_chunking_logs)

    # Wrap generator to add chunk count on complete
    output_file = os.path.join(absolute_processing_path, 'output_chunks.json')
    timeout = _subprocess_timeout(1800, chat_albert)

    async def sse_with_count():
        """Relay the script events, adding the output count to the completion event."""
        async for event in run_subprocess_with_sse(cmd, parser, session_folder=_session_key(path), timeout=timeout, env=subprocess_env, job_ticket=job_ticket):
            # Intercept complete event to add count
            if '"type": "complete"' in event and '"message": "Process completed successfully"' in event:
                try:
                    if os.path.exists(output_file):
                        count = await asyncio.to_thread(_json_list_count, output_file)
                        yield f'data: {{"type": "complete", "message": "Process completed successfully", "count": {count}}}\n\n'
                        continue
                except Exception:
                    pass
            yield event

    return StreamingResponse(sse_with_count(), media_type="text/event-stream")


@router.post("/dense_embedding_generation_sse")
async def dense_embedding_generation_sse(
    path: str = Form(...),
    embedding_provider: str = Form(None),
    server: str = Form(None),
    model: str = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    SSE version of dense_embedding_generation for real-time progress updates.
    Uses multilevel progress parser for dual progress bars (documents + chunks).
    Requires OpenAI API key for embedding generation.
    \f
    The session folder is checked first (``_pipeline_session_refusal``, with
    the database session ``db`` and ``current_user``): one error event with
    status 400 outside uploads/, 403 for a registered session of an
    inaccessible project.
    """
    refusal = _pipeline_session_refusal(db, path, current_user)
    if refusal is not None:
        return _session_refusal_sse(refusal)
    ticket, busy = _acquire_session_job(path, job_control.GROUP_PIPELINE, current_user)
    if busy is not None:
        return _session_refusal_sse(busy)
    try:
        response = await _dense_embedding_generation_sse_impl(path=path, embedding_provider=embedding_provider, server=server,
                                                              model=model, db=db, current_user=current_user,
                                                              job_ticket=ticket)
    except BaseException:
        ticket.release()
        raise
    return _release_after_stream(response, ticket)


async def _dense_embedding_generation_sse_impl(
    path: str = Form(...),
    embedding_provider: str = Form(None),
    server: str = Form(None),
    model: str = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    job_ticket=None,
):
    """Body of ``/dense_embedding_generation_sse``: the route has checked the session and holds its job ticket (audit A12)."""
    from app.utils.sse_helpers import (
        run_subprocess_with_sse, create_combined_parser,
        parse_multilevel_progress, parse_tqdm_progress, parse_chunking_logs
    )

    refusal = _pipeline_session_refusal(db, path, current_user)
    if refusal is not None:
        return _session_refusal_sse(refusal)

    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    input_chunks = os.path.join(absolute_processing_path, 'output_chunks.json')

    if not os.path.exists(input_chunks):
        async def error_generator():
            """Single SSE error event of an early refusal of this route."""
            yield f"data: {{\"type\": \"error\", \"message\": \"output_chunks.json not found\"}}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")

    # Couple EMBEDDING_SERVER + EMBEDDING_MODEL (sprint « configuration unifiée », lot L6)
    try:
        embedding_choice = _explicit_embedding_target(server, model)
    except ServiceConfigError as e:
        return _sse_error_response({"type": "error", "message": str(e), "model_not_configured": True,
                                    "variables": list(e.variables)})
    if embedding_choice is not None:
        provider = ALBERT_DB_CHOICE if embedding_choice.server.key == ALBERT_DB_CHOICE else None
        try:
            subprocess_env = _explicit_embedding_env(current_user, embedding_choice)
        except CredentialMissingError as e:
            return _sse_error_response(_credential_error_payload(e))
        except ValueError as e:
            return _sse_error_response(dict(policy_error_body(e), type="error", message=str(e)))
    else:
        # Embedding provider of the request (None: historical OpenAI behaviour)
        try:
            provider = _resolve_embedding_provider(embedding_provider)
        except ValueError as e:
            return _sse_error_response({"type": "error", "message": str(e)})

        # Build secure subprocess environment with user credentials. The error
        # event is built here: the except block unbinds ``e`` when it ends.
        try:
            subprocess_env = build_subprocess_env(current_user, required_keys=_embedding_required_keys(provider))
        except CredentialMissingError as e:
            return _sse_error_response(_credential_error_payload(e))
        restrict_subprocess_env(subprocess_env)
        if provider is not None:
            subprocess_env[EMBEDDING_PROVIDER_ENV] = provider

    script_path = os.path.join(RAGPY_DIR, "scripts", "rad_chunk.py")
    cmd = [
        "python3", "-u", script_path,  # -u for unbuffered output
        "--input", input_chunks,
        "--output", absolute_processing_path,
        "--phase", "dense"
    ]

    # Prioritize structured PROGRESS logs for multilevel progress display
    parser = create_combined_parser(parse_multilevel_progress, parse_tqdm_progress, parse_chunking_logs)

    # Wrap generator to add chunk count on complete
    output_file = os.path.join(absolute_processing_path, 'output_chunks_with_embeddings.json')
    logger.info(f"Dense embedding expecting output at: {output_file}")
    timeout = _subprocess_timeout(1800, provider == ALBERT_DB_CHOICE)

    async def sse_with_count():
        """Relay the script events, adding the output count to the completion event."""
        async for event in run_subprocess_with_sse(cmd, parser, session_folder=_session_key(path), timeout=timeout, env=subprocess_env, job_ticket=job_ticket):
            if '"type": "complete"' in event and '"message": "Process completed successfully"' in event:
                try:
                    logger.info(f"Dense complete event received, checking for output file: {output_file}")
                    if os.path.exists(output_file):
                        count = await asyncio.to_thread(_json_list_count, output_file)
                        logger.info(f"Dense embedding file found with {count} chunks")
                        yield f'data: {{"type": "complete", "message": "Process completed successfully", "count": {count}}}\n\n'
                        continue
                    else:
                        logger.warning(f"Dense output file not found: {output_file}")
                except Exception as e:
                    logger.error(f"Error reading dense output file: {e}")
            yield event

    return StreamingResponse(sse_with_count(), media_type="text/event-stream")


@router.post("/sparse_embedding_generation_sse")
async def sparse_embedding_generation_sse(
    path: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    SSE version of sparse_embedding_generation for real-time progress updates.
    Uses multilevel progress parser for chunk-level progress display.
    Uses local spaCy model - no external API credentials required.
    \f
    The session folder is checked first (``_pipeline_session_refusal``, with
    the database session ``db`` and ``current_user``): one error event with
    status 400 outside uploads/, 403 for a registered session of an
    inaccessible project.
    """
    refusal = _pipeline_session_refusal(db, path, current_user)
    if refusal is not None:
        return _session_refusal_sse(refusal)
    ticket, busy = _acquire_session_job(path, job_control.GROUP_PIPELINE, current_user)
    if busy is not None:
        return _session_refusal_sse(busy)
    try:
        response = await _sparse_embedding_generation_sse_impl(path=path, db=db, current_user=current_user, job_ticket=ticket)
    except BaseException:
        ticket.release()
        raise
    return _release_after_stream(response, ticket)


async def _sparse_embedding_generation_sse_impl(
    path: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    job_ticket=None,
):
    """Body of ``/sparse_embedding_generation_sse``: the route has checked the session and holds its job ticket (audit A12)."""
    from app.utils.sse_helpers import (
        run_subprocess_with_sse, create_combined_parser,
        parse_multilevel_progress, parse_tqdm_progress, parse_chunking_logs
    )

    refusal = _pipeline_session_refusal(db, path, current_user)
    if refusal is not None:
        return _session_refusal_sse(refusal)

    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    input_file = os.path.join(absolute_processing_path, 'output_chunks_with_embeddings.json')

    if not os.path.exists(input_file):
        async def error_generator():
            """Single SSE error event of an early refusal of this route."""
            yield f"data: {{\"type\": \"error\", \"message\": \"output_chunks_with_embeddings.json not found\"}}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")

    # Build subprocess environment (no external credentials required for sparse/spaCy)
    # Still use build_subprocess_env for proper credential isolation
    subprocess_env = restrict_subprocess_env(build_subprocess_env(current_user))

    script_path = os.path.join(RAGPY_DIR, "scripts", "rad_chunk.py")
    cmd = [
        "python3", "-u", script_path,  # -u for unbuffered output
        "--input", input_file,
        "--output", absolute_processing_path,
        "--phase", "sparse"
    ]

    # Prioritize structured PROGRESS logs for progress display
    parser = create_combined_parser(parse_multilevel_progress, parse_tqdm_progress, parse_chunking_logs)

    # Wrap generator to add chunk count on complete
    output_file = os.path.join(absolute_processing_path, 'output_chunks_with_embeddings_sparse.json')
    logger.info(f"Sparse embedding expecting output at: {output_file}")

    async def sse_with_count():
        """Relay the script events, adding the output count to the completion event."""
        async for event in run_subprocess_with_sse(cmd, parser, session_folder=_session_key(path), timeout=1800, env=subprocess_env, job_ticket=job_ticket):
            if '"type": "complete"' in event and '"message": "Process completed successfully"' in event:
                try:
                    logger.info(f"Sparse complete event received, checking for output file: {output_file}")
                    if os.path.exists(output_file):
                        count = await asyncio.to_thread(_json_list_count, output_file)
                        logger.info(f"Sparse embedding file found with {count} chunks")
                        yield f'data: {{"type": "complete", "message": "Process completed successfully", "count": {count}}}\n\n'
                        continue
                    else:
                        logger.warning(f"Sparse output file not found: {output_file}")
                except Exception as e:
                    logger.error(f"Error reading sparse output file: {e}")
            yield event

    return StreamingResponse(sse_with_count(), media_type="text/event-stream")


# =============================================================================
# DOCUMENT CLUSTERING (Step 4.b)
# =============================================================================

@router.post("/cluster_documents")
async def cluster_documents(
    session_folder: str = Form(...),
    session_name: str = Form(...),
    min_cluster_size: int = Form(None),
    aggregation: str = Form("mean"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Cluster documents based on their embeddings using UMAP + HDBSCAN.

    This endpoint performs document-level clustering on embeddings and generates
    Zotero-compatible tags for automatic organization. It's designed to run after
    embedding generation (Step 4) as Step 4.b.

    Args:
        session_folder: Relative path to session folder in uploads/ directory.
        session_name: Name for tag generation (e.g., 'MaBiblio').
                     Tags will be formatted as '_MaBiblio_01', '_MaBiblio_02', etc.
        min_cluster_size: Minimum documents per cluster. If None, auto-calculated
                         to achieve approximately N/10 clusters.
        aggregation: Method to aggregate chunk embeddings per document.
                    Options: 'mean' (default), 'max', 'first'.

    Returns:
        JSONResponse with:
            - success (bool): Whether clustering completed successfully
            - n_documents (int): Total documents clustered
            - n_clusters (int): Number of clusters found
            - n_noise (int): Number of unclustered (noise) documents
            - cluster_sizes (dict): Mapping of cluster_id to document count
            - output_path (str): Path to clustering_results.json

    Raises:
        HTTPException 400: If embeddings file not found
        HTTPException 500: If clustering fails

    Example:
        POST /cluster_documents
        Form data:
            session_folder: "abc123_MaBiblio"
            session_name: "MaBiblio"
    \f
    The session folder is checked first (``_pipeline_session_refusal``, with
    the database session ``db`` and ``current_user``): 400 outside uploads/,
    403 for a registered session of an inaccessible project.
    """
    refusal = _pipeline_session_refusal(db, session_folder, current_user)
    if refusal is not None:
        return _session_refusal_json(refusal)
    ticket, busy = _acquire_session_job(session_folder, job_control.GROUP_CLUSTERING, current_user)
    if busy is not None:
        return _session_refusal_json(busy)
    try:
        return await _cluster_documents_impl(session_folder=session_folder, session_name=session_name, min_cluster_size=min_cluster_size, aggregation=aggregation, db=db, current_user=current_user)
    finally:
        ticket.release()


async def _cluster_documents_impl(
    session_folder: str = Form(...),
    session_name: str = Form(...),
    min_cluster_size: int = Form(None),
    aggregation: str = Form("mean"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Body of ``/cluster_documents``: the route has checked the session and holds its job ticket (audit A12)."""
    refusal = _pipeline_session_refusal(db, session_folder, current_user)
    if refusal is not None:
        return _session_refusal_json(refusal)

    # Validate session folder exists
    abs_session = os.path.abspath(os.path.join(UPLOAD_DIR, session_folder))

    # Look for embeddings file (prefer sparse, fallback to dense)
    embeddings_path = os.path.join(abs_session, "output_chunks_with_embeddings_sparse.json")
    if not os.path.exists(embeddings_path):
        embeddings_path = os.path.join(abs_session, "output_chunks_with_embeddings.json")
        if not os.path.exists(embeddings_path):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Embeddings file not found. Run embedding generation first.",
                    "missing_file": "output_chunks_with_embeddings.json"
                }
            )

    try:
        # Import clustering module
        import sys
        sys.path.insert(0, RAGPY_DIR)
        from scripts.rad_clustering import run_clustering_pipeline

        # Run clustering pipeline
        results = run_clustering_pipeline(
            embeddings_json_path=embeddings_path,
            output_dir=abs_session,
            session_name=session_name,
            min_cluster_size=min_cluster_size if min_cluster_size else None,
            aggregation_method=aggregation
        )

        return JSONResponse({
            "success": True,
            "n_documents": results["n_documents"],
            "n_clusters": results["n_clusters"],
            "n_noise": results["n_noise"],
            "noise_ratio": results["noise_ratio"],
            "cluster_sizes": results["cluster_sizes"],
            "output_path": os.path.join(session_folder, "clustering_results.json")
        })

    except ValueError as e:
        logger.warning(f"Clustering validation error: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )

    except ImportError as e:
        logger.error(f"Clustering dependency missing: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Clustering dependencies not installed. Run: pip install umap-learn hdbscan",
                "details": str(e)
            }
        )

    except Exception as e:
        logger.exception(f"Clustering failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Clustering failed: {str(e)}"}
        )


@router.post("/cluster_documents_sse")
async def cluster_documents_sse(
    session_folder: str = Form(...),
    session_name: str = Form(...),
    min_cluster_size: int = Form(None),
    aggregation: str = Form("mean"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    SSE version of cluster_documents for real-time progress updates.

    Streams progress events during clustering:
    - init: Starting clustering
    - progress: Step completion (1/5, 2/5, etc.)
    - complete: Clustering finished with results
    - error: If clustering fails

    Uses subprocess to run rad_clustering.py with PROGRESS logging.
    \f
    The session folder is checked first (``_pipeline_session_refusal``, with
    the database session ``db`` and ``current_user``): status 400 outside
    uploads/, 403 for a registered session of an inaccessible project. The
    refusal is a JSON body ``{"error": <message>}``, not an SSE event: the
    clustering handler of the pipeline page reads ``response.json()`` on a
    refusal status, and would otherwise show the bare status text.
    """
    refusal = _pipeline_session_refusal(db, session_folder, current_user)
    if refusal is not None:
        return _session_refusal_json(refusal)
    ticket, busy = _acquire_session_job(session_folder, job_control.GROUP_CLUSTERING, current_user)
    if busy is not None:
        return _session_refusal_json(busy)
    try:
        response = await _cluster_documents_sse_impl(session_folder=session_folder, session_name=session_name, min_cluster_size=min_cluster_size, aggregation=aggregation, db=db, current_user=current_user, job_ticket=ticket)
    except BaseException:
        ticket.release()
        raise
    return _release_after_stream(response, ticket)


async def _cluster_documents_sse_impl(
    session_folder: str = Form(...),
    session_name: str = Form(...),
    min_cluster_size: int = Form(None),
    aggregation: str = Form("mean"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    job_ticket=None,
):
    """Body of ``/cluster_documents_sse``: the route has checked the session and holds its job ticket (audit A12)."""
    from app.utils.sse_helpers import (
        run_subprocess_with_sse, create_combined_parser,
        parse_multilevel_progress
    )

    refusal = _pipeline_session_refusal(db, session_folder, current_user)
    if refusal is not None:
        return _session_refusal_json(refusal)

    # Validate session folder
    abs_session = os.path.abspath(os.path.join(UPLOAD_DIR, session_folder))

    # Look for embeddings file
    embeddings_path = os.path.join(abs_session, "output_chunks_with_embeddings_sparse.json")
    if not os.path.exists(embeddings_path):
        embeddings_path = os.path.join(abs_session, "output_chunks_with_embeddings.json")
        if not os.path.exists(embeddings_path):
            async def error_generator():
                """Single SSE error event of an early refusal of this route."""
                yield 'data: {"type": "error", "message": "Embeddings file not found. Run embedding generation first."}\n\n'
            return StreamingResponse(error_generator(), media_type="text/event-stream")

    # Build subprocess environment (no external credentials needed for clustering)
    subprocess_env = restrict_subprocess_env(build_subprocess_env(current_user))

    # Build command
    script_path = os.path.join(RAGPY_DIR, "scripts", "rad_clustering.py")
    cmd = [
        "python3", "-u", script_path,
        "--input", embeddings_path,
        "--output", abs_session,
        "--session-name", session_name,
        "--aggregation", aggregation
    ]

    if min_cluster_size:
        cmd.extend(["--min-cluster-size", str(min_cluster_size)])

    # Use multilevel progress parser
    parser = create_combined_parser(parse_multilevel_progress)

    # Wrap generator to add cluster count on complete
    output_file = os.path.join(abs_session, "clustering_results.json")

    async def sse_with_results():
        """Relay the clustering events, then send the results event."""
        async for event in run_subprocess_with_sse(cmd, parser, session_folder=_session_key(session_folder), timeout=600, env=subprocess_env, job_ticket=job_ticket):
            if '"type": "complete"' in event and '"message": "Process completed successfully"' in event:
                try:
                    if os.path.exists(output_file):
                        with open(output_file, 'r', encoding='utf-8') as f:
                            results = json.load(f)
                        # Format expected by JavaScript: {type, total, results: {n_documents, n_clusters, n_noise}}
                        complete_event = {
                            "type": "complete",
                            "total": 5,
                            "results": {
                                "n_documents": results["n_documents"],
                                "n_clusters": results["n_clusters"],
                                "n_noise": results["n_noise"]
                            }
                        }
                        yield f'data: {json.dumps(complete_event)}\n\n'
                        continue
                except Exception as e:
                    logger.error(f"Error reading clustering results: {e}")
            yield event

    return StreamingResponse(sse_with_results(), media_type="text/event-stream")


@router.post("/apply_cluster_tags_zotero")
async def apply_cluster_tags_zotero(
    session_folder: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Apply cluster tags to Zotero items.

    Reads clustering_results.json and adds cluster tags to corresponding
    Zotero items based on their itemKey field. Tags follow the format
    '_SessionName_01', '_SessionName_02', etc.

    Args:
        session_folder: Relative path to session folder containing clustering_results.json

    Returns:
        JSONResponse with:
            - success (bool): Whether tagging completed
            - tagged_count (int): Number of items successfully tagged
            - failed_count (int): Number of failed items
            - errors (list): List of error details (truncated to 10)

    Raises:
        HTTPException 400: If clustering_results.json not found
        HTTPException 403: If Zotero credentials missing

    Note:
        Requires Zotero API credentials (zotero_api_key, zotero_user_id or zotero_group_id)
        to be configured in user settings.
    \f
    The session folder is checked first (``_pipeline_session_refusal``, with
    the database session ``db`` and ``current_user``): 400 outside uploads/,
    403 for a registered session of an inaccessible project.
    """
    refusal = _pipeline_session_refusal(db, session_folder, current_user)
    if refusal is not None:
        return _session_refusal_json(refusal)
    ticket, busy = _acquire_session_job(session_folder, job_control.GROUP_CLUSTERING, current_user)
    if busy is not None:
        return _session_refusal_json(busy)
    try:
        return await _apply_cluster_tags_zotero_impl(session_folder=session_folder, db=db, current_user=current_user)
    finally:
        ticket.release()


async def _apply_cluster_tags_zotero_impl(
    session_folder: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Body of ``/apply_cluster_tags_zotero``: the route has checked the session and holds its job ticket (audit A12)."""
    refusal = _pipeline_session_refusal(db, session_folder, current_user)
    if refusal is not None:
        return _session_refusal_json(refusal)

    # Get Zotero credentials
    zotero_api_key = get_credential_or_env(current_user, "zotero_api_key")
    zotero_user_id = get_credential_or_env(current_user, "zotero_user_id")
    zotero_group_id = get_credential_or_env(current_user, "zotero_group_id")

    if not zotero_api_key:
        return JSONResponse(
            status_code=403,
            content={
                "error": get_credential_error_message("zotero_api_key"),
                "credential_required": "zotero_api_key",
                "configure_url": "/settings/credentials"
            }
        )

    library_type = "groups" if zotero_group_id else "users"
    library_id = zotero_group_id or zotero_user_id

    if not library_id:
        return JSONResponse(
            status_code=403,
            content={
                "error": get_credential_error_message("zotero_user_id"),
                "credential_required": "zotero_user_id",
                "configure_url": "/settings/credentials"
            }
        )

    # Load clustering results
    abs_session = os.path.abspath(os.path.join(UPLOAD_DIR, session_folder))
    results_path = os.path.join(abs_session, "clustering_results.json")

    if not os.path.exists(results_path):
        return JSONResponse(
            status_code=400,
            content={
                "error": "clustering_results.json not found. Run clustering first.",
                "missing_file": "clustering_results.json"
            }
        )

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            clustering_results = json.load(f)

        # Build tag mapping from itemKey to cluster_tag
        tag_mapping = {}
        for doc in clustering_results.get("documents", []):
            item_key = doc.get("itemKey")
            cluster_tag = doc.get("cluster_tag")
            if item_key and cluster_tag:
                tag_mapping[item_key] = cluster_tag

        if not tag_mapping:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "No Zotero item keys found in clustering results. Ensure documents have itemKey field."
                }
            )

        # Import and call Zotero tag function
        from app.utils.zotero_client import add_tags_to_items

        result = add_tags_to_items(
            library_type=library_type,
            library_id=library_id,
            tag_mapping=tag_mapping,
            api_key=zotero_api_key
        )

        return JSONResponse({
            "success": True,
            "tagged_count": result["success_count"],
            "failed_count": result["failed_count"],
            "total_items": len(tag_mapping),
            "errors": result.get("errors", [])[:10]  # Limit error details
        })

    except Exception as e:
        logger.exception(f"Failed to apply Zotero tags: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to apply tags: {str(e)}"}
        )
