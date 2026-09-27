"""
Celery Task Routes
==================

This module provides FastAPI endpoints for managing Celery tasks.
It supports a dual-mode architecture where processing can be done via
either subprocess (legacy) or Celery (production).

Features:
    - Task submission endpoints for all pipeline stages
    - Task status polling endpoint
    - Task cancellation endpoint
    - Automatic fallback to subprocess when Celery unavailable

Security:
    - Every endpoint requires an authenticated, active user.
    - Submission checks the edit right on the pipeline session (``path``
      must be a session of a project the user owns or collaborates on, and
      ``session_id`` its id): a read-only member is refused (audit A02).
    - Missing credentials are refused at submission with the same 403 body
      as the HTTP routes (``error``, ``credential_required``,
      ``configure_url``).
    - Tasks receive ``user_id`` only, never a secret: the worker rebuilds the
      user's subprocess environment (``app.tasks.runner``).
    - The owner of each task is recorded in Redis; status and cancellation
      are reserved to the owner or an administrator (administrators only
      when the owner registry is unavailable).
    - ``/status`` and ``/workers`` are reserved to administrators.

Albert (DINUM, opt-in): same rules as the HTTP routes. A recoding model or
an embedding provider selecting Albert while it is disabled gives a 400
without queueing; the ``albert`` upload target (fields read from the raw
form, never declared, so the OpenAPI schema stays unchanged) is handled only
while Albert is enabled, before the sparse file check, and requires the
retention acknowledgement (audited at submission). A submission selecting
Albert is queued with ``apply_async`` and time limits above the Albert
script timeout; any other one keeps ``.delay`` and the global limits.

Environment Variables:
    ENABLE_CELERY: Set to 'true' to enable Celery mode (default: false)

Endpoints:
    POST /api/celery/process_dataframe - Submit extraction task
    POST /api/celery/initial_chunking - Submit chunking task
    POST /api/celery/dense_embedding - Submit dense embedding task
    POST /api/celery/sparse_embedding - Submit sparse embedding task
    POST /api/celery/upload_vectordb - Submit vector DB upload task
    GET /api/celery/task/{task_id}/status - Get task status
    POST /api/celery/task/{task_id}/cancel - Cancel task
    GET /api/celery/status - Celery system status (admin)
    GET /api/celery/workers - Active workers (admin)
"""
import os
import logging
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.celery_app import (
    CELERY_ENABLED,
    is_celery_available,
    get_task_status,
    revoke_task
)
from app.core.config import UPLOAD_DIR
from app.core.credentials import CredentialMissingError, albert_enabled
from app.core.upload_safety import UnsafePathError, confined_path
from app.services import job_control
from app.database.session import get_db
from app.middleware.auth import get_current_active_user, require_admin
from app.models.user import User
from app.routes.pipeline import verify_session_access

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/celery", tags=["celery"])

CREDENTIALS_CONFIGURE_URL = "/settings/credentials"
# Upload target handled by _submit_albert_upload (Albert enabled only).
ALBERT_DB_CHOICE = "albert"


def _check_celery_available() -> None:
    """
    Check if Celery is enabled and available.

    Raises:
        HTTPException: If Celery is not available
    """
    if not CELERY_ENABLED:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Celery is not enabled",
                "message": "Set ENABLE_CELERY=true in .env to enable Celery mode"
            }
        )

    if not is_celery_available():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Celery broker unavailable",
                "message": "Redis broker is not reachable. Check if Redis is running."
            }
        )


def _runner():
    """
    Import the task runner lazily.

    Importing ``app.tasks`` loads every task module; doing it at request time
    keeps the web application import free of the tasks package.

    Returns:
        The ``app.tasks.runner`` module.
    """
    from app.tasks import runner

    return runner


def _session_directory(db: Session, path: str, session_id: int, user: User) -> str:
    """
    Resolve the session directory after checking the user's access to it.

    Args:
        db: Database session
        path: Session folder under uploads/ (must be a registered session)
        session_id: Database ID announced for this session
        user: The current user

    Returns:
        Absolute path of the session directory

    Raises:
        HTTPException: 404 unknown session, 403 no access to its project or
            read-only access (a queued stage writes into the session, audit
            A02), 400 when ``session_id`` is not the id of that session or
            when the folder is not strictly under uploads/, 409 while a
            pipeline stage runs on the session (audit A12)
    """
    session = verify_session_access(db, path, user, write=True)
    if session.id != session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id ne correspond pas à la session indiquée par path"
        )
    try:
        directory = os.path.abspath(confined_path(UPLOAD_DIR, path))
    except UnsafePathError:
        raise HTTPException(
            status_code=400,
            detail="Chemin de session invalide : le dossier doit se trouver sous uploads/."
        )
    # A stage already running on the session (web route or task): refused now
    # rather than failing in the worker (audit A12; the worker holds the lock too).
    if job_control.session_busy(_runner().session_lock_key(directory), job_control.GROUP_PIPELINE):
        raise HTTPException(
            status_code=409,
            detail=job_control.SESSION_BUSY_MESSAGE.format(label=job_control.GROUP_LABELS[job_control.GROUP_PIPELINE]),
        )
    return directory


def _credential_error(error: CredentialMissingError) -> JSONResponse:
    """
    Build the 403 response of a missing credential (same body as the HTTP routes).

    Args:
        error: The credential error raised by the environment builder

    Returns:
        JSONResponse with status 403
    """
    return JSONResponse(status_code=403, content={
        "error": str(error),
        "credential_required": error.credential_key,
        "configure_url": CREDENTIALS_CONFIGURE_URL
    })


def _check_credentials(user: User, stage: str, **params) -> Optional[JSONResponse]:
    """
    Check the user's credentials for a pipeline stage before queueing it.

    Uses the same environment builder as the worker, so the refusal happens
    at submission rather than in the task.

    Args:
        user: The current user
        stage: Pipeline stage (``runner.STAGES``)
        **params: Stage parameters (``model``, ``db_choice``, ``embedding_provider``)

    Returns:
        The 403 response when a credential is missing, a 400 response when
        the selection is refused (an Albert model or provider while Albert
        is disabled, an unknown provider), else None
    """
    try:
        _runner().build_task_env(user, stage, **params)
    except CredentialMissingError as e:
        logger.warning(f"User {user.id} missing credential: {e.credential_key}")
        return _credential_error(e)
    except ValueError as e:
        logger.warning(f"User {user.id} selection refused for stage {stage}: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})
    return None


def _albert_upload_target(form: Any) -> Tuple[Optional[Dict[str, Any]], Optional[JSONResponse]]:
    """
    Validate the raw form fields of an albert upload (same rules as the HTTP route).

    Args:
        form: The request form

    Returns:
        ``(target, None)`` with ``collection_id``, ``collection_name`` and
        ``create_collection``, or ``(None, response)`` with a 400 response
        (acknowledgement missing: ``gdpr_ack_required``)
    """
    from app.routes.processing import ALBERT_FORM_GDPR_ACK, ALBERT_GDPR_MESSAGE, albert_collection_target, form_flag

    if not form_flag(form.get(ALBERT_FORM_GDPR_ACK)):
        return None, JSONResponse(status_code=400, content={
            "error": ALBERT_GDPR_MESSAGE,
            "gdpr_ack_required": True
        })
    return albert_collection_target(form)


def _queue_task(task: Any, payload: Dict[str, Any], albert_selected: bool) -> Any:
    """
    Queue a pipeline task with keyword arguments only.

    A task that does not select Albert is queued exactly as before
    (``task.delay(**payload)``, global Celery time limits). A task that
    selects Albert (``albert/`` recoding model, ``embedding_provider=albert``,
    active Albert OCR link, ``db_choice=albert``) is queued with
    ``apply_async`` and the time limits of ``runner.albert_time_limits()``,
    above the Albert script timeout, so that the global soft limit (1 h)
    never kills a longer Albert job.

    Args:
        task: The Celery task.
        payload: Keyword arguments of the task (no secret).
        albert_selected: True when the submission selects Albert.

    Returns:
        The ``AsyncResult`` of the queued task.
    """
    if not albert_selected:
        return task.delay(**payload)
    return task.apply_async(kwargs=payload, **_runner().albert_time_limits())


def _record_owner(task_id: str, user: User) -> None:
    """
    Record the submitting user as the owner of a queued task.

    Args:
        task_id: Celery task ID
        user: The submitting user
    """
    if not _runner().record_task_owner(task_id, user.id):
        logger.warning(f"Owner of task {task_id} not recorded: status reserved to administrators")


def _check_task_access(task_id: str, user: User) -> None:
    """
    Allow the owner of a task or an administrator.

    Args:
        task_id: Celery task ID
        user: The current user

    Raises:
        HTTPException: 403 when the user is neither the owner nor an admin
    """
    if user.is_admin:
        return
    owner_id = _runner().get_task_owner(task_id)
    if owner_id is None or owner_id != user.id:
        raise HTTPException(status_code=403, detail="Accès non autorisé à cette tâche")


@router.post("/process_dataframe")
async def submit_extraction_task(
    path: str = Form(...),
    session_id: int = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Submit PDF extraction task to Celery queue.

    This endpoint queues an extraction task for processing Zotero JSON
    and associated PDFs. Returns immediately with a task ID for polling.

    Args:
        path: Relative path to session directory under uploads/
        session_id: Database session ID for tracking
        db: Database session
        current_user: Authenticated user (owner of the task)

    Returns:
        JSONResponse: {
            "task_id": str,
            "status": "queued",
            "message": str
        }
    """
    _check_celery_available()

    # Resolve paths
    absolute_path = _session_directory(db, path, session_id, current_user)

    if not os.path.isdir(absolute_path):
        raise HTTPException(
            status_code=400,
            detail=f"Session directory not found: {path}"
        )

    # Find JSON file
    excluded_prefixes = ('output_', 'output.', 'generated_')
    try:
        json_files = [
            f for f in os.listdir(absolute_path)
            if f.lower().endswith('.json') and not f.startswith(excluded_prefixes)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list directory: {e}")

    if not json_files:
        raise HTTPException(
            status_code=400,
            detail="No Zotero JSON file found (excluding pipeline-generated files)"
        )

    json_path = os.path.join(absolute_path, json_files[0])
    output_path = os.path.join(absolute_path, 'output.csv')

    runner = _runner()
    denied = _check_credentials(current_user, runner.STAGE_EXTRACTION)
    if denied is not None:
        return denied

    # Submit task
    from app.tasks.extraction import process_dataframe_task

    task = _queue_task(process_dataframe_task, dict(
        json_path=json_path,
        base_dir=absolute_path,
        output_path=output_path,
        session_id=session_id,
        user_id=current_user.id
    ), runner.ocr_albert_active(current_user))
    _record_owner(task.id, current_user)

    logger.info(f"Submitted extraction task {task.id} for session {session_id}")

    return JSONResponse({
        "task_id": task.id,
        "status": "queued",
        "message": "Extraction task queued for processing"
    })


@router.post("/initial_chunking")
async def submit_chunking_task(
    path: str = Form(...),
    session_id: int = Form(...),
    model: str = Form("gpt-4o-mini"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Submit text chunking task to Celery queue.

    Args:
        path: Relative path to session directory
        session_id: Database session ID
        model: LLM model for GPT recoding (default: gpt-4o-mini)
        db: Database session
        current_user: Authenticated user (owner of the task)

    Returns:
        JSONResponse with task_id
    """
    _check_celery_available()

    absolute_path = _session_directory(db, path, session_id, current_user)
    input_csv = os.path.join(absolute_path, 'output.csv')

    if not os.path.exists(input_csv):
        raise HTTPException(
            status_code=400,
            detail="output.csv not found. Complete extraction step first."
        )

    runner = _runner()
    model = model or runner.DEFAULT_CHUNKING_MODEL

    denied = _check_credentials(current_user, runner.STAGE_CHUNKING, model=model)
    if denied is not None:
        return denied

    from app.tasks.chunking import initial_chunking_task

    task = _queue_task(initial_chunking_task, dict(
        input_csv=input_csv,
        output_dir=absolute_path,
        session_id=session_id,
        model=model,
        user_id=current_user.id
    ), runner.chunking_selects_albert(model))
    _record_owner(task.id, current_user)

    logger.info(f"Submitted chunking task {task.id} for session {session_id}")

    return JSONResponse({
        "task_id": task.id,
        "status": "queued",
        "message": f"Chunking task queued (model: {model})"
    })


@router.post("/dense_embedding")
async def submit_dense_embedding_task(
    path: str = Form(...),
    session_id: int = Form(...),
    embedding_provider: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Submit dense embedding generation task to Celery queue.

    Args:
        path: Relative path to session directory
        session_id: Database session ID
        db: Database session
        current_user: Authenticated user (owner of the task)

    Returns:
        JSONResponse with task_id
    \f
    ``embedding_provider``: dense embedding provider, same rule as the HTTP
    route (ignored while Albert is disabled, except ``albert``: 400). A task
    selecting Albert is queued with the Albert time limits (``_queue_task``).
    """
    _check_celery_available()

    absolute_path = _session_directory(db, path, session_id, current_user)
    input_file = os.path.join(absolute_path, 'output_chunks.json')

    if not os.path.exists(input_file):
        raise HTTPException(
            status_code=400,
            detail="output_chunks.json not found. Complete chunking step first."
        )

    runner = _runner()
    try:
        provider = runner.resolve_embedding_provider(embedding_provider)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    params = {} if provider is None else {"embedding_provider": provider}
    denied = _check_credentials(current_user, runner.STAGE_DENSE, **params)
    if denied is not None:
        return denied

    from app.tasks.embeddings import dense_embedding_task

    task = _queue_task(dense_embedding_task, dict(
        input_file=input_file,
        output_dir=absolute_path,
        session_id=session_id,
        user_id=current_user.id,
        **params
    ), provider == runner.ALBERT_DB_CHOICE)
    _record_owner(task.id, current_user)

    logger.info(f"Submitted dense embedding task {task.id} for session {session_id}")

    return JSONResponse({
        "task_id": task.id,
        "status": "queued",
        "message": "Dense embedding task queued"
    })


@router.post("/sparse_embedding")
async def submit_sparse_embedding_task(
    path: str = Form(...),
    session_id: int = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Submit sparse embedding generation task to Celery queue.

    Args:
        path: Relative path to session directory
        session_id: Database session ID
        db: Database session
        current_user: Authenticated user (owner of the task)

    Returns:
        JSONResponse with task_id
    """
    _check_celery_available()

    absolute_path = _session_directory(db, path, session_id, current_user)
    input_file = os.path.join(absolute_path, 'output_chunks_with_embeddings.json')

    if not os.path.exists(input_file):
        raise HTTPException(
            status_code=400,
            detail="output_chunks_with_embeddings.json not found. Complete dense embedding first."
        )

    from app.tasks.embeddings import sparse_embedding_task

    task = sparse_embedding_task.delay(
        input_file=input_file,
        output_dir=absolute_path,
        session_id=session_id,
        user_id=current_user.id
    )
    _record_owner(task.id, current_user)

    logger.info(f"Submitted sparse embedding task {task.id} for session {session_id}")

    return JSONResponse({
        "task_id": task.id,
        "status": "queued",
        "message": "Sparse embedding task queued"
    })


async def _submit_albert_upload(
    request: Request,
    path: str,
    session_id: int,
    absolute_path: str,
    db: Session,
    current_user: User
) -> JSONResponse:
    """
    Queue an upload to an Albert collection (Albert enabled only).

    Same checks as the albert branch of the HTTP ``/upload_db`` route: input
    chosen by ``resolve_albert_input`` (sparse, dense, then plain chunks),
    retention acknowledgement required (400 ``gdpr_ack_required``),
    collection fields validated, ``albert_api_key`` required (403). The
    acknowledgement is written to the audit log before queueing
    (``ALBERT_COLLECTION_UPLOAD``); the task is queued with the Albert time
    limits (``_queue_task``), never auto-retries and refuses a redelivered
    message.

    Args:
        request: The request (raw form fields ``albert_*``)
        path: Session folder, relative to uploads/
        session_id: Database session ID
        absolute_path: Absolute session folder
        db: Database session (audit log)
        current_user: Authenticated user (owner of the task)

    Returns:
        JSONResponse with task_id, or an error response
    """
    from app.models.audit import create_audit_log
    from app.routes.processing import ALBERT_COLLECTION_RESOURCE, ALBERT_UPLOAD_AUDIT_ACTION
    from scripts.rad_albert.collections import resolve_albert_input

    runner = _runner()
    input_file = resolve_albert_input(absolute_path)
    if not input_file:
        raise HTTPException(
            status_code=400,
            detail="Chunks file not found. Complete the chunking step first."
        )

    target, error = _albert_upload_target(await request.form())
    if error is not None:
        return error

    denied = _check_credentials(current_user, runner.STAGE_VECTORDB, db_choice=runner.ALBERT_DB_CHOICE)
    if denied is not None:
        return denied

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
        logger.error(f"Albert upload refused, audit log unavailable: {type(exc).__name__}")
        try:
            db.rollback()
        except Exception:
            pass
        return JSONResponse(status_code=500, content={
            "error": "Journal d'audit indisponible : envoi vers Albert refusé."
        })

    from app.tasks.vectordb import upload_to_vectordb_task

    task = _queue_task(upload_to_vectordb_task, dict(
        input_file=input_file,
        session_id=session_id,
        db_choice=runner.ALBERT_DB_CHOICE,
        user_id=current_user.id,
        albert_collection_id=target["collection_id"],
        albert_collection_name=target["collection_name"],
        albert_create_collection=target["create_collection"],
        albert_gdpr_ack=True
    ), True)
    _record_owner(task.id, current_user)

    logger.info(f"Submitted Albert upload task {task.id} for session {session_id}")

    return JSONResponse({
        "task_id": task.id,
        "status": "queued",
        "message": f"Upload task queued for {runner.ALBERT_DB_CHOICE}"
    })


@router.post("/upload_vectordb")
async def submit_vectordb_upload_task(
    request: Request,
    path: str = Form(...),
    session_id: int = Form(...),
    db_choice: str = Form(...),
    pinecone_index_name: Optional[str] = Form(None),
    pinecone_namespace: Optional[str] = Form(None),
    weaviate_class_name: Optional[str] = Form(None),
    weaviate_tenant_name: Optional[str] = Form(None),
    qdrant_collection_name: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Submit vector database upload task to Celery queue.

    Args:
        path: Relative path to session directory
        session_id: Database session ID
        db_choice: Target database ('pinecone', 'weaviate', 'qdrant')
        pinecone_index_name: Pinecone index name (if applicable)
        pinecone_namespace: Pinecone namespace (optional)
        weaviate_class_name: Weaviate class name (if applicable)
        weaviate_tenant_name: Weaviate tenant name (optional)
        qdrant_collection_name: Qdrant collection name (if applicable)
        db: Database session
        current_user: Authenticated user (owner of the task)

    Returns:
        JSONResponse with task_id
    \f
    The albert target is handled by ``_submit_albert_upload`` while Albert is
    enabled, before the sparse file check (raw form fields); otherwise the
    historical checks (sparse file, then allowed choices) apply unchanged.
    """
    _check_celery_available()

    absolute_path = _session_directory(db, path, session_id, current_user)

    if db_choice == ALBERT_DB_CHOICE and albert_enabled():
        return await _submit_albert_upload(request, path, session_id, absolute_path, db, current_user)

    input_file = os.path.join(absolute_path, 'output_chunks_with_embeddings_sparse.json')

    if not os.path.exists(input_file):
        raise HTTPException(
            status_code=400,
            detail="Embeddings file not found. Complete embedding generation first."
        )

    # Validate db_choice
    if db_choice not in ('pinecone', 'weaviate', 'qdrant'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid db_choice: {db_choice}. Must be pinecone, weaviate, or qdrant."
        )

    denied = _check_credentials(current_user, _runner().STAGE_VECTORDB, db_choice=db_choice)
    if denied is not None:
        return denied

    from app.tasks.vectordb import upload_to_vectordb_task

    task = upload_to_vectordb_task.delay(
        input_file=input_file,
        session_id=session_id,
        db_choice=db_choice,
        pinecone_index_name=pinecone_index_name,
        pinecone_namespace=pinecone_namespace,
        weaviate_class_name=weaviate_class_name,
        weaviate_tenant_name=weaviate_tenant_name,
        qdrant_collection_name=qdrant_collection_name,
        user_id=current_user.id
    )
    _record_owner(task.id, current_user)

    logger.info(f"Submitted vectordb upload task {task.id} to {db_choice}")

    return JSONResponse({
        "task_id": task.id,
        "status": "queued",
        "message": f"Upload task queued for {db_choice}"
    })


@router.get("/task/{task_id}/status")
async def get_celery_task_status(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get the status of a Celery task.

    This endpoint provides real-time status updates for queued tasks.
    Poll this endpoint to track task progress. Reserved to the owner of the
    task or an administrator.

    Args:
        task_id: The Celery task ID returned when task was submitted
        current_user: Authenticated user

    Returns:
        JSONResponse: {
            "state": str,  # PENDING, STARTED, PROGRESS, SUCCESS, FAILURE, REVOKED
            "current": int,  # Current progress (if PROGRESS)
            "total": int,  # Total items (if PROGRESS)
            "percent": int,  # Progress percentage (if PROGRESS)
            "item": str,  # Current item being processed (if PROGRESS)
            "status": str,  # Human-readable status message
            "result": dict,  # Task result (if SUCCESS)
            "error": str  # Error message (if FAILURE)
        }
    """
    _check_task_access(task_id, current_user)
    status = get_task_status(task_id)
    return JSONResponse(status)


@router.post("/task/{task_id}/cancel")
async def cancel_celery_task(
    task_id: str,
    terminate: bool = Query(False, description="Force terminate running task"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Cancel a Celery task.

    Reserved to the owner of the task or an administrator. With
    ``terminate``, the worker kills the task's script process group.

    Args:
        task_id: The Celery task ID to cancel
        terminate: If True, forcefully terminate running task (SIGTERM)
        current_user: Authenticated user

    Returns:
        JSONResponse: {
            "success": bool,
            "message": str
        }
    """
    _check_task_access(task_id, current_user)
    result = revoke_task(task_id, terminate=terminate)

    if result.get('success'):
        logger.info(f"Task {task_id} cancelled (terminate={terminate})")
        return JSONResponse(result)
    else:
        raise HTTPException(status_code=500, detail=result.get('error'))


@router.get("/status")
async def get_celery_status(
    current_user: User = Depends(require_admin)
):
    """
    Get overall Celery system status (administrators only).

    Args:
        current_user: Authenticated administrator

    Returns:
        JSONResponse: {
            "enabled": bool,
            "available": bool,
            "broker_url": str (masked)
        }
    """
    from app.celery_app import CELERY_BROKER_URL

    # Mask sensitive parts of broker URL
    masked_url = CELERY_BROKER_URL
    if '@' in masked_url:
        # Hide password in URL
        parts = masked_url.split('@')
        masked_url = parts[0].split(':')[0] + ':***@' + parts[1]

    return JSONResponse({
        "enabled": CELERY_ENABLED,
        "available": is_celery_available() if CELERY_ENABLED else False,
        "broker_url": masked_url
    })


@router.get("/workers")
async def get_celery_workers(
    current_user: User = Depends(require_admin)
):
    """
    Get information about active Celery workers (administrators only).

    Args:
        current_user: Authenticated administrator

    Returns:
        JSONResponse with worker information
    """
    _check_celery_available()

    try:
        from app.celery_app import celery_app

        # Get active workers
        inspector = celery_app.control.inspect(timeout=2)
        active = inspector.active() or {}
        stats = inspector.stats() or {}

        workers = []
        for worker_name, worker_stats in stats.items():
            workers.append({
                "name": worker_name,
                "active_tasks": len(active.get(worker_name, [])),
                "pool": worker_stats.get('pool', {}).get('implementation', 'unknown'),
                "concurrency": worker_stats.get('pool', {}).get('max-concurrency', 0),
                "processed": worker_stats.get('total', {})
            })

        return JSONResponse({
            "workers": workers,
            "total_workers": len(workers)
        })

    except Exception as e:
        logger.error(f"Failed to get worker info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
