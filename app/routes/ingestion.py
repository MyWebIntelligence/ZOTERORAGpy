"""
Ingestion Routes
================

This module handles file ingestion and initial processing. It supports uploading
ZIP archives containing documents or direct CSV uploads for structured data.
It manages the creation of `PipelineSession` records and file extraction.

Key Features:
- ZIP Upload: Extracts and organizes files for processing.
- CSV Upload: Direct ingestion of structured data into DataFrames.
- Stage Upload: Allows uploading intermediate artifacts for specific pipeline stages.

Session folders (decision of 2026-09-27, lot 9):
- ``/upload_stage_file/{stage}`` requires an authenticated user (the
  ``Authorization`` header or the ``access_token`` cookie sent by the page)
  and checks the session folder before writing anything, with the helpers of
  ``app/routes/processing.py``: 400 when the folder does not resolve strictly
  under ``UPLOAD_DIR``, 403 for a registered session of a project the
  non-admin user cannot access; folders unrelated to any ``PipelineSession``
  row keep the historical behaviour.
- ZIP extraction refuses the whole archive, before writing anything, when a
  member would land outside the extraction directory (``..`` components,
  absolute names): ``/upload_zip`` answers with its "not a valid ZIP
  archive" 400.
- ``/upload_zip`` and ``/upload_csv`` (same decision, extended to every
  route touching sessions or projects) require an authenticated user (the
  ``Authorization`` header or the ``access_token`` cookie sent by the page),
  401 otherwise. When a ``project_id`` is given, the check of the
  ``/api/pipeline/projects/{id}/upload_*`` routes runs first, before
  anything is written (``verify_project_access``, then the edit right):
  404 for an unknown project, 403 for a project the user cannot access or
  may only read; owner, collaborator and administrator go on.

Upload confinement and ownership (audit A01/A02, 2026-09-27):
- Every storage name is generated on the server (``<8 hex>_<safe stem>``
  folders, ``_source_upload.csv`` for the CSV being converted); the client
  file name only gives a sanitised stem and is kept as metadata. Every
  destination and cleanup target is checked strictly under ``UPLOAD_DIR``
  (``app.core.upload_safety``): an absolute name, ``..`` segments, POSIX or
  Windows separators or an empty name never write, overwrite or delete
  anything outside the new session folder, even when parsing fails.
- Sizes are bounded: ``UPLOAD_MAX_MB`` per uploaded file,
  ``UPLOAD_MAX_UNZIPPED_MB`` and ``UPLOAD_MAX_ZIP_ENTRIES`` per archive
  (413 above, nothing left on disk).
- Every upload has an owner: the project (``PipelineSession``) when a
  ``project_id`` is given, else the uploader (``SessionOwner``). When that
  record cannot be written, the upload is removed and the route answers
  500: a folder is never left without an owner.
- ``/upload_stage_file/{stage}`` requires the write right on the session
  (``app.core.session_access``): a read-only project member, another user
  or an unowned legacy folder is refused with 403.
- Blocking work (copies, extraction, pandas) runs in a worker thread, so a
  large upload does not stall the other requests and SSE streams.
"""
import asyncio
import os
import zipfile
import logging
import sys
import json
import csv
from typing import Optional, Tuple

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.config import RAGPY_DIR, UPLOAD_DIR
from app.core.session_access import SessionAction, canonical_session_folder, record_session_owner
from app.services import job_control
from app.core.upload_safety import (
    FILENAME_CORRUPTION_MAP,  # noqa: F401  (re-exported for existing imports)
    UnsafePathError,
    UnsafeZipMemberError,  # noqa: F401  (re-exported for existing imports)
    UploadTooLargeError,
    confined_path,
    extract_zip_with_encoding_fix,
    filename_extension,
    fix_zip_filename_encoding,  # noqa: F401  (re-exported for existing imports)
    max_upload_bytes,
    new_session_folder_name,
    safe_remove_file,
    safe_remove_tree,
    save_upload,
    save_upload_atomic,
    split_processing_path,
    zip_member_target as _zip_member_target,  # noqa: F401  (historical name)
)
from app.database.session import get_db
from app.middleware.auth import get_current_active_user
from app.models.pipeline_session import PipelineSession, SessionStatus
from app.models.project import Project
from app.models.user import User
from app.routes.pipeline import verify_project_access
from app.routes.processing import _session_refusal_for, _session_refusal_json

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter()

# Server-generated name of the uploaded CSV while it is converted to output.csv.
CSV_SOURCE_FILENAME = "_source_upload.csv"
UPLOAD_OWNER_ERROR_MESSAGE = (
    "Import annulé : la session n'a pas pu être enregistrée (base de données indisponible). Réessayez."
)


# --- Configuration for Stage Uploads ---
BASE_CHUNK_OUTPUT_NAME = "output"

STAGE_UPLOAD_CONFIG = {
    "initial": {
        "filename": f"{BASE_CHUNK_OUTPUT_NAME}.csv",
        "allowed_extensions": [".csv"],
        "summary_type": "csv",
        "description": "Output CSV from dataframe processing"
    },
    "dense": {
        "filename": f"{BASE_CHUNK_OUTPUT_NAME}_chunks.json",
        "allowed_extensions": [".json"],
        "summary_type": "json_list",
        "description": "Chunks JSON from initial chunking"
    },
    "sparse": {
        "filename": f"{BASE_CHUNK_OUTPUT_NAME}_chunks_with_embeddings.json",
        "allowed_extensions": [".json"],
        "summary_type": "json_list",
        "description": "Dense embeddings JSON"
    }
}

def summarize_uploaded_stage(stage_key: str, file_path: str) -> dict:
    """Build a lightweight summary for uploaded intermediate files."""
    config = STAGE_UPLOAD_CONFIG.get(stage_key, {})
    summary_type = config.get("summary_type")
    summary: dict = {}

    if summary_type == "csv":
        try:
            with open(file_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                headers = next(reader, None)
                row_count = sum(1 for _ in reader)
            summary["rows"] = row_count
            if headers is not None:
                summary["columns"] = len(headers)
        except Exception as exc:
            summary["parse_warning"] = f"Failed to analyse CSV: {exc}"
    elif summary_type == "json_list":
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                summary["count"] = len(data)
            else:
                summary["parse_warning"] = "Uploaded JSON is not a list; unable to count items."
        except Exception as exc:
            summary["parse_warning"] = f"Failed to analyse JSON: {exc}"

    return summary


# Refusal message of the /api/pipeline/projects/{id}/upload_* routes (read-only member).
PROJECT_UPLOAD_DENIED_MESSAGE = "Vous n'avez pas les droits pour ajouter des fichiers à ce projet"


def _upload_project_or_refusal(
    db: Session, project_id: Optional[int], user: User
) -> Tuple[Optional[Project], Optional[JSONResponse]]:
    """
    Check the project an upload is attached to, before anything is written.

    Same check as ``/api/pipeline/projects/{id}/upload_zip`` and
    ``upload_csv``: ``verify_project_access`` (the project exists, the user
    owns it, is a member of it or is an administrator), then the edit right
    (owner, collaborator or administrator), since the upload creates a
    ``PipelineSession`` and replaces ``project.session_folder``.

    Args:
        db: Database session.
        project_id: Project sent by the client, or None when absent.
        user: The authenticated user.

    Returns:
        ``(project, None)`` when the upload may go on (``project`` is None
        without ``project_id``); ``(None, response)`` otherwise, the response
        being a JSON ``{"error": <message>}`` with status 404 (unknown
        project) or 403 (no access, or read-only member).
    """
    if project_id is None:
        return None, None
    try:
        project = verify_project_access(db, project_id, user)
    except HTTPException as exc:
        logger.warning(f"Upload refused for user {user.id}: project {project_id} ({exc.status_code})")
        return None, JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
    if not project.can_edit(user.id) and not user.is_admin:
        logger.warning(f"Upload refused for user {user.id}: read-only access to project {project_id}")
        return None, JSONResponse(status_code=403, content={"error": PROJECT_UPLOAD_DENIED_MESSAGE})
    return project, None


def _register_upload_session(
    db: Session,
    project: Project,
    session_folder: str,
    original_filename: str,
    source_type: str,
    status: SessionStatus,
    row_count: Optional[int] = None,
) -> Optional[int]:
    """
    Record an upload as a ``PipelineSession`` of ``project`` and make it the project's active folder.

    A database failure is logged and rolled back; the caller then removes
    the upload and answers 500 (audit A02: a folder is never left on disk
    without its owner record).

    Args:
        db: Database session.
        project: Project checked by ``_upload_project_or_refusal``.
        session_folder: Folder of the upload, relative to ``UPLOAD_DIR``.
        original_filename: Name of the uploaded file (metadata only).
        source_type: ``"zip"`` or ``"csv"``.
        status: Initial status of the session.
        row_count: Number of rows (CSV uploads), or None.

    Returns:
        The id of the new ``PipelineSession``, or None on a database failure.
    """
    try:
        pipeline_session = PipelineSession(
            project_id=project.id,
            session_folder=session_folder,
            original_filename=(original_filename or "")[:255] or None,
            source_type=source_type,
            status=status,
            row_count=row_count,
        )
        db.add(pipeline_session)
        project.session_folder = session_folder
        db.commit()
        db.refresh(pipeline_session)
        logger.info(f"Created PipelineSession {pipeline_session.id} for project {project.id}")
        return pipeline_session.id
    except Exception as e:
        logger.error(f"Failed to create PipelineSession: {e}")
        try:
            db.rollback()
        except Exception as rollback_error:
            logger.error(f"Rollback after PipelineSession failure failed: {rollback_error}")
        return None


def _record_upload_owner(
    db: Session,
    project: Optional[Project],
    user: User,
    session_folder: str,
    original_filename: Optional[str],
    source_type: str,
    status: SessionStatus,
    row_count: Optional[int] = None,
) -> Tuple[bool, Optional[int]]:
    """
    Record the owner of a new upload: its project, or else its uploader.

    Args:
        db: Database session.
        project: Project checked by ``_upload_project_or_refusal``, or None.
        user: The uploader.
        session_folder: Folder of the upload, relative to ``UPLOAD_DIR``.
        original_filename: Name of the uploaded file (metadata only).
        source_type: ``"zip"`` or ``"csv"``.
        status: Initial status of the project session.
        row_count: Number of rows (CSV uploads), or None.

    Returns:
        ``(recorded, session_id)``: ``recorded`` is False on a database
        failure (the caller removes the upload); ``session_id`` is the new
        ``PipelineSession`` id for a project upload, else None.
    """
    if project is not None:
        session_id = _register_upload_session(
            db, project, session_folder, original_filename, source_type, status, row_count=row_count
        )
        return session_id is not None, session_id
    try:
        record_session_owner(
            db, session_folder, user.id, source_type=source_type, original_filename=original_filename
        )
    except Exception as e:
        logger.error(f"Failed to record the owner of upload '{session_folder}': {e}")
        return False, None
    return True, None


def _remove_upload(dst_dir: Optional[str], archive_path: Optional[str] = None) -> None:
    """
    Remove a refused or failed upload (session folder and uploaded archive), confined to ``UPLOAD_DIR``.

    Args:
        dst_dir: Session folder created for the upload, or None.
        archive_path: Uploaded archive kept next to it, or None.
    """
    if dst_dir:
        safe_remove_tree(dst_dir, UPLOAD_DIR)
    if archive_path:
        safe_remove_file(archive_path, UPLOAD_DIR)


def _too_large(exc: Exception) -> JSONResponse:
    """413 response of an upload above a size limit (``{"error": <message>}``)."""
    return JSONResponse(status_code=413, content={"error": str(exc)})


@router.post("/upload_zip")
async def upload_zip(
    file: UploadFile = File(...),
    project_id: Optional[int] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Upload a ZIP archive (Zotero export) and extract it under ``UPLOAD_DIR``.
    \f
    Requires an authenticated user (``Authorization`` header or
    ``access_token`` cookie). A given ``project_id`` is checked first, before
    anything is written (``_upload_project_or_refusal``: 404 unknown project,
    403 no access or read-only); the new session is then recorded in that
    project and becomes its active folder. Without ``project_id`` the upload
    is recorded as owned by the uploader (``SessionOwner``).

    Storage names are generated on the server; 413 above ``UPLOAD_MAX_MB``
    or the archive limits, 400 for an invalid archive or a member outside
    the extraction folder, 500 when the owner record cannot be written
    (nothing is left on disk in these cases).
    """
    project, refusal = _upload_project_or_refusal(db, project_id, current_user)
    if refusal is not None:
        return refusal

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dst_dir_name = new_session_folder_name(file.filename)
    try:
        zip_path = confined_path(UPLOAD_DIR, f"{dst_dir_name}.zip")
        dst_dir = confined_path(UPLOAD_DIR, dst_dir_name)
    except UnsafePathError as e:  # pragma: no cover - generated names never leave UPLOAD_DIR
        logger.error(f"Refused ZIP storage path: {e}")
        return JSONResponse(status_code=400, content={"error": "Invalid file name."})

    try:
        await asyncio.to_thread(save_upload, file.file, zip_path, max_upload_bytes())
    except UploadTooLargeError as e:
        logger.warning(f"ZIP upload refused (size): {e}")
        return _too_large(e)
    except OSError as e:
        logger.error(f"Failed to save ZIP file: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to save ZIP file.", "details": str(e)})
    logger.info(f"Uploaded ZIP saved to: {zip_path}")

    if os.path.exists(dst_dir):
        logger.warning(f"Extraction directory {dst_dir} already exists. Overwriting.")
        safe_remove_tree(dst_dir, UPLOAD_DIR)
    os.makedirs(dst_dir, exist_ok=True)

    try:
        # Use custom extraction with filename encoding fix for French/accented characters
        corrected_count = await asyncio.to_thread(extract_zip_with_encoding_fix, zip_path, dst_dir)
        logger.info(f"ZIP content extracted to initial directory: {dst_dir}")
        if corrected_count > 0:
            logger.info(f"Corrected encoding for {corrected_count} filenames (macOS/French character fix)")
    except UploadTooLargeError as e:
        logger.warning(f"ZIP upload refused (archive limits): {e}")
        _remove_upload(dst_dir, zip_path)
        return _too_large(e)
    except zipfile.BadZipFile:
        logger.error(f"Failed to extract ZIP: Bad ZIP file {zip_path}")
        _remove_upload(dst_dir, zip_path)
        return JSONResponse(status_code=400, content={"error": "Uploaded file is not a valid ZIP archive."})
    except Exception as e:
        logger.error(f"Failed to extract ZIP {zip_path} to {dst_dir}: {str(e)}")
        _remove_upload(dst_dir, zip_path)
        return JSONResponse(status_code=500, content={"error": "Failed to extract ZIP file.", "details": str(e)})

    # Single root folder -> processing path; file tree for the page
    processing_path, tree = await asyncio.to_thread(split_processing_path, dst_dir)
    if processing_path != dst_dir:
        logger.info(f"ZIP extracted to a single root folder. Processing path: {processing_path}")

    relative_processing_path = os.path.relpath(processing_path, UPLOAD_DIR)
    logger.info(f"Returning relative processing path: {relative_processing_path}")

    recorded, session_id = _record_upload_owner(
        db, project, current_user, relative_processing_path, file.filename, "zip", SessionStatus.CREATED
    )
    if not recorded:
        _remove_upload(dst_dir, zip_path)
        return JSONResponse(status_code=500, content={"error": UPLOAD_OWNER_ERROR_MESSAGE})

    return JSONResponse({
        "path": relative_processing_path,
        "tree": tree,
        "project_id": project_id,
        "session_id": session_id
    })


def _convert_uploaded_csv(temp_csv_path: str, output_csv_path: str) -> int:
    """
    Convert an uploaded CSV to the canonical ``output.csv`` and remove the source copy.

    Runs in a worker thread (pandas).

    Args:
        temp_csv_path: Server-named copy of the uploaded CSV.
        output_csv_path: ``output.csv`` of the session folder.

    Returns:
        Number of rows written.

    Raises:
        ImportError: The ingestion module cannot be imported.
        Exception: Any ingestion error (``CSVIngestionError``, pandas...).
    """
    if RAGPY_DIR not in sys.path:
        sys.path.insert(0, RAGPY_DIR)
    from ingestion import ingest_csv_to_dataframe

    logger.info(f"Converting CSV to DataFrame using ingestion module: {temp_csv_path}")
    df = ingest_csv_to_dataframe(temp_csv_path)
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"DataFrame saved as output.csv: {output_csv_path}")
    safe_remove_file(temp_csv_path)
    logger.info(f"Temporary CSV deleted: {temp_csv_path}")
    return len(df)


@router.post("/upload_csv")
async def upload_csv_endpoint(
    file: UploadFile = File(...),
    project_id: Optional[int] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Upload CSV file for direct ingestion (bypass PDF/OCR).
    \f
    Requires an authenticated user (``Authorization`` header or
    ``access_token`` cookie). A given ``project_id`` is checked first, before
    anything is written (``_upload_project_or_refusal``: 404 unknown project,
    403 no access or read-only); the new session is then recorded in that
    project and becomes its active folder, else it is recorded as owned by
    the uploader (``SessionOwner``).

    The uploaded file is stored under a server-generated name inside a new
    session folder; its client name is never used as a path (audit A01). A
    parsing failure removes that folder only.
    """
    project, refusal = _upload_project_or_refusal(db, project_id, current_user)
    if refusal is not None:
        return refusal

    file_extension = filename_extension(file.filename)
    if file_extension != ".csv":
        logger.error(f"Invalid file extension for CSV upload: {file_extension}")
        return JSONResponse(status_code=400, content={"error": "Only .csv files are accepted."})

    dst_dir_name = new_session_folder_name(file.filename)
    try:
        dst_dir = confined_path(UPLOAD_DIR, dst_dir_name)
        temp_csv_path = confined_path(dst_dir, CSV_SOURCE_FILENAME)
        output_csv_path = confined_path(dst_dir, "output.csv")
    except UnsafePathError as e:  # pragma: no cover - generated names never leave UPLOAD_DIR
        logger.error(f"Refused CSV storage path: {e}")
        return JSONResponse(status_code=400, content={"error": "Invalid file name."})
    os.makedirs(dst_dir, exist_ok=True)

    try:
        await asyncio.to_thread(save_upload, file.file, temp_csv_path, max_upload_bytes())
        logger.info(f"Uploaded CSV saved to: {temp_csv_path}")
    except UploadTooLargeError as e:
        logger.warning(f"CSV upload refused (size): {e}")
        _remove_upload(dst_dir)
        return _too_large(e)
    except Exception as e:
        logger.error(f"Failed to save CSV file: {e}")
        _remove_upload(dst_dir)
        return JSONResponse(status_code=500, content={"error": "Failed to save CSV file.", "details": str(e)})

    try:
        row_count = await asyncio.to_thread(_convert_uploaded_csv, temp_csv_path, output_csv_path)
    except ImportError as e:
        logger.error(f"Failed to import ingestion module: {e}")
        _remove_upload(dst_dir)
        return JSONResponse(status_code=500, content={"error": "Server configuration error: CSV ingestion module not found.", "details": str(e)})
    except Exception as e:
        logger.error(f"Failed to process CSV: {e}", exc_info=True)
        _remove_upload(dst_dir)
        return JSONResponse(status_code=500, content={"error": "Failed to process CSV file.", "details": str(e)})

    tree = ["output.csv"]
    relative_processing_path = os.path.relpath(dst_dir, UPLOAD_DIR)
    logger.info(f"CSV ingestion successful. Returning path: {relative_processing_path}")

    recorded, session_id = _record_upload_owner(
        db, project, current_user, relative_processing_path, file.filename, "csv", SessionStatus.EXTRACTED,
        row_count=row_count,
    )
    if not recorded:
        _remove_upload(dst_dir)
        return JSONResponse(status_code=500, content={"error": UPLOAD_OWNER_ERROR_MESSAGE})

    return JSONResponse({
        "path": relative_processing_path,
        "tree": tree,
        "message": f"CSV ingested successfully: {row_count} rows processed.",
        "project_id": project_id,
        "session_id": session_id
    })


@router.post("/upload_stage_file/{stage}")
async def upload_stage_file(
    stage: str,
    path: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Allow operators to upload intermediate artifacts for any stage.
    \f
    Requires an authenticated user (``Authorization`` header or
    ``access_token`` cookie). The session folder is checked first, before
    anything is written, for the write right (``_session_refusal_for`` of
    ``app/routes/processing.py``, on the very path this route writes to):
    400 when it does not resolve strictly under ``UPLOAD_DIR``, 403 for a
    session the user may not modify (another project, read-only member,
    another uploader, unowned legacy folder). The file is refused with 413
    above ``UPLOAD_MAX_MB``; its target name is fixed by the stage and it
    replaces the previous artifact atomically (a refused or failed upload
    leaves it untouched). While a pipeline stage runs on the session, the
    upload is refused with 409 (``job_control``, audit A12).
    """
    logger.info(f"Received upload for stage '{stage}' targeting path '{path}' with original filename '{file.filename}'")

    refusal = _session_refusal_for(
        db, path, os.path.join(UPLOAD_DIR, path), current_user, SessionAction.WRITE, upload_dir=UPLOAD_DIR
    )
    if refusal is not None:
        return _session_refusal_json(refusal)

    config = STAGE_UPLOAD_CONFIG.get(stage)
    if not config:
        logger.error(f"Unknown stage '{stage}' supplied to upload endpoint")
        return JSONResponse(status_code=400, content={"error": f"Unknown stage: {stage}"})

    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    if not os.path.isdir(absolute_processing_path):
        logger.error(f"Upload target directory does not exist: {absolute_processing_path}")
        return JSONResponse(status_code=400, content={"error": f"Processing directory not found: {path}"})

    ext = filename_extension(file.filename)
    allowed_exts = config.get("allowed_extensions", [])
    if allowed_exts and ext not in allowed_exts:
        logger.error(f"File extension '{ext}' is not allowed for stage '{stage}' (allowed: {allowed_exts})")
        return JSONResponse(status_code=400, content={"error": f"Invalid file type for stage {stage}.", "allowed_extensions": allowed_exts})

    target_filename = config["filename"]
    try:
        target_path = confined_path(absolute_processing_path, target_filename)
    except UnsafePathError as e:  # a symbolic link planted as the target name
        logger.error(f"Refused stage upload target: {e}")
        return JSONResponse(status_code=400, content={"error": f"Invalid target for stage {stage}."})

    # Never replace an artifact under a running stage of the same session (audit A12)
    ticket, busy = job_control.acquire_job(
        canonical_session_folder(path, UPLOAD_DIR) or path, job_control.GROUP_PIPELINE,
        current_user.id, admission=False,
    )
    if busy is not None:
        return _session_refusal_json(busy)
    try:
        file.file.seek(0)
        await asyncio.to_thread(save_upload_atomic, file.file, target_path, max_upload_bytes())
    except UploadTooLargeError as e:
        logger.warning(f"Stage upload refused (size): {e}")
        return _too_large(e)
    except Exception as exc:
        logger.error(f"Failed to store upload for stage '{stage}' at '{target_path}': {exc}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Failed to write uploaded file: {exc}"})
    finally:
        ticket.release()

    summary = await asyncio.to_thread(summarize_uploaded_stage, stage, target_path)
    logger.info(f"Stored uploaded file for stage '{stage}' at '{target_path}' with summary: {summary}")

    response_payload = {
        "status": "success",
        "stage": stage,
        "filename": target_filename,
        "relative_path": target_filename,
        "details": summary
    }

    return JSONResponse(response_payload)
