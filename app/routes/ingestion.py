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
  may only read; owner, collaborator and administrator go on. Without
  ``project_id`` the historical behaviour is kept for the authenticated
  user (folder created, no ``PipelineSession``).
"""
import os
import shutil
import zipfile
import uuid
import logging
import sys
import json
import csv
import unicodedata
from typing import Optional, Tuple

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.config import RAGPY_DIR, UPLOAD_DIR
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


# =============================================================================
# ZIP Filename Encoding Fix
# =============================================================================
# macOS Zotero exports use NFD (decomposed) Unicode for filenames.
# When extracted on some systems, the encoding gets corrupted.
# This map fixes common corruption patterns.

FILENAME_CORRUPTION_MAP = {
    # Combining accent sequences (macOS NFD decomposition artifacts)
    'e\u0308\u0300': 'è', 'e\u0308': 'é', 'e\u0301': 'é', 'e\u0300': 'è',
    'e╠ü': 'é', 'e╠Ç': 'è',
    'a╠Ç': 'à', 'a\u0300': 'à',
    'i\u0302': 'î', 'o\u0302': 'ô',
    'u\u0300': 'ù', 'u\u0302': 'û',
    'c╠º': 'ç', 'c\u0327': 'ç',
    # Windows CP1252/UTF-8 misinterpretation artifacts
    'ΓÇÖ': "'", 'ΓÇô': '–', 'ΓÇ£': '"', 'ΓÇ¥': '"',
    'ΓÇª': '…', 'ΓÇö': '—',
    '┬½': '«', '┬╗': '»',
    'Γé¼': '€',
    # Double-encoded accents
    'é╠ü': 'é', 'è╠Ç': 'è',
    # Explicit NFD sequences
    '\u0065\u0301': 'é', '\u0065\u0300': 'è',
    '\u0061\u0300': 'à', '\u0063\u0327': 'ç',
}


def fix_zip_filename_encoding(filename: str) -> str:
    """
    Fix corrupted filename encoding from ZIP extraction.

    This handles common issues when extracting ZIP files created on macOS
    with French/accented characters:
    - NFD (decomposed) Unicode normalization
    - CP437/UTF-8 encoding mismatches
    - Windows codepage artifacts

    Args:
        filename: The potentially corrupted filename

    Returns:
        The corrected filename with proper Unicode (NFC normalized)
    """
    fixed = filename

    # Apply corruption fixes (longer patterns first for correct replacement)
    for corrupt, correct in sorted(FILENAME_CORRUPTION_MAP.items(), key=lambda x: -len(x[0])):
        fixed = fixed.replace(corrupt, correct)

    # Normalize to NFC (composed form) - this is the standard for most systems
    fixed = unicodedata.normalize('NFC', fixed)

    return fixed


class UnsafeZipMemberError(zipfile.BadZipFile):
    """A ZIP member whose name would be extracted outside the destination directory.

    Subclass of ``zipfile.BadZipFile``: callers that already refuse invalid
    archives (``/upload_zip`` answers 400) refuse these the same way.
    """


def _zip_member_target(dst_dir: str, member_name: str) -> str:
    """
    Target path of a ZIP member, refused when it does not stay under ``dst_dir``.

    ``..`` components and absolute names (which ``os.path.join`` would keep
    as is) are resolved with symbolic links followed, then compared
    component by component with the resolved destination.

    Args:
        dst_dir: Extraction directory.
        member_name: Member name after the encoding fix.

    Returns:
        ``os.path.join(dst_dir, member_name)``.

    Raises:
        UnsafeZipMemberError: When the resolved target is not ``dst_dir`` or
            a path under it (or cannot be resolved).
    """
    target_path = os.path.join(dst_dir, member_name)
    root = os.path.realpath(dst_dir)
    try:
        resolved = os.path.realpath(target_path)
        inside = os.path.commonpath([root, resolved]) == root
    except ValueError:  # embedded NUL byte, or different drives (Windows)
        inside = False
    if not inside:
        logger.warning(f"Refused ZIP member outside the extraction directory: {member_name!r}")
        raise UnsafeZipMemberError(f"ZIP member outside the extraction directory: {member_name!r}")
    return target_path


def extract_zip_with_encoding_fix(zip_path: str, dst_dir: str) -> int:
    """
    Extract ZIP file with automatic filename encoding correction.

    This function extracts files from a ZIP archive while fixing common
    encoding issues with French/accented filenames. Every member name is
    checked before anything is written: when one would land outside
    ``dst_dir`` (zip slip), the whole archive is refused.

    Args:
        zip_path: Path to the ZIP file
        dst_dir: Destination directory for extraction

    Returns:
        Number of files that had their names corrected

    Raises:
        zipfile.BadZipFile: If the ZIP file is invalid, including
            ``UnsafeZipMemberError`` for a member outside ``dst_dir``
    """
    corrected_count = 0

    with zipfile.ZipFile(zip_path, 'r') as z:
        members = [(member, fix_zip_filename_encoding(member)) for member in z.namelist()]
        targets = [_zip_member_target(dst_dir, fixed_name) for _member, fixed_name in members]

        for (member, fixed_name), target_path in zip(members, targets):
            # Track if we made corrections
            if fixed_name != member:
                corrected_count += 1
                logger.debug(f"Fixed filename: {member} -> {fixed_name}")

            # Handle directories
            if member.endswith('/'):
                os.makedirs(target_path, exist_ok=True)
                continue

            # Ensure parent directory exists
            parent_dir = os.path.dirname(target_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            # Extract the file content and write with corrected name
            with z.open(member) as src, open(target_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)

    if corrected_count > 0:
        logger.info(f"Fixed encoding for {corrected_count} filenames during ZIP extraction")

    return corrected_count


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

    A database failure is logged and leaves the upload in place, as before
    (the response then carries ``session_id: null``).

    Args:
        db: Database session.
        project: Project checked by ``_upload_project_or_refusal``.
        session_folder: Folder of the upload, relative to ``UPLOAD_DIR``.
        original_filename: Name of the uploaded file.
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
            original_filename=original_filename,
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
    is not attached to any project.
    """
    project, refusal = _upload_project_or_refusal(db, project_id, current_user)
    if refusal is not None:
        return refusal

    # Generate a unique prefix for the filename
    unique_id = str(uuid.uuid4().hex)[:8]
    original_filename, file_extension = os.path.splitext(file.filename)
    unique_filename = f"{unique_id}_{original_filename}{file_extension}"

    zip_path = os.path.join(UPLOAD_DIR, unique_filename)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    with open(zip_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info(f"Uploaded ZIP saved to: {zip_path}")

    # Extract ZIP contents
    dst_dir_name = f"{unique_id}_{original_filename}"
    dst_dir = os.path.join(UPLOAD_DIR, dst_dir_name)
    
    if os.path.exists(dst_dir):
        logger.warning(f"Extraction directory {dst_dir} already exists. Overwriting.")
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    try:
        # Use custom extraction with filename encoding fix for French/accented characters
        corrected_count = extract_zip_with_encoding_fix(zip_path, dst_dir)
        logger.info(f"ZIP content extracted to initial directory: {dst_dir}")
        if corrected_count > 0:
            logger.info(f"Corrected encoding for {corrected_count} filenames (macOS/French character fix)")
    except zipfile.BadZipFile:
        logger.error(f"Failed to extract ZIP: Bad ZIP file {zip_path}")
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        return JSONResponse(status_code=400, content={"error": "Uploaded file is not a valid ZIP archive."})
    except Exception as e:
        logger.error(f"Failed to extract ZIP {zip_path} to {dst_dir}: {str(e)}")
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        return JSONResponse(status_code=500, content={"error": "Failed to extract ZIP file.", "details": str(e)})

    # Check extraction structure
    extracted_items = os.listdir(dst_dir)
    processing_path = dst_dir

    if len(extracted_items) == 1:
        single_item_path = os.path.join(dst_dir, extracted_items[0])
        if os.path.isdir(single_item_path):
            logger.info(f"ZIP extracted to a single root folder: {extracted_items[0]}. Adjusting processing path.")
            processing_path = single_item_path
        else:
            logger.info(f"ZIP extracted a single file: {extracted_items[0]}. Processing path remains {dst_dir}.")
    else:
        logger.info(f"ZIP extracted multiple items or no items into {dst_dir}. Processing path remains {dst_dir}.")

    # Build file tree
    tree = []
    for root, dirs, files in os.walk(processing_path):
        for d in dirs:
            tree.append(os.path.relpath(os.path.join(root, d), processing_path) + '/')
        for fname in files:
            tree.append(os.path.relpath(os.path.join(root, fname), processing_path))
            
    relative_processing_path = os.path.relpath(processing_path, UPLOAD_DIR)
    logger.info(f"Returning relative processing path: {relative_processing_path}")

    # Create PipelineSession if project_id is provided (checked above)
    session_id = None
    if project is not None:
        session_id = _register_upload_session(
            db, project, relative_processing_path, file.filename, "zip", SessionStatus.CREATED
        )

    return JSONResponse({
        "path": relative_processing_path,
        "tree": tree,
        "project_id": project_id,
        "session_id": session_id
    })


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
    project and becomes its active folder.
    """
    project, refusal = _upload_project_or_refusal(db, project_id, current_user)
    if refusal is not None:
        return refusal

    unique_id = str(uuid.uuid4().hex)[:8]
    original_filename, file_extension = os.path.splitext(file.filename)

    if file_extension.lower() != ".csv":
        logger.error(f"Invalid file extension for CSV upload: {file_extension}")
        return JSONResponse(status_code=400, content={"error": "Only .csv files are accepted."})

    dst_dir_name = f"{unique_id}_{original_filename}"
    dst_dir = os.path.join(UPLOAD_DIR, dst_dir_name)
    os.makedirs(dst_dir, exist_ok=True)

    temp_csv_path = os.path.join(dst_dir, f"{original_filename}.csv")
    try:
        with open(temp_csv_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"Uploaded CSV saved to: {temp_csv_path}")
    except Exception as e:
        logger.error(f"Failed to save CSV file: {e}")
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        return JSONResponse(status_code=500, content={"error": "Failed to save CSV file.", "details": str(e)})

    # Import ingestion module
    try:
        if RAGPY_DIR not in sys.path:
            sys.path.insert(0, RAGPY_DIR)
        from ingestion import ingest_csv_to_dataframe
    except ImportError as e:
        logger.error(f"Failed to import ingestion module: {e}")
        return JSONResponse(status_code=500, content={"error": "Server configuration error: CSV ingestion module not found.", "details": str(e)})

    try:
        logger.info(f"Converting CSV to DataFrame using ingestion module: {temp_csv_path}")
        df = ingest_csv_to_dataframe(temp_csv_path)

        output_csv_path = os.path.join(dst_dir, "output.csv")
        df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"DataFrame saved as output.csv: {output_csv_path}")

        if os.path.abspath(temp_csv_path) != os.path.abspath(output_csv_path):
            os.remove(temp_csv_path)
            logger.info(f"Temporary CSV deleted: {temp_csv_path}")

        tree = ["output.csv"]
        relative_processing_path = os.path.relpath(dst_dir, UPLOAD_DIR)
        logger.info(f"CSV ingestion successful. Returning path: {relative_processing_path}")

        session_id = None
        if project is not None:
            session_id = _register_upload_session(
                db, project, relative_processing_path, file.filename, "csv", SessionStatus.EXTRACTED,
                row_count=len(df)
            )

        return JSONResponse({
            "path": relative_processing_path,
            "tree": tree,
            "message": f"CSV ingested successfully: {len(df)} rows processed.",
            "project_id": project_id,
            "session_id": session_id
        })

    except Exception as e:
        logger.error(f"Failed to process CSV: {e}", exc_info=True)
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        return JSONResponse(status_code=500, content={"error": "Failed to process CSV file.", "details": str(e)})


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
    anything is written, with the check of the pipeline routes
    (``_session_refusal_for`` of ``app/routes/processing.py``, on the very
    path this route writes to): 400 when it does not resolve strictly under
    ``UPLOAD_DIR``, 403 for a registered session of a project the non-admin
    user cannot access. A folder unrelated to any ``PipelineSession`` row
    keeps the historical behaviour.
    """
    logger.info(f"Received upload for stage '{stage}' targeting path '{path}' with original filename '{file.filename}'")

    refusal = _session_refusal_for(db, path, os.path.join(UPLOAD_DIR, path), current_user)
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

    _, ext = os.path.splitext(file.filename or "")
    ext = ext.lower()
    allowed_exts = config.get("allowed_extensions", [])
    if allowed_exts and ext not in allowed_exts:
        logger.error(f"File extension '{ext}' is not allowed for stage '{stage}' (allowed: {allowed_exts})")
        return JSONResponse(status_code=400, content={"error": f"Invalid file type for stage {stage}.", "allowed_extensions": allowed_exts})

    target_filename = config["filename"]
    target_path = os.path.join(absolute_processing_path, target_filename)

    try:
        file.file.seek(0)
        with open(target_path, "wb") as handled:
            shutil.copyfileobj(file.file, handled)
    except Exception as exc:
        logger.error(f"Failed to store upload for stage '{stage}' at '{target_path}': {exc}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Failed to write uploaded file: {exc}"})

    summary = summarize_uploaded_stage(stage, target_path)
    logger.info(f"Stored uploaded file for stage '{stage}' at '{target_path}' with summary: {summary}")

    response_payload = {
        "status": "success",
        "stage": stage,
        "filename": target_filename,
        "relative_path": target_filename,
        "details": summary
    }

    return JSONResponse(response_payload)
