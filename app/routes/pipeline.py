"""
Pipeline Session Routes
=======================

This module manages the RAG pipeline sessions within projects. It handles the creation,
retrieval, and management of pipeline sessions, including file uploads (ZIP, CSV)
and status tracking.

Key Features:
- Session Management: List, create, and delete pipeline sessions.
- File Uploads: Handle ZIP archives and CSV files for ingestion.
- Status Tracking: Update and retrieve the status of processing sessions.
- File Verification: Check for the existence of intermediate files (chunks, embeddings).

Access rights (audit A01/A02, 2026-09-27):
- Reading a session (``/verify``, ``/files``, the session list) needs the
  membership of its project; changing it (``PATCH .../status``, Celery
  submissions) needs the edit right (owner or collaborator), like the
  pipeline routes (``app.core.session_access``). Deleting stays reserved to
  the project owner and the administrators.
- Uploads use server-generated storage names, bounded sizes and the shared
  ZIP extraction of ``app.core.upload_safety``; the ``PipelineSession`` row
  is written before the response, and a failed write removes the upload.
  Every removal is confined to ``UPLOAD_DIR``.
"""
import asyncio
import os
import zipfile
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request, Query, status
from fastapi import status as http_status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.upload_safety import (
    UnsafePathError,
    UploadTooLargeError,
    confined_path,
    extract_zip_with_encoding_fix,
    filename_extension,
    max_upload_bytes,
    new_session_folder_name,
    remove_session_files,
    safe_remove_file,
    safe_remove_tree,
    save_upload,
    split_processing_path,
)
from app.database.session import get_db
from app.models.user import User
from app.models.project import Project
from app.models.pipeline_session import PipelineSession, SessionStatus
from app.middleware.auth import get_current_active_user

router = APIRouter(prefix="/api/pipeline", tags=["Pipeline"])

# Directory configuration
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAGPY_DIR = os.path.dirname(APP_DIR)
UPLOAD_DIR = os.path.join(RAGPY_DIR, "uploads")


# --- Schemas ---

class SessionResponse(BaseModel):
    id: int
    project_id: int
    session_folder: str
    original_filename: Optional[str]
    status: str
    source_type: Optional[str]
    row_count: Optional[int]
    chunk_count: Optional[int]
    vector_db_type: Optional[str]
    index_name: Optional[str]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


class SessionListResponse(BaseModel):
    sessions: List[SessionResponse]
    total: int
    page: int
    per_page: int
    pages: int


# --- Helper functions ---

# Vector space fields written in the chunks only outside the default OpenAI
# space (for instance Albert bge-m3, 1024 dimensions).
EMBEDDING_SPACE_FIELDS = ("embedding_provider", "embedding_model", "embedding_dim")


def _embedding_space_fields(chunks) -> dict:
    """
    Return the vector space fields present in a chunks file, if any.

    The first chunk carrying at least one of ``EMBEDDING_SPACE_FIELDS`` gives
    the values (a file holds one space only: the connectors refuse mixed
    files). Files of the default space have none of these fields, so the
    result is empty and the session status JSON is unchanged.

    Args:
        chunks: The parsed chunks file (a list of dicts).

    Returns:
        A dict holding the fields found (possibly empty).
    """
    if not isinstance(chunks, list):
        return {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        found = {name: chunk[name] for name in EMBEDDING_SPACE_FIELDS if name in chunk}
        if found:
            return found
    return {}


def verify_project_access(db: Session, project_id: int, user: User) -> Project:
    """
    Verifies that a user has access to a specific project.

    This function checks if the project exists and if the user is either the
    project owner, a project member, or an administrator. If access is not
    granted, it raises an HTTPException.

    Args:
        db (Session): The database session.
        project_id (int): The ID of the project to check.
        user (User): The user object to verify access for.

    Returns:
        Project: The project object if the user has access.

    Raises:
        HTTPException: If the project is not found (404) or if the user does
                       not have access (403).
    """
    project = db.query(Project).filter(Project.id == project_id).first()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Projet non trouvé"
        )

    # Check access (owner, member, or admin)
    if not project.has_access(user.id) and not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès non autorisé à ce projet"
        )

    return project


SESSION_EDIT_DENIED_MESSAGE = (
    "Accès en lecture seule à cette session : seuls le propriétaire et les collaborateurs "
    "du projet peuvent la modifier."
)


def verify_project_edit(project: Project, user: User, detail: str = SESSION_EDIT_DENIED_MESSAGE) -> None:
    """
    Require the edit right on ``project`` (owner, collaborator or administrator).

    Args:
        project: Project already checked by ``verify_project_access``.
        user: The user object to verify.
        detail: Message of the 403 refusal.

    Raises:
        HTTPException: 403 when the user may only read the project (viewer).
    """
    if not project.can_edit(user.id) and not user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


def verify_session_access(
    db: Session, session_folder: str, user: User, write: bool = False
) -> PipelineSession:
    """
    Verifies that a user has access to a specific pipeline session.

    This function finds a pipeline session by its folder name and then uses
    `verify_project_access` to ensure the user has rights to the parent project.

    Args:
        db (Session): The database session.
        session_folder (str): The unique folder name of the pipeline session.
        user (User): The user object to verify access for.
        write (bool): Require the edit right (owner, collaborator or
            administrator) instead of the membership only (audit A02: a
            viewer reads, never modifies).

    Returns:
        PipelineSession: The session object if the user has access.

    Raises:
        HTTPException: If the session is not found (404) or if the user does
                       not have access to the parent project, or only reads
                       it when ``write`` is set (403).
    """
    session = db.query(PipelineSession).filter(
        PipelineSession.session_folder == session_folder
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session non trouvée"
        )

    # Verify project access
    project = verify_project_access(db, session.project_id, user)
    if write:
        verify_project_edit(project, user)

    return session


def _session_directory_path(session_folder: str) -> str:
    """
    Absolute path of a recorded session folder, confined to ``UPLOAD_DIR``.

    Args:
        session_folder: Folder of a ``PipelineSession`` row.

    Returns:
        The absolute folder path.

    Raises:
        HTTPException: 400 when the stored folder does not resolve strictly
            under ``UPLOAD_DIR`` (a legacy row with an unsafe name).
    """
    try:
        return os.path.abspath(confined_path(UPLOAD_DIR, session_folder))
    except UnsafePathError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chemin de session invalide : le dossier doit se trouver sous uploads/."
        )


# --- Routes ---

@router.get("/projects/{project_id}/sessions", response_model=SessionListResponse)
async def list_project_sessions(
    project_id: int,
    page: int = Query(1, ge=1, description="Page number (starts at 1)"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page (max 100)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Lists the pipeline sessions for a specific project with pagination.

    This endpoint retrieves a paginated list of all pipeline sessions associated
    with a given project ID. The user must have access to the project.

    Args:
        project_id (int): The ID of the project whose sessions are to be listed.
        page (int, optional): The page number for pagination. Defaults to 1.
        per_page (int, optional): The number of sessions to return per page.
                                 Defaults to 10, with a maximum of 100.
        db (Session): The database session dependency.
        current_user (User): The currently authenticated active user.

    Returns:
        SessionListResponse: A response object containing the list of sessions
                             for the requested page, along with pagination details.
    """
    # Verify access
    project = verify_project_access(db, project_id, current_user)

    # Build base query
    query = db.query(PipelineSession).filter(
        PipelineSession.project_id == project_id
    )

    # Get total count
    total = query.count()

    # Calculate pagination
    offset = (page - 1) * per_page
    pages = (total + per_page - 1) // per_page if total > 0 else 1

    # Apply pagination
    sessions = query.order_by(PipelineSession.created_at.desc())\
                   .offset(offset)\
                   .limit(per_page)\
                   .all()

    return SessionListResponse(
        sessions=[SessionResponse(
            id=s.id,
            project_id=s.project_id,
            session_folder=s.session_folder,
            original_filename=s.original_filename,
            status=s.status.value if s.status else "unknown",
            source_type=s.source_type,
            row_count=s.row_count,
            chunk_count=s.chunk_count,
            vector_db_type=s.vector_db_type,
            index_name=s.index_name,
            created_at=s.created_at,
            updated_at=s.updated_at
        ) for s in sessions],
        total=total,
        page=page,
        per_page=per_page,
        pages=pages
    )


@router.post("/projects/{project_id}/upload_zip")
async def upload_zip_to_project(
    project_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Uploads a ZIP archive to a project, creating a new pipeline session.

    This endpoint handles the upload of a ZIP file. It performs the following steps:
    1. Verifies that the user has edit permissions for the project.
    2. Creates a unique session folder for the upload.
    3. Saves and extracts the ZIP archive into this folder.
    4. Creates a `PipelineSession` record in the database, linking it to the project.
    5. Updates the project's active session to this new session.
    6. Returns the new session's path and a file tree of the extracted contents.

    Args:
        project_id (int): The ID of the project to upload the file to.
        file (UploadFile): The ZIP file being uploaded.
        db (Session): The database session dependency.
        current_user (User): The currently authenticated active user.

    Returns:
        JSONResponse: A response containing the relative path to the session folder,
                      a tree of the extracted files, the new session ID, and the project ID.

    Raises:
        HTTPException: If the user lacks permissions (403), the file is not a
                       valid ZIP (400), or an internal error occurs (500).
    """
    # Verify access (must be able to edit)
    project = verify_project_access(db, project_id, current_user)

    if not project.can_edit(current_user.id) and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Vous n'avez pas les droits pour ajouter des fichiers à ce projet"
        )

    # Server-generated storage names (audit A01): the client name is metadata only
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    session_folder = new_session_folder_name(file.filename)
    zip_path = confined_path(UPLOAD_DIR, f"{session_folder}.zip")
    dst_dir = confined_path(UPLOAD_DIR, session_folder)

    try:
        await asyncio.to_thread(save_upload, file.file, zip_path, max_upload_bytes())
    except UploadTooLargeError as e:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(e))
    except OSError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de l'enregistrement: {str(e)}"
        )

    try:
        # Extract ZIP (same checks as /upload_zip: members, sizes, encoding fix)
        if os.path.exists(dst_dir):
            safe_remove_tree(dst_dir, UPLOAD_DIR)
        os.makedirs(dst_dir, exist_ok=True)
        await asyncio.to_thread(extract_zip_with_encoding_fix, zip_path, dst_dir)

        # Handle single root directory case
        processing_path, tree = await asyncio.to_thread(split_processing_path, dst_dir)
        relative_path = os.path.relpath(processing_path, UPLOAD_DIR)

        # Create pipeline session record
        pipeline_session = PipelineSession(
            project_id=project_id,
            session_folder=relative_path,
            original_filename=(file.filename or "")[:255] or None,
            source_type="zip",
            status=SessionStatus.CREATED
        )
        db.add(pipeline_session)

        # Update project's active session
        project.session_folder = relative_path

        db.commit()
        db.refresh(pipeline_session)

        return JSONResponse({
            "path": relative_path,
            "tree": tree,
            "session_id": pipeline_session.id,
            "project_id": project_id
        })

    except UploadTooLargeError as e:
        _discard_upload(db, dst_dir, zip_path)
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(e))
    except zipfile.BadZipFile:
        _discard_upload(db, dst_dir, zip_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le fichier n'est pas une archive ZIP valide"
        )
    except Exception as e:
        _discard_upload(db, dst_dir, zip_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de l'extraction: {str(e)}"
        )


def _discard_upload(db: Session, dst_dir: Optional[str], archive_path: Optional[str] = None) -> None:
    """
    Roll back and remove a failed project upload (folder and archive), confined to ``UPLOAD_DIR``.

    Args:
        db: Database session (rolled back: no row is left for a removed folder).
        dst_dir: Session folder created for the upload, or None.
        archive_path: Uploaded archive, or None.
    """
    try:
        db.rollback()
    except Exception:
        pass
    if dst_dir:
        safe_remove_tree(dst_dir, UPLOAD_DIR)
    if archive_path:
        safe_remove_file(archive_path, UPLOAD_DIR)


@router.post("/projects/{project_id}/upload_csv")
async def upload_csv_to_project(
    project_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Uploads a CSV file to a project, creating a new pipeline session.

    This endpoint handles the upload of a CSV file. It performs the following steps:
    1. Verifies that the user has edit permissions for the project.
    2. Validates that the uploaded file has a '.csv' extension.
    3. Creates a unique session folder.
    4. Processes the CSV using the `ingestion` module to create a standardized DataFrame.
    5. Saves the processed data as 'output.csv' in the session folder.
    6. Creates a `PipelineSession` record with a status of 'EXTRACTED'.
    7. Updates the project's active session.
    8. Returns the session path and a success message.

    Args:
        project_id (int): The ID of the project to upload the file to.
        file (UploadFile): The CSV file being uploaded.
        db (Session): The database session dependency.
        current_user (User): The currently authenticated active user.

    Returns:
        JSONResponse: A response containing the session path, file tree, session ID,
                      project ID, and a success message with the row count.

    Raises:
        HTTPException: If the user lacks permissions (403), the file is not a
                       CSV (400), or an internal error occurs (500).
    """
    # Verify access (must be able to edit)
    project = verify_project_access(db, project_id, current_user)

    if not project.can_edit(current_user.id) and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Vous n'avez pas les droits pour ajouter des fichiers à ce projet"
        )

    # Validate extension
    if filename_extension(file.filename) != ".csv":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Seuls les fichiers .csv sont acceptés"
        )

    # Server-generated storage names (audit A01): the client name is metadata only
    session_folder = new_session_folder_name(file.filename)
    dst_dir = confined_path(UPLOAD_DIR, session_folder)
    temp_csv_path = confined_path(dst_dir, "_source_upload.csv")
    output_csv_path = confined_path(dst_dir, "output.csv")

    os.makedirs(dst_dir, exist_ok=True)

    try:
        # Save CSV temporarily
        await asyncio.to_thread(save_upload, file.file, temp_csv_path, max_upload_bytes())

        # Import and process with ingestion module
        import sys
        if RAGPY_DIR not in sys.path:
            sys.path.insert(0, RAGPY_DIR)

        from ingestion import ingest_csv_to_dataframe

        df = await asyncio.to_thread(ingest_csv_to_dataframe, temp_csv_path)

        # Save as output.csv
        await asyncio.to_thread(df.to_csv, output_csv_path, index=False, encoding="utf-8-sig")

        # Clean up the uploaded copy
        safe_remove_file(temp_csv_path, UPLOAD_DIR)

        relative_path = os.path.relpath(dst_dir, UPLOAD_DIR)

        # Create pipeline session record
        pipeline_session = PipelineSession(
            project_id=project_id,
            session_folder=relative_path,
            original_filename=(file.filename or "")[:255] or None,
            source_type="csv",
            status=SessionStatus.EXTRACTED,  # CSV skips extraction
            row_count=len(df)
        )
        db.add(pipeline_session)

        # Update project's active session
        project.session_folder = relative_path

        db.commit()
        db.refresh(pipeline_session)

        return JSONResponse({
            "path": relative_path,
            "tree": ["output.csv"],
            "session_id": pipeline_session.id,
            "project_id": project_id,
            "message": f"CSV importé avec succès: {len(df)} lignes"
        })

    except UploadTooLargeError as e:
        _discard_upload(db, dst_dir)
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(e))
    except Exception as e:
        _discard_upload(db, dst_dir)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du traitement CSV: {str(e)}"
        )


@router.get("/sessions/{session_folder:path}/verify")
async def verify_session(
    session_folder: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Verifies that the current user has access to a specific session folder.

    This endpoint is a security check to authorize actions on a session. It uses
    `verify_session_access` to confirm that the session exists and that the
    user has rights to the parent project.

    Args:
        session_folder (str): The path-like identifier of the session folder.
        db (Session): The database session dependency.
        current_user (User): The currently authenticated active user.

    Returns:
        JSONResponse: A response indicating that access is authorized, along
                      with the session and project IDs and ``can_edit``
                      (False for a read-only member: the pipeline routes
                      refuse its modifications with 403).

    Raises:
        HTTPException: If the session is not found (404) or if the user
                       lacks access permissions (403).
    """
    session = verify_session_access(db, session_folder, current_user)
    project = db.query(Project).filter(Project.id == session.project_id).first()
    can_edit = bool(current_user.is_admin or (project is not None and project.can_edit(current_user.id)))

    return JSONResponse({
        "authorized": True,
        "session_id": session.id,
        "project_id": session.project_id,
        "status": session.status.value if session.status else "unknown",
        "can_edit": can_edit
    })


@router.patch("/sessions/{session_id}/status")
async def update_session_status(
    session_id: int,
    status: str = Form(...),
    row_count: Optional[int] = Form(None),
    chunk_count: Optional[int] = Form(None),
    error_message: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Updates the status and metadata of a pipeline session.

    This endpoint allows for updating the state of a session as it progresses
    through the pipeline. It also allows for storing counts (rows, chunks)
    and error messages.

    Args:
        session_id (int): The ID of the session to update.
        status (str): The new status for the session (must be a valid
                      `SessionStatus` enum value).
        row_count (Optional[int], optional): The number of rows extracted.
        chunk_count (Optional[int], optional): The number of chunks generated.
        error_message (Optional[str], optional): Any error message to record.
        db (Session): The database session dependency.
        current_user (User): The currently authenticated active user.

    Returns:
        JSONResponse: A confirmation response with the new status.

    Raises:
        HTTPException: If the session is not found (404), the user lacks
                       access or may only read the project (403), or the
                       status value is invalid (400).
    """
    session = db.query(PipelineSession).filter(PipelineSession.id == session_id).first()

    if not session:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail="Session non trouvée"
        )

    # Verify project access, then the edit right (audit A02: a viewer never modifies)
    project = verify_project_access(db, session.project_id, current_user)
    verify_project_edit(project, current_user)

    # Update status
    try:
        session.status = SessionStatus(status)
    except ValueError:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=f"Statut invalide: {status}"
        )

    if row_count is not None:
        session.row_count = row_count
    if chunk_count is not None:
        session.chunk_count = chunk_count
    if error_message is not None:
        session.error_message = error_message

    if status == SessionStatus.COMPLETED.value:
        session.completed_at = datetime.utcnow()

    db.commit()

    return JSONResponse({"status": "updated", "new_status": session.status.value})


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Deletes a pipeline session and its associated files.

    This action is restricted to the project owner or an administrator.
    It removes the session record from the database and deletes the
    corresponding session folder from the filesystem.

    Args:
        session_id (int): The ID of the session to delete.
        db (Session): The database session dependency.
        current_user (User): The currently authenticated active user.

    Returns:
        JSONResponse: A confirmation message indicating successful deletion.

    Raises:
        HTTPException: If the session is not found (404) or if the user
                       is not authorized to perform the deletion (403).
    """
    session = db.query(PipelineSession).filter(PipelineSession.id == session_id).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session non trouvée"
        )

    # Get project and verify ownership (a row of a deleted project: administrators only)
    project = db.query(Project).filter(Project.id == session.project_id).first()

    if not current_user.is_admin and (project is None or project.owner_id != current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Seul le propriétaire peut supprimer une session"
        )

    # Delete session folder, empty parent and uploaded archive, confined to
    # UPLOAD_DIR (a legacy row with an unsafe folder name removes nothing)
    remove_session_files(session.session_folder, UPLOAD_DIR)

    # Update project if this was the active session
    if project is not None and project.session_folder == session.session_folder:
        project.session_folder = None

    db.delete(session)
    db.commit()

    return JSONResponse({"message": "Session supprimée avec succès"})


@router.get("/sessions/{session_folder:path}/files")
async def get_session_files(
    session_folder: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Detects existing files in a session to determine the current pipeline stage.

    This endpoint is used to resume a pipeline. It inspects the session folder
    for output files from each stage (extraction, chunking, embedding) and
    returns a summary of which stages are complete.

    Args:
        session_folder (str): The path-like identifier of the session folder.
        db (Session): The database session dependency.
        current_user (User): The currently authenticated active user.

    Returns:
        JSONResponse: An object containing the current state of the session,
                      including which files exist, their corresponding counts
                      (rows/chunks), and the determined `current_stage`.
    """
    # Verify session access
    session = verify_session_access(db, session_folder, current_user)

    session_path = _session_directory_path(session_folder)

    if not os.path.exists(session_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dossier de session non trouvé"
        )

    # Define expected files at each stage
    files_status = {
        "upload": {
            "completed": False,
            "files": []
        },
        "extraction": {
            "completed": False,
            "file": "output.csv",
            "exists": False,
            "row_count": None
        },
        "chunking": {
            "completed": False,
            "file": "output_chunks.json",
            "exists": False,
            "chunk_count": None
        },
        "dense_embedding": {
            "completed": False,
            "file": "output_chunks_with_embeddings.json",
            "exists": False,
            "chunk_count": None
        },
        "sparse_embedding": {
            "completed": False,
            "file": "output_chunks_with_embeddings_sparse.json",
            "exists": False,
            "chunk_count": None
        }
    }

    # Check upload (any files present)
    all_files = os.listdir(session_path)
    files_status["upload"]["files"] = all_files
    files_status["upload"]["completed"] = len(all_files) > 0

    # Check extraction (output.csv)
    csv_path = os.path.join(session_path, "output.csv")
    if os.path.exists(csv_path):
        files_status["extraction"]["exists"] = True
        files_status["extraction"]["completed"] = True
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, nrows=0)
            # Count rows more efficiently
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                row_count = sum(1 for _ in f) - 1  # minus header
            files_status["extraction"]["row_count"] = row_count
        except Exception:
            pass

    # Check chunking (output_chunks.json)
    chunks_path = os.path.join(session_path, "output_chunks.json")
    if os.path.exists(chunks_path):
        files_status["chunking"]["exists"] = True
        files_status["chunking"]["completed"] = True
        try:
            import json
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            files_status["chunking"]["chunk_count"] = len(chunks)
        except Exception:
            pass

    # Check dense embeddings (output_chunks_with_embeddings.json)
    dense_path = os.path.join(session_path, "output_chunks_with_embeddings.json")
    if os.path.exists(dense_path):
        files_status["dense_embedding"]["exists"] = True
        files_status["dense_embedding"]["completed"] = True
        try:
            import json
            with open(dense_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            files_status["dense_embedding"]["chunk_count"] = len(chunks)
            # embedding_* exposed only when present (non-default space)
            files_status["dense_embedding"].update(_embedding_space_fields(chunks))
        except Exception:
            pass

    # Check sparse embeddings (output_chunks_with_embeddings_sparse.json)
    sparse_path = os.path.join(session_path, "output_chunks_with_embeddings_sparse.json")
    if os.path.exists(sparse_path):
        files_status["sparse_embedding"]["exists"] = True
        files_status["sparse_embedding"]["completed"] = True
        try:
            import json
            with open(sparse_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            files_status["sparse_embedding"]["chunk_count"] = len(chunks)
            # embedding_* exposed only when present (non-default space)
            files_status["sparse_embedding"].update(_embedding_space_fields(chunks))
        except Exception:
            pass

    # Determine current stage
    current_stage = "upload"
    if files_status["sparse_embedding"]["completed"]:
        current_stage = "destination"
    elif files_status["dense_embedding"]["completed"]:
        current_stage = "sparse_embedding"
    elif files_status["chunking"]["completed"]:
        current_stage = "dense_embedding"
    elif files_status["extraction"]["completed"]:
        current_stage = "chunking"
    elif files_status["upload"]["completed"]:
        current_stage = "extraction"

    return JSONResponse({
        "session_id": session.id,
        "session_folder": session_folder,
        "project_id": session.project_id,
        "current_stage": current_stage,
        "files": files_status,
        "session_status": session.status.value if session.status else "unknown"
    })
