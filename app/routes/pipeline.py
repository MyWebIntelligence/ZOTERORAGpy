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
"""
import os
import shutil
import uuid
import zipfile
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

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


def verify_session_access(db: Session, session_folder: str, user: User) -> PipelineSession:
    """
    Verifies that a user has access to a specific pipeline session.

    This function finds a pipeline session by its folder name and then uses
    `verify_project_access` to ensure the user has rights to the parent project.

    Args:
        db (Session): The database session.
        session_folder (str): The unique folder name of the pipeline session.
        user (User): The user object to verify access for.

    Returns:
        PipelineSession: The session object if the user has access.

    Raises:
        HTTPException: If the session is not found (404) or if the user does
                       not have access to the parent project (403).
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
    verify_project_access(db, session.project_id, user)

    return session


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

    # Generate unique session folder
    unique_id = str(uuid.uuid4().hex)[:8]
    original_filename, file_extension = os.path.splitext(file.filename)
    session_folder = f"{unique_id}_{original_filename}"

    zip_path = os.path.join(UPLOAD_DIR, f"{session_folder}{file_extension}")
    dst_dir = os.path.join(UPLOAD_DIR, session_folder)

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    try:
        # Save ZIP file
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Extract ZIP
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        os.makedirs(dst_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(dst_dir)

        # Handle single root directory case
        extracted_items = os.listdir(dst_dir)
        processing_path = dst_dir

        if len(extracted_items) == 1:
            single_item_path = os.path.join(dst_dir, extracted_items[0])
            if os.path.isdir(single_item_path):
                processing_path = single_item_path

        relative_path = os.path.relpath(processing_path, UPLOAD_DIR)

        # Create pipeline session record
        pipeline_session = PipelineSession(
            project_id=project_id,
            session_folder=relative_path,
            original_filename=file.filename,
            source_type="zip",
            status=SessionStatus.CREATED
        )
        db.add(pipeline_session)

        # Update project's active session
        project.session_folder = relative_path

        db.commit()
        db.refresh(pipeline_session)

        # Build file tree
        tree = []
        for root, dirs, files in os.walk(processing_path):
            for d in dirs:
                tree.append(os.path.relpath(os.path.join(root, d), processing_path) + '/')
            for fname in files:
                tree.append(os.path.relpath(os.path.join(root, fname), processing_path))

        return JSONResponse({
            "path": relative_path,
            "tree": tree,
            "session_id": pipeline_session.id,
            "project_id": project_id
        })

    except zipfile.BadZipFile:
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le fichier n'est pas une archive ZIP valide"
        )
    except Exception as e:
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de l'extraction: {str(e)}"
        )


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
    original_filename, file_extension = os.path.splitext(file.filename)
    if file_extension.lower() != ".csv":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Seuls les fichiers .csv sont acceptés"
        )

    # Generate unique session folder
    unique_id = str(uuid.uuid4().hex)[:8]
    session_folder = f"{unique_id}_{original_filename}"
    dst_dir = os.path.join(UPLOAD_DIR, session_folder)

    os.makedirs(dst_dir, exist_ok=True)

    try:
        # Save CSV temporarily
        temp_csv_path = os.path.join(dst_dir, f"{original_filename}.csv")
        with open(temp_csv_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Import and process with ingestion module
        import sys
        if RAGPY_DIR not in sys.path:
            sys.path.insert(0, RAGPY_DIR)

        from ingestion import ingest_csv_to_dataframe
        import pandas as pd

        df = ingest_csv_to_dataframe(temp_csv_path)

        # Save as output.csv
        output_csv_path = os.path.join(dst_dir, "output.csv")
        df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

        # Clean up temp file if different
        if os.path.abspath(temp_csv_path) != os.path.abspath(output_csv_path):
            os.remove(temp_csv_path)

        relative_path = os.path.relpath(dst_dir, UPLOAD_DIR)

        # Create pipeline session record
        pipeline_session = PipelineSession(
            project_id=project_id,
            session_folder=relative_path,
            original_filename=file.filename,
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

    except Exception as e:
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
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
                      with the session and project IDs.

    Raises:
        HTTPException: If the session is not found (404) or if the user
                       lacks access permissions (403).
    """
    session = verify_session_access(db, session_folder, current_user)

    return JSONResponse({
        "authorized": True,
        "session_id": session.id,
        "project_id": session.project_id,
        "status": session.status.value if session.status else "unknown"
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
                       access (403), or the status value is invalid (400).
    """
    session = db.query(PipelineSession).filter(PipelineSession.id == session_id).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session non trouvée"
        )

    # Verify project access
    verify_project_access(db, session.project_id, current_user)

    # Update status
    try:
        session.status = SessionStatus(status)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
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

    # Get project and verify ownership
    project = db.query(Project).filter(Project.id == session.project_id).first()

    if project.owner_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Seul le propriétaire peut supprimer une session"
        )

    # Delete files
    session_path = os.path.join(UPLOAD_DIR, session.session_folder)
    if os.path.exists(session_path):
        shutil.rmtree(session_path)

    # Update project if this was the active session
    if project.session_folder == session.session_folder:
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

    session_path = os.path.join(UPLOAD_DIR, session_folder)

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
