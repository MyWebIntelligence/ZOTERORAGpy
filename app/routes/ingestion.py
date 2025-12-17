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
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.config import RAGPY_DIR, UPLOAD_DIR
from app.database.session import get_db
from app.models.pipeline_session import PipelineSession, SessionStatus
from app.models.project import Project

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


def extract_zip_with_encoding_fix(zip_path: str, dst_dir: str) -> int:
    """
    Extract ZIP file with automatic filename encoding correction.

    This function extracts files from a ZIP archive while fixing common
    encoding issues with French/accented filenames.

    Args:
        zip_path: Path to the ZIP file
        dst_dir: Destination directory for extraction

    Returns:
        Number of files that had their names corrected

    Raises:
        zipfile.BadZipFile: If the ZIP file is invalid
    """
    corrected_count = 0

    with zipfile.ZipFile(zip_path, 'r') as z:
        for member in z.namelist():
            # Fix the filename encoding
            fixed_name = fix_zip_filename_encoding(member)

            # Track if we made corrections
            if fixed_name != member:
                corrected_count += 1
                logger.debug(f"Fixed filename: {member} -> {fixed_name}")

            # Build the target path
            target_path = os.path.join(dst_dir, fixed_name)

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


@router.post("/upload_zip")
async def upload_zip(
    file: UploadFile = File(...),
    project_id: int = Form(None)
):
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

    # Create PipelineSession if project_id is provided
    session_id = None
    if project_id:
        try:
            db = next(get_db())
            try:
                project = db.query(Project).filter(Project.id == project_id).first()
                if project:
                    pipeline_session = PipelineSession(
                        project_id=project_id,
                        session_folder=relative_processing_path,
                        original_filename=file.filename,
                        source_type="zip",
                        status=SessionStatus.CREATED
                    )
                    db.add(pipeline_session)
                    project.session_folder = relative_processing_path
                    db.commit()
                    db.refresh(pipeline_session)
                    session_id = pipeline_session.id
                    logger.info(f"Created PipelineSession {session_id} for project {project_id}")
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to create PipelineSession: {e}")

    return JSONResponse({
        "path": relative_processing_path,
        "tree": tree,
        "project_id": project_id,
        "session_id": session_id
    })


@router.post("/upload_csv")
async def upload_csv_endpoint(
    file: UploadFile = File(...),
    project_id: int = Form(None)
):
    """
    Upload CSV file for direct ingestion (bypass PDF/OCR).
    """
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
        if project_id:
            try:
                db = next(get_db())
                try:
                    project = db.query(Project).filter(Project.id == project_id).first()
                    if project:
                        pipeline_session = PipelineSession(
                            project_id=project_id,
                            session_folder=relative_processing_path,
                            original_filename=file.filename,
                            source_type="csv",
                            status=SessionStatus.EXTRACTED,
                            row_count=len(df)
                        )
                        db.add(pipeline_session)
                        project.session_folder = relative_processing_path
                        db.commit()
                        db.refresh(pipeline_session)
                        session_id = pipeline_session.id
                        logger.info(f"Created PipelineSession {session_id} for project {project_id}")
                finally:
                    db.close()
            except Exception as e:
                logger.error(f"Failed to create PipelineSession: {e}")

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
async def upload_stage_file(stage: str, path: str = Form(...), file: UploadFile = File(...)):
    """Allow operators to upload intermediate artifacts for any stage."""
    logger.info(f"Received upload for stage '{stage}' targeting path '{path}' with original filename '{file.filename}'")

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
