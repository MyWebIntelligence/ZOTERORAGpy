import logging
import os
import shutil
import zipfile
import subprocess
import uuid # Import uuid for generating unique names
import json
import csv
import sys

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)
import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from logging.handlers import RotatingFileHandler

# Import Zotero utilities
from app.utils import zotero_client, llm_note_generator, zotero_parser

# --- Path Definitions ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))  # Should be /.../__RAG/ragpy/app
RAGPY_DIR = os.path.dirname(APP_DIR)                  # Should be /.../__RAG/ragpy
LOG_DIR = os.path.join(RAGPY_DIR, "logs")
UPLOAD_DIR = os.path.join(RAGPY_DIR, "uploads")
STATIC_DIR = os.path.join(APP_DIR, "static")
TEMPLATES_DIR = os.path.join(APP_DIR, "templates")

# Ensure LOG_DIR and UPLOAD_DIR exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Logging Configuration ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# Console Handler (keeps default FastAPI/Uvicorn console logging)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
# File Handler for app.log
app_log_file = os.path.join(LOG_DIR, "app.log")
file_handler = RotatingFileHandler(app_log_file, maxBytes=10*1024*1024, backupCount=5) # 10MB per file, 5 backups
file_handler.setFormatter(log_formatter)
# Get root logger and add handlers
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
# Clear existing handlers if any (e.g., from basicConfig) to avoid duplicate console logs
if root_logger.hasHandlers():
    root_logger.handlers.clear()
root_logger.addHandler(console_handler) # Keep console output
root_logger.addHandler(file_handler)    # Add file output

logger = logging.getLogger(__name__) # Get logger for this module

logger.info(f"APP_DIR initialized to: {APP_DIR}")
logger.info(f"RAGPY_DIR initialized to: {RAGPY_DIR}")
logger.info(f"LOG_DIR initialized to: {LOG_DIR}")
logger.info(f"UPLOAD_DIR initialized to: {UPLOAD_DIR}")
logger.info(f"Application log file: {app_log_file}")
# --- End Logging and Path Setup ---

# Initialize FastAPI app
app = FastAPI()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    # Serve the main UI
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_zip")
async def upload_zip(file: UploadFile = File(...)):
    # Generate a unique prefix for the filename
    unique_id = str(uuid.uuid4().hex)[:8] # Use first 8 chars of a UUID hex
    original_filename, file_extension = os.path.splitext(file.filename)
    unique_filename = f"{unique_id}_{original_filename}{file_extension}"
    
    zip_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    # Ensure UPLOAD_DIR exists (it should from startup, but good practice)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    with open(zip_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info(f"Uploaded ZIP saved to: {zip_path}")

    # Extract ZIP contents to a directory named after the unique zip file (without extension)
    dst_dir_name = f"{unique_id}_{original_filename}"
    dst_dir = os.path.join(UPLOAD_DIR, dst_dir_name)
    
    # Ensure dst_dir doesn't already exist, or handle as needed (e.g. clear or error)
    # For simplicity, let's assume it won't exist due to unique_id. If it could, add handling.
    if os.path.exists(dst_dir):
        logger.warning(f"Extraction directory {dst_dir} already exists. Overwriting.")
        shutil.rmtree(dst_dir) # Example: remove if exists
    os.makedirs(dst_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(dst_dir)
        logger.info(f"ZIP content extracted to initial directory: {dst_dir}")
    except zipfile.BadZipFile:
        logger.error(f"Failed to extract ZIP: Bad ZIP file {zip_path}")
        # Clean up the created dst_dir if extraction fails
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        return JSONResponse(status_code=400, content={"error": "Uploaded file is not a valid ZIP archive."})
    except Exception as e:
        logger.error(f"Failed to extract ZIP {zip_path} to {dst_dir}: {str(e)}")
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        return JSONResponse(status_code=500, content={"error": "Failed to extract ZIP file.", "details": str(e)})

    # Check if the ZIP extracted into a single root directory matching original_filename (common case)
    # or if it extracted into a single directory that might be different from original_filename
    extracted_items = os.listdir(dst_dir)
    processing_path = dst_dir # Path to be used by subsequent pipeline steps

    if len(extracted_items) == 1:
        single_item_path = os.path.join(dst_dir, extracted_items[0])
        if os.path.isdir(single_item_path):
            logger.info(f"ZIP extracted to a single root folder: {extracted_items[0]}. Adjusting processing path.")
            processing_path = single_item_path
        else: # Single file extracted, not a directory - use dst_dir
            logger.info(f"ZIP extracted a single file: {extracted_items[0]}. Processing path remains {dst_dir}.")
    else: # Multiple items or no items at the root of extraction
        logger.info(f"ZIP extracted multiple items or no items into {dst_dir}. Processing path remains {dst_dir}.")


    # Build file tree relative to the actual processing_path
    tree = []
    for root, dirs, files in os.walk(processing_path):
        for d in dirs:
            tree.append(os.path.relpath(os.path.join(root, d), processing_path) + '/')
        for fname in files:
            tree.append(os.path.relpath(os.path.join(root, fname), processing_path))
            
    relative_processing_path = os.path.relpath(processing_path, UPLOAD_DIR)
    logger.info(f"Returning relative processing path: {relative_processing_path} to client. (Absolute was: {processing_path})")
    return JSONResponse({"path": relative_processing_path, "tree": tree})

@app.post("/upload_csv")
async def upload_csv_endpoint(file: UploadFile = File(...)):
    """
    Upload CSV file for direct ingestion (bypass PDF/OCR).
    Converts CSV to DataFrame using ingestion.csv_ingestion module,
    then saves it as output.csv for the pipeline.
    """
    # Generate unique directory for this CSV upload
    unique_id = str(uuid.uuid4().hex)[:8]
    original_filename, file_extension = os.path.splitext(file.filename)

    # Validate file extension
    if file_extension.lower() != ".csv":
        logger.error(f"Invalid file extension for CSV upload: {file_extension}")
        return JSONResponse(status_code=400, content={"error": "Only .csv files are accepted."})

    dst_dir_name = f"{unique_id}_{original_filename}"
    dst_dir = os.path.join(UPLOAD_DIR, dst_dir_name)

    os.makedirs(dst_dir, exist_ok=True)

    # Save uploaded CSV temporarily
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
        import sys
        if RAGPY_DIR not in sys.path:
            sys.path.insert(0, RAGPY_DIR)
        from ingestion import ingest_csv_to_dataframe
    except ImportError as e:
        logger.error(f"Failed to import ingestion module: {e}")
        return JSONResponse(status_code=500, content={"error": "Server configuration error: CSV ingestion module not found.", "details": str(e)})

    # Convert CSV to DataFrame using ingestion module
    try:
        logger.info(f"Converting CSV to DataFrame using ingestion module: {temp_csv_path}")
        df = ingest_csv_to_dataframe(temp_csv_path)

        # Save as output.csv (compatible with pipeline)
        output_csv_path = os.path.join(dst_dir, "output.csv")
        df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"DataFrame saved as output.csv: {output_csv_path}")

        # Delete temporary CSV
        os.remove(temp_csv_path)

        # Build file tree
        tree = ["output.csv"]

        relative_processing_path = os.path.relpath(dst_dir, UPLOAD_DIR)
        logger.info(f"CSV ingestion successful. Returning path: {relative_processing_path}")

        return JSONResponse({
            "path": relative_processing_path,
            "tree": tree,
            "message": f"CSV ingested successfully: {len(df)} rows processed."
        })

    except Exception as e:
        logger.error(f"Failed to process CSV: {e}", exc_info=True)
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        return JSONResponse(status_code=500, content={"error": "Failed to process CSV file.", "details": str(e)})

@app.post("/stop_all_scripts")
async def stop_all_scripts():
    command = 'pkill -SIGTERM -f "python3 scripts/rad_"'
    action_taken = False
    details = ""
    status_message = "Attempting to stop scripts..."
    try:
        # Using shell=True for pkill with pattern matching.
        # pkill returns 0 if processes were signaled, 1 if no processes matched, >1 for errors.
        process = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=900)
        
        if process.returncode == 0:
            logger.info(f"Successfully sent SIGTERM to processes matching 'python3 scripts/rad_'. stdout: {process.stdout.strip()}, stderr: {process.stderr.strip()}")
            action_taken = True
            details = f"SIGTERM signal sent. pkill stdout: '{process.stdout.strip()}', stderr: '{process.stderr.strip()}'"
            status_message = "Stop signal sent to running scripts."
        elif process.returncode == 1: # No processes matched
            logger.info("No 'python3 scripts/rad_' processes found by pkill.")
            details = "No matching script processes were found running."
            status_message = "No relevant scripts found running."
        else: # Other pkill errors
            logger.error(f"pkill command failed with return code {process.returncode}. stderr: {process.stderr.strip()}")
            details = f"pkill command error (code {process.returncode}): {process.stderr.strip()}"
            status_message = "Error attempting to stop scripts."
        
        return JSONResponse({"status": status_message, "action_taken": action_taken, "details": details})

    except subprocess.TimeoutExpired:
        logger.error("pkill command timed out.")
        return JSONResponse(status_code=500, content={"error": "Stop command timed out.", "details": "pkill command took too long to execute."})
    except Exception as e:
        logger.error(f"Exception while trying to stop scripts: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Failed to execute stop command.", "details": str(e)})

@app.post("/process_dataframe")
async def process_dataframe(path: str = Form(...)): # path is now relative to UPLOAD_DIR
    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    logger.info(f"Received relative path: '{path}', resolved to absolute: '{absolute_processing_path}'")

    # Find first JSON in directory
    try:
        if not os.path.isdir(absolute_processing_path):
            logger.error(f"Processing directory does not exist: {absolute_processing_path}")
            return JSONResponse(status_code=400, content={"error": f"Processing directory not found: {path}"})

        out_csv = os.path.join(absolute_processing_path, 'output.csv')

        # Check if output.csv already exists with texteocr content
        # Determine OCR mode: "full", "skip", or "csv_cleanup"
        ocr_mode = "full"  # Default: run full OCR via rad_dataframe.py
        existing_df = None
        removed_rows_info = []  # Info about removed rows for user feedback

        if os.path.exists(out_csv):
            try:
                existing_df = pd.read_csv(out_csv, dtype=str, keep_default_na=False)
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
                        existing_df.to_csv(out_csv, index=False, encoding='utf-8-sig')

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
        json_path = None
        if ocr_mode == "full":
            json_files = [f for f in os.listdir(absolute_processing_path) if f.lower().endswith('.json')]
            if not json_files:
                logger.error(f"No JSON file found in {absolute_processing_path} and no existing texteocr content")
                return JSONResponse(status_code=400, content={
                    "error": "No JSON file found and output.csv does not contain texteocr content."
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
                # Construct absolute path to the script using RAGPY_DIR
                # RAGPY_DIR is /.../ragpy
                project_scripts_dir = os.path.join(RAGPY_DIR, "scripts") # Scripts are in ragpy/scripts
                script_path = os.path.join(project_scripts_dir, "rad_dataframe.py")
                # Ensure it's absolute and normalized (though it should be already if RAGPY_DIR was derived from abspath)
                script_path = os.path.abspath(script_path)

                logger.info(f"  APP_DIR: {APP_DIR}")
                logger.info(f"  RAGPY_DIR: {RAGPY_DIR}")
                logger.info(f"  Calculated project_scripts_dir: {project_scripts_dir}")
                logger.info(f"  Calculated script_path (final for subprocess): {script_path}")

                # Log the exact command and arguments being used
                cmd_to_run = [
                    "python3", script_path,
                    "--json", os.path.abspath(json_path),
                    "--dir", os.path.abspath(absolute_processing_path),
                    "--output", os.path.abspath(out_csv)
                ]
                logger.info(f"Executing rad_dataframe.py with command: {' '.join(cmd_to_run)}")
                logger.info(f"  Resolved --json path: {os.path.abspath(json_path)}")
                logger.info(f"  Resolved --dir path: {os.path.abspath(absolute_processing_path)}")
                logger.info(f"  Resolved --output path: {os.path.abspath(out_csv)}")
                logger.info(f"  Existence check for --json ({os.path.abspath(json_path)}): {os.path.exists(os.path.abspath(json_path))}")
                logger.info(f"  Existence check for --dir ({os.path.abspath(absolute_processing_path)}): {os.path.exists(os.path.abspath(absolute_processing_path))}")

                # Use shell=False for better security and explicit argument passing
                result = subprocess.run([
                    "python3", script_path,
                    "--json", json_path, # Already absolute
                    "--dir", absolute_processing_path, # Pass absolute path to script
                    "--output", out_csv # Already absolute
                ], check=False, capture_output=True, text=True)  # Don't use check=True, handle ourselves

                # Manually check the return code and handle
                if result.returncode != 0:
                    logger.error(f"Extraction script failed with code {result.returncode}. stderr: {result.stderr}")
                    return JSONResponse(status_code=500, content={
                        "error": f"Extraction script failed with code {result.returncode}.",
                        "details": result.stderr,
                        "stdout": result.stdout[:500]  # Include first part of stdout for debugging
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
            df = pd.read_csv(out_csv, escapechar='\\', dtype=str, keep_default_na=False)
        except pd.errors.ParserError:
            logger.warning(f"Failed to parse CSV {out_csv} with escapechar='\\', dtype=str. Retrying without escapechar.")
            try:
                df = pd.read_csv(out_csv, dtype=str, keep_default_na=False)
            except Exception as e_inner: # Catch any error from the second read attempt
                logger.error(f"Failed to read CSV {out_csv} even with dtype=str and no escapechar: {str(e_inner)}")
                return JSONResponse(status_code=500, content={"error": "CSV parsing failed.", "details": str(e_inner)})
        except Exception as e_outer: # Catch other errors from the first read attempt
             logger.error(f"Failed to read CSV {out_csv} with escapechar='\\', dtype=str: {str(e_outer)}")
             return JSONResponse(status_code=500, content={"error": "CSV reading failed.", "details": str(e_outer)})

            
        if df.empty:
            logger.warning(f"CSV file {out_csv} is empty or contains no data after reading.")
            return JSONResponse(status_code=500, content={"error": "CSV file is empty or contains no data."})
            
        # df.head(5) ensures we don't process too much for preview
        # .fillna('') ensures that any NaN/NaT values (if keep_default_na was True or dtype=str didn't prevent them) become empty strings
        # to_dict(orient='records') should be safe if all data is string
        preview_df = df.head(5).fillna('') 
        preview = preview_df.to_dict(orient='records')
        
        # Final check for JSON serializability of the preview itself, though fillna('') and dtype=str should make it safe.
        try:
            json.dumps(preview)
        except TypeError as te:
            logger.error(f"Preview data is not JSON serializable even after dtype=str and fillna: {str(te)}")
            # Fallback: convert all values in preview records to strings manually
            safer_preview = []
            for record in preview:
                safer_record = {k: str(v) for k, v in record.items()}
                safer_preview.append(safer_record)
            preview = safer_preview
            
        # Build response with optional removed rows info
        response_data = {"csv": out_csv, "preview": preview}

        if removed_rows_info:
            response_data["warning"] = (
                f"{len(removed_rows_info)} row(s) with empty text content were removed from the CSV."
            )
            response_data["removed_rows"] = removed_rows_info

        return JSONResponse(response_data)
    except Exception as e: # Catch-all for any other unexpected error in this block
        logger.error(f"General error in processing/previewing CSV {out_csv}: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "CSV read or preview failed.", "details": str(e)})


@app.post("/upload_stage_file/{stage}")
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
    except Exception as exc:  # noqa: BLE001
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

# --- Phased Chunking Endpoints ---
BASE_CHUNK_OUTPUT_NAME = "output" # Consistent name from output.csv

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
        except Exception as exc:  # noqa: BLE001
            summary["parse_warning"] = f"Failed to analyse CSV: {exc}"
    elif summary_type == "json_list":
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                summary["count"] = len(data)
            else:
                summary["parse_warning"] = "Uploaded JSON is not a list; unable to count items."
        except Exception as exc:  # noqa: BLE001
            summary["parse_warning"] = f"Failed to analyse JSON: {exc}"

    return summary

from fastapi import Query, Body

from fastapi.middleware.cors import CORSMiddleware

# Allow CORS for local development (optional, but helps with frontend dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/get_credentials")
async def get_credentials():
    """
    Get credentials from ragpy/.env for the settings form.
    Returns a JSON with keys for:
      - OPENAI_API_KEY
      - OPENROUTER_API_KEY, OPENROUTER_DEFAULT_MODEL
      - MISTRAL_API_KEY, MISTRAL_OCR_MODEL, MISTRAL_API_BASE_URL
      - PINECONE_API_KEY, PINECONE_ENV
      - WEAVIATE_API_KEY, WEAVIATE_URL
      - QDRANT_API_KEY, QDRANT_URL
    """
    env_path = os.path.join(RAGPY_DIR, ".env") # Use RAGPY_DIR for consistency
    env_path = os.path.abspath(env_path)
    
    logger.info(f"Attempting to read .env file for get_credentials at: {env_path}")
    if not os.path.exists(env_path):
        logger.error(f".env file not found at: {env_path}")
        # Return empty strings for all keys if .env doesn't exist, so frontend form is populated correctly
        credential_keys_on_missing = [
            "OPENAI_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_DEFAULT_MODEL",
            "MISTRAL_API_KEY", "MISTRAL_OCR_MODEL", "MISTRAL_API_BASE_URL",
            "PINECONE_API_KEY", "PINECONE_ENV",
            "WEAVIATE_API_KEY", "WEAVIATE_URL", "QDRANT_API_KEY", "QDRANT_URL",
            "ZOTERO_API_KEY", "ZOTERO_USER_ID", "ZOTERO_GROUP_ID"
        ]
        empty_credentials = {k: "" for k in credential_keys_on_missing}
        logger.info("Returning empty credentials as .env file was not found.")
        return JSONResponse(status_code=200, content=empty_credentials) # Return 200 with empty strings

    # Read existing .env
    env_vars = {}
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env_vars[k.strip()] = v.strip() # Strip keys and values
        logger.info(f"Read {len(env_vars)} environment variables from {env_path}")
    except Exception as e:
        logger.error(f"Error reading .env file: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Failed to read credentials: {str(e)}"})

    # Return just the credentials keys that we need for the form
    credential_keys = [
        "OPENAI_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_DEFAULT_MODEL",
        "MISTRAL_API_KEY", "MISTRAL_OCR_MODEL", "MISTRAL_API_BASE_URL",
        "PINECONE_API_KEY", "PINECONE_ENV",
        "WEAVIATE_API_KEY", "WEAVIATE_URL",
        "QDRANT_API_KEY", "QDRANT_URL",
        "ZOTERO_API_KEY", "ZOTERO_USER_ID", "ZOTERO_GROUP_ID"
    ]
    
    # Return all requested credential keys (even if empty)
    credentials = {k: env_vars.get(k, "") for k in credential_keys}
    
    return credentials

@app.post("/save_credentials")
async def save_credentials(
    data: dict = Body(...)
):
    """
    Save credentials for OpenAI, OpenRouter, Mistral, Pinecone, Weaviate, Qdrant to ragpy/.env.
    Expects a JSON body with keys:
      - OPENAI_API_KEY
      - OPENROUTER_API_KEY, OPENROUTER_DEFAULT_MODEL
      - MISTRAL_API_KEY, MISTRAL_OCR_MODEL, MISTRAL_API_BASE_URL
      - PINECONE_API_KEY, PINECONE_ENV
      - WEAVIATE_API_KEY, WEAVIATE_URL
      - QDRANT_API_KEY, QDRANT_URL
    """
    env_path = os.path.join(RAGPY_DIR, ".env") # Use RAGPY_DIR for consistency
    env_path = os.path.abspath(env_path)
    logger.info(f"Attempting to save credentials to .env file at: {env_path}")

    # Read existing .env content to preserve other variables
    env_vars = {}
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        env_vars[k.strip()] = v.strip() # Strip keys and values
            logger.info(f"Successfully read {len(env_vars)} existing variables from {env_path}")
        except Exception as e:
            logger.error(f"Error reading existing .env file at {env_path}: {str(e)}", exc_info=True)
            # Decide if we should proceed or return an error. For now, let's proceed but log heavily.
            # If the file is corrupted, overwriting might be acceptable if we only care about these specific keys.

    # Update env_vars with data from the request
    # Only update keys that are explicitly part of the credentials form
    # This prevents accidental overwriting of other .env variables if `data` contains more keys.
    credential_keys_to_save = [
        "OPENAI_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_DEFAULT_MODEL",
        "MISTRAL_API_KEY", "MISTRAL_OCR_MODEL", "MISTRAL_API_BASE_URL",
        "PINECONE_API_KEY", "PINECONE_ENV",
        "WEAVIATE_API_KEY", "WEAVIATE_URL", "QDRANT_API_KEY", "QDRANT_URL",
        "ZOTERO_API_KEY", "ZOTERO_USER_ID", "ZOTERO_GROUP_ID"
    ]
    updated_keys = []
    for key in credential_keys_to_save:
        if key in data: # Check if the key was provided in the request
            value = data[key]
            if value is not None and value.strip() != "": # If a non-empty value is provided
                env_vars[key] = value.strip()
                updated_keys.append(key)
            elif value == "" and key in env_vars: # If an empty value is provided for an existing key, remove it
                del env_vars[key]
                updated_keys.append(key) # Indicate it was "processed" by removal
            elif value == "" and key not in env_vars: # Empty value for a non-existing key, do nothing
                pass


    # Write all variables (original + updated/new) back to .env
    try:
        with open(env_path, "w", encoding="utf-8") as f:
            for k, v in env_vars.items():
                f.write(f"{k}={v}\n")
        logger.info(f"Successfully wrote {len(env_vars)} variables to {env_path}. Updated/Removed keys from request: {updated_keys}")
        return JSONResponse({"status": "success", "message": f"Credentials saved to {env_path}. Processed keys: {updated_keys}", "saved_path": env_path})
    except Exception as e:
        logger.error(f"Failed to write to .env file at {env_path}: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Failed to write credentials to {env_path}", "details": str(e)})


@app.post("/generate_zotero_notes")
async def generate_zotero_notes(
    session: str = Form(...),
    model: str = Form(""),
    extended_analysis: str = Form("true")
):
    """
    Generate reading notes for Zotero items and push them to Zotero.

    This endpoint:
    1. Reads the output.csv from the session
    2. For each item with an itemKey:
       - Generates a reading note using LLM
       - Pushes it as a child note to Zotero
    3. Returns the status for each item

    Args:
        session: Session directory name (e.g., "abc123_MyLibrary")
        model: LLM model to use. If empty string, uses OPENROUTER_DEFAULT_MODEL from .env.
               Examples: "gpt-4o-mini" (OpenAI), "openai/gemini-2.5-flash" (OpenRouter)
        extended_analysis: "true" for exhaustive analysis (8000-12000 words), "false" for quick summary (200-300 words)

    Returns:
        JSON with list of items and their status
    """
    # Convert string to boolean
    use_extended = extended_analysis.lower() == "true"
    logger.info(f"Starting Zotero notes generation for session: {session}, model: {model or 'default'}, extended_analysis: {use_extended}")

    # Build session directory path
    session_dir = os.path.join(UPLOAD_DIR, session)

    if not os.path.exists(session_dir):
        logger.error(f"Session directory not found: {session_dir}")
        return JSONResponse(
            status_code=404,
            content={"error": f"Session directory not found: {session}"}
        )

    # Extract library info from Zotero JSON
    library_info = zotero_parser.extract_library_info_from_session(session_dir)

    if not library_info.get("success"):
        logger.error(f"Failed to extract library info: {library_info.get('error')}")
        return JSONResponse(
            status_code=400,
            content={"error": library_info.get("error")}
        )

    library_type = library_info["library_type"]
    library_id = library_info["library_id"]
    logger.info(f"Detected library: type={library_type}, id={library_id}")

    # Get Zotero credentials from environment
    zotero_api_key = os.getenv("ZOTERO_API_KEY")

    if not zotero_api_key:
        logger.error("ZOTERO_API_KEY not found in environment")
        return JSONResponse(
            status_code=400,
            content={"error": "ZOTERO_API_KEY not configured. Please set it in Settings."}
        )

    # Verify API key first
    try:
        key_info = zotero_client.verify_api_key(zotero_api_key)
        logger.info(f"Zotero API key verified: {key_info.get('username', 'Unknown user')}")
    except zotero_client.ZoteroAPIError as e:
        logger.error(f"Invalid Zotero API key: {e}")
        return JSONResponse(
            status_code=401,
            content={"error": f"Invalid Zotero API key: {e.message}"}
        )

    # Read the output.csv file
    csv_path = os.path.join(session_dir, "output.csv")

    if not os.path.exists(csv_path):
        logger.error(f"output.csv not found at: {csv_path}")
        return JSONResponse(
            status_code=404,
            content={"error": "output.csv not found. Please run 'Generate CSV' first."}
        )

    # Load CSV with pandas
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        logger.info(f"Loaded CSV with {len(df)} rows")
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to read CSV: {str(e)}"}
        )

    # Filter rows with itemKey
    # Note: itemKey might be stored in different column names depending on rad_dataframe.py
    # Let's check for common variations
    itemkey_column = None
    for col in ["itemKey", "item_key", "key"]:
        if col in df.columns:
            itemkey_column = col
            break

    if not itemkey_column:
        logger.error("No itemKey column found in CSV")
        return JSONResponse(
            status_code=400,
            content={"error": "No itemKey column found in CSV. Make sure the data comes from a Zotero export."}
        )

    # Filter items with non-null itemKey
    df_items = df[df[itemkey_column].notna()].copy()
    logger.info(f"Found {len(df_items)} items with itemKey")

    if len(df_items) == 0:
        return JSONResponse(
            status_code=400,
            content={"error": "No items with itemKey found in CSV"}
        )

    # Process each item
    results = []

    for idx, row in df_items.iterrows():
        item_key = str(row[itemkey_column])
        item_result = {
            "itemKey": item_key,
            "title": row.get("title", "Untitled"),
            "status": "pending",
            "message": ""
        }

        try:
            # Build metadata dictionary
            metadata = {
                "title": row.get("title", "Untitled"),
                "authors": row.get("authors", "N/A"),
                "date": row.get("date", "N/A"),
                "abstract": row.get("abstractNote", row.get("abstract", "")),
                "doi": row.get("doi", ""),
                "url": row.get("url", ""),
                "language": row.get("language", "")
            }

            # Get text content (texteocr)
            text_content = row.get("texteocr", "")

            if not text_content and not metadata.get("abstract"):
                item_result["status"] = "skipped"
                item_result["message"] = "No text content or abstract available"
                logger.warning(f"Skipping item {item_key}: no content")
                results.append(item_result)
                continue

            # Convert empty string to None to use default model
            llm_model = model if model else None

            if use_extended:
                # EXTENDED MODE: Generate a full reading note (HTML) as child note
                logger.info(f"Generating extended note for item {item_key}")
                sentinel, note_html = llm_note_generator.build_note_html(
                    metadata=metadata,
                    text_content=text_content,
                    model=llm_model,
                    use_llm=True,
                    extended_analysis=True
                )

                # Check if note already exists
                logger.info(f"Checking if note exists for item {item_key}")
                note_exists = zotero_client.check_note_exists(
                    library_type=library_type,
                    library_id=library_id,
                    item_key=item_key,
                    sentinel=sentinel,
                    api_key=zotero_api_key
                )

                if note_exists:
                    item_result["status"] = "exists"
                    item_result["message"] = "Note already exists (idempotent)"
                    item_result["sentinel"] = sentinel
                    logger.info(f"Note already exists for item {item_key}")
                    results.append(item_result)
                    continue

                # Create the note
                logger.info(f"Creating note for item {item_key}")
                create_result = zotero_client.create_child_note(
                    library_type=library_type,
                    library_id=library_id,
                    item_key=item_key,
                    note_html=note_html,
                    tags=["ragpy", "fiche-lecture"],
                    api_key=zotero_api_key
                )

                if create_result.get("success"):
                    item_result["status"] = "created"
                    item_result["message"] = "Note created successfully"
                    item_result["noteKey"] = create_result.get("note_key")
                    item_result["sentinel"] = sentinel
                    item_result["zotero_url"] = f"zotero://select/library/items/{create_result.get('note_key')}"
                    logger.info(f"Successfully created note for item {item_key}")
                else:
                    item_result["status"] = "error"
                    item_result["message"] = create_result.get("message", "Unknown error")
                    logger.error(f"Failed to create note for item {item_key}: {item_result['message']}")

            else:
                # SHORT MODE: Generate abstract summary and update abstractNote field
                logger.info(f"Generating abstract summary for item {item_key}")
                try:
                    abstract_text = llm_note_generator.build_abstract_text(
                        metadata=metadata,
                        text_content=text_content,
                        model=llm_model
                    )

                    # Update the item's abstractNote in Zotero
                    logger.info(f"Updating abstractNote for item {item_key}")
                    update_result = zotero_client.update_item_abstract(
                        library_type=library_type,
                        library_id=library_id,
                        item_key=item_key,
                        new_abstract=abstract_text,
                        api_key=zotero_api_key
                    )

                    if update_result.get("success"):
                        item_result["status"] = "updated"
                        item_result["message"] = "Abstract enriched successfully"
                        item_result["new_abstract_length"] = update_result.get("new_abstract_length", 0)
                        logger.info(f"Successfully updated abstract for item {item_key}")
                    else:
                        item_result["status"] = "error"
                        item_result["message"] = update_result.get("message", "Unknown error")
                        logger.error(f"Failed to update abstract for item {item_key}: {item_result['message']}")

                except ValueError as e:
                    item_result["status"] = "error"
                    item_result["message"] = str(e)
                    logger.error(f"Abstract generation error for item {item_key}: {e}")

        except zotero_client.ZoteroAPIError as e:
            item_result["status"] = "error"
            item_result["message"] = f"Zotero API error {e.status_code}: {e.message}"
            logger.error(f"Zotero API error for item {item_key}: {e}")
        except Exception as e:
            item_result["status"] = "error"
            item_result["message"] = f"Unexpected error: {str(e)}"
            logger.error(f"Unexpected error for item {item_key}: {e}", exc_info=True)

        results.append(item_result)

    # Build summary
    summary = {
        "total": len(results),
        "created": sum(1 for r in results if r["status"] == "created"),
        "updated": sum(1 for r in results if r["status"] == "updated"),
        "exists": sum(1 for r in results if r["status"] == "exists"),
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
        "errors": sum(1 for r in results if r["status"] == "error")
    }

    logger.info(f"Zotero notes generation complete: {summary}")

    return JSONResponse({
        "success": True,
        "summary": summary,
        "items": results
    })


@app.get("/get_first_chunk")
async def get_first_chunk(path: str = Query(...), filetype: str = Query(...)): # path is now relative to UPLOAD_DIR
    """
    Returns the first chunk and its embedding from the specified output file.
    filetype: one of 'initial', 'dense', 'sparse'
    """
    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    logger.info(f"get_first_chunk received relative path: '{path}', resolved to absolute: '{absolute_processing_path}' for filetype '{filetype}'")

    # Determine file path using absolute_processing_path
    if filetype == "initial":
        chunk_file = os.path.join(absolute_processing_path, f"{BASE_CHUNK_OUTPUT_NAME}_chunks.json")
    elif filetype == "dense":
        chunk_file = os.path.join(absolute_processing_path, f"{BASE_CHUNK_OUTPUT_NAME}_chunks_with_embeddings.json")
    elif filetype == "sparse":
        chunk_file = os.path.join(absolute_processing_path, f"{BASE_CHUNK_OUTPUT_NAME}_chunks_with_embeddings_sparse.json")
    else:
        return JSONResponse(status_code=400, content={"error": "Invalid filetype."})

    if not os.path.exists(chunk_file):
        return JSONResponse(status_code=404, content={"error": f"File not found: {chunk_file}"})

    try:
        with open(chunk_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data or not isinstance(data, list):
            return JSONResponse(status_code=404, content={"error": "No chunks found in file."})
        first_chunk = data[0]
        # Sanitize embedding for display
        def sanitize_embedding(emb):
            if isinstance(emb, list):
                # Truncate and round for display
                return [round(float(x), 4) for x in emb[:10]] + (["..."] if len(emb) > 10 else [])
            if isinstance(emb, dict):
                # For sparse: show a few indices/values
                indices = emb.get("indices", [])[:10]
                values = emb.get("values", [])[:10]
                return {"indices": indices, "values": [round(float(v), 4) for v in values] + (["..."] if len(values) > 10 else [])}
            return emb

        result = {
            "text": (first_chunk.get("text") or first_chunk.get("chunk_text", ""))[:500],  # Truncate for safety
            "embedding": None,
            "sparse_embedding": None
        }
        if filetype == "dense" or filetype == "sparse":
            result["embedding"] = sanitize_embedding(first_chunk.get("embedding"))
        if filetype == "sparse":
            result["sparse_embedding"] = sanitize_embedding(first_chunk.get("sparse_embedding"))
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to read chunk file: {e}"})

@app.post("/initial_text_chunking")
async def initial_text_chunking(path: str = Form(...), model: str = Form(None)): # path is now relative to UPLOAD_DIR
    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    logger.info(f"initial_text_chunking received relative path: '{path}', resolved to absolute: '{absolute_processing_path}'")
    if model:
        logger.info(f"LLM model specified for recoding: {model}")

    input_csv = os.path.join(absolute_processing_path, f"{BASE_CHUNK_OUTPUT_NAME}.csv")
    output_json = os.path.join(absolute_processing_path, f"{BASE_CHUNK_OUTPUT_NAME}_chunks.json")

    if not os.path.exists(input_csv):
        logger.error(f"Input CSV not found for initial chunking: {input_csv}")
        return JSONResponse(status_code=400, content={"error": f"Input CSV not found: {input_csv}"})

    try:
        # Ensure script path is robust
        script_path = os.path.abspath(os.path.join(RAGPY_DIR, "scripts", "rad_chunk.py"))
        logger.info(f"Executing rad_chunk.py (initial) with script: {script_path}, input: {input_csv}, output dir: {absolute_processing_path}")

        # Build command with optional --model parameter
        cmd = [
            "python3", script_path,
            "--phase", "initial",
            "--input", input_csv,
            "--output", absolute_processing_path
        ]
        if model:
            cmd.extend(["--model", model])

        # Modifié pour que stdout/stderr du script aillent vers la console uvicorn
        subprocess.run(cmd, check=True, timeout=3600) # Timeout augmenté à 1 heure (3600 secondes)
    except subprocess.TimeoutExpired:
        logger.error(f"Initial chunking script timed out after 1 hour. Path: {absolute_processing_path}")
        return JSONResponse(status_code=504, content={"error": "Initial chunking timed out (1 hour)."})
    except subprocess.CalledProcessError as e:
        # e.stderr ne sera pas peuplé si capture_output=False. L'erreur du script sera dans la console.
        logger.error(f"Initial chunking script failed with code {e.returncode}. Path: {absolute_processing_path}. Script error output should be in console.")
        return JSONResponse(status_code=500, content={"error": "Initial chunking failed.", "details": f"Script exited with code {e.returncode}. Check server console for details."})
    except Exception as e:
        logger.error(f"Unexpected error in initial chunking. Path: {absolute_processing_path}. Error: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Unexpected error in initial chunking.", "details": str(e)})

    if os.path.exists(output_json):
        try:
            with open(output_json, 'r', encoding='utf-8') as f: data = json.load(f)
            return JSONResponse({"status": "success", "file": output_json, "count": len(data)})
        except Exception as e:
            return JSONResponse({"status": "success_file_unreadable", "file": output_json, "error": str(e)})
    else:
        return JSONResponse(status_code=500, content={"error": "Output JSON from initial chunking not found.", "file": output_json})

@app.post("/dense_embedding_generation")
async def dense_embedding_generation(path: str = Form(...)): # path is now relative to UPLOAD_DIR
    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    logger.info(f"dense_embedding_generation received relative path: '{path}', resolved to absolute: '{absolute_processing_path}'")

    input_json = os.path.join(absolute_processing_path, f"{BASE_CHUNK_OUTPUT_NAME}_chunks.json")
    output_json = os.path.join(absolute_processing_path, f"{BASE_CHUNK_OUTPUT_NAME}_chunks_with_embeddings.json")

    if not os.path.exists(input_json):
        logger.error(f"Input JSON for dense embeddings not found: {input_json}")
        return JSONResponse(status_code=400, content={"error": f"Input JSON for dense embeddings not found: {input_json}"})

    try:
        # Ensure script path is robust
        script_path = os.path.abspath(os.path.join(RAGPY_DIR, "scripts", "rad_chunk.py"))
        logger.info(f"Executing rad_chunk.py (dense) with script: {script_path}, input: {input_json}, output dir: {absolute_processing_path}")

        # Modifié pour que stdout/stderr du script aillent vers la console uvicorn
        subprocess.run([
            "python3", script_path,
            "--phase", "dense",
            "--input", input_json, 
            "--output", absolute_processing_path
        ], check=True, timeout=3600) # Timeout augmenté à 1 heure (3600 secondes)
    except subprocess.TimeoutExpired:
        logger.error(f"Dense embedding script timed out after 1 hour. Path: {absolute_processing_path}")
        return JSONResponse(status_code=504, content={"error": "Dense embedding timed out (1 hour)."})
    except subprocess.CalledProcessError as e:
        logger.error(f"Dense embedding script failed with code {e.returncode}. Path: {absolute_processing_path}. Script error output should be in console.")
        return JSONResponse(status_code=500, content={"error": "Dense embedding failed.", "details": f"Script exited with code {e.returncode}. Check server console for details."})
    except Exception as e:
        logger.error(f"Unexpected error in dense embedding. Path: {absolute_processing_path}. Error: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Unexpected error in dense embedding.", "details": str(e)})

    if os.path.exists(output_json):
        try:
            with open(output_json, 'r', encoding='utf-8') as f: data = json.load(f)
            return JSONResponse({"status": "success", "file": output_json, "count": len(data)})
        except Exception as e:
            return JSONResponse({"status": "success_file_unreadable", "file": output_json, "error": str(e)})
    else:
        return JSONResponse(status_code=500, content={"error": "Output JSON from dense embedding not found.", "file": output_json})

@app.post("/sparse_embedding_generation")
async def sparse_embedding_generation(path: str = Form(...)): # path is now relative to UPLOAD_DIR
    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    logger.info(f"sparse_embedding_generation received relative path: '{path}', resolved to absolute: '{absolute_processing_path}'")

    input_json = os.path.join(absolute_processing_path, f"{BASE_CHUNK_OUTPUT_NAME}_chunks_with_embeddings.json")
    output_json = os.path.join(absolute_processing_path, f"{BASE_CHUNK_OUTPUT_NAME}_chunks_with_embeddings_sparse.json")

    if not os.path.exists(input_json):
        logger.error(f"Input JSON for sparse embeddings not found: {input_json}")
        return JSONResponse(status_code=400, content={"error": f"Input JSON for sparse embeddings not found: {input_json}"})

    try:
        # Ensure script path is robust
        script_path = os.path.abspath(os.path.join(RAGPY_DIR, "scripts", "rad_chunk.py"))
        logger.info(f"Executing rad_chunk.py (sparse) with script: {script_path}, input: {input_json}, output dir: {absolute_processing_path}")

        # Modifié pour que stdout/stderr du script aillent vers la console uvicorn
        subprocess.run([
            "python3", script_path,
            "--phase", "sparse",
            "--input", input_json,
            "--output", absolute_processing_path
        ], check=True, timeout=3600) # Timeout augmenté à 1 heure (3600 secondes)
    except subprocess.TimeoutExpired:
        logger.error(f"Sparse embedding script timed out after 1 hour. Path: {absolute_processing_path}")
        return JSONResponse(status_code=504, content={"error": "Sparse embedding timed out (1 hour)."})
    except subprocess.CalledProcessError as e:
        logger.error(f"Sparse embedding script failed with code {e.returncode}. Path: {absolute_processing_path}. Script error output should be in console.")
        return JSONResponse(status_code=500, content={"error": "Sparse embedding failed.", "details": f"Script exited with code {e.returncode}. Check server console for details."})
    except Exception as e:
        logger.error(f"Unexpected error in sparse embedding. Path: {absolute_processing_path}. Error: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Unexpected error in sparse embedding.", "details": str(e)})

    if os.path.exists(output_json):
        try:
            with open(output_json, 'r', encoding='utf-8') as f: data = json.load(f)
            return JSONResponse({"status": "success", "file": output_json, "count": len(data)})
        except Exception as e:
            return JSONResponse({"status": "success_file_unreadable", "file": output_json, "error": str(e)})
    else:
        return JSONResponse(status_code=500, content={"error": "Output JSON from sparse embedding not found.", "file": output_json})


@app.post("/upload_db") # Added decorator back
async def upload_db(
    path: str = Form(...), 
    db_choice: str = Form(...),
    pinecone_index_name: str = Form(None),
    pinecone_namespace: str = Form(None),
    weaviate_class_name: str = Form(None),
    weaviate_tenant_name: str = Form(None),
    qdrant_collection_name: str = Form(None)
): # path is now relative to UPLOAD_DIR
    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    logger.info(f"/upload_db called with relative path: '{path}', resolved to absolute: '{absolute_processing_path}', db_choice: '{db_choice}'")
    logger.info(f"Pinecone index: {pinecone_index_name}, namespace: {pinecone_namespace}, Weaviate class: {weaviate_class_name}, Qdrant collection: {qdrant_collection_name}")

    chunks_json = os.path.join(absolute_processing_path, f"{BASE_CHUNK_OUTPUT_NAME}_chunks_with_embeddings_sparse.json")
    if not os.path.exists(chunks_json):
        logger.error(f"Chunks file not found for DB upload: {chunks_json}")
        return JSONResponse(status_code=400, content={"error": f"Required chunks file not found: {chunks_json}"})

    # Load credentials
    env_path_relative = os.path.join(os.path.dirname(__file__), "..", ".env")
    env_path_abs = os.path.abspath(env_path_relative)
    
    current_env_vars = {}
    if os.path.exists(env_path_abs):
        try:
            with open(env_path_abs, "r", encoding="utf-8") as f:
                for line in f:
                    if "=" in line and not line.strip().startswith("#"):
                        k, v = line.strip().split("=", 1)
                        current_env_vars[k.strip()] = v.strip()
            logger.info(f"Loaded {len(current_env_vars)} credentials from {env_path_abs} for DB upload.")
        except Exception as e:
            logger.error(f"Error reading .env file for DB upload: {str(e)}")
            return JSONResponse(status_code=500, content={"error": f"Failed to read credentials for DB upload: {str(e)}"})
    else:
        logger.warning(f".env file not found at {env_path_abs} for DB upload. Operations might fail if API keys are required.")

    # Import necessary functions from rad_vectordb.py
    # This assumes 'scripts' is in PYTHONPATH or accessible.
    # If running with `uvicorn app.main:app` from `ragpy` directory, this should work.
    try:
        from scripts.rad_vectordb import insert_to_pinecone, insert_to_weaviate_hybrid, insert_to_qdrant
    except ImportError as e:
        logger.error(f"Failed to import vector DB functions: {e}. Ensure scripts directory is in PYTHONPATH.")
        # Attempt to add parent of scripts to sys.path if running from app dir
        import sys
        scripts_dir_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if scripts_dir_parent not in sys.path:
            sys.path.append(scripts_dir_parent)
        try:
            from scripts.rad_vectordb import insert_to_pinecone, insert_to_weaviate_hybrid, insert_to_qdrant
            logger.info("Successfully imported DB functions after sys.path modification.")
        except ImportError as e_retry:
            logger.error(f"Still failed to import vector DB functions after sys.path modification: {e_retry}")
            return JSONResponse(status_code=500, content={"error": f"Server configuration error: Cannot import DB scripts. {e_retry}"})


    try:
        if db_choice == "pinecone":
            api_key = current_env_vars.get("PINECONE_API_KEY")
            # PINECONE_ENV is typically handled by the Pinecone client library via environment variable
            # or if the index is serverless, it might not be needed for pc.Index(index_name)
            # The insert_to_pinecone function doesn't explicitly take PINECONE_ENV.
            if not api_key:
                return JSONResponse(status_code=400, content={"error": "Pinecone API Key not found in credentials."})
            if not pinecone_index_name:
                return JSONResponse(status_code=400, content={"error": "Pinecone Index Name is required."})
            namespace_value = (pinecone_namespace or "").strip() or None
            
            logger.info(f"Starting Pinecone upload to index: {pinecone_index_name}, namespace: {namespace_value} with file {chunks_json}")
            pinecone_result = insert_to_pinecone( # Capture the result
                embeddings_json_file=chunks_json,
                index_name=pinecone_index_name,
                pinecone_api_key=api_key,
                namespace=namespace_value
            )
            # Use the status and message from pinecone_result for the response
            if pinecone_result.get("status") == "success" or pinecone_result.get("status") == "success_partial_data":
                logger.info(f"Pinecone upload successful/partially successful: {pinecone_result.get('message')}")
                response_content = {
                    "status": "success", # Simplified status for frontend if needed, or pass pinecone_result["status"]
                    "message": pinecone_result.get("message", "Pinecone operation completed."),
                    "inserted_count": pinecone_result.get("inserted_count", 0),
                    "db_choice": db_choice,
                    "namespace": namespace_value
                }
                logger.info(f"Returning 200 OK for Pinecone success: {response_content}")
                return JSONResponse(response_content)
            else: # Handle error or partial_error from insert_to_pinecone
                logger.error(f"Pinecone upload failed or had issues: {pinecone_result.get('message')}")
                return JSONResponse(status_code=500, content={ # Or a more appropriate status code
                    "error": pinecone_result.get("message", "Pinecone operation failed."),
                    "status": pinecone_result.get("status", "error"), # Pass along the specific status
                    "inserted_count": pinecone_result.get("inserted_count", 0),
                    "db_choice": db_choice,
                    "namespace": namespace_value
                })

        elif db_choice == "weaviate":
            api_key = current_env_vars.get("WEAVIATE_API_KEY")
            url = current_env_vars.get("WEAVIATE_URL")
            if not api_key or not url:
                return JSONResponse(status_code=400, content={"error": "Weaviate API Key or URL not found in credentials."})
            if not weaviate_class_name:
                return JSONResponse(status_code=400, content={"error": "Weaviate Class Name is required."})
            if not weaviate_tenant_name: # tenant_name is optional in function but good to ensure it's passed if provided
                logger.warning("Weaviate Tenant Name not provided, using default if any in function.")
            
            logger.info(f"Starting Weaviate upload to URL: {url}, Class: {weaviate_class_name}, Tenant: {weaviate_tenant_name or 'default'}")
            inserted_count = insert_to_weaviate_hybrid( # Capture result
                embeddings_json_file=chunks_json,
                url=url,
                api_key=api_key,
                class_name=weaviate_class_name,
                tenant_name=weaviate_tenant_name
            )
            # insert_to_weaviate_hybrid returns an int (count) or 0 on error
            if inserted_count > 0:
                status_message = f"Weaviate upload successful. {inserted_count} items inserted."
                logger.info(status_message)
                response_content = {
                    "status": "success", 
                    "message": status_message,
                    "inserted_count": inserted_count,
                    "db_choice": db_choice
                }
                logger.info(f"Returning 200 OK for Weaviate success: {response_content}")
                return JSONResponse(response_content)
            else:
                # This path is taken if file not found, or major error in setup, or 0 items inserted from a valid file.
                # The function insert_to_weaviate_hybrid prints its own errors.
                status_message = f"Weaviate upload to class '{weaviate_class_name}' (tenant: '{weaviate_tenant_name or 'default'}') completed, but 0 items were inserted. Check server logs for details."
                logger.warning(status_message) # It might not be a full error if the file was empty but processed.
                response_content = {
                    "status": "warning", # Or "error" depending on how strict we want to be
                    "message": status_message,
                    "inserted_count": 0,
                    "db_choice": db_choice
                }
                logger.info(f"Returning 200 OK for Weaviate (0 inserted): {response_content}")
                return JSONResponse(status_code=200, content=response_content)


        elif db_choice == "qdrant":
            api_key = current_env_vars.get("QDRANT_API_KEY") # Can be None for local/unsecured
            url = current_env_vars.get("QDRANT_URL")
            if not url: # URL is essential
                return JSONResponse(status_code=400, content={"error": "Qdrant URL not found in credentials."})
            if not qdrant_collection_name:
                return JSONResponse(status_code=400, content={"error": "Qdrant Collection Name is required."})

            logger.info(f"Starting Qdrant upload to URL: {url}, Collection: {qdrant_collection_name}")
            inserted_count = insert_to_qdrant( # Capture result
                embeddings_json_file=chunks_json,
                collection_name=qdrant_collection_name,
                qdrant_url=url,
                qdrant_api_key=api_key
            )
            # insert_to_qdrant returns an int (count) or 0 on error
            if inserted_count > 0:
                status_message = f"Qdrant upload successful. {inserted_count} items inserted."
                logger.info(status_message)
                response_content = {
                    "status": "success",
                    "message": status_message,
                    "inserted_count": inserted_count,
                    "db_choice": db_choice
                }
                logger.info(f"Returning 200 OK for Qdrant success: {response_content}")
                return JSONResponse(response_content)
            else:
                status_message = f"Qdrant upload to collection '{qdrant_collection_name}' completed, but 0 items were inserted. Check server logs for details."
                logger.warning(status_message)
                response_content = {
                    "status": "warning", 
                    "message": status_message,
                    "inserted_count": 0,
                    "db_choice": db_choice
                }
                logger.info(f"Returning 200 OK for Qdrant (0 inserted): {response_content}")
                return JSONResponse(status_code=200, content=response_content)
        else:
            logger.error(f"Invalid DB choice: {db_choice}")
            return JSONResponse(status_code=400, content={"error": "Invalid database choice."})

        # This part of the original code is now largely unreachable due to returns within each db_choice block.
        # Kept for safety, but ideally refactor to ensure all paths return explicitly.
        # logger.info(f"DB upload process completed for {db_choice}.") # This log might be misleading now
        # return JSONResponse({"status": status_message, "db_choice": db_choice}) # status_message might be from the last successful DB type

    except ValueError as ve: # Catch specific ValueErrors from the DB functions (e.g. missing keys, bad URL)
        logger.error(f"ValueError during DB upload for {db_choice}: {str(ve)}")
        return JSONResponse(status_code=400, content={"error": f"Configuration error for {db_choice}: {str(ve)}"})
    except Exception as e:
        logger.error(f"An unexpected error occurred during DB upload for {db_choice}: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"An unexpected error occurred during {db_choice} upload.", "details": str(e)})

import asyncio

# ========================================
# SSE Endpoints for Progress Tracking
# ========================================

@app.post("/process_dataframe_sse")
async def process_dataframe_sse(path: str = Form(...)):
    """
    Process DataFrame with Server-Sent Events for progress tracking.
    Returns SSE stream with progress updates.
    """
    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    logger.info(f"SSE: Processing dataframe for path: '{absolute_processing_path}'")

    async def event_generator():
        try:
            # Find JSON file
            if not os.path.isdir(absolute_processing_path):
                yield f"data: {json.dumps({'type': 'error', 'message': f'Directory not found: {path}'})}\n\n"
                return

            json_files = [f for f in os.listdir(absolute_processing_path) if f.lower().endswith('.json')]
            if not json_files:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No JSON file found'})}\n\n"
                return

            json_path = os.path.join(absolute_processing_path, json_files[0])

            # Count items in JSON
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, list):
                    total_items = len(data)
                elif isinstance(data, dict) and "items" in data:
                    total_items = len(data["items"])
                else:
                    total_items = 0

                yield f"data: {json.dumps({'type': 'init', 'total': total_items, 'message': f'Found {total_items} items to process'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Failed to read JSON: {str(e)}'})}\n\n"
                return

            # Start subprocess
            out_csv = os.path.join(absolute_processing_path, 'output.csv')
            script_path = os.path.abspath(os.path.join(RAGPY_DIR, "scripts", "rad_dataframe.py"))

            # Use unbuffered Python output and set environment for real-time progress
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            process = await asyncio.create_subprocess_exec(
                "python3", "-u", script_path,
                "--json", json_path,
                "--dir", absolute_processing_path,
                "--output", out_csv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env
            )

            # Monitor progress by reading the progress file created by rad_dataframe.py
            progress_file = os.path.splitext(out_csv)[0] + ".progress.json"
            last_progress = 0

            while True:
                # Check if process has finished
                if process.returncode is not None:
                    break

                # Try to read progress from file
                try:
                    if os.path.exists(progress_file):
                        with open(progress_file, 'r', encoding='utf-8') as pf:
                            progress_data = json.load(pf)
                            current_progress = len(progress_data.get("processed_keys", []))
                            if current_progress > last_progress:
                                last_progress = current_progress
                                yield f"data: {json.dumps({'type': 'progress', 'current': current_progress, 'total': total_items})}\n\n"
                except (json.JSONDecodeError, IOError):
                    pass  # File may be being written

                # Wait a bit before checking again
                await asyncio.sleep(0.5)

                # Check process status without blocking
                try:
                    await asyncio.wait_for(process.wait(), timeout=0.1)
                except asyncio.TimeoutError:
                    pass  # Process still running

            # Final wait for process completion
            await process.wait()

            if process.returncode == 0 and os.path.exists(out_csv):
                yield f"data: {json.dumps({'type': 'complete', 'file': 'output.csv', 'message': 'Processing complete'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Process failed with code {process.returncode}'})}\n\n"

        except Exception as e:
            logger.error(f"SSE error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/generate_zotero_notes_sse")
async def generate_zotero_notes_sse(
    session: str = Form(...),
    model: str = Form(""),
    extended_analysis: str = Form("true")
):
    """
    Generate Zotero notes with Server-Sent Events for progress tracking.
    """
    use_extended = extended_analysis.lower() == "true"
    session_dir = os.path.join(UPLOAD_DIR, session)

    async def event_generator():
        try:
            if not os.path.exists(session_dir):
                yield f"data: {json.dumps({'type': 'error', 'message': f'Session not found: {session}'})}\n\n"
                return

            # Extract library info
            library_info = zotero_parser.extract_library_info_from_session(session_dir)
            if not library_info.get("success"):
                yield f"data: {json.dumps({'type': 'error', 'message': library_info.get('error')})}\n\n"
                return

            library_type = library_info["library_type"]
            library_id = library_info["library_id"]

            # Get API key
            zotero_api_key = os.getenv("ZOTERO_API_KEY")
            if not zotero_api_key:
                yield f"data: {json.dumps({'type': 'error', 'message': 'ZOTERO_API_KEY not configured'})}\n\n"
                return

            # Verify API key
            try:
                zotero_client.verify_api_key(zotero_api_key)
            except zotero_client.ZoteroAPIError as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Invalid API key: {e.message}'})}\n\n"
                return

            # Read CSV
            csv_path = os.path.join(session_dir, "output.csv")
            if not os.path.exists(csv_path):
                yield f"data: {json.dumps({'type': 'error', 'message': 'output.csv not found'})}\n\n"
                return

            df = pd.read_csv(csv_path, encoding="utf-8-sig")

            # Find itemKey column
            itemkey_column = None
            for col in ["itemKey", "item_key", "key"]:
                if col in df.columns:
                    itemkey_column = col
                    break

            if not itemkey_column:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No itemKey column found'})}\n\n"
                return

            df_items = df[df[itemkey_column].notna()].copy()
            total_items = len(df_items)

            yield f"data: {json.dumps({'type': 'init', 'total': total_items, 'message': f'Found {total_items} items'})}\n\n"

            results = {"created": 0, "exists": 0, "skipped": 0, "errors": 0}

            for idx, (row_idx, row) in enumerate(df_items.iterrows()):
                item_key = str(row[itemkey_column])
                title = row.get("title", "Untitled")

                try:
                    metadata = {
                        "title": title,
                        "authors": row.get("authors", "N/A"),
                        "date": row.get("date", "N/A"),
                        "abstract": row.get("abstractNote", row.get("abstract", "")),
                        "doi": row.get("doi", ""),
                        "url": row.get("url", ""),
                        "language": row.get("language", "")
                    }

                    text_content = row.get("texteocr", "")

                    if not text_content and not metadata.get("abstract"):
                        results["skipped"] += 1
                        yield f"data: {json.dumps({'type': 'progress', 'current': idx + 1, 'total': total_items, 'item': title, 'status': 'skipped'})}\n\n"
                        continue

                    llm_model = model if model else None

                    if use_extended:
                        # EXTENDED MODE: Generate full reading note
                        sentinel, note_html = llm_note_generator.build_note_html(
                            metadata=metadata,
                            text_content=text_content,
                            model=llm_model,
                            use_llm=True,
                            extended_analysis=True
                        )

                        note_exists = zotero_client.check_note_exists(
                            library_type=library_type,
                            library_id=library_id,
                            item_key=item_key,
                            sentinel=sentinel,
                            api_key=zotero_api_key
                        )

                        if note_exists:
                            results["exists"] += 1
                            yield f"data: {json.dumps({'type': 'progress', 'current': idx + 1, 'total': total_items, 'item': title, 'status': 'exists'})}\n\n"
                            continue

                        create_result = zotero_client.create_child_note(
                            library_type=library_type,
                            library_id=library_id,
                            item_key=item_key,
                            note_html=note_html,
                            tags=["ragpy", "fiche-lecture"],
                            api_key=zotero_api_key
                        )

                        if create_result.get("success"):
                            results["created"] += 1
                            yield f"data: {json.dumps({'type': 'progress', 'current': idx + 1, 'total': total_items, 'item': title, 'status': 'created'})}\n\n"
                        else:
                            results["errors"] += 1
                            yield f"data: {json.dumps({'type': 'progress', 'current': idx + 1, 'total': total_items, 'item': title, 'status': 'error', 'message': create_result.get('message', '')})}\n\n"

                    else:
                        # SHORT MODE: Generate abstract summary and update abstractNote
                        abstract_text = llm_note_generator.build_abstract_text(
                            metadata=metadata,
                            text_content=text_content,
                            model=llm_model
                        )

                        update_result = zotero_client.update_item_abstract(
                            library_type=library_type,
                            library_id=library_id,
                            item_key=item_key,
                            new_abstract=abstract_text,
                            api_key=zotero_api_key
                        )

                        if update_result.get("success"):
                            results["updated"] = results.get("updated", 0) + 1
                            yield f"data: {json.dumps({'type': 'progress', 'current': idx + 1, 'total': total_items, 'item': title, 'status': 'updated'})}\n\n"
                        else:
                            results["errors"] += 1
                            yield f"data: {json.dumps({'type': 'progress', 'current': idx + 1, 'total': total_items, 'item': title, 'status': 'error', 'message': update_result.get('message', '')})}\n\n"

                except Exception as e:
                    results["errors"] += 1
                    yield f"data: {json.dumps({'type': 'progress', 'current': idx + 1, 'total': total_items, 'item': title, 'status': 'error', 'message': str(e)})}\n\n"

                # Small delay to avoid overwhelming the client
                await asyncio.sleep(0.1)

            yield f"data: {json.dumps({'type': 'complete', 'summary': results})}\n\n"

        except Exception as e:
            logger.error(f"SSE Zotero error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/download_file")
async def download_file(session_path: str = Query(...), filename: str = Query(...)):
    """
    Allows downloading a file from a session directory.
    session_path: The relative path of the session within UPLOAD_DIR.
    filename: The name of the file to download.
    """
    logger.info(f"Download request for session_path='{session_path}', filename='{filename}'")
    
    # Construct the full, absolute path to the file
    # UPLOAD_DIR is the absolute path to the main uploads directory
    # session_path is relative to UPLOAD_DIR (e.g., "unique_id_original_filename")
    # filename is the name of the file within that session directory
    
    # Basic security: prevent path traversal attacks in filename and session_path
    if ".." in filename or "/" in filename or "\\" in filename:
        logger.error(f"Invalid characters in filename: {filename}")
        return JSONResponse(status_code=400, content={"error": "Invalid filename."})
    if ".." in session_path: # session_path can contain subdirs, but not '..'
        logger.error(f"Invalid characters in session_path: {session_path}")
        return JSONResponse(status_code=400, content={"error": "Invalid session path."})

    # Normalize paths to prevent issues and ensure they are within UPLOAD_DIR
    # Create the base session directory path
    base_session_dir = os.path.abspath(os.path.join(UPLOAD_DIR, session_path))
    
    # Create the full file path
    file_path_abs = os.path.abspath(os.path.join(base_session_dir, filename))

    # Security check: Ensure the resolved file_path_abs is still within UPLOAD_DIR
    # This is crucial to prevent accessing files outside the intended scope.
    # os.path.commonpath can be used, or check if UPLOAD_DIR is a prefix of file_path_abs.
    # Note: os.path.commonprefix is not suitable for path validation.
    
    # Ensure UPLOAD_DIR is absolute and normalized for comparison
    normalized_upload_dir = os.path.abspath(UPLOAD_DIR)
    
    if not file_path_abs.startswith(normalized_upload_dir):
        logger.error(f"Attempt to access file outside UPLOAD_DIR. Requested: '{file_path_abs}', UPLOAD_DIR: '{normalized_upload_dir}'")
        return JSONResponse(status_code=403, content={"error": "Access denied: File is outside the allowed directory."})

    if not os.path.exists(file_path_abs) or not os.path.isfile(file_path_abs):
        logger.error(f"File not found or is not a file: {file_path_abs}")
        return JSONResponse(status_code=404, content={"error": "File not found."})

    logger.info(f"Serving file: {file_path_abs}")
    # Use FileResponse to send the file
    # The `filename` parameter in FileResponse sets the name for the download dialog.
    return FileResponse(path=file_path_abs, filename=filename, media_type='application/octet-stream')
