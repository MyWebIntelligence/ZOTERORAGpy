import os
import subprocess
import logging
import json
import pandas as pd
import asyncio
from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.config import APP_DIR, RAGPY_DIR, UPLOAD_DIR

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/stop_all_scripts")
async def stop_all_scripts():
    command = 'pkill -SIGTERM -f "python3 scripts/rad_"'
    action_taken = False
    details = ""
    status_message = "Attempting to stop scripts..."
    try:
        # Using shell=True for pkill with pattern matching.
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


@router.post("/process_dataframe")
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
                project_scripts_dir = os.path.join(RAGPY_DIR, "scripts")
                script_path = os.path.join(project_scripts_dir, "rad_dataframe.py")
                script_path = os.path.abspath(script_path)

                logger.info(f"Executing rad_dataframe.py with command: python3 {script_path} ...")

                # Use shell=False for better security and explicit argument passing
                result = subprocess.run([
                    "python3", script_path,
                    "--json", json_path,
                    "--dir", absolute_processing_path,
                    "--output", out_csv
                ], check=False, capture_output=True, text=True)

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
            df = pd.read_csv(out_csv, escapechar='\\', dtype=str, keep_default_na=False)
        except pd.errors.ParserError:
            logger.warning(f"Failed to parse CSV {out_csv} with escapechar='\\', dtype=str. Retrying without escapechar.")
            try:
                df = pd.read_csv(out_csv, dtype=str, keep_default_na=False)
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
async def initial_text_chunking(path: str = Form(...), model: str = Form(None)):
    """
    Generate initial text chunks from the CSV file using rad_chunk.py.
    
    Args:
        path: Relative path to the session directory (contains output.csv)
        model: Optional LLM model for text recoding (e.g., "gpt-4o-mini" or "openai/gemini-2.5-flash")
    """
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
    
    # Set default model if not provided
    if not model:
        model = "gpt-4o-mini"
    
    logger.info(f"Running chunking script: {script_path}")
    logger.info(f"  Input CSV: {input_csv}")
    logger.info(f"  Output dir: {absolute_processing_path}")
    logger.info(f"  Model: {model}")
    
    try:
        # Run rad_chunk.py with phase=initial
        result = subprocess.run([
            "python3", script_path,
            "--input", input_csv,
            "--output", absolute_processing_path,
            "--phase", "initial",
            "--model", model
        ], check=False, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
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
            with open(output_chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
                chunk_count = len(chunks_data) if isinstance(chunks_data, list) else 0
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
async def process_dataframe_sse(path: str = Form(...)):
    """
    Process dataframe with Server-Sent Events for real-time progress updates.
    Streams progress from rad_dataframe.py execution.
    """
    from app.utils.sse_helpers import run_subprocess_with_sse, create_combined_parser, parse_tqdm_progress, parse_dataframe_logs
    
    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    logger.info(f"SSE dataframe processing for path: '{path}', resolved to: '{absolute_processing_path}'")
    
    if not os.path.isdir(absolute_processing_path):
        async def error_generator():
            yield f"data: {{\"type\": \"error\", \"message\": \"Processing directory not found: {path}\"}}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")
    
    # Find JSON file
    try:
        json_files = [f for f in os.listdir(absolute_processing_path) if f.lower().endswith('.json')]
    except Exception as e:
        async def error_generator():
            yield f"data: {{\"type\": \"error\", \"message\": \"Failed to list directory: {str(e)}\"}}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")
    
    if not json_files:
        async def error_generator():
            yield f"data: {{\"type\": \"error\", \"message\": \"No JSONfile found in directory\"}}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")
    
    json_path = os.path.join(absolute_processing_path, json_files[0])
    out_csv = os.path.join(absolute_processing_path, 'output.csv')
    
    # Build command
    script_path = os.path.join(RAGPY_DIR, "scripts", "rad_dataframe.py")
    cmd = ["python3", script_path, "--json", json_path, "--dir", absolute_processing_path, "--output", out_csv]
    
    logger.info(f"Executing: {' '.join(cmd)}")
    
    # Use combined parser for tqdm and custom logs
    parser = create_combined_parser(parse_tqdm_progress, parse_dataframe_logs)
    
    return StreamingResponse(
        run_subprocess_with_sse(cmd, parser, timeout=1800),
        media_type="text/event-stream"
    )


@router.post("/dense_embedding_generation")
async def dense_embedding_generation(path: str = Form(...)):
    """
    Generate dense embeddings using rad_chunk.py with phase=dense.
    """
    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    logger.info(f"Dense embedding generation for path: '{path}'")
    
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
    
    try:
        result = subprocess.run([
            "python3", script_path,
            "--input", input_chunks,
            "--output", absolute_processing_path,
            "--phase", "dense"
        ], check=False, capture_output=True, text=True, timeout=1800)
        
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
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                count = len(data) if isinstance(data, list) else 0
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
async def sparse_embedding_generation(path: str = Form(...)):
    """
    Generate sparse embeddings using rad_chunk.py with phase=sparse.
    """
    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    logger.info(f"Sparse embedding generation for path: '{path}'")
    
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
    
    try:
        result = subprocess.run([
            "python3", script_path,
            "--input", input_file,
            "--output", absolute_processing_path,
            "--phase", "sparse"
        ], check=False, capture_output=True, text=True, timeout=1800)
        
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
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                count = len(data) if isinstance(data, list) else 0
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


@router.post("/upload_db")
async def upload_db(
    path: str = Form(...),
    db_choice: str = Form(...),
    pinecone_index_name: str = Form(None),
    pinecone_namespace: str = Form(None),
    weaviate_class_name: str = Form(None),
    weaviate_tenant_name: str = Form(None),
    qdrant_collection_name: str = Form(None)
):
    """
    Upload embeddings to vector database using rad_vectordb.py.
    """
    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    logger.info(f"Vector DB upload for path: '{path}', db: {db_choice}")
    
    if not os.path.isdir(absolute_processing_path):
        return JSONResponse(status_code=400, content={"error": f"Directory not found: {path}"})
    
    # Input should be sparse embeddings file
    input_file = os.path.join(absolute_processing_path, 'output_chunks_with_embeddings_sparse.json')
    if not os.path.exists(input_file):
        return JSONResponse(status_code=400, content={
            "error": "Embeddings file not found. Please complete embedding generation first."
        })
    
    script_path = os.path.join(RAGPY_DIR, "scripts", "rad_vectordb.py")
    if not os.path.exists(script_path):
        return JSONResponse(status_code=500, content={"error": "Vector DB script not found"})
    
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
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            logger.error(f"Vector DB upload failed: {result.stderr}")
            return JSONResponse(status_code=500, content={
                "error": "Vector DB upload failed",
                "details": result.stderr[:1000]
            })
        
        # Try to parse output for count
        inserted_count = None
        try:
            # Look for patterns like "Inserted X vectors" in stdout
            if "inserted" in result.stdout.lower():
                import re
                match = re.search(r'(\d+)', result.stdout)
                if match:
                    inserted_count = int(match.group(1))
        except:
            pass
        
        response = {
            "status": "success",
            "message": f"Uploaded to {db_choice}"
        }
        if inserted_count:
            response["inserted_count"] = inserted_count
        
        return JSONResponse(response)
    except subprocess.TimeoutExpired:
        return JSONResponse(status_code=500, content={"error": "Upload timed out (1h limit)"})
    except Exception as e:
        logger.error(f"DB upload error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/generate_zotero_notes_sse")
async def generate_zotero_notes_sse(
    session: str = Form(...),
    extended_analysis: str = Form("false"),
    model: str = Form(None)
):
    """
    Generate Zotero notes with SSE progress updates.
    This implementation reads the Zotero JSON export, counts items, and streams
    dummy progress events for each item (simulating note generation).
    """
    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, session))
    logger.info(f"Zotero notes generation for session: '{session}'")
    
    async def event_generator():
        try:
            # Find Zotero JSON export (first .json file, ignoring chunks file)
            json_files = [f for f in os.listdir(absolute_processing_path) if f.endswith('.json') and f != 'output_chunks.json']
            if not json_files:
                yield f"data: {{\"type\": \"error\", \"message\": \"No Zotero JSON found. This feature requires a Zotero export.\"}}\n\n"
                return

            json_path = os.path.join(absolute_processing_path, json_files[0])
            # Load JSON content
            with open(json_path, 'r', encoding='utf-8') as jf:
                data = json.load(jf)

            # Zotero export can be a list of items or a dict with 'items'
            items = data if isinstance(data, list) else data.get('items', [])
            total_items = len(items)

            # Init event
            yield f"data: {{\"type\": \"init\", \"total\": {total_items}, \"message\": \"Starting Zotero notes generation...\"}}\n\n"

            # Simulate processing each item
            for idx, item in enumerate(items, start=1):
                # Simulate some work (could be replaced by real processing)
                await asyncio.sleep(0.05)
                # Emit progress event with current/total and a simple message
                title = item.get('title') if isinstance(item, dict) else str(item)
                safe_title = title.replace('"', '\\"') if isinstance(title, str) else ''
                # Simulated status; in real implementation this would reflect actual processing outcome
                status = "created"
                yield f"data: {{\"type\": \"progress\", \"current\": {idx}, \"total\": {total_items}, \"item\": \"{safe_title}\", \"status\": \"{status}\", \"message\": \"Processing {idx}/{total_items}: {safe_title}\"}}\n\n"

            # Completion event with summary counts
            summary = {
                "created": total_items,
                "exists": 0,
                "skipped": 0,
                "errors": 0
            }
            yield f"data: {{\"type\": \"complete\", \"message\": \"Zotero notes generation completed successfully\", \"summary\": {json.dumps(summary)}}}\n\n"
        except Exception as e:
            logger.error(f"Zotero notes SSE error: {e}", exc_info=True)
            yield f"data: {{\"type\": \"error\", \"message\": \"{str(e)}\"}}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ============================================================================
# Optional SSE versions for better UX on long-running operations
# ============================================================================

@router.post("/initial_text_chunking_sse")
async def initial_text_chunking_sse(path: str = Form(...), model: str = Form(None)):
    """
    SSE version of initial_text_chunking for real-time progress updates.
    """
    from app.utils.sse_helpers import run_subprocess_with_sse, create_combined_parser, parse_tqdm_progress, parse_chunking_logs
    
    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    logger.info(f"SSE chunking for path: '{path}'")
    
    if not os.path.isdir(absolute_processing_path):
        async def error_generator():
            yield f"data: {{\"type\": \"error\", \"message\": \"Directory not found: {path}\"}}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")
    
    input_csv = os.path.join(absolute_processing_path, 'output.csv')
    if not os.path.exists(input_csv):
        async def error_generator():
            yield f"data: {{\"type\": \"error\", \"message\": \"output.csv not found. Complete extraction first.\"}}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")
    
    script_path = os.path.join(RAGPY_DIR, "scripts", "rad_chunk.py")
    model = model or "gpt-4o-mini"
    
    cmd = [
        "python3", script_path,
        "--input", input_csv,
        "--output", absolute_processing_path,
        "--phase", "initial",
        "--model", model
    ]
    
    parser = create_combined_parser(parse_tqdm_progress, parse_chunking_logs)
    
    return StreamingResponse(
        run_subprocess_with_sse(cmd, parser, timeout=1800),
        media_type="text/event-stream"
    )


@router.post("/dense_embedding_generation_sse")
async def dense_embedding_generation_sse(path: str = Form(...)):
    """
    SSE version of dense_embedding_generation for real-time progress updates.
    """
    from app.utils.sse_helpers import run_subprocess_with_sse, create_combined_parser, parse_tqdm_progress, parse_chunking_logs
    
    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    input_chunks = os.path.join(absolute_processing_path, 'output_chunks.json')
    
    if not os.path.exists(input_chunks):
        async def error_generator():
            yield f"data: {{\"type\": \"error\", \"message\": \"output_chunks.json not found\"}}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")
    
    script_path = os.path.join(RAGPY_DIR, "scripts", "rad_chunk.py")
    cmd = [
        "python3", script_path,
        "--input", input_chunks,
        "--output", absolute_processing_path,
        "--phase", "dense"
    ]
    
    parser = create_combined_parser(parse_tqdm_progress, parse_chunking_logs)
    
    return StreamingResponse(
        run_subprocess_with_sse(cmd, parser, timeout=1800),
        media_type="text/event-stream"
    )


@router.post("/sparse_embedding_generation_sse")
async def sparse_embedding_generation_sse(path: str = Form(...)):
    """
    SSE version of sparse_embedding_generation for real-time progress updates.
    """
    from app.utils.sse_helpers import run_subprocess_with_sse, create_combined_parser, parse_tqdm_progress, parse_chunking_logs
    
    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    input_file = os.path.join(absolute_processing_path, 'output_chunks_with_embeddings.json')
    
    if not os.path.exists(input_file):
        async def error_generator():
            yield f"data: {{\"type\": \"error\", \"message\": \"output_chunks_with_embeddings.json not found\"}}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")
    
    script_path = os.path.join(RAGPY_DIR, "scripts", "rad_chunk.py")
    cmd = [
        "python3", script_path,
        "--input", input_file,
        "--output", absolute_processing_path,
        "--phase", "sparse"
    ]
    
    parser = create_combined_parser(parse_tqdm_progress, parse_chunking_logs)
    
    return StreamingResponse(
        run_subprocess_with_sse(cmd, parser, timeout=1800),
        media_type="text/event-stream"
    )

