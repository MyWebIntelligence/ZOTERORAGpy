import os
import subprocess
import logging
import json
import pandas as pd
import asyncio
from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.config import APP_DIR, RAGPY_DIR, UPLOAD_DIR
from app.services.process_manager import process_manager

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter()


async def run_tracked_subprocess(
    cmd: list,
    session_folder: str,
    timeout: int = 1800
) -> subprocess.CompletedProcess:
    """
    Lance un subprocess ASYNC avec tracking PID pour permettre l'arrêt par session.

    IMPORTANT: Cette fonction est async pour ne pas bloquer le serveur,
    permettant ainsi de traiter les requêtes de stop en parallèle.

    Args:
        cmd: Commande à exécuter (liste d'arguments)
        session_folder: Identifiant de la session pour le tracking
        timeout: Timeout en secondes (défaut: 30 min)

    Returns:
        subprocess.CompletedProcess avec stdout, stderr, returncode
    """
    # Utiliser asyncio.create_subprocess_exec pour ne pas bloquer
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    # Enregistrer le PID pour permettre l'arrêt
    process_manager.register(session_folder, process.pid)
    logger.info(f"Started async process PID {process.pid} for session '{session_folder}'")

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
        process.kill()
        await process.communicate()  # Clean up
        raise subprocess.TimeoutExpired(cmd, timeout)
    finally:
        # Toujours désenregistrer le PID à la fin
        process_manager.unregister(session_folder, process.pid)
        logger.info(f"Process PID {process.pid} finished for session '{session_folder}'")


@router.post("/stop_all_scripts")
async def stop_all_scripts(session: str = Form(...)):
    """
    Arrête les processus de traitement d'une session spécifique.

    IMPORTANT: Le paramètre session est OBLIGATOIRE pour garantir l'isolation
    entre utilisateurs. Chaque utilisateur ne peut arrêter que les processus
    de sa propre session.

    Args:
        session: Identifiant de la session (chemin relatif du dossier session)

    Returns:
        JSONResponse avec le statut de l'opération
    """
    if not session or not session.strip():
        logger.warning("stop_all_scripts called without session parameter")
        return JSONResponse(status_code=400, content={
            "error": "Session parameter is required",
            "details": "You must specify which session's processes to stop."
        })

    session = session.strip()

    # Valider que le dossier session existe
    session_path = os.path.join(UPLOAD_DIR, session)
    if not os.path.isdir(session_path):
        logger.warning(f"stop_all_scripts: session not found: {session}")
        return JSONResponse(status_code=404, content={
            "error": f"Session not found: {session}",
            "details": "The specified session directory does not exist."
        })

    try:
        # Utiliser le ProcessManager pour arrêter uniquement les processus de cette session
        # Exécuter dans un thread pool pour ne pas bloquer pendant l'attente SIGTERM → SIGKILL
        result = await asyncio.to_thread(process_manager.stop_session, session)
        logger.info(f"stop_all_scripts result for session '{session}': {result}")
        return JSONResponse(result)

    except Exception as e:
        logger.error(f"Exception in stop_all_scripts for session '{session}': {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={
            "error": "Failed to execute stop command.",
            "details": str(e)
        })


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
                # Construct absolute path to the script using RAGPY_DIR
                project_scripts_dir = os.path.join(RAGPY_DIR, "scripts")
                script_path = os.path.join(project_scripts_dir, "rad_dataframe.py")
                script_path = os.path.abspath(script_path)

                logger.info(f"Executing rad_dataframe.py with command: python3 {script_path} ...")

                # Use tracked subprocess for session-aware process management
                result = await run_tracked_subprocess(
                    cmd=[
                        "python3", script_path,
                        "--json", json_path,
                        "--dir", absolute_processing_path,
                        "--output", out_csv
                    ],
                    session_folder=path,
                    timeout=1800
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
        # Run rad_chunk.py with phase=initial (tracked for session-aware stop)
        result = await run_tracked_subprocess(
            cmd=[
                "python3", script_path,
                "--input", input_csv,
                "--output", absolute_processing_path,
                "--phase", "initial",
                "--model", model
            ],
            session_folder=path,
            timeout=1800
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
    from app.utils.sse_helpers import (
        run_subprocess_with_sse, create_combined_parser,
        parse_tqdm_progress, parse_dataframe_logs, parse_multilevel_progress
    )

    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    logger.info(f"SSE dataframe processing for path: '{path}', resolved to: '{absolute_processing_path}'")
    
    if not os.path.isdir(absolute_processing_path):
        async def error_generator():
            yield f"data: {{\"type\": \"error\", \"message\": \"Processing directory not found: {path}\"}}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")
    
    # Find Zotero JSON file (exclude pipeline-generated output files)
    # Pipeline generates: output_chunks.json, output_chunks_with_embeddings.json, etc.
    excluded_prefixes = ('output_', 'output.', 'generated_')
    try:
        json_files = [f for f in os.listdir(absolute_processing_path)
                      if f.lower().endswith('.json') and not f.startswith(excluded_prefixes)]
    except Exception as e:
        async def error_generator():
            yield f"data: {{\"type\": \"error\", \"message\": \"Failed to list directory: {str(e)}\"}}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")

    if not json_files:
        async def error_generator():
            yield f"data: {{\"type\": \"error\", \"message\": \"No Zotero JSON file found in directory (excluding output_*.json)\"}}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")
    
    json_path = os.path.join(absolute_processing_path, json_files[0])
    out_csv = os.path.join(absolute_processing_path, 'output.csv')
    
    # Build command
    script_path = os.path.join(RAGPY_DIR, "scripts", "rad_dataframe.py")
    cmd = ["python3", "-u", script_path, "--json", json_path, "--dir", absolute_processing_path, "--output", out_csv]
    
    logger.info(f"Executing: {' '.join(cmd)}")

    # Use combined parser: prioritize structured PROGRESS logs, then tqdm, then custom logs
    parser = create_combined_parser(parse_multilevel_progress, parse_tqdm_progress, parse_dataframe_logs)

    # Wrap generator to add document count on complete
    async def sse_with_count():
        async for event in run_subprocess_with_sse(cmd, parser, session_folder=path, timeout=1800):
            if '"type": "complete"' in event and '"message": "Process completed successfully"' in event:
                try:
                    if os.path.exists(out_csv):
                        df = pd.read_csv(out_csv, dtype=str, keep_default_na=False)
                        count = len(df)
                        yield f'data: {{"type": "complete", "message": "Process completed successfully", "count": {count}}}\n\n'
                        continue
                except Exception:
                    pass
            yield event

    return StreamingResponse(sse_with_count(), media_type="text/event-stream")


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
        result = await run_tracked_subprocess(
            cmd=[
                "python3", script_path,
                "--input", input_chunks,
                "--output", absolute_processing_path,
                "--phase", "dense"
            ],
            session_folder=path,
            timeout=1800
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
        result = await run_tracked_subprocess(
            cmd=[
                "python3", script_path,
                "--input", input_file,
                "--output", absolute_processing_path,
                "--phase", "sparse"
            ],
            session_folder=path,
            timeout=1800
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
        result = await run_tracked_subprocess(
            cmd=cmd,
            session_folder=path,
            timeout=3600  # 1h for large uploads
        )

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

    This implementation:
    1. Reads output.csv (contains texteocr for each document)
    2. Generates notes via LLM using build_note_html()
    3. Creates notes in Zotero via API (if credentials available)
    4. Streams real-time progress via SSE
    """
    from dotenv import load_dotenv
    from app.utils.llm_note_generator import (
        build_note_html_async, build_abstract_text_async, sentinel_in_html
    )
    from app.utils.zotero_client import (
        verify_api_key, create_child_note, check_note_exists,
        update_item_abstract, ZoteroAPIError
    )

    # Reload .env to pick up any credential changes
    load_dotenv(override=True)

    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, session))
    logger.info(f"Zotero notes generation for session: '{session}', extended: {extended_analysis}, model: {model}")

    # Parse extended_analysis flag
    use_extended = extended_analysis.lower() in ("true", "1", "yes")

    async def event_generator():
        try:
            # Check for output.csv (contains texteocr from pipeline)
            csv_path = os.path.join(absolute_processing_path, 'output.csv')
            if not os.path.exists(csv_path):
                yield f"data: {{\"type\": \"error\", \"message\": \"output.csv not found. Please complete the extraction step first.\"}}\n\n"
                return

            # Load CSV with documents
            try:
                df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
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

            # Check Zotero credentials
            zotero_api_key = os.getenv("ZOTERO_API_KEY", "")
            zotero_user_id = os.getenv("ZOTERO_USER_ID", "")
            zotero_group_id = os.getenv("ZOTERO_GROUP_ID", "")

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

            # Counters for summary
            created = 0
            exists = 0
            skipped = 0
            errors = 0

            # Storage for generated notes (if local mode or for backup)
            generated_notes = []

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
                    }

                    # Branch based on analysis mode
                    loop = asyncio.get_event_loop()
                    status = "created"

                    if use_extended:
                        # EXTENDED MODE: Generate HTML note and create child note
                        # Uses global semaphore for concurrency control
                        sentinel, note_html = await build_note_html_async(
                            metadata=metadata,
                            text_content=texteocr,
                            model=model,
                            use_llm=True,
                            extended_analysis=True
                        )

                        # Store generated note
                        generated_notes.append({
                            "item_key": item_key,
                            "title": title,
                            "sentinel": sentinel,
                            "note_html": note_html,
                            "mode": "extended"
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
                                    # Create the child note
                                    result = await loop.run_in_executor(
                                        None,
                                        lambda lt=library_type, lid=library_id, ik=item_key, nh=note_html, ak=zotero_api_key: create_child_note(
                                            library_type=lt,
                                            library_id=lid,
                                            item_key=ik,
                                            note_html=nh,
                                            tags=["ragpy-generated"],
                                            api_key=ak
                                        )
                                    )

                                    if result.get("success"):
                                        created += 1
                                        status = "created"
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
                            model=model
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

                except Exception as e:
                    errors += 1
                    error_msg = str(e).replace('"', '\\"').replace('\n', ' ')[:100]
                    logger.error(f"Error processing document {doc_num}: {e}", exc_info=True)
                    yield f"data: {{\"type\": \"progress\", \"current\": {doc_num}, \"total\": {total_items}, \"item\": \"{safe_title}\", \"status\": \"error\", \"message\": \"Error: {error_msg}\"}}\n\n"

                # Small delay to prevent overwhelming the API
                await asyncio.sleep(0.1)

            # Save generated notes to file for backup/review
            if generated_notes:
                notes_file = os.path.join(absolute_processing_path, 'generated_notes.json')
                try:
                    with open(notes_file, 'w', encoding='utf-8') as f:
                        json.dump(generated_notes, f, ensure_ascii=False, indent=2)
                    logger.info(f"Saved {len(generated_notes)} notes to {notes_file}")
                except Exception as e:
                    logger.warning(f"Could not save notes file: {e}")

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

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ============================================================================
# Optional SSE versions for better UX on long-running operations
# ============================================================================

@router.post("/initial_text_chunking_sse")
async def initial_text_chunking_sse(path: str = Form(...), model: str = Form(None)):
    """
    SSE version of initial_text_chunking for real-time progress updates.
    Uses multilevel progress parser for dual progress bars (documents + chunks).
    """
    from app.utils.sse_helpers import (
        run_subprocess_with_sse, create_combined_parser,
        parse_multilevel_progress, parse_tqdm_progress, parse_chunking_logs
    )

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

    async def sse_with_count():
        async for event in run_subprocess_with_sse(cmd, parser, session_folder=path, timeout=1800):
            # Intercept complete event to add count
            if '"type": "complete"' in event and '"message": "Process completed successfully"' in event:
                try:
                    if os.path.exists(output_file):
                        with open(output_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            count = len(data) if isinstance(data, list) else 0
                        yield f'data: {{"type": "complete", "message": "Process completed successfully", "count": {count}}}\n\n'
                        continue
                except Exception:
                    pass
            yield event

    return StreamingResponse(sse_with_count(), media_type="text/event-stream")


@router.post("/dense_embedding_generation_sse")
async def dense_embedding_generation_sse(path: str = Form(...)):
    """
    SSE version of dense_embedding_generation for real-time progress updates.
    Uses multilevel progress parser for dual progress bars (documents + chunks).
    """
    from app.utils.sse_helpers import (
        run_subprocess_with_sse, create_combined_parser,
        parse_multilevel_progress, parse_tqdm_progress, parse_chunking_logs
    )

    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    input_chunks = os.path.join(absolute_processing_path, 'output_chunks.json')

    if not os.path.exists(input_chunks):
        async def error_generator():
            yield f"data: {{\"type\": \"error\", \"message\": \"output_chunks.json not found\"}}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")

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

    async def sse_with_count():
        async for event in run_subprocess_with_sse(cmd, parser, session_folder=path, timeout=1800):
            if '"type": "complete"' in event and '"message": "Process completed successfully"' in event:
                try:
                    logger.info(f"Dense complete event received, checking for output file: {output_file}")
                    if os.path.exists(output_file):
                        with open(output_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            count = len(data) if isinstance(data, list) else 0
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
async def sparse_embedding_generation_sse(path: str = Form(...)):
    """
    SSE version of sparse_embedding_generation for real-time progress updates.
    Uses multilevel progress parser for chunk-level progress display.
    """
    from app.utils.sse_helpers import (
        run_subprocess_with_sse, create_combined_parser,
        parse_multilevel_progress, parse_tqdm_progress, parse_chunking_logs
    )

    absolute_processing_path = os.path.abspath(os.path.join(UPLOAD_DIR, path))
    input_file = os.path.join(absolute_processing_path, 'output_chunks_with_embeddings.json')

    if not os.path.exists(input_file):
        async def error_generator():
            yield f"data: {{\"type\": \"error\", \"message\": \"output_chunks_with_embeddings.json not found\"}}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")

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
        async for event in run_subprocess_with_sse(cmd, parser, session_folder=path, timeout=1800):
            if '"type": "complete"' in event and '"message": "Process completed successfully"' in event:
                try:
                    logger.info(f"Sparse complete event received, checking for output file: {output_file}")
                    if os.path.exists(output_file):
                        with open(output_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            count = len(data) if isinstance(data, list) else 0
                        logger.info(f"Sparse embedding file found with {count} chunks")
                        yield f'data: {{"type": "complete", "message": "Process completed successfully", "count": {count}}}\n\n'
                        continue
                    else:
                        logger.warning(f"Sparse output file not found: {output_file}")
                except Exception as e:
                    logger.error(f"Error reading sparse output file: {e}")
            yield event

    return StreamingResponse(sse_with_count(), media_type="text/event-stream")
