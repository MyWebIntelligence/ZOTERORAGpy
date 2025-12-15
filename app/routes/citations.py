"""
Citation Import Routes
======================

This module handles the import of citations from Publish or Perish JSON files to Zotero,
with LLM-based relevance filtering and user preview/validation workflow.

Key Features:
- Upload Publish or Perish JSON files with configuration
- LLM-based citation filtering with real-time SSE progress
- Preview interface with relevant/skipped citations
- Import selected citations to Zotero (create or update)
- Progress tracking via PipelineSession model

Workflow:
1. Upload JSON + config → Create session
2. Filter citations via LLM → Save preview.json
3. User validates selection → Import to Zotero
4. Return final stats (created/updated/errors)
"""
import os

# Default LLM model (uses OpenRouter format by default)
# Format "provider/model" → OpenRouter (e.g., google/gemini-2.5-flash)
# Format "model" → OpenAI direct (e.g., gpt-4o-mini)
DEFAULT_LLM_MODEL = os.getenv("OPENROUTER_DEFAULT_MODEL", "gpt-4o-mini")
import json
import uuid
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.core.config import UPLOAD_DIR
from app.database.session import get_db
from app.models.pipeline_session import PipelineSession, SessionStatus
from app.models.project import Project
from app.middleware.auth import get_current_active_user
from app.models.user import User
from app.core.credentials import get_credential_or_env, get_credential_error_message
from app.utils.publishorperish_parser import parse_pop_json
from app.utils.citation_fetcher import fetch_citation_content
from app.utils.citation_filter import filter_citation_with_llm
from app.utils.llm_note_generator import get_llm_semaphore
from app.utils.parallel_citation_processor import (
    process_citations_parallel,
    DEFAULT_BATCH_SIZE
)
from app.utils.zotero_client import (
    get_or_create_collection,
    create_or_update_item,
    upload_file_attachment,
    fetch_collection_items,
    ZoteroAPIError
)
from app.utils.pdf_downloader import download_pdf
from app.utils.state_persistence import (
    append_filtering_state,
    get_processed_indices,
    get_filtering_stats,
    rebuild_preview_from_state,
    append_import_state,
    get_imported_indices,
    get_import_stats,
    get_session_progress_summary,
    read_filtering_state,
    read_import_state
)

# Setup logger
logger = logging.getLogger(__name__)

# Main router for project-related endpoints
router = APIRouter(prefix="/api/projects", tags=["Citations"])

# Separate router for session-related endpoints (without /projects prefix)
session_router = APIRouter(prefix="/api", tags=["Citation Sessions"])


@router.post("/{project_id}/upload_pop_json")
async def upload_pop_json(
    project_id: int,
    json_file: UploadFile = File(...),
    collection_name: str = Form(...),
    collection_description: str = Form(""),
    model: str = Form(DEFAULT_LLM_MODEL),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Upload Publish or Perish JSON file and create citation import session.

    Args:
        project_id: ID of the target project
        json_file: Uploaded JSON file from Publish or Perish export
        collection_name: Target Zotero collection name (created if doesn't exist)
        collection_description: Description for the bibliography context
        model: LLM model to use for filtering (gpt-4o-mini, google/gemini-2.5-flash, etc.)
        db: Database session
        current_user: Authenticated user

    Returns:
        JSONResponse with:
            - session_id: Pipeline session ID
            - session_folder: Relative path to session folder
            - total_citations: Number of citations in the file
            - requires_confirmation: True if citation count > 100

    Raises:
        HTTPException 404: Project not found
        HTTPException 403: User doesn't have access to project
        HTTPException 400: Invalid JSON file format
    """
    # Verify project access
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Check user has access (owner or member)
    if project.owner_id != current_user.id:
        # TODO: Check project members table when implemented
        # For now, only owner can import citations
        raise HTTPException(
            status_code=403,
            detail="You don't have permission to import citations for this project"
        )

    # Generate unique session folder
    unique_id = str(uuid.uuid4().hex)[:8]
    session_folder_name = f"pop_{unique_id}_{project.name[:20]}"
    session_folder = os.path.join(UPLOAD_DIR, session_folder_name)
    os.makedirs(session_folder, exist_ok=True)
    logger.info(f"Created session folder: {session_folder}")

    # Save uploaded JSON file
    json_path = os.path.join(session_folder, "publishorperish.json")
    try:
        with open(json_path, "wb") as f:
            content = await json_file.read()
            f.write(content)
        logger.info(f"Saved uploaded JSON to: {json_path}")
    except Exception as e:
        logger.error(f"Failed to save uploaded JSON: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    # Parse and count citations
    try:
        citations = parse_pop_json(json_path)
        total_citations = len(citations)
        logger.info(f"Parsed {total_citations} citations from uploaded JSON")
    except ValueError as e:
        logger.error(f"Invalid JSON format: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid Publish or Perish JSON format: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to parse JSON: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse JSON file: {str(e)}")

    # Save configuration for later use
    config = {
        "collection_name": collection_name,
        "collection_description": collection_description,
        "model": model,
        "project_id": project_id,
        "project_name": project.name,
        "project_description": project.description or ""
    }
    config_path = os.path.join(session_folder, "config.json")
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved configuration to: {config_path}")
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save configuration: {str(e)}")

    # Create PipelineSession
    try:
        relative_session_folder = os.path.relpath(session_folder, UPLOAD_DIR)
        pipeline_session = PipelineSession(
            project_id=project_id,
            session_folder=relative_session_folder,
            original_filename=json_file.filename,
            source_type="pop_json",
            status=SessionStatus.CREATED,
            row_count=total_citations  # Total citations to process
        )
        db.add(pipeline_session)
        db.commit()
        db.refresh(pipeline_session)
        session_id = pipeline_session.id
        logger.info(f"Created PipelineSession {session_id} for citation import (project {project_id})")
    except Exception as e:
        logger.error(f"Failed to create PipelineSession: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

    # Check if confirmation required (> 100 citations)
    requires_confirmation = total_citations > 100

    return JSONResponse({
        "session_id": session_id,
        "session_folder": relative_session_folder,
        "total_citations": total_citations,
        "requires_confirmation": requires_confirmation
    })


@router.post("/{project_id}/filter_citations_sse")
async def filter_citations_sse(
    project_id: int,
    session_id: int = Form(...),
    batch_size: int = Form(DEFAULT_BATCH_SIZE),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Filter citations using LLM with Server-Sent Events for real-time progress.

    This endpoint performs parallel batch processing for LLM filtering with web content fetching.
    It streams progress in real-time and saves results to preview.json for user validation.

    Args:
        project_id: ID of the project
        session_id: Pipeline session ID
        batch_size: Number of citations to process in parallel per batch (1-50, default: 10)
        db: Database session
        current_user: Authenticated user

    Returns:
        StreamingResponse with SSE events:
            - init: {"type": "init", "total": N, "batch_size": M}
            - progress: {"type": "progress", "current": i, "status": "relevant"|"skipped", "title": "..."}
            - complete: {"type": "complete", "relevant": X, "skipped": Y}
            - error: {"type": "error", "message": "..."}

    Raises:
        HTTPException 404: Session or project not found
        HTTPException 403: User doesn't have access
    """
    async def event_generator():
        try:
            # Get session from database
            pipeline_session = db.query(PipelineSession).filter(
                PipelineSession.id == session_id,
                PipelineSession.project_id == project_id
            ).first()

            if not pipeline_session:
                yield f'data: {{"type": "error", "message": "Session not found"}}\n\n'
                return

            # Verify project access
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project or project.owner_id != current_user.id:
                yield f'data: {{"type": "error", "message": "Access denied"}}\n\n'
                return

            # Build absolute paths
            session_folder = os.path.join(UPLOAD_DIR, pipeline_session.session_folder)
            json_path = os.path.join(session_folder, "publishorperish.json")
            config_path = os.path.join(session_folder, "config.json")
            preview_path = os.path.join(session_folder, "preview.json")

            # Check files exist
            if not os.path.exists(json_path):
                yield f'data: {{"type": "error", "message": "Citations JSON not found"}}\n\n'
                return
            if not os.path.exists(config_path):
                yield f'data: {{"type": "error", "message": "Configuration not found"}}\n\n'
                return

            # Load citations
            try:
                citations = parse_pop_json(json_path)
                logger.info(f"Loaded {len(citations)} citations for filtering")
            except Exception as e:
                logger.error(f"Failed to parse citations: {e}")
                yield f'data: {{"type": "error", "message": "Failed to parse citations: {str(e)}"}}\n\n'
                return

            # Load config
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                yield f'data: {{"type": "error", "message": "Failed to load configuration: {str(e)}"}}\n\n'
                return

            # ========== RESUME SUPPORT ==========
            # Check for existing filtering progress (for crash recovery)
            processed_indices = get_processed_indices(session_folder)
            total_citations = len(citations)
            already_processed = len(processed_indices)
            is_resume = already_processed > 0

            if is_resume:
                logger.info(
                    f"RESUME MODE: Found {already_processed}/{total_citations} citations already processed"
                )
                # Rebuild preview_results from existing state
                preview_results = rebuild_preview_from_state(session_folder, citations)

                # Filter citations to only process remaining ones
                # Create list of (original_index, citation) for unprocessed
                remaining_citations = [
                    (idx, c) for idx, c in enumerate(citations)
                    if idx not in processed_indices
                ]
                logger.info(f"Remaining citations to process: {len(remaining_citations)}")
            else:
                remaining_citations = list(enumerate(citations))
                preview_results = {"relevant": [], "skipped": []}
            # ====================================

            # Get LLM credentials (user credentials with admin fallback to .env)
            openai_api_key = get_credential_or_env(current_user, "openai_api_key")
            openrouter_api_key = get_credential_or_env(current_user, "openrouter_api_key")

            # Validate credentials based on model type
            model_name = config.get("model", DEFAULT_LLM_MODEL)
            if "/" in model_name:  # OpenRouter model format (e.g., google/gemini-2.5-flash)
                if not openrouter_api_key:
                    error_msg = get_credential_error_message("openrouter_api_key")
                    yield f'data: {{"type": "error", "message": "{error_msg}", "credential_required": "openrouter_api_key"}}\n\n'
                    return
            else:  # OpenAI model
                if not openai_api_key:
                    error_msg = get_credential_error_message("openai_api_key")
                    yield f'data: {{"type": "error", "message": "{error_msg}", "credential_required": "openai_api_key"}}\n\n'
                    return

            # If all citations already processed, skip to completion
            if is_resume and len(remaining_citations) == 0:
                logger.info("All citations already processed, skipping to completion")
                pipeline_session.status = SessionStatus.FILTERED
                db.commit()
                relevant_count = len(preview_results["relevant"])
                skipped_count = len(preview_results["skipped"])
                yield f'data: {{"type": "init", "total": {total_citations}, "already_processed": {already_processed}, "remaining": 0, "batch_size": {batch_size}}}\n\n'
                yield f'data: {{"type": "complete", "relevant": {relevant_count}, "skipped": {skipped_count}, "resumed": true}}\n\n'
                return

            # Update session status
            pipeline_session.status = SessionStatus.FILTERING_CITATIONS
            db.commit()
            logger.info(f"Session {session_id} status → FILTERING_CITATIONS")

            # Validate and constrain batch_size (1-50)
            effective_batch_size = min(max(batch_size, 1), 50)

            logger.info(
                f"Starting parallel citation filtering: {len(remaining_citations)} citations "
                f"(total: {total_citations}, already processed: {already_processed}), "
                f"batch_size={effective_batch_size}"
            )

            # Extract just the citations for processing (keeping track of original indices)
            citations_to_process = [c for _, c in remaining_citations]
            index_mapping = {i: orig_idx for i, (orig_idx, _) in enumerate(remaining_citations)}

            # Process citations in parallel batches
            async for event_type, event_data in process_citations_parallel(
                citations=citations_to_process,
                config={
                    "project_name": config["project_name"],
                    "project_description": config["project_description"],
                    "collection_name": config["collection_name"],
                    "collection_description": config.get("collection_description", ""),
                    "model": config.get("model", DEFAULT_LLM_MODEL)
                },
                openai_api_key=openai_api_key,
                openrouter_api_key=openrouter_api_key,
                batch_size=effective_batch_size
            ):
                if event_type == "init":
                    # Include resume info in init event
                    yield f'data: {{"type": "init", "total": {total_citations}, "already_processed": {already_processed}, "remaining": {len(remaining_citations)}, "batch_size": {event_data["batch_size"]}}}\n\n'

                elif event_type == "progress":
                    # Map processor index to original index
                    processor_idx = event_data["current"] - 1
                    original_idx = index_mapping.get(processor_idx, processor_idx)

                    # Calculate overall progress (including already processed)
                    overall_current = already_processed + event_data["current"]

                    status = event_data["status"]
                    citation_dict = event_data["citation"]
                    filter_result = event_data.get("filter_result")
                    web_source = event_data.get("web_source", "none")

                    # Build state entry for progressive saving (use ORIGINAL index)
                    state_entry = {
                        "index": original_idx,
                        "relevant": status == "relevant",
                        "web_source": web_source
                    }

                    if status == "relevant" and filter_result:
                        zotero_data = filter_result.get("zotero_item", filter_result)
                        preview_results["relevant"].append({
                            "index": original_idx,
                            "citation": citation_dict,
                            "zotero_data": zotero_data,
                            "relevance_score": filter_result.get("relevance_score"),
                            "relevance_reason": filter_result.get("relevance_reason"),
                            "web_source": web_source
                        })
                        # Add to state entry for crash recovery
                        state_entry["relevance_score"] = filter_result.get("relevance_score")
                        state_entry["relevance_reason"] = filter_result.get("relevance_reason")
                        state_entry["zotero_data"] = zotero_data
                        logger.info(f"Citation {original_idx} marked as RELEVANT")
                    elif status == "error":
                        error_msg = event_data.get('error_message', 'Unknown')
                        preview_results["skipped"].append({
                            "index": original_idx,
                            "citation": citation_dict,
                            "reason": f"Error during filtering: {error_msg}"
                        })
                        state_entry["relevant"] = None  # None indicates error
                        state_entry["error"] = error_msg
                        logger.warning(f"Citation {original_idx} marked as ERROR")
                    else:
                        preview_results["skipped"].append({
                            "index": original_idx,
                            "citation": citation_dict,
                            "reason": "Not relevant (LLM decision)"
                        })
                        state_entry["reason"] = "Not relevant (LLM decision)"
                        logger.info(f"Citation {original_idx} marked as SKIPPED")

                    # PROGRESSIVE SAVE: Append state immediately (crash-safe)
                    try:
                        append_filtering_state(session_folder, state_entry)
                    except Exception as save_err:
                        logger.error(f"Failed to save progressive state for idx {original_idx}: {save_err}")

                    # Yield progress event with global progress counters
                    title_escaped = event_data["title"].replace('"', '\\"').replace('\n', ' ')
                    yield f'data: {{"type": "progress", "current": {overall_current}, "total": {total_citations}, "status": "{status}", "title": "{title_escaped}"}}\n\n'

                elif event_type == "complete":
                    # Don't yield complete here - we'll do it after saving
                    pass

            # Save preview results
            try:
                with open(preview_path, "w", encoding="utf-8") as f:
                    json.dump(preview_results, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved preview results to: {preview_path}")
            except Exception as e:
                logger.error(f"Failed to save preview: {e}")
                yield f'data: {{"type": "error", "message": "Failed to save preview results: {str(e)}"}}\n\n'
                return

            # Update session status to FILTERED (ready for import)
            pipeline_session.status = SessionStatus.FILTERED
            db.commit()
            logger.info(f"Session {session_id} status → FILTERED (filtering complete)")

            # Send completion event
            relevant_count = len(preview_results["relevant"])
            skipped_count = len(preview_results["skipped"])
            yield f'data: {{"type": "complete", "relevant": {relevant_count}, "skipped": {skipped_count}}}\n\n'

        except Exception as e:
            logger.error(f"Unexpected error in filter_citations_sse: {e}", exc_info=True)
            yield f'data: {{"type": "error", "message": "Unexpected error: {str(e)}"}}\n\n'

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/{project_id}/import_citations_sse")
async def import_citations_sse(
    project_id: int,
    session_id: int = Form(...),
    selected_indices: str = Form(...),  # JSON array "[0,1,2,...]"
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Import selected citations to Zotero with Server-Sent Events for progress tracking.

    This endpoint performs the actual import to Zotero, creating or updating items
    as needed. It uses the triple deduplication strategy (DOI → URL → Title).

    Args:
        project_id: ID of the project
        session_id: Pipeline session ID
        selected_indices: JSON array of selected citation indices from preview
        db: Database session
        current_user: Authenticated user

    Returns:
        StreamingResponse with SSE events:
            - init: {"type": "init", "total": N}
            - progress: {"type": "progress", "current": i, "action": "created"|"updated", "title": "..."}
            - error: {"type": "error", "index": i, "message": "..."}
            - complete: {"type": "complete", "created": X, "updated": Y, "errors": Z}

    Raises:
        HTTPException 404: Session or project not found
        HTTPException 403: User doesn't have access
    """
    async def event_generator():
        try:
            # Get session from database
            pipeline_session = db.query(PipelineSession).filter(
                PipelineSession.id == session_id,
                PipelineSession.project_id == project_id
            ).first()

            if not pipeline_session:
                yield f'data: {{"type": "error", "message": "Session not found"}}\n\n'
                return

            # Verify project access
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project or project.owner_id != current_user.id:
                yield f'data: {{"type": "error", "message": "Access denied"}}\n\n'
                return

            # Build absolute paths
            session_folder = os.path.join(UPLOAD_DIR, pipeline_session.session_folder)
            preview_path = os.path.join(session_folder, "preview.json")
            config_path = os.path.join(session_folder, "config.json")

            # Check files exist
            if not os.path.exists(preview_path):
                yield f'data: {{"type": "error", "message": "Preview results not found. Please run filtering first."}}\n\n'
                return
            if not os.path.exists(config_path):
                yield f'data: {{"type": "error", "message": "Configuration not found"}}\n\n'
                return

            # Load preview results
            try:
                with open(preview_path, "r", encoding="utf-8") as f:
                    preview = json.load(f)
                logger.info(f"Loaded preview with {len(preview['relevant'])} relevant citations")
            except Exception as e:
                logger.error(f"Failed to load preview: {e}")
                yield f'data: {{"type": "error", "message": "Failed to load preview results: {str(e)}"}}\n\n'
                return

            # Load config
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                yield f'data: {{"type": "error", "message": "Failed to load configuration: {str(e)}"}}\n\n'
                return

            # Parse selected indices
            try:
                selected = json.loads(selected_indices)
                if not isinstance(selected, list):
                    raise ValueError("selected_indices must be a JSON array")
            except Exception as e:
                logger.error(f"Invalid selected_indices format: {e}")
                yield f'data: {{"type": "error", "message": "Invalid selected indices format: {str(e)}"}}\n\n'
                return

            # Filter selected items
            try:
                to_import = [preview["relevant"][i] for i in selected]
                logger.info(f"Selected {len(to_import)} citations to import")
            except IndexError as e:
                logger.error(f"Invalid index in selection: {e}")
                yield f'data: {{"type": "error", "message": "Invalid citation index in selection"}}\n\n'
                return

            # ========== RESUME SUPPORT FOR IMPORT ==========
            # Check for existing import progress (for crash recovery)
            already_imported_indices = get_imported_indices(session_folder)
            total_to_import = len(to_import)
            already_imported_count = len([i for i in selected if i in already_imported_indices])
            is_import_resume = already_imported_count > 0

            if is_import_resume:
                logger.info(
                    f"IMPORT RESUME MODE: Found {already_imported_count}/{total_to_import} citations already imported"
                )
                # Load existing import stats
                existing_import_stats = get_import_stats(session_folder)
            else:
                existing_import_stats = {"created": 0, "updated": 0, "errors": 0}
            # ===============================================

            # Get Zotero credentials (user credentials with admin fallback to .env)
            zotero_api_key = get_credential_or_env(current_user, "zotero_api_key")
            zotero_user_id = get_credential_or_env(current_user, "zotero_user_id")
            zotero_group_id = get_credential_or_env(current_user, "zotero_group_id")

            # Determine library type based on available credentials
            library_type = "groups" if zotero_group_id else "users"
            library_id = zotero_group_id or zotero_user_id

            if not zotero_api_key:
                error_msg = get_credential_error_message("zotero_api_key")
                yield f'data: {{"type": "error", "message": "{error_msg}", "credential_required": "zotero_api_key"}}\n\n'
                return
            if not library_id:
                error_msg = get_credential_error_message("zotero_user_id")
                yield f'data: {{"type": "error", "message": "{error_msg}", "credential_required": "zotero_user_id"}}\n\n'
                return

            # Update session status
            pipeline_session.status = SessionStatus.IMPORTING_CITATIONS
            db.commit()
            logger.info(f"Session {session_id} status → IMPORTING_CITATIONS")

            # Get or create collection
            try:
                coll_result = get_or_create_collection(
                    library_type=library_type,
                    library_id=library_id,
                    collection_name=config["collection_name"],
                    description=config["collection_description"],
                    api_key=zotero_api_key
                )
                collection_key = coll_result["key"]
                logger.info(f"Using Zotero collection: {collection_key} (created={coll_result['created']})")
            except ZoteroAPIError as e:
                logger.error(f"Failed to get/create collection: {e}")
                yield f'data: {{"type": "error", "message": "Failed to access Zotero collection: {str(e)}"}}\n\n'
                return

            # Send init event with resume info
            total = len(to_import)
            yield f'data: {{"type": "init", "total": {total}, "already_imported": {already_imported_count}, "remaining": {total - already_imported_count}}}\n\n'

            # Initialize counters with existing stats if resuming
            created = existing_import_stats["created"]
            updated = existing_import_stats["updated"]
            errors = existing_import_stats["errors"]
            error_details = []

            # Import each citation (skip already imported)
            processed_count = already_imported_count
            for idx, item in enumerate(to_import):
                # Get the original index from selected array
                original_idx = selected[idx]

                # Skip if already imported (resume support)
                if original_idx in already_imported_indices:
                    logger.debug(f"Skipping already imported citation {original_idx}")
                    continue

                processed_count += 1
                try:
                    # Prepare Zotero item data
                    zotero_data = item["zotero_data"].copy()
                    zotero_data["collections"] = [collection_key]

                    # Clean up "N/A" values that LLM may have generated (Zotero rejects these)
                    na_patterns = {"N/A", "n/a", "N.A.", "n.a.", "NA", "na", "None", "null", "undefined", "-"}
                    fields_to_clean = ["DOI", "ISSN", "ISBN", "pages", "volume", "issue", "callNumber"]
                    for field in fields_to_clean:
                        if field in zotero_data and zotero_data[field] in na_patterns:
                            logger.debug(f"Cleaning N/A value from field {field}")
                            zotero_data[field] = ""

                    # DEBUG: Log the zotero_data being sent
                    logger.info(f"Citation {idx} itemType: {zotero_data.get('itemType')}, title: {zotero_data.get('title', '')[:50]}")

                    # Create or update item
                    result = create_or_update_item(
                        library_type=library_type,
                        library_id=library_id,
                        item_data=zotero_data,
                        api_key=zotero_api_key
                    )

                    if result["success"]:
                        if result["action"] == "created":
                            created += 1
                        else:
                            updated += 1

                        # PROGRESSIVE SAVE: Record successful import immediately
                        try:
                            append_import_state(session_folder, {
                                "index": original_idx,
                                "imported": True,
                                "zotero_key": result["item_key"],
                                "action": result["action"]
                            })
                        except Exception as save_err:
                            logger.error(f"Failed to save import state for idx {original_idx}: {save_err}")

                        # Yield progress event
                        title_escaped = item["citation"]["title"].replace('"', '\\"').replace('\n', ' ')[:50]
                        yield f'data: {{"type": "progress", "current": {processed_count}, "total": {total}, "action": "{result["action"]}", "title": "{title_escaped}"}}\n\n'
                        logger.info(f"Citation {original_idx} {result['action']}: {result['item_key']}")

                        # Attempt PDF attachment if fulltext_url available
                        fulltext_url = item["citation"].get("fulltext_url")
                        if fulltext_url:
                            try:
                                logger.info(f"Downloading PDF for citation {original_idx}: {str(fulltext_url)[:80]}...")

                                # Download PDF
                                pdf_result = await download_pdf(
                                    fulltext_url=str(fulltext_url),
                                    timeout=60,
                                    max_size_mb=50,
                                    convert_html=True
                                )

                                if pdf_result.success and pdf_result.pdf_bytes:
                                    # Upload to Zotero
                                    attach_result = upload_file_attachment(
                                        library_type=library_type,
                                        library_id=library_id,
                                        parent_item_key=result["item_key"],
                                        pdf_bytes=pdf_result.pdf_bytes,
                                        filename=pdf_result.filename,
                                        md5_hash=pdf_result.md5_hash,
                                        mtime=pdf_result.mtime,
                                        api_key=zotero_api_key,
                                        original_url=str(fulltext_url)
                                    )

                                    if attach_result["success"]:
                                        logger.info(f"PDF attached to item {result['item_key']}: {attach_result['attachment_key']}")
                                        yield f'data: {{"type": "attachment", "item_key": "{result["item_key"]}", "status": "success", "source": "{pdf_result.source}"}}\n\n'
                                    else:
                                        logger.warning(f"Failed to attach PDF: {attach_result['message']}")
                                        yield f'data: {{"type": "attachment", "item_key": "{result["item_key"]}", "status": "failed", "message": "{attach_result["message"][:100]}"}}\n\n'
                                else:
                                    logger.warning(f"PDF download failed: {pdf_result.error}")
                                    # Don't send error event for download failures (graceful degradation)

                            except Exception as e:
                                logger.warning(f"PDF attachment error (non-blocking): {e}")
                                # Continue without attachment - graceful degradation
                    else:
                        errors += 1
                        error_msg = result.get("message", "Unknown error")
                        error_details.append({"index": original_idx, "message": error_msg})

                        # PROGRESSIVE SAVE: Record failed import
                        try:
                            append_import_state(session_folder, {
                                "index": original_idx,
                                "imported": False,
                                "error": error_msg
                            })
                        except Exception as save_err:
                            logger.error(f"Failed to save import state for idx {original_idx}: {save_err}")

                        yield f'data: {{"type": "error", "index": {original_idx}, "message": "{error_msg}"}}\n\n'
                        logger.error(f"Failed to import citation {original_idx}: {error_msg}")

                except Exception as e:
                    errors += 1
                    error_msg = str(e).replace('"', '\\"')
                    error_details.append({"index": original_idx, "message": error_msg})

                    # PROGRESSIVE SAVE: Record exception
                    try:
                        append_import_state(session_folder, {
                            "index": original_idx,
                            "imported": False,
                            "error": error_msg
                        })
                    except Exception as save_err:
                        logger.error(f"Failed to save import state for idx {original_idx}: {save_err}")

                    yield f'data: {{"type": "error", "index": {original_idx}, "message": "{error_msg}"}}\n\n'
                    logger.error(f"Exception importing citation {original_idx}: {e}", exc_info=True)

            # Save error log if any
            if error_details:
                error_log_path = os.path.join(session_folder, "import_errors.json")
                try:
                    with open(error_log_path, "w", encoding="utf-8") as f:
                        json.dump(error_details, f, indent=2, ensure_ascii=False)
                    logger.info(f"Saved error log to: {error_log_path}")
                except Exception as e:
                    logger.warning(f"Failed to save error log: {e}")

            # Update session with final stats
            pipeline_session.status = SessionStatus.COMPLETED
            pipeline_session.row_count = total  # Total processed
            pipeline_session.chunk_count = created + updated  # Successfully imported
            db.commit()
            logger.info(f"Session {session_id} status → COMPLETED (created={created}, updated={updated}, errors={errors})")

            # Send completion event
            yield f'data: {{"type": "complete", "created": {created}, "updated": {updated}, "errors": {errors}}}\n\n'

        except Exception as e:
            logger.error(f"Unexpected error in import_citations_sse: {e}", exc_info=True)
            yield f'data: {{"type": "error", "message": "Unexpected error: {str(e)}"}}\n\n'

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/{project_id}/batch_import_citations_sse")
async def batch_import_citations_sse(
    project_id: int,
    session_id: int = Form(...),
    batch_size: int = Form(10),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Progressive batch import: Filter + Create Zotero + Attach PDF in batches.

    This endpoint processes citations in batches (default: 10), providing
    immediate feedback as items are created in Zotero. Unlike the traditional
    two-step process (preview → import), this creates items progressively.

    Flow per batch:
    1. Filter batch with LLM (~2 min for 10 citations)
    2. Create Zotero items (~30s)
    3. Download + attach PDFs (~1-2 min)
    4. Send progress event

    Benefits:
    - First items visible in Zotero after ~4 minutes (vs 3+ hours)
    - Graceful handling of interruptions (completed batches are preserved)
    - Lower memory usage (only current batch in memory)

    Args:
        project_id: ID of the project
        session_id: Pipeline session ID
        batch_size: Number of citations per batch (default: 10, max: 50)
        db: Database session
        current_user: Authenticated user

    Returns:
        StreamingResponse with SSE events:
            - init: {"type": "init", "total": N, "batch_size": M, "existing_items": X}
            - batch_start: {"type": "batch_start", "batch": i, "start_idx": X, "end_idx": Y}
            - filter_progress: {"type": "filter_progress", "current": i, "status": "relevant"|"skipped"}
            - import_progress: {"type": "import_progress", "current": i, "action": "created"|"updated"|"skipped"}
            - attachment: {"type": "attachment", "item_key": "...", "status": "success"|"failed"}
            - batch_complete: {"type": "batch_complete", "batch": i, "created": X, "skipped": Y}
            - complete: {"type": "complete", "total_created": X, "total_skipped": Y, "total_errors": Z}
    """
    async def event_generator():
        try:
            # Validate batch_size
            effective_batch_size = min(max(batch_size, 1), 50)

            # Get session from database
            pipeline_session = db.query(PipelineSession).filter(
                PipelineSession.id == session_id,
                PipelineSession.project_id == project_id
            ).first()

            if not pipeline_session:
                yield f'data: {{"type": "error", "message": "Session not found"}}\n\n'
                return

            # Verify project access
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project or project.owner_id != current_user.id:
                yield f'data: {{"type": "error", "message": "Access denied"}}\n\n'
                return

            # Build paths
            session_folder = os.path.join(UPLOAD_DIR, pipeline_session.session_folder)
            json_path = os.path.join(session_folder, "publishorperish.json")
            config_path = os.path.join(session_folder, "config.json")

            if not os.path.exists(json_path) or not os.path.exists(config_path):
                yield f'data: {{"type": "error", "message": "Required files not found"}}\n\n'
                return

            # Load citations and config
            try:
                citations = parse_pop_json(json_path)
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception as e:
                yield f'data: {{"type": "error", "message": "Failed to load data: {str(e)}"}}\n\n'
                return

            # Get credentials
            openai_api_key = get_credential_or_env(current_user, "openai_api_key")
            openrouter_api_key = get_credential_or_env(current_user, "openrouter_api_key")
            zotero_api_key = get_credential_or_env(current_user, "zotero_api_key")
            zotero_user_id = get_credential_or_env(current_user, "zotero_user_id")
            zotero_group_id = get_credential_or_env(current_user, "zotero_group_id")

            # Validate credentials
            model_name = config.get("model", DEFAULT_LLM_MODEL)
            if "/" in model_name and not openrouter_api_key:
                yield f'data: {{"type": "error", "message": "OpenRouter API key required", "credential_required": "openrouter_api_key"}}\n\n'
                return
            elif "/" not in model_name and not openai_api_key:
                yield f'data: {{"type": "error", "message": "OpenAI API key required", "credential_required": "openai_api_key"}}\n\n'
                return

            if not zotero_api_key:
                yield f'data: {{"type": "error", "message": "Zotero API key required", "credential_required": "zotero_api_key"}}\n\n'
                return

            library_type = "groups" if zotero_group_id else "users"
            library_id = zotero_group_id or zotero_user_id
            if not library_id:
                yield f'data: {{"type": "error", "message": "Zotero user/group ID required"}}\n\n'
                return

            # Update session status
            pipeline_session.status = SessionStatus.FILTERING_CITATIONS
            db.commit()

            # Get or create collection
            try:
                coll_result = get_or_create_collection(
                    library_type=library_type,
                    library_id=library_id,
                    collection_name=config["collection_name"],
                    description=config.get("collection_description", ""),
                    api_key=zotero_api_key
                )
                collection_key = coll_result["key"]
            except ZoteroAPIError as e:
                yield f'data: {{"type": "error", "message": "Failed to access Zotero collection: {str(e)}"}}\n\n'
                return

            # Fetch existing items for deduplication
            try:
                existing_items = fetch_collection_items(
                    library_type=library_type,
                    library_id=library_id,
                    collection_key=collection_key,
                    api_key=zotero_api_key
                )
                existing_dois = {item["DOI"].lower() for item in existing_items if item.get("DOI")}
                existing_titles = {item["title"].lower().strip() for item in existing_items if item.get("title")}
                logger.info(f"Found {len(existing_items)} existing items ({len(existing_dois)} with DOIs)")
            except Exception as e:
                logger.warning(f"Could not fetch existing items: {e}")
                existing_dois = set()
                existing_titles = set()

            # Send init event
            total = len(citations)
            num_batches = (total + effective_batch_size - 1) // effective_batch_size
            yield f'data: {{"type": "init", "total": {total}, "batch_size": {effective_batch_size}, "num_batches": {num_batches}, "existing_items": {len(existing_items)}}}\n\n'

            # Global counters
            total_created = 0
            total_updated = 0
            total_skipped = 0
            total_errors = 0
            semaphore = get_llm_semaphore()

            # Process in batches
            for batch_num in range(num_batches):
                start_idx = batch_num * effective_batch_size
                end_idx = min(start_idx + effective_batch_size, total)
                batch_citations = citations[start_idx:end_idx]

                yield f'data: {{"type": "batch_start", "batch": {batch_num + 1}, "start_idx": {start_idx}, "end_idx": {end_idx}}}\n\n'

                batch_created = 0
                batch_skipped = 0
                batch_errors = 0
                batch_results = []

                # Phase 1: Filter batch with LLM
                for local_idx, citation in enumerate(batch_citations):
                    global_idx = start_idx + local_idx
                    citation_dict = citation.model_dump(mode='json')

                    # Check for duplicates before LLM call
                    doi = citation_dict.get("doi", "")
                    title = citation_dict.get("title", "")

                    if doi and doi.lower() in existing_dois:
                        logger.info(f"Citation {global_idx} skipped (duplicate DOI: {doi})")
                        batch_skipped += 1
                        yield f'data: {{"type": "filter_progress", "current": {global_idx + 1}, "status": "skipped_duplicate"}}\n\n'
                        continue

                    if title and title.lower().strip() in existing_titles:
                        logger.info(f"Citation {global_idx} skipped (duplicate title)")
                        batch_skipped += 1
                        yield f'data: {{"type": "filter_progress", "current": {global_idx + 1}, "status": "skipped_duplicate"}}\n\n'
                        continue

                    # Fetch web content
                    try:
                        content, source = await fetch_citation_content(
                            article_url=str(citation.article_url) if citation.article_url else None,
                            fulltext_url=str(citation.fulltext_url) if citation.fulltext_url else None,
                            timeout=30,
                            max_chars=10000
                        )
                    except Exception as e:
                        content, source = "", "none"

                    # Filter with LLM
                    try:
                        async with semaphore:
                            filter_result = await filter_citation_with_llm(
                                citation=citation_dict,
                                web_content=content,
                                web_content_source=source,
                                project_name=config["project_name"],
                                project_description=config["project_description"],
                                collection_name=config["collection_name"],
                                collection_description=config.get("collection_description", ""),
                                model=model_name,
                                openai_api_key=openai_api_key,
                                openrouter_api_key=openrouter_api_key
                            )

                        if isinstance(filter_result, dict):
                            batch_results.append({
                                "global_idx": global_idx,
                                "citation": citation_dict,
                                "zotero_data": filter_result.get("zotero_item", filter_result)
                            })
                            yield f'data: {{"type": "filter_progress", "current": {global_idx + 1}, "status": "relevant"}}\n\n'
                        else:
                            batch_skipped += 1
                            yield f'data: {{"type": "filter_progress", "current": {global_idx + 1}, "status": "skipped"}}\n\n'

                    except Exception as e:
                        logger.error(f"LLM filtering failed for citation {global_idx}: {e}")
                        batch_errors += 1
                        yield f'data: {{"type": "filter_progress", "current": {global_idx + 1}, "status": "error"}}\n\n'

                # Phase 2: Create Zotero items for filtered results
                for item in batch_results:
                    global_idx = item["global_idx"]
                    zotero_data = item["zotero_data"].copy()
                    zotero_data["collections"] = [collection_key]

                    # Clean up "N/A" values that LLM may have generated (Zotero rejects these)
                    na_patterns = {"N/A", "n/a", "N.A.", "n.a.", "NA", "na", "None", "null", "undefined", "-"}
                    fields_to_clean = ["DOI", "ISSN", "ISBN", "pages", "volume", "issue", "callNumber"]
                    for field in fields_to_clean:
                        if field in zotero_data and zotero_data[field] in na_patterns:
                            logger.debug(f"Cleaning N/A value from field {field}")
                            zotero_data[field] = ""

                    try:
                        result = create_or_update_item(
                            library_type=library_type,
                            library_id=library_id,
                            item_data=zotero_data,
                            api_key=zotero_api_key
                        )

                        if result["success"]:
                            # Extract item key (can be string or dict)
                            raw_key = result["item_key"]
                            item_key = raw_key["key"] if isinstance(raw_key, dict) else raw_key

                            if result["action"] == "created":
                                batch_created += 1
                                # Add to dedup sets
                                if zotero_data.get("DOI"):
                                    existing_dois.add(zotero_data["DOI"].lower())
                                if zotero_data.get("title"):
                                    existing_titles.add(zotero_data["title"].lower().strip())
                            else:
                                batch_created += 1  # Updated counts as created for stats

                            yield f'data: {{"type": "import_progress", "current": {global_idx + 1}, "action": "{result["action"]}", "item_key": "{item_key}"}}\n\n'

                            # Phase 3: Attach PDF if available
                            fulltext_url = item["citation"].get("fulltext_url")
                            if fulltext_url:
                                try:
                                    pdf_result = await download_pdf(
                                        fulltext_url=str(fulltext_url),
                                        timeout=60,
                                        max_size_mb=50,
                                        convert_html=True
                                    )

                                    if pdf_result.success and pdf_result.pdf_bytes:
                                        attach_result = upload_file_attachment(
                                            library_type=library_type,
                                            library_id=library_id,
                                            parent_item_key=item_key,
                                            pdf_bytes=pdf_result.pdf_bytes,
                                            filename=pdf_result.filename,
                                            md5_hash=pdf_result.md5_hash,
                                            mtime=pdf_result.mtime,
                                            api_key=zotero_api_key,
                                            original_url=str(fulltext_url)
                                        )

                                        if attach_result["success"]:
                                            yield f'data: {{"type": "attachment", "item_key": "{item_key}", "status": "success"}}\n\n'
                                        else:
                                            yield f'data: {{"type": "attachment", "item_key": "{item_key}", "status": "failed", "error": "{attach_result.get("message", "Unknown error")[:100]}"}}\n\n'
                                    else:
                                        logger.debug(f"PDF download failed for {fulltext_url[:50]}: {pdf_result.error}")
                                except Exception as e:
                                    logger.warning(f"PDF attachment failed: {e}")
                        else:
                            batch_errors += 1
                            yield f'data: {{"type": "import_progress", "current": {global_idx + 1}, "action": "error"}}\n\n'

                    except Exception as e:
                        logger.error(f"Failed to create Zotero item: {e}")
                        batch_errors += 1

                # Update totals
                total_created += batch_created
                total_skipped += batch_skipped
                total_errors += batch_errors

                # Batch complete event
                yield f'data: {{"type": "batch_complete", "batch": {batch_num + 1}, "created": {batch_created}, "skipped": {batch_skipped}, "errors": {batch_errors}}}\n\n'

            # Update session status
            pipeline_session.status = SessionStatus.COMPLETED
            pipeline_session.row_count = total
            pipeline_session.chunk_count = total_created
            db.commit()

            # Final completion event
            yield f'data: {{"type": "complete", "total_created": {total_created}, "total_skipped": {total_skipped}, "total_errors": {total_errors}}}\n\n'

        except Exception as e:
            logger.error(f"Unexpected error in batch_import_citations_sse: {e}", exc_info=True)
            yield f'data: {{"type": "error", "message": "Unexpected error: {str(e)}"}}\n\n'

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@session_router.get("/sessions/{session_id}/state")
async def get_session_state(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get complete session state for resume capability.

    This endpoint returns the full state of a citation import session,
    including filtering and import progress. Used to detect if a session
    can be resumed and to pre-fill the interface with saved configuration.

    Args:
        session_id: Pipeline session ID
        db: Database session
        current_user: Authenticated user

    Returns:
        JSONResponse with:
            {
                "session_id": int,
                "status": str,
                "config": {...},
                "filtering": {
                    "progress": int,
                    "total": int,
                    "relevant": int,
                    "skipped": int,
                    "can_resume": bool
                },
                "import": {
                    "progress": int,
                    "total": int,
                    "created": int,
                    "updated": int,
                    "can_resume": bool
                }
            }

    Raises:
        HTTPException 404: Session not found
        HTTPException 403: User doesn't have access
    """
    # Get session from database
    pipeline_session = db.query(PipelineSession).filter(
        PipelineSession.id == session_id
    ).first()

    if not pipeline_session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Verify project access
    project = db.query(Project).filter(Project.id == pipeline_session.project_id).first()
    if not project or project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Build absolute paths
    session_folder = os.path.join(UPLOAD_DIR, pipeline_session.session_folder)

    # Load config
    config_path = os.path.join(session_folder, "config.json")
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    # Load total citations count
    json_path = os.path.join(session_folder, "publishorperish.json")
    total_citations = 0
    if os.path.exists(json_path):
        try:
            citations = parse_pop_json(json_path)
            total_citations = len(citations)
        except Exception as e:
            logger.warning(f"Failed to count citations: {e}")

    # Get filtering stats from progressive state
    filtering_stats = get_filtering_stats(session_folder)

    # Get import stats from progressive state
    import_stats = get_import_stats(session_folder)

    # Determine if can resume filtering
    filtering_in_progress = pipeline_session.status in [
        SessionStatus.FILTERING_CITATIONS,
        SessionStatus.FILTERING_PAUSED,
        SessionStatus.CREATED
    ]
    filtering_incomplete = filtering_stats["total"] < total_citations
    can_resume_filtering = filtering_in_progress and filtering_incomplete and total_citations > 0

    # Determine if can resume import
    import_in_progress = pipeline_session.status in [
        SessionStatus.IMPORTING_CITATIONS,
        SessionStatus.IMPORTING_PAUSED,
        SessionStatus.FILTERED
    ]
    import_total = filtering_stats["relevant"]  # Only relevant citations are importable
    import_incomplete = import_stats["total"] < import_total
    can_resume_import = import_in_progress and import_incomplete and import_total > 0

    return JSONResponse({
        "session_id": session_id,
        "status": pipeline_session.status.value if pipeline_session.status else None,
        "config": config,
        "filtering": {
            "progress": filtering_stats["total"],
            "total": total_citations,
            "relevant": filtering_stats["relevant"],
            "skipped": filtering_stats["skipped"],
            "errors": filtering_stats["errors"],
            "can_resume": can_resume_filtering
        },
        "import": {
            "progress": import_stats["total"],
            "total": import_total,
            "created": import_stats["created"],
            "updated": import_stats["updated"],
            "errors": import_stats["errors"],
            "can_resume": can_resume_import
        }
    })


@session_router.get("/sessions/{session_id}/preview")
async def get_preview_results(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get preview results (relevant and skipped citations) for user validation.

    This endpoint retrieves the preview.json file created during the filtering step,
    allowing the frontend to display the results for user review.

    Args:
        session_id: Pipeline session ID
        db: Database session
        current_user: Authenticated user

    Returns:
        JSONResponse with:
            {
                "relevant": [
                    {
                        "index": int,
                        "citation": {...},
                        "zotero_data": {...},
                        "web_source": "pdf"|"html"|"none"
                    },
                    ...
                ],
                "skipped": [
                    {
                        "index": int,
                        "citation": {...},
                        "reason": str
                    },
                    ...
                ]
            }

    Raises:
        HTTPException 404: Session not found or preview not available
        HTTPException 403: User doesn't have access to this session
    """
    # Get session from database
    pipeline_session = db.query(PipelineSession).filter(
        PipelineSession.id == session_id
    ).first()

    if not pipeline_session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Verify project access
    project = db.query(Project).filter(Project.id == pipeline_session.project_id).first()
    if not project or project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Build path to preview file
    session_folder = os.path.join(UPLOAD_DIR, pipeline_session.session_folder)
    preview_path = os.path.join(session_folder, "preview.json")

    # Check if preview exists
    if not os.path.exists(preview_path):
        raise HTTPException(
            status_code=404,
            detail="Preview not available. Please run the filtering step first."
        )

    # Load and return preview results
    try:
        with open(preview_path, "r", encoding="utf-8") as f:
            preview_results = json.load(f)
        logger.info(f"Loaded preview for session {session_id}: {len(preview_results['relevant'])} relevant, {len(preview_results['skipped'])} skipped")
        return JSONResponse(preview_results)
    except Exception as e:
        logger.error(f"Failed to load preview results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load preview results: {str(e)}")
